use num::{FromPrimitive, Num, ToPrimitive};
use opencv::{
    core::{
        bitwise_not_def, greater_than_mat_f64, kmeans_def, less_than_mat_f64, mean, min_max_loc,
        no_array, split, subtract_def, Mat, MatExpr, MatExprTraitConst, MatTraitConst,
        MatTraitConstManual, Moments, Point, Point2d, Point2f, Point2i, Point_, Rect, Rect_,
        RotatedRect, Scalar, Size, TermCriteria, ToInputArray, ToInputOutputArray, VecN, Vector,
        _InputArrayTraitConst, BORDER_CONSTANT, CV_8UC1,
    },
    imgcodecs::{imread, imwrite_def},
    imgproc::{
        arc_length, bounding_rect, canny_def, circle, connected_components, contour_area_def,
        cvt_color_def, dilate, distance_transform_def, draw_contours, erode, find_contours_def,
        gaussian_blur_def, line_def, median_blur, min_area_rect, min_enclosing_circle, moments,
        morphology_default_border_value, morphology_ex, point_polygon_test, put_text_def,
        rectangle_def, threshold, watershed, DIST_L2, DIST_MASK_5, FILLED, LINE_8,
    },
    Result,
};
use std::path::Path;

// pub fn color(index: i32) -> VecN<f64, 4> {
//     let golden_ratio: f64 = (5.0_f64.sqrt() - 1.0) / 2.0; // 0.61803398875
//     let h = index as f64 * golden_ratio;
//     let rgba = Srgb::from_color(Hsv::new(h, 0.85, 0.5)).into_format::<u8>();
//     VecN([rgba.blue as _, rgba.green as _, rgba.red as _, 255.0])
// }

/// Contour
pub trait Contour {
    fn area(&self) -> Result<f64>;
    fn bounding_rectangle(&self) -> Result<Rect>;
    fn incircle(&self, point: Point_<impl ToPrimitive>) -> Result<Circle>;
    fn max_incircle(&self) -> Result<Circle>;
    fn min_circumcircle(&self) -> Result<Circle>;
    fn moments(&self, binary_image: bool) -> Result<Moments>;
    fn perimeter(&self, closed: bool) -> Result<f64>;
    fn rotated_rectangle(&self) -> Result<RotatedRect>;
}

impl Contour for Mat {
    fn area(&self) -> Result<f64> {
        Ok(contour_area_def(self)?)
    }

    fn bounding_rectangle(&self) -> Result<Rect> {
        Ok(bounding_rect(self)?)
    }

    fn incircle(&self, point: Point_<impl ToPrimitive>) -> Result<Circle> {
        let center = point.to().unwrap();
        let radius = point_polygon_test(self, center, true)?;
        Ok(Circle {
            center,
            radius: radius as _,
        })
    }

    fn max_incircle(&self) -> Result<Circle> {
        const WHITE: Scalar = Scalar::all(255.0);

        let mut width = 0;
        let mut height = 0;
        for (_, value) in self.iter::<VecN<i32, 2>>()? {
            width = width.max(value[0]);
            height = height.max(value[1]);
        }
        let mut mask = Mat::zeros_size(Size::new(width, height), CV_8UC1)?.to_mat()?;
        mask.draw_contour(self, WHITE, FILLED)?;
        let distance_transform = mask.distance_transform(DIST_L2, DIST_MASK_5)?;
        let (center, radius) = distance_transform.max(Some(&mask))?;
        Ok(Circle {
            center: center.to().unwrap(),
            radius: radius as _,
        })
    }

    fn min_circumcircle(&self) -> Result<Circle> {
        let mut center = Point2f::default();
        let mut radius = 0.0;
        min_enclosing_circle(self, &mut center, &mut radius)?;
        Ok(Circle { center, radius })
    }

    fn moments(&self, binary_image: bool) -> Result<Moments> {
        Ok(moments(self, binary_image)?)
    }

    fn perimeter(&self, closed: bool) -> Result<f64> {
        Ok(arc_length(self, closed)?)
    }

    fn rotated_rectangle(&self) -> Result<RotatedRect> {
        Ok(min_area_rect(self)?)
    }
}

/// Draw
pub trait Draw {
    fn draw_circle(
        &mut self,
        center: Point_<impl ToPrimitive>,
        radius: impl ToPrimitive,
        color: Scalar,
        thickness: i32,
    ) -> Result<()>;
    fn draw_contour(
        &mut self,
        contour: &impl ToInputArray,
        color: Scalar,
        thickness: i32,
    ) -> Result<()>;
    fn draw_contours(
        &mut self,
        contours: &impl ToInputArray,
        color: Scalar,
        thickness: i32,
    ) -> Result<()>;
    fn draw_rectangle(&mut self, rectangle: Rect, color: Scalar) -> Result<()>;
    fn draw_rotated_rectangle(
        &mut self,
        rotated_rectangle: RotatedRect,
        color: Scalar,
    ) -> Result<()>;
    fn draw_text(
        &mut self,
        text: impl AsRef<str>,
        origin: Point_<impl ToPrimitive>,
        font_face: i32,
        font_scale: f64,
        color: Scalar,
    ) -> Result<()>;
}

impl<T: ToInputOutputArray> Draw for T {
    fn draw_circle(
        &mut self,
        center: Point_<impl ToPrimitive>,
        radius: impl ToPrimitive,
        color: Scalar,
        thickness: i32,
    ) -> Result<()> {
        Ok(circle(
            self,
            center.to().unwrap(),
            radius.to_i32().unwrap(),
            color,
            thickness,
            LINE_8,
            0,
        )?)
    }

    fn draw_contour(
        &mut self,
        contour: &impl ToInputArray,
        color: Scalar,
        thickness: i32,
    ) -> Result<()> {
        let input_array = contour.input_array()?;
        assert!(input_array.is_mat()?);
        let contours = Vector::<Mat>::from_elem(input_array.get_mat_def()?, 1);
        draw_contours(
            self,
            &contours,
            0,
            color,
            thickness,
            LINE_8,
            &no_array(),
            i32::MAX,
            Point::new(0, 0),
        )?;
        Ok(())
    }

    fn draw_contours(
        &mut self,
        contours: &impl ToInputArray,
        color: Scalar,
        thickness: i32,
    ) -> Result<()> {
        assert!(contours.input_array()?.is_mat_vector()?);
        draw_contours(
            self,
            contours,
            -1,
            color,
            thickness,
            LINE_8,
            &no_array(),
            i32::MAX,
            Point::new(0, 0),
        )?;
        Ok(())
    }

    fn draw_rectangle(&mut self, rectangle: Rect, color: Scalar) -> Result<()> {
        Ok(rectangle_def(self, rectangle, color)?)
    }

    fn draw_rotated_rectangle(
        &mut self,
        rotated_rectangle: RotatedRect,
        color: Scalar,
    ) -> Result<()> {
        let mut points = Vector::with_capacity(4);
        rotated_rectangle.points_vec(&mut points)?;
        for index in 0..4 {
            line_def(
                self,
                points.get(index)?.to().unwrap(),
                points.get((index + 1) % 4)?.to().unwrap(),
                color,
            )?;
        }
        Ok(())
    }

    fn draw_text(
        &mut self,
        text: impl AsRef<str>,
        origin: Point_<impl ToPrimitive>,
        font_face: i32,
        font_scale: f64,
        color: Scalar,
    ) -> Result<()> {
        Ok(put_text_def(
            self,
            text.as_ref(),
            origin.to().unwrap(),
            font_face,
            font_scale,
            color,
        )?)
    }
}

pub trait ToInputArrayExt {
    fn bitwise_not(&self) -> Result<Mat>;
    fn canny(&self, threshold1: f64, threshold2: f64) -> Result<Mat>;
    fn connected_components(&self, connectivity: i32, ltype: i32) -> Result<Mat>;
    fn convert_color(&self, code: i32) -> Result<Mat>;
    fn dilate(&self, kernel: &impl ToInputArray, iterations: i32) -> Result<Mat>;
    fn distance_transform(&self, distance_type: i32, mask_size: i32) -> Result<Mat>;
    fn erode(&self, kernel: &impl ToInputArray, iterations: i32) -> Result<Mat>;
    fn find_contours(&self, mode: i32, method: i32) -> Result<Vector<Mat>>;
    fn gaussian_blur(&self, ksize: Size, sigma_x: f64) -> Result<Mat>;
    fn kmeans(
        &self,
        k: i32,
        criteria: TermCriteria,
        attempts: i32,
        flags: i32,
    ) -> Result<(f64, Mat)>;
    fn max(&self, mask: Option<&impl ToInputArray>) -> Result<(Point2i, f64)>;
    fn mean(&self, mask: &impl ToInputArray) -> Result<VecN<f64, 4>>;
    fn median_blur(&self, ksize: i32) -> Result<Mat>;
    fn min(&self, mask: Option<&impl ToInputArray>) -> Result<(Point2i, f64)>;
    fn morphology(&self, op: i32, kernel: &impl ToInputArray, iterations: i32) -> Result<Mat>;
    fn read(filename: impl AsRef<Path>, flags: i32) -> Result<Mat>;
    fn split(&self) -> Result<Vector<Mat>>;
    fn subtract(&self, rhs: &impl ToInputArray) -> Result<Mat>;
    fn threshold(&self, thresh: f64, maxval: f64, typ: i32) -> Result<Mat>;
    fn watershed(&self, markers: &mut impl ToInputOutputArray) -> Result<()>;
    fn write(&self, path: impl AsRef<Path>) -> Result<()>;
}

impl<T: ToInputArray> ToInputArrayExt for T {
    fn bitwise_not(&self) -> Result<Mat> {
        let mut dst = Mat::default();
        bitwise_not_def(self, &mut dst)?;
        Ok(dst)
    }

    fn canny(&self, threshold1: f64, threshold2: f64) -> Result<Mat> {
        let mut edges = Mat::default();
        canny_def(self, &mut edges, threshold1, threshold2)?;
        Ok(edges)
    }

    fn connected_components(&self, connectivity: i32, ltype: i32) -> Result<Mat> {
        let mut labels = Mat::default();
        connected_components(self, &mut labels, connectivity, ltype)?;
        Ok(labels)
    }

    fn convert_color(&self, code: i32) -> Result<Mat> {
        let mut dst = Mat::default();
        cvt_color_def(self, &mut dst, code)?;
        Ok(dst)
    }

    fn dilate(&self, kernel: &impl ToInputArray, iterations: i32) -> Result<Mat> {
        let mut dst = Mat::default();
        dilate(
            self,
            &mut dst,
            kernel,
            Default::default(),
            iterations,
            Default::default(),
            Default::default(),
        )?;
        Ok(dst)
    }

    fn distance_transform(&self, distance_type: i32, mask_size: i32) -> Result<Mat> {
        let mut dst = Mat::default();
        distance_transform_def(self, &mut dst, distance_type, mask_size)?;
        Ok(dst)
    }

    fn erode(&self, kernel: &impl ToInputArray, iterations: i32) -> Result<Mat> {
        let mut dst = Mat::default();
        erode(
            self,
            &mut dst,
            kernel,
            Point::new(-1, -1),
            iterations,
            BORDER_CONSTANT,
            morphology_default_border_value()?,
        )?;
        Ok(dst)
    }

    fn find_contours(&self, mode: i32, method: i32) -> Result<Vector<Mat>> {
        let mut contours = Vector::<Mat>::new();
        find_contours_def(self, &mut contours, mode, method)?;
        Ok(contours)
    }

    fn gaussian_blur(&self, ksize: Size, sigma_x: f64) -> Result<Mat> {
        let mut dst = Mat::default();
        gaussian_blur_def(self, &mut dst, ksize, sigma_x)?;
        Ok(dst)
    }

    fn kmeans(
        &self,
        k: i32,
        criteria: TermCriteria,
        attempts: i32,
        flags: i32,
    ) -> Result<(f64, Mat)> {
        let mut best_labels = Mat::default();
        let compactness = kmeans_def(self, k, &mut best_labels, criteria, attempts, flags)?;
        Ok((compactness, best_labels))
    }

    fn max(&self, mask: Option<&impl ToInputArray>) -> Result<(Point2i, f64)> {
        let mut point = Point2i::default();
        let mut value = 0.0;
        match mask {
            Some(mask) => min_max_loc(self, None, Some(&mut value), None, Some(&mut point), mask)?,
            None => min_max_loc(
                self,
                None,
                Some(&mut value),
                None,
                Some(&mut point),
                &no_array(),
            )?,
        }
        Ok((point, value))
    }

    fn mean(&self, mask: &impl ToInputArray) -> Result<VecN<f64, 4>> {
        Ok(mean(self, mask)?)
    }

    fn min(&self, mask: Option<&impl ToInputArray>) -> Result<(Point2i, f64)> {
        let mut point = Point2i::default();
        let mut value = 0.0;
        match mask {
            Some(mask) => min_max_loc(self, Some(&mut value), None, Some(&mut point), None, mask)?,
            None => min_max_loc(
                self,
                Some(&mut value),
                None,
                Some(&mut point),
                None,
                &no_array(),
            )?,
        }
        Ok((point, value))
    }

    fn median_blur(&self, ksize: i32) -> Result<Mat> {
        let mut dst = Mat::default();
        median_blur(self, &mut dst, ksize)?;
        Ok(dst)
    }

    // * To remove any small white noises in the image, we can use morphological opening.
    // * To remove any small holes in the object, we can use morphological closing.
    fn morphology(&self, op: i32, kernel: &impl ToInputArray, iterations: i32) -> Result<Mat> {
        let mut dst = Mat::default();
        morphology_ex(
            self,
            &mut dst,
            op,
            kernel,
            Default::default(),
            iterations,
            Default::default(),
            Default::default(),
        )?;
        Ok(dst)
    }

    fn read(filename: impl AsRef<Path>, flags: i32) -> Result<Mat> {
        Ok(imread(&filename.as_ref().to_string_lossy(), flags)?)
    }

    fn split(&self) -> Result<Vector<Mat>> {
        let mut mv = Vector::<Mat>::with_capacity(3);
        split(self, &mut mv)?;
        Ok(mv)
    }

    fn subtract(&self, rhs: &impl ToInputArray) -> Result<Mat> {
        let mut dst = Mat::default();
        subtract_def(self, rhs, &mut dst)?;
        Ok(dst)
    }

    fn threshold(&self, thresh: f64, maxval: f64, r#type: i32) -> Result<Mat> {
        let mut dst = Mat::default();
        threshold(self, &mut dst, thresh, maxval, r#type)?;
        Ok(dst)
    }

    fn watershed(&self, markers: &mut impl ToInputOutputArray) -> Result<()> {
        watershed(self, markers)?;
        Ok(())
    }

    fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        imwrite_def(&path.as_ref().to_string_lossy(), self)?;
        Ok(())
    }
}

/// [`Mat`]
pub trait MatExt {
    fn to(&self, rtype: i32) -> Result<Mat>;
}

impl MatExt for Mat {
    fn to(&self, rtype: i32) -> Result<Mat> {
        let mut m = Mat::default();
        self.convert_to_def(&mut m, rtype)?;
        Ok(m)
    }
}

/// [`MatTraitConst`]
pub trait MatTraitConstExt {
    fn greater_than(&self, s: f64) -> Result<MatExpr>;
    fn less_than(&self, s: f64) -> Result<MatExpr>;
}

impl<T: MatTraitConst> MatTraitConstExt for T {
    fn greater_than(&self, s: f64) -> Result<MatExpr> {
        Ok(greater_than_mat_f64(self, s)?)
    }

    fn less_than(&self, s: f64) -> Result<MatExpr> {
        Ok(less_than_mat_f64(self, s)?)
    }
}

/// [`Moments`]
pub trait MomentsExt {
    fn centroid(&self) -> Point2d;
}

impl MomentsExt for Moments {
    fn centroid(&self) -> Point2d {
        Point2d::new(self.m10 / self.m00, self.m01 / self.m00)
    }
}

/// [`Rect_`]
pub trait RectExt<T> {
    fn center(self) -> Point_<T>
    where
        T: FromPrimitive;
    fn tr(self) -> Point_<T>;
}

impl<T: Num> RectExt<T> for Rect_<T> {
    fn center(self) -> Point_<T>
    where
        T: FromPrimitive,
    {
        Point_::new(
            self.x + self.width / T::from_u8(2).unwrap(),
            self.y + self.height / T::from_u8(2).unwrap(),
        )
    }

    fn tr(self) -> Point_<T> {
        Point_::new(self.x + self.width, self.y)
    }
}

/// Circle
#[derive(Clone, Copy, Debug, Default)]
pub struct Circle {
    pub center: Point2f,
    pub radius: f32,
}

// mod error {
//     use thiserror::Error;

//     /// Result
//     pub type Result<T, E = Error> = std::result::Result<T, E>;

//     /// Error
//     #[derive(Debug, Error)]
//     pub enum Error {
//         #[error("cast error")]
//         Cast,
//         #[error(transparent)]
//         Disconnect(#[from] opencv::Error),
//     }
// }
