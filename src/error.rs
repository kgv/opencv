use thiserror::Error;

/// Result
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Error
#[derive(Debug, Error)]
pub enum Error {
    #[error("not found error")]
    NotFound,
    #[error(transparent)]
    OpenCV(#[from] opencv::Error),
}
