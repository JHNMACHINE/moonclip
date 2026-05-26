use thiserror::Error;

#[derive(Error, Debug)]
pub enum RevolverError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Checkpoint not found: {0}")]
    NotFound(String),

    #[error("Integrity check failed: expected {expected}, got {actual}")]
    IntegrityError { expected: String, actual: String },

    #[error("Delta error: {0}")]
    Delta(String),

    #[error("Storage backend error: {0}")]
    Storage(String),

    #[error("Invalid configuration: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, RevolverError>;

impl From<RevolverError> for pyo3::PyErr {
    fn from(err: RevolverError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}
