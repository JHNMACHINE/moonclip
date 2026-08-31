use thiserror::Error;

#[derive(Error, Debug)]
pub enum MoonclipError {
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

pub type Result<T> = std::result::Result<T, MoonclipError>;

/// The message out of a `catch_unwind` payload, for the three background
/// threads that turn a panic into an error rather than dying of it.
///
/// A payload is `&str` for `panic!("literal")` and `String` for a formatted
/// one; anything else is a `panic_any` with a type this crate cannot name, and
/// saying so beats printing nothing.
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown payload".into())
}

#[cfg(feature = "python")]
impl From<MoonclipError> for pyo3::PyErr {
    fn from(err: MoonclipError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}
