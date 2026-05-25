use thiserror::Error;

#[derive(Debug, Error)]
pub enum DagMlError {
    #[error("invalid identifier `{value}`: {reason}")]
    InvalidIdentifier { value: String, reason: &'static str },

    #[error("graph validation failed: {0}")]
    GraphValidation(String),

    #[error("OOF validation failed: {0}")]
    OofValidation(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, DagMlError>;
