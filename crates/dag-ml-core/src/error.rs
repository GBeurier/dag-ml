use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OofLeakageViolation {
    pub producer_node: String,
    pub partition: String,
    pub fold_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OofLeakageReport {
    pub node_id: String,
    pub violators: Vec<OofLeakageViolation>,
    pub allow_train_predictions_as_features: bool,
    pub remediation: String,
}

#[derive(Debug, Error)]
pub enum DagMlError {
    #[error("invalid identifier `{value}`: {reason}")]
    InvalidIdentifier { value: String, reason: &'static str },

    #[error("graph validation failed: {0}")]
    GraphValidation(String),

    #[error("controller validation failed: {0}")]
    ControllerValidation(String),

    #[error("campaign validation failed: {0}")]
    CampaignValidation(String),

    #[error("planning failed: {0}")]
    Planning(String),

    #[error("runtime validation failed: {0}")]
    RuntimeValidation(String),

    #[error("OOF validation failed: {0}")]
    OofValidation(String),

    #[error("OOF leakage at `{}`: {} violator(s); {}", .0.node_id, .0.violators.len(), .0.remediation)]
    OofLeakage(Box<OofLeakageReport>),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, DagMlError>;
