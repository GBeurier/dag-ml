use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Phase {
    Compile,
    Plan,
    FitCv,
    Select,
    Refit,
    Predict,
    Explain,
}

impl Phase {
    pub fn is_training(self) -> bool {
        matches!(self, Self::FitCv | Self::Refit)
    }
}
