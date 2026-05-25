//! Core contracts for the DAG-ML control engine.
//!
//! This crate intentionally contains no host-runtime dependency and no heavy
//! data buffers. It validates control structures and leakage-sensitive
//! prediction flows that every binding must preserve.

pub mod campaign;
pub mod error;
pub mod fold;
pub mod graph;
pub mod ids;
pub mod oof;
pub mod phase;
pub mod rng;

pub use campaign::*;
pub use error::{DagMlError, Result};
pub use fold::*;
pub use graph::*;
pub use ids::*;
pub use oof::*;
pub use phase::*;
pub use rng::*;
