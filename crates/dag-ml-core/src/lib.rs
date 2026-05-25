//! Core contracts for the DAG-ML control engine.
//!
//! This crate intentionally contains no host-runtime dependency and no heavy
//! data buffers. It validates control structures and leakage-sensitive
//! prediction flows that every binding must preserve.

pub mod aggregation;
pub mod campaign;
pub mod controller;
pub mod data;
pub mod error;
pub mod fold;
pub mod generation;
pub mod graph;
pub mod ids;
pub mod oof;
pub mod phase;
pub mod plan;
pub mod policy;
pub mod relation;
pub mod rng;
pub mod runtime;

pub use aggregation::*;
pub use campaign::*;
pub use controller::*;
pub use data::*;
pub use error::{DagMlError, Result};
pub use fold::*;
pub use generation::*;
pub use graph::*;
pub use ids::*;
pub use oof::*;
pub use phase::*;
pub use plan::*;
pub use policy::*;
pub use relation::*;
pub use rng::*;
pub use runtime::*;
