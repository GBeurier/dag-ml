//! Additive closed Methods profile compilation and read-only RAW inspection.
use std::collections::BTreeMap;
use pyo3::prelude::*;
use serde_json::Value;
use crate::py_core_error;

#[pyfunction]
pub fn methods_pls_role_pipeline_contract_json(params_json: &str) -> PyResult<String> {
    dag_ml_core::canonical::parse_typed_json(params_json).map_err(|e| py_core_error(dag_ml_core::DagMlError::RuntimeValidation(e.to_string())))?;
    let params: BTreeMap<String,Value> = serde_json::from_str(params_json).map_err(|e| py_core_error(e.into()))?;
    let contract=dag_ml_core::methods_pls_role_pipeline_contract(&params).map_err(py_core_error)?;
    serde_json::to_string(&contract).map_err(|e| py_core_error(e.into()))
}

#[pyfunction]
pub fn inspect_methods_role_pipeline_params_json(payload_bytes: Vec<u8>, methods_library_path: &str) -> PyResult<String> {
    #[cfg(feature="methods-optimizer")]
    {
        let runtime=dag_ml_core::MethodsRuntime::configure(methods_library_path).map_err(|e|py_core_error(dag_ml_core::DagMlError::RuntimeValidation(e.to_string())))?;
        let inspected=dag_ml_core::inspect_methods_role_pipeline_params(&payload_bytes,&runtime).map_err(py_core_error)?;
        serde_json::to_string(&inspected).map_err(|e|py_core_error(e.into()))
    }
    #[cfg(not(feature="methods-optimizer"))]
    {
        let _=(payload_bytes,methods_library_path);
        Err(py_core_error(dag_ml_core::DagMlError::RuntimeValidation("Methods RAW inspection requires a methods-optimizer enabled DAG binding".into())))
    }
}
