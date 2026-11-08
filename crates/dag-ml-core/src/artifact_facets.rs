//! Owner facets independent of any container or host model loader.
//!
//! Descriptors are declarations. Only [`ValidatedOwnerArtifact`] values made
//! from current owner bytes supply DAG facts. No function in this module
//! reads an archive or fetches a reference. Inspection and validation never
//! hydrate or execute. Explicit admitted native actions are separate below.

use std::collections::{BTreeMap, BTreeSet};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    canonical::{deserialize_external_contract, parse_typed_json_bytes, TypedCanonicalValue},
    ArtifactLoadMode, BundlePredictionCachePayloadSet, DagMlError, ExecutionBundle, ExecutionPlan,
    GraphSpec, InitialFullRefitPackage, PortablePredictorPackage, PortableRefitPackageV3, ScoreSet,
    TrainingOutcome,
};

#[path = "artifact_facets_native.rs"]
mod native;
pub use native::*;

pub const FACET_GRAPH_SPEC_PROFILE: &str = "dagml.facet.graph-spec.tcv1.v1";
pub const FACET_EXECUTION_BUNDLE_PROFILE: &str = "dagml.facet.execution-bundle.tcv1.v1";
pub const FACET_SCORE_SET_PROFILE: &str = "dagml.facet.score-set.tcv1.v1";
pub const FACET_PREDICTION_CACHES_PROFILE: &str =
    "dagml.facet.prediction-cache-payload-set.tcv1.v1";
pub const FACET_EXECUTION_PLAN_PROFILE: &str = "dagml.facet.execution-plan.tcv1.v1";
/// Accepted profiles for the DAG payloads supported by this API. This does not
/// change the meaning of any existing internal or archive fingerprint.
pub const DAG_FACET_FINGERPRINT_PROFILES: &[&str] = &[
    FACET_GRAPH_SPEC_PROFILE,
    FACET_EXECUTION_BUNDLE_PROFILE,
    FACET_SCORE_SET_PROFILE,
    FACET_PREDICTION_CACHES_PROFILE,
    FACET_EXECUTION_PLAN_PROFILE,
    "dagml_tcv1",
];

pub const FACET_GRAPH_SPEC: &str = "dagml.graph_spec";
pub const FACET_EXECUTION_BUNDLE: &str = "dagml.execution_bundle";
pub const FACET_SCORE_SET: &str = "dagml.score_set";
pub const FACET_TRAINING_OUTCOME: &str = "dagml.training_outcome";
pub const FACET_PREDICTOR_PACKAGE: &str = "dagml.portable_predictor_package";
pub const FACET_INITIAL_REFIT: &str = "dagml.initial_full_refit_package";
pub const FACET_REFIT_V3: &str = "dagml.portable_refit_package";
pub const FACET_PREDICTION_CACHES: &str = "dagml.prediction_cache_payload_set";
pub const FACET_EXECUTION_PLAN: &str = "dagml.execution_plan";

/// Typed owner refusals. None of these values is executable admission.
#[derive(Debug, thiserror::Error)]
pub enum FacetRefusal {
    #[error("unsupported facet operation `{operation}`")]
    UnsupportedOperation { operation: String },
    #[error("owner content for `{artifact_id}` is unresolved")]
    UnresolvedOwnerContent { artifact_id: String },
    #[error("facet mismatch: {detail}")]
    Mismatch { detail: String },
    #[error("explicit caller trust is required before host serializer hydration")]
    MissingTrust,
    #[error("owner evidence remains unverified: {detail}")]
    UnverifiedEvidence { detail: String },
    #[error("missing declared prerequisite capability `{capability}`")]
    MissingCapability { capability: String },
    #[error("Methods owner runtime is unavailable: {source}")]
    RuntimeUnavailable {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("unsupported facet payload `{content_type}`")]
    UnsupportedPayload { content_type: String },
    #[error("invalid facet payload: {detail}")]
    InvalidPayload { detail: String },
    #[error("invalid facet structure: {detail}")]
    InvalidStructure { detail: String },
    #[error("invalid owner JSON: {source}")]
    Json {
        #[from]
        source: serde_json::Error,
    },
    #[error("{context}: {source}")]
    OwnerValidation {
        context: &'static str,
        #[source]
        source: DagMlError,
    },
}

impl From<DagMlError> for FacetRefusal {
    fn from(source: DagMlError) -> Self {
        Self::OwnerValidation {
            context: "current owner contract validation",
            source,
        }
    }
}

pub type FacetResult<T> = std::result::Result<T, FacetRefusal>;

fn error(message: impl Into<String>) -> FacetRefusal {
    FacetRefusal::InvalidStructure {
        detail: message.into(),
    }
}
fn mismatch(message: impl Into<String>) -> FacetRefusal {
    FacetRefusal::Mismatch {
        detail: message.into(),
    }
}
fn invalid_payload(message: impl Into<String>) -> FacetRefusal {
    FacetRefusal::InvalidPayload {
        detail: message.into(),
    }
}
fn unsupported_payload(content_type: impl Into<String>) -> FacetRefusal {
    FacetRefusal::UnsupportedPayload {
        content_type: content_type.into(),
    }
}

fn nonempty(value: &str) -> FacetResult<()> {
    if value.trim().is_empty() {
        return Err(error("identifiers must not be empty"));
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn check_sha256(value: &str) -> FacetResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(error(
            "raw SHA-256 must be 64 lowercase hexadecimal characters",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacetContentType {
    pub id: String,
    pub version: u32,
}

impl FacetContentType {
    pub fn new(id: impl Into<String>, version: u32) -> Self {
        Self {
            id: id.into(),
            version,
        }
    }

    pub fn validate(&self) -> FacetResult<()> {
        nonempty(&self.id)?;
        if self.version == 0 {
            return Err(error("content type version must be positive"));
        }
        Ok(())
    }
}

/// The same complete identity is mandatory for embedded and external facets.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerIdentity {
    pub owner: String,
    pub schema: String,
    pub schema_version: u32,
    pub fingerprint_profile: String,
    pub fingerprint: String,
}

impl OwnerIdentity {
    pub fn validate(&self) -> FacetResult<()> {
        for value in [
            &self.owner,
            &self.schema,
            &self.fingerprint_profile,
            &self.fingerprint,
        ] {
            nonempty(value)?;
        }
        if self.schema_version == 0 {
            return Err(error("owner schema_version must be positive"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFacetKind {
    Recipe,
    Dataset,
    Model,
    Run,
    Result,
    Score,
    Prediction,
    Residual,
    TrainingEvidence,
    FullRefit,
    Provenance,
    Attachment,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ArtifactFacetStorage {
    Embedded {
        root_member: String,
        member_paths: Vec<String>,
    },
    Reference {
        #[serde(
            default,
            deserialize_with = "optional_raw_sha256",
            skip_serializing_if = "Option::is_none"
        )]
        root_sha256: Option<String>,
    },
}

/// Minimal reference to a declared owner requirement artifact. Its meaning is
/// owner-defined; generic validation checks only shape and ID existence.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FacetRequirementReference {
    pub artifact_id: String,
}

impl<'de> Deserialize<'de> for FacetRequirementReference {
    fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> std::result::Result<Self, D::Error> {
        struct ReferenceVisitor;
        impl<'de> serde::de::Visitor<'de> for ReferenceVisitor {
            type Value = FacetRequirementReference;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object containing only artifact_id")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut artifact_id = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key != "artifact_id" {
                        return Err(serde::de::Error::unknown_field(&key, &["artifact_id"]));
                    }
                    if artifact_id.is_some() {
                        return Err(serde::de::Error::duplicate_field("artifact_id"));
                    }
                    artifact_id = Some(map.next_value::<String>()?);
                }
                Ok(FacetRequirementReference {
                    artifact_id: artifact_id
                        .ok_or_else(|| serde::de::Error::missing_field("artifact_id"))?,
                })
            }
        }
        decoder.deserialize_map(ReferenceVisitor)
    }
}

fn optional_raw_sha256<'de, D: serde::Deserializer<'de>>(
    decoder: D,
) -> std::result::Result<Option<String>, D::Error> {
    let value = String::deserialize(decoder)?;
    check_sha256(&value).map_err(serde::de::Error::custom)?;
    Ok(Some(value))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFacet {
    pub id: String,
    pub kind: ArtifactFacetKind,
    pub owner: String,
    pub content_type: FacetContentType,
    pub identity: OwnerIdentity,
    pub storage: ArtifactFacetStorage,
    // A custom decoder makes missing distinct from explicitly null.
    #[serde(deserialize_with = "required_requirement")]
    pub requirements: Option<FacetRequirementReference>,
}

fn required_requirement<'de, D: serde::Deserializer<'de>>(
    decoder: D,
) -> std::result::Result<Option<FacetRequirementReference>, D::Error> {
    Option::deserialize(decoder)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFacetRelation {
    RecipeOf,
    DatasetOf,
    ProducedBy,
    Evaluates,
    ResidualOf,
    EvidenceFor,
    RefitOf,
}

impl ArtifactFacetRelation {
    pub fn permits(self, source: ArtifactFacetKind, target: ArtifactFacetKind) -> bool {
        use ArtifactFacetKind::*;
        match self {
            Self::RecipeOf => {
                source == Recipe
                    && matches!(target, Run | Result | TrainingEvidence | Model | FullRefit)
            }
            Self::DatasetOf => {
                source == Dataset
                    && matches!(
                        target,
                        Run | Result
                            | TrainingEvidence
                            | Model
                            | Prediction
                            | Residual
                            | Score
                            | FullRefit
                    )
            }
            Self::ProducedBy => {
                matches!(
                    source,
                    Model | Result | TrainingEvidence | Prediction | Residual | Score | FullRefit
                ) && target == Run
            }
            Self::Evaluates => {
                source == Score && matches!(target, Prediction | Result | TrainingEvidence)
            }
            Self::ResidualOf => source == Residual && target == Prediction,
            Self::EvidenceFor => source == TrainingEvidence && target == Model,
            Self::RefitOf => source == FullRefit && target == TrainingEvidence,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFacetLink {
    pub id: String,
    pub owner: String,
    pub relation: ArtifactFacetRelation,
    pub source_artifact: String,
    pub target_artifact: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFacetGraph {
    pub artifacts: Vec<ArtifactFacet>,
    pub links: Vec<ArtifactFacetLink>,
}

impl ArtifactFacetGraph {
    pub fn from_json(json: &str) -> FacetResult<Self> {
        let graph: Self = strict_decode(json)?;
        graph.validate_structure()?;
        Ok(graph)
    }

    /// Structural closure only. There is no kind-to-operation inference and no
    /// one-dataset-per-run constraint. Core owns transport path validation.
    pub fn validate_structure(&self) -> FacetResult<()> {
        let mut artifacts = BTreeMap::new();
        let mut members = BTreeSet::new();
        for artifact in &self.artifacts {
            nonempty(&artifact.id)?;
            nonempty(&artifact.owner)?;
            artifact.content_type.validate()?;
            artifact.identity.validate()?;
            if artifact.owner != artifact.identity.owner {
                return Err(error("artifact owner differs from identity owner"));
            }
            if artifacts.insert(artifact.id.as_str(), artifact).is_some() {
                return Err(error("duplicate artifact ID"));
            }
            match &artifact.storage {
                ArtifactFacetStorage::Embedded {
                    root_member,
                    member_paths,
                } => {
                    if !member_paths.contains(root_member) || member_paths.is_empty() {
                        return Err(error("embedded root must occur in member_paths"));
                    }
                    for path in member_paths {
                        nonempty(path)?;
                        if !members.insert(path) {
                            return Err(error("a member must belong to exactly one facet"));
                        }
                    }
                }
                ArtifactFacetStorage::Reference {
                    root_sha256: Some(raw),
                } => check_sha256(raw)?,
                ArtifactFacetStorage::Reference { root_sha256: None } => {}
            }
        }
        for artifact in &self.artifacts {
            if let Some(reference) = &artifact.requirements {
                nonempty(&reference.artifact_id)?;
                if !artifacts.contains_key(reference.artifact_id.as_str()) {
                    return Err(error("undeclared requirements artifact"));
                }
            }
        }
        let mut ids = BTreeSet::new();
        let mut edges = BTreeSet::new();
        for link in &self.links {
            nonempty(&link.id)?;
            nonempty(&link.owner)?;
            if !ids.insert(&link.id) {
                return Err(error("duplicate link ID"));
            }
            let source = artifacts
                .get(link.source_artifact.as_str())
                .ok_or_else(|| error("undeclared source endpoint"))?;
            let target = artifacts
                .get(link.target_artifact.as_str())
                .ok_or_else(|| error("undeclared target endpoint"))?;
            if source.id == target.id || !link.relation.permits(source.kind, target.kind) {
                return Err(error("self-link or invalid relation direction"));
            }
            if !edges.insert((link.relation, &source.id, &target.id)) {
                return Err(error("duplicate semantic edge"));
            }
        }
        Ok(())
    }
}

// Strict parsing applies even to owner types whose historical parser accepts
// unknown fields. Shape validation rejects serde's positional struct syntax.
fn strict_decode<T: DeserializeOwned + Serialize>(json: &str) -> FacetResult<T> {
    let value: T =
        deserialize_external_contract(json, "owner facet", DagMlError::RuntimeValidation)?;
    let mut ignored = Vec::new();
    let mut decoder = serde_json::Deserializer::from_str(json);
    let _: T = serde_ignored::deserialize(&mut decoder, |path| ignored.push(path.to_string()))?;
    decoder.end()?;
    if !ignored.is_empty() {
        return Err(error(format!(
            "unknown owner fields: {}",
            ignored.join(", ")
        )));
    }
    Ok(value)
}

#[derive(Clone, Debug)]
enum OwnerPayload {
    Graph(GraphSpec),
    Bundle(Box<ExecutionBundle>),
    Scores(ScoreSet),
    Outcome(Box<TrainingOutcome>),
    Predictor(Box<PortablePredictorPackage>),
    InitialRefit(Box<InitialFullRefitPackage>),
    ChildRefit(Box<PortableRefitPackageV3>),
    Caches(BundlePredictionCachePayloadSet),
    Plan(Box<ExecutionPlan>),
}

/// Non-deserializable proof of current owner validation, retaining original
/// bytes. Private fields prevent callers from creating a claimed JSON proof.
///
/// ```compile_fail
/// use dag_ml_core::ValidatedOwnerArtifact;
/// let forged: ValidatedOwnerArtifact = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct ValidatedOwnerArtifact {
    bytes: Vec<u8>,
    content_type: FacetContentType,
    identity: OwnerIdentity,
    raw_sha256: String,
    payload: OwnerPayload,
}

impl ValidatedOwnerArtifact {
    pub fn original_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn content_type(&self) -> &FacetContentType {
        &self.content_type
    }
    pub fn identity(&self) -> &OwnerIdentity {
        &self.identity
    }
    pub fn raw_sha256(&self) -> &str {
        &self.raw_sha256
    }
    pub fn training_outcome(&self) -> Option<&TrainingOutcome> {
        match &self.payload {
            OwnerPayload::Outcome(value) => Some(value),
            _ => None,
        }
    }
    pub fn predictor_package(&self) -> Option<&PortablePredictorPackage> {
        match &self.payload {
            OwnerPayload::Predictor(value) => Some(value),
            _ => None,
        }
    }
    pub fn initial_refit_package(&self) -> Option<&InitialFullRefitPackage> {
        match &self.payload {
            OwnerPayload::InitialRefit(value) => Some(value),
            _ => None,
        }
    }
    pub fn refit_package(&self) -> Option<&PortableRefitPackageV3> {
        match &self.payload {
            OwnerPayload::ChildRefit(value) => Some(value),
            _ => None,
        }
    }
    pub fn prediction_caches(&self) -> Option<&BundlePredictionCachePayloadSet> {
        match &self.payload {
            OwnerPayload::Caches(value) => Some(value),
            _ => None,
        }
    }
}

fn typed(bytes: &[u8]) -> FacetResult<TypedCanonicalValue> {
    parse_typed_json_bytes(bytes).map_err(|cause| invalid_payload(cause.to_string()))
}

/// New profile preimage: TCV1 of the object
/// `{profile: PROFILE, content_type: {id, version}, payload: TYPED_JSON}`.
/// The payload is the validated typed owner projection, safely serialized and
/// parsed directly by TCV1. Original bytes are independently retained and pass
/// strict token preflight before decoding; no lossy Value pass is used here.
fn facet_domain(
    profile: &str,
    content: &FacetContentType,
    payload: TypedCanonicalValue,
) -> FacetResult<String> {
    TypedCanonicalValue::Object(vec![
        (
            "profile".into(),
            TypedCanonicalValue::String(profile.into()),
        ),
        ("content_type".into(), typed(&serde_json::to_vec(content)?)?),
        ("payload".into(), payload),
    ])
    .fingerprint()
    .map_err(|cause| error(cause.to_string()))
}

/// Validate known DAG content without replacing supplied bytes. Unknown owner
/// content returns `None` and remains opaque/unverified. A known malformed or
/// unsupported-version payload is a refusal, never an opaque fallback.
pub fn validate_owner_facet_bytes(
    content: &FacetContentType,
    bytes: &[u8],
) -> FacetResult<Option<ValidatedOwnerArtifact>> {
    content.validate()?;
    if !matches!(
        content.id.as_str(),
        FACET_GRAPH_SPEC
            | FACET_EXECUTION_BUNDLE
            | FACET_SCORE_SET
            | FACET_TRAINING_OUTCOME
            | FACET_PREDICTOR_PACKAGE
            | FACET_INITIAL_REFIT
            | FACET_REFIT_V3
            | FACET_PREDICTION_CACHES
            | FACET_EXECUTION_PLAN
    ) {
        return Ok(None);
    }
    let raw = typed(bytes)?;
    let json = std::str::from_utf8(bytes).map_err(|cause| invalid_payload(cause.to_string()))?;
    let (payload, version, profile, self_field) = match content.id.as_str() {
        FACET_GRAPH_SPEC => {
            let _: GraphSpec = strict_decode(json)?;
            (
                OwnerPayload::Graph(GraphSpec::from_json(json)?),
                1,
                FACET_GRAPH_SPEC_PROFILE,
                None,
            )
        }
        FACET_EXECUTION_BUNDLE => {
            let _: ExecutionBundle = strict_decode(json)?;
            let value = ExecutionBundle::from_json(json)?;
            let version = value.schema_version;
            (
                OwnerPayload::Bundle(Box::new(value)),
                version,
                FACET_EXECUTION_BUNDLE_PROFILE,
                None,
            )
        }
        FACET_SCORE_SET => {
            let value: ScoreSet = strict_decode(json)?;
            value.validate()?;
            let version = value.schema_version;
            (
                OwnerPayload::Scores(value),
                version,
                FACET_SCORE_SET_PROFILE,
                None,
            )
        }
        FACET_TRAINING_OUTCOME => {
            let _: TrainingOutcome = strict_decode(json)?;
            let value = TrainingOutcome::from_json(json)?;
            let version = value.schema_version;
            (
                OwnerPayload::Outcome(Box::new(value)),
                version,
                "dagml_tcv1",
                Some("outcome_fingerprint"),
            )
        }
        FACET_PREDICTOR_PACKAGE => {
            let _: PortablePredictorPackage = strict_decode(json)?;
            let value = PortablePredictorPackage::from_json(json)?;
            let version = value.schema_version;
            (
                OwnerPayload::Predictor(Box::new(value)),
                version,
                "dagml_tcv1",
                Some("package_fingerprint"),
            )
        }
        FACET_INITIAL_REFIT => {
            let _: InitialFullRefitPackage = strict_decode(json)?;
            let value = InitialFullRefitPackage::from_json(json)?;
            let version = value.schema_version;
            (
                OwnerPayload::InitialRefit(Box::new(value)),
                version,
                "dagml_tcv1",
                Some("package_fingerprint"),
            )
        }
        FACET_REFIT_V3 => {
            let _: PortableRefitPackageV3 = strict_decode(json)?;
            let value = PortableRefitPackageV3::from_json(json)?;
            let version = value.schema_version;
            (
                OwnerPayload::ChildRefit(Box::new(value)),
                version,
                "dagml_tcv1",
                Some("package_fingerprint"),
            )
        }
        FACET_PREDICTION_CACHES => {
            let value: BundlePredictionCachePayloadSet = strict_decode(json)?;
            value.validate()?;
            let version = value.schema_version;
            (
                OwnerPayload::Caches(value),
                version,
                FACET_PREDICTION_CACHES_PROFILE,
                None,
            )
        }
        FACET_EXECUTION_PLAN => {
            let _: ExecutionPlan = strict_decode(json)?;
            (
                OwnerPayload::Plan(Box::new(ExecutionPlan::from_json(json)?)),
                1,
                FACET_EXECUTION_PLAN_PROFILE,
                None,
            )
        }
        _ => unreachable!("known content type matched above"),
    };
    if content.version != version {
        return Err(error(
            "content type version differs from current owner payload version",
        ));
    }
    validate_owner_composites(&payload)?;
    let fingerprint = if profile != "dagml_tcv1" {
        let projection = match &payload {
            OwnerPayload::Graph(value) => serde_json::to_vec(value)?,
            OwnerPayload::Bundle(value) => serde_json::to_vec(value)?,
            OwnerPayload::Scores(value) => serde_json::to_vec(value)?,
            OwnerPayload::Caches(value) => serde_json::to_vec(value)?,
            OwnerPayload::Plan(value) => serde_json::to_vec(value)?,
            _ => return Err(error("new facet profile lacks a typed owner projection")),
        };
        facet_domain(profile, content, typed(&projection)?)?
    } else {
        match self_field {
            Some(field) => raw.fingerprint_without(field),
            None => raw.fingerprint(),
        }
        .map_err(|cause| error(cause.to_string()))?
    };
    Ok(Some(ValidatedOwnerArtifact {
        bytes: bytes.to_vec(),
        content_type: content.clone(),
        raw_sha256: sha256(bytes),
        identity: OwnerIdentity {
            owner: "dag-ml".into(),
            schema: content.id.clone(),
            schema_version: version,
            fingerprint_profile: profile.into(),
            fingerprint,
        },
        payload,
    }))
}

fn owner_object<T: Serialize>(
    id: &str,
    version: u32,
    value: &T,
) -> FacetResult<ValidatedOwnerArtifact> {
    validate_owner_facet_bytes(
        &FacetContentType::new(id, version),
        &crate::stable_json::to_vec(value)?,
    )?
    .ok_or_else(|| error("unsupported builder content"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FacetSemanticStatus {
    Validated,
    Unverified,
    Mismatch,
}

/// Process-local verdict, serialized only as non-authoritative diagnostics.
/// ```compile_fail
/// use dag_ml_core::FacetLinkVerdict;
/// let forged: FacetLinkVerdict = serde_json::from_str("{}").unwrap();
/// forged.require_validated().unwrap();
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FacetLinkVerdict {
    pub status: FacetSemanticStatus,
    pub reason: String,
}

impl FacetLinkVerdict {
    pub fn unverified(reason: impl Into<String>) -> Self {
        Self {
            status: FacetSemanticStatus::Unverified,
            reason: reason.into(),
        }
    }
    fn comparison(equal: bool, reason: &str) -> Self {
        Self {
            status: if equal {
                FacetSemanticStatus::Validated
            } else {
                FacetSemanticStatus::Mismatch
            },
            reason: reason.into(),
        }
    }
    pub fn require_validated(&self) -> FacetResult<()> {
        if self.status != FacetSemanticStatus::Validated {
            return Err(match self.status {
                FacetSemanticStatus::Mismatch => mismatch(&self.reason),
                _ => FacetRefusal::UnverifiedEvidence {
                    detail: self.reason.clone(),
                },
            });
        }
        Ok(())
    }
}

/// Diagnostic report entry; JSON cannot mint a live validated result.
/// ```compile_fail
/// use dag_ml_core::FacetLinkValidation;
/// let forged: FacetLinkValidation = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FacetLinkValidation {
    pub link_id: String,
    pub verdict: FacetLinkVerdict,
}

/// Separate, explicitly trusted composition port. Implementors must validate
/// actual IO bytes/identity AND the IO-to-DMD/DAG envelope/binding projection
/// for this target/cohort. Equality to TrainingDataIdentity is NOT that proof.
/// This port must not hydrate models or obtain feature buffers in DAG.
pub trait FacetDatasetBindingValidator {
    fn validate_dataset_binding(
        &self,
        dataset: &ArtifactFacet,
        original_dataset_bytes: Option<&[u8]>,
        target: &ValidatedOwnerArtifact,
    ) -> FacetResult<FacetLinkVerdict>;
}

#[derive(Default)]
pub struct FacetValidationContext<'a> {
    pub owner_proofs: BTreeMap<String, ValidatedOwnerArtifact>,
    /// Supplied candidate bytes, not owner validation. The known raw identity
    /// and optional root SHA are checked before a separate semantic port runs.
    pub external_bytes: BTreeMap<String, Vec<u8>>,
    pub dataset_binding_validator: Option<&'a dyn FacetDatasetBindingValidator>,
}

impl FacetValidationContext<'_> {
    /// Attach a live composition authority without cloning retained owner bytes
    /// or requiring the IO proof/provider to have a static lifetime.
    pub fn with_dataset_binding_validator(
        self,
        validator: &dyn FacetDatasetBindingValidator,
    ) -> FacetValidationContext<'_> {
        FacetValidationContext {
            owner_proofs: self.owner_proofs,
            external_bytes: self.external_bytes,
            dataset_binding_validator: Some(validator),
        }
    }
}

fn payload_accepts_kind(payload: &OwnerPayload, kind: ArtifactFacetKind) -> bool {
    use ArtifactFacetKind::*;
    if matches!(kind, Attachment | Provenance) {
        // Storage/inspection only: neither kind is a scientific endpoint.
        return true;
    }
    match payload {
        OwnerPayload::Graph(_) => kind == Recipe,
        OwnerPayload::Bundle(_) => kind == Run,
        OwnerPayload::Scores(_) => kind == Score,
        OwnerPayload::Outcome(_) => matches!(kind, Result | TrainingEvidence | Run),
        OwnerPayload::Predictor(_) => kind == Model,
        OwnerPayload::InitialRefit(_) => matches!(kind, FullRefit | Model | Run),
        OwnerPayload::ChildRefit(_) => matches!(kind, FullRefit | Run),
        OwnerPayload::Caches(_) => kind == Prediction,
        OwnerPayload::Plan(_) => false,
    }
}

fn check_candidate_bytes(artifact: &ArtifactFacet, bytes: &[u8]) -> FacetResult<()> {
    let raw = sha256(bytes);
    if artifact.identity.fingerprint_profile == "sha256.raw.v1" {
        check_sha256(&artifact.identity.fingerprint)?;
        if artifact.identity.fingerprint != raw {
            return Err(mismatch("supplied bytes mismatch sha256.raw.v1 identity"));
        }
    }
    if let ArtifactFacetStorage::Reference {
        root_sha256: Some(expected),
    } = &artifact.storage
    {
        if expected != &raw {
            return Err(mismatch(
                "supplied candidate bytes mismatch reference root SHA",
            ));
        }
    }
    Ok(())
}

fn matches_proof(artifact: &ArtifactFacet, proof: &ValidatedOwnerArtifact) -> FacetResult<()> {
    if !payload_accepts_kind(&proof.payload, artifact.kind) {
        return Err(mismatch(format!(
            "known owner payload is incompatible with scientific kind for `{}`",
            artifact.id
        )));
    }
    if artifact.content_type != proof.content_type
        || artifact.identity != proof.identity
        || artifact.owner != proof.identity.owner
    {
        return Err(mismatch(format!(
            "supplied proof identity mismatch for `{}`",
            artifact.id
        )));
    }
    if let ArtifactFacetStorage::Reference {
        root_sha256: Some(raw),
    } = &artifact.storage
    {
        if raw != &proof.raw_sha256 {
            return Err(mismatch("reference raw SHA mismatch"));
        }
    }
    Ok(())
}

/// Inspect links using only live owner proofs. Mismatches are returned for
/// diagnostic use; use [`validate_artifact_facet_links`] before any action.
pub fn inspect_artifact_facet_links(
    graph: &ArtifactFacetGraph,
    context: &FacetValidationContext<'_>,
) -> FacetResult<Vec<FacetLinkValidation>> {
    graph.validate_structure()?;
    for (id, proof) in &context.owner_proofs {
        let artifact = graph
            .artifacts
            .iter()
            .find(|artifact| &artifact.id == id)
            .ok_or_else(|| error("proof supplied for undeclared artifact"))?;
        matches_proof(artifact, proof)?;
    }
    let artifacts = graph
        .artifacts
        .iter()
        .map(|artifact| (artifact.id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    for (id, bytes) in &context.external_bytes {
        let artifact = artifacts
            .get(id.as_str())
            .ok_or_else(|| error("candidate bytes supplied for undeclared artifact"))?;
        check_candidate_bytes(artifact, bytes)?;
    }
    graph.links.iter().map(|link| {
        let source = artifacts[link.source_artifact.as_str()];
        let target = artifacts[link.target_artifact.as_str()];
        let verdict = if link.owner != "dag-ml" {
            FacetLinkVerdict::unverified("relation owner is unsupported")
        } else if link.relation == ArtifactFacetRelation::DatasetOf {
            match (context.dataset_binding_validator, context.owner_proofs.get(&target.id)) {
                (Some(port), Some(target_proof)) => {
                    let bytes = context.external_bytes.get(&source.id).map(Vec::as_slice);
                    port.validate_dataset_binding(source, bytes, target_proof)?
                }
                _ => FacetLinkVerdict::unverified("actual IO identity and IO-to-DMD/DAG envelope/binding proof are absent"),
            }
        } else {
            match (context.owner_proofs.get(&source.id), context.owner_proofs.get(&target.id)) {
                (Some(source), Some(target)) => compare_link(link, source, target, graph, context)?,
                _ => FacetLinkVerdict::unverified("owner bytes/proof are unresolved; a reference is not a missing container body"),
            }
        };
        Ok(FacetLinkValidation { link_id: link.id.clone(), verdict })
    }).collect()
}

/// Refuse every known mismatch, including mismatched supplied proofs. Missing
/// evidence stays explicitly unverified; callers must require the particular
/// validated links needed by their operation.
pub fn validate_artifact_facet_links(
    graph: &ArtifactFacetGraph,
    context: &FacetValidationContext<'_>,
) -> FacetResult<Vec<FacetLinkValidation>> {
    let report = inspect_artifact_facet_links(graph, context)?;
    if let Some(link) = report
        .iter()
        .find(|link| link.verdict.status == FacetSemanticStatus::Mismatch)
    {
        return Err(mismatch(format!(
            "link `{}` mismatch: {}",
            link.link_id, link.verdict.reason
        )));
    }
    Ok(report)
}

fn proof_identity_of_object<T: Serialize>(
    proof: &ValidatedOwnerArtifact,
    id: &str,
    version: u32,
    object: &T,
) -> FacetResult<bool> {
    Ok(proof.identity == owner_object(id, version, object)?.identity)
}

fn optional_caches_match(
    proof: &ValidatedOwnerArtifact,
    caches: Option<&BundlePredictionCachePayloadSet>,
) -> FacetResult<bool> {
    match caches {
        Some(caches) => proof_identity_of_object(
            proof,
            FACET_PREDICTION_CACHES,
            caches.schema_version,
            caches,
        ),
        None => Ok(false),
    }
}

fn graph_of(payload: &OwnerPayload) -> Option<&GraphSpec> {
    match payload {
        OwnerPayload::Graph(graph) => Some(graph),
        OwnerPayload::Outcome(outcome) => Some(&outcome.effective_plan.graph_plan.graph),
        OwnerPayload::Predictor(package) => Some(&package.template.graph),
        OwnerPayload::InitialRefit(package) => Some(&package.effective_plan.graph_plan.graph),
        OwnerPayload::ChildRefit(package) => Some(&package.outcome.effective_plan.graph_plan.graph),
        OwnerPayload::Plan(plan) => Some(&plan.graph_plan.graph),
        _ => None,
    }
}

fn compare_link(
    link: &ArtifactFacetLink,
    source: &ValidatedOwnerArtifact,
    target: &ValidatedOwnerArtifact,
    graph: &ArtifactFacetGraph,
    context: &FacetValidationContext<'_>,
) -> FacetResult<FacetLinkVerdict> {
    use ArtifactFacetRelation::*;
    use OwnerPayload::*;
    match link.relation {
        RecipeOf => {
            if matches!(source.payload, Graph(_)) {
                if let Some(graph) = graph_of(&target.payload) {
                    return Ok(FacetLinkVerdict::comparison(
                        proof_identity_of_object(source, FACET_GRAPH_SPEC, 1, graph)?,
                        "compiled GraphSpec identity compared with owner target graph",
                    ));
                }
                if let Bundle(bundle) = &target.payload {
                    return Ok(FacetLinkVerdict::comparison(
                        crate::campaign::stable_json_fingerprint(
                            graph_of(&source.payload).expect("graph payload checked"),
                        )? == bundle.graph_fingerprint,
                        "current owner graph fingerprint compared with exact bundle graph binding",
                    ));
                }
            }
        }
        EvidenceFor => {
            if let (Outcome(outcome), Predictor(package)) = (&source.payload, &target.payload) {
                return Ok(FacetLinkVerdict::comparison(
                    predictor_matches_outcome(outcome, package)?,
                    "current outcome/package selection, output, data and fitted-state closure",
                ));
            }
        }
        RefitOf => {
            if let (ChildRefit(child), Outcome(parent)) = (&source.payload, &target.payload) {
                if child.outcome.recipe.parent_outcome != parent.to_reference()? {
                    return Ok(FacetLinkVerdict::comparison(
                        false,
                        "child refit parent outcome reference mismatch",
                    ));
                }
                // Only a declared evidence_for edge may supply the source
                // package witness. Unlinked context proofs have no authority.
                let mut witness: Option<&ValidatedOwnerArtifact> = None;
                for edge in graph.links.iter().filter(|edge| {
                    edge.owner == "dag-ml"
                        && edge.relation == EvidenceFor
                        && edge.source_artifact == link.target_artifact
                }) {
                    if let Some(proof) = context.owner_proofs.get(&edge.target_artifact) {
                        if let Predictor(package) = &proof.payload {
                            if package.package_fingerprint
                                == child.outcome.recipe.parent_package_fingerprint
                            {
                                // The validated PPC self-fingerprint includes its
                                // package ID/version. Exact parent FP uniquely
                                // selects this owner identity (absent collision).
                                witness = Some(proof);
                            }
                        }
                    }
                }
                if let Some(ValidatedOwnerArtifact {
                    payload: Predictor(package),
                    ..
                }) = witness
                {
                    child
                        .outcome
                        .recipe
                        .validate_against_source_package(package)?;
                    return Ok(FacetLinkVerdict::comparison(predictor_matches_outcome(parent, package)?,
                        "declared source package and parent outcome match refit recipe; target evidence remains distinct"));
                }
                return Ok(FacetLinkVerdict::unverified(
                    "parent outcome agrees, but declared source-package evidence_for witness is unresolved",
                ));
            }
            if matches!(source.payload, InitialRefit(_)) {
                return Ok(FacetLinkVerdict::comparison(
                    false,
                    "initial full refit has no CV parent",
                ));
            }
        }
        Evaluates => {
            if let (Scores(scores), Outcome(outcome)) = (&source.payload, &target.payload) {
                if !proof_identity_of_object(
                    source,
                    FACET_SCORE_SET,
                    outcome.score_set.schema_version,
                    &outcome.score_set,
                )? || scores.plan_id != outcome.effective_plan.id
                {
                    return Ok(FacetLinkVerdict::comparison(
                        false,
                        "score does not match validated outcome score/plan",
                    ));
                }
            }
            return Ok(FacetLinkVerdict::unverified("complete evaluation population, target/class and scope evidence is not established by ScoreSet alone"));
        }
        ResidualOf => {
            return Ok(FacetLinkVerdict::unverified(
                "complete owner residual source/observed-target/population evidence is absent",
            ))
        }
        ProducedBy => {
            if matches!(target.payload, ChildRefit(_) | InitialRefit(_)) {
                let equal = source.identity == target.identity;
                return Ok(FacetLinkVerdict::comparison(
                    equal,
                    if equal {
                        "same validated refit package carries run identity"
                    } else {
                        "known owner source does not identify this refit run package"
                    },
                ));
            }
            if let Outcome(outcome) = &target.payload {
                if matches!(source.payload, ChildRefit(_) | InitialRefit(_)) {
                    return Ok(FacetLinkVerdict::comparison(
                        false,
                        "refit package does not identify this outcome run",
                    ));
                }
                let equal = match &source.payload {
                    Outcome(_) => Some(source.identity == target.identity),
                    Predictor(package) => Some(predictor_matches_outcome(outcome, package)?),
                    Scores(_) => Some(proof_identity_of_object(
                        source,
                        FACET_SCORE_SET,
                        outcome.score_set.schema_version,
                        &outcome.score_set,
                    )?),
                    Caches(_) => Some(optional_caches_match(
                        source,
                        outcome.portable_prediction_caches.as_ref(),
                    )?),
                    _ => None,
                };
                if let Some(equal) = equal {
                    return Ok(FacetLinkVerdict::comparison(equal, "owner-fingerprinted outcome binds producer run, outputs and exact owner content (no independent evaluation claim)"));
                }
            }
            if let Bundle(bundle) = &target.payload {
                let mut equal = None;
                match &source.payload {
                    Outcome(outcome) => {
                        equal = Some(proof_identity_of_object(
                            target,
                            FACET_EXECUTION_BUNDLE,
                            outcome.execution_bundle.schema_version,
                            &outcome.execution_bundle,
                        )?)
                    }
                    Predictor(package) => {
                        equal = Some(proof_identity_of_object(
                            target,
                            FACET_EXECUTION_BUNDLE,
                            package.execution_bundle.schema_version,
                            &package.execution_bundle,
                        )?)
                    }
                    Caches(caches) => equal = Some(caches.validate_against_bundle(bundle).is_ok()),
                    ChildRefit(_) | InitialRefit(_) => equal = Some(false),
                    Scores(_) => {
                        if let Some(scores) = &bundle.scores {
                            equal = Some(proof_identity_of_object(
                                source,
                                FACET_SCORE_SET,
                                scores.schema_version,
                                scores,
                            )?);
                        }
                    }
                    _ => {}
                }
                if let Some(equal) = equal {
                    return Ok(if equal {
                        FacetLinkVerdict::unverified(
                            "bundle has no run label; an owner-fingerprinted outcome run identity is required",
                        )
                    } else {
                        FacetLinkVerdict::comparison(
                            false,
                            "execution bundle producer closure mismatch",
                        )
                    });
                }
            }
        }
        DatasetOf => unreachable!("dataset binding uses its separate port"),
    }
    Ok(FacetLinkVerdict::unverified(
        "current owner payloads do not prove this relation",
    ))
}

fn predictor_matches_outcome(
    outcome: &TrainingOutcome,
    package: &PortablePredictorPackage,
) -> FacetResult<bool> {
    // Both validators are always called, including when invoked independently
    // of the public builder. No old archive builder/reader is used.
    outcome.validate()?;
    package.validate()?;
    Ok(package.training_outcome == outcome.to_reference()?
        && package.execution_bundle == outcome.execution_bundle
        && package.effective_plan == outcome.effective_plan
        && package.template.graph == outcome.effective_plan.graph_plan.graph)
}

/// Neutral descriptor graph plus exact members. Proof/context is intentionally
/// process-local and is not serialized as an archive authorization assertion.
#[derive(Debug)]
pub struct ArtifactFacetExport {
    pub graph: ArtifactFacetGraph,
    pub members: BTreeMap<String, Vec<u8>>,
    pub context: FacetValidationContext<'static>,
}

impl std::fmt::Debug for FacetValidationContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FacetValidationContext")
            .field("owner_proofs", &self.owner_proofs)
            .field("external_byte_ids", &self.external_bytes.keys())
            .field(
                "has_dataset_binding_validator",
                &self.dataset_binding_validator.is_some(),
            )
            .finish()
    }
}

impl ArtifactFacetExport {
    fn new() -> Self {
        Self {
            graph: ArtifactFacetGraph::default(),
            members: BTreeMap::new(),
            context: FacetValidationContext::default(),
        }
    }

    fn add(
        &mut self,
        id: &str,
        kind: ArtifactFacetKind,
        proof: ValidatedOwnerArtifact,
        path: Option<&str>,
    ) -> FacetResult<()> {
        let storage = if let Some(path) = path {
            if self
                .members
                .insert(path.into(), proof.bytes.clone())
                .is_some()
            {
                return Err(error("duplicate builder member"));
            }
            ArtifactFacetStorage::Embedded {
                root_member: path.into(),
                member_paths: vec![path.into()],
            }
        } else {
            ArtifactFacetStorage::Reference {
                root_sha256: Some(proof.raw_sha256.clone()),
            }
        };
        self.graph.artifacts.push(ArtifactFacet {
            id: id.into(),
            kind,
            owner: proof.identity.owner.clone(),
            content_type: proof.content_type.clone(),
            identity: proof.identity.clone(),
            storage,
            requirements: None,
        });
        if self.context.owner_proofs.insert(id.into(), proof).is_some() {
            return Err(error("duplicate builder artifact"));
        }
        Ok(())
    }

    fn link(&mut self, relation: ArtifactFacetRelation, source: &str, target: &str) {
        self.graph.links.push(ArtifactFacetLink {
            id: format!("link:{}", self.graph.links.len()),
            owner: "dag-ml".into(),
            relation,
            source_artifact: source.into(),
            target_artifact: target.into(),
        });
    }

    fn add_raw_members(
        &mut self,
        facet_id: &str,
        payloads: &BTreeMap<crate::ArtifactId, Vec<u8>>,
    ) -> FacetResult<()> {
        let artifact = self
            .graph
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.id == facet_id)
            .ok_or_else(|| error("raw member owner is absent"))?;
        let ArtifactFacetStorage::Embedded { member_paths, .. } = &mut artifact.storage else {
            return Err(error("raw member owner must be embedded"));
        };
        let mut paths = BTreeSet::new();
        for bytes in payloads.values() {
            let path = raw_state_path(facet_id, bytes);
            if let Some(existing) = self.members.get(&path) {
                if existing != bytes {
                    return Err(mismatch("raw content-address collision"));
                }
            } else {
                self.members.insert(path.clone(), bytes.clone());
            }
            paths.insert(path);
        }
        member_paths.extend(paths);
        Ok(())
    }

    /// Validate exact physical ownership without creating or opening a ZIP.
    pub fn validate(&self) -> FacetResult<Vec<FacetLinkValidation>> {
        validate_facet_export_members(&self.graph, &self.context, &self.members)
    }
}

fn validate_facet_export_members(
    graph: &ArtifactFacetGraph,
    context: &FacetValidationContext<'_>,
    members: &BTreeMap<String, Vec<u8>>,
) -> FacetResult<Vec<FacetLinkValidation>> {
    let report = validate_artifact_facet_links(graph, context)?;
    let mut claimed = BTreeSet::new();
    for artifact in &graph.artifacts {
        if let ArtifactFacetStorage::Embedded {
            root_member,
            member_paths,
        } = &artifact.storage
        {
            if let Some(proof) = context.owner_proofs.get(&artifact.id) {
                validate_facet_members(artifact, proof, members)?;
            } else if matches!(
                artifact.kind,
                ArtifactFacetKind::Model | ArtifactFacetKind::FullRefit
            ) {
                return Err(FacetRefusal::UnresolvedOwnerContent {
                    artifact_id: artifact.id.clone(),
                });
            } else if !members.contains_key(root_member) {
                return Err(error("missing embedded root bytes"));
            }
            for path in member_paths {
                claimed.insert(path);
            }
        }
    }
    if claimed != members.keys().collect::<BTreeSet<_>>() {
        return Err(error("exact member ownership mismatch"));
    }
    Ok(report)
}

/// Expected raw state mapping derived from a currently validated root package.
/// It is an inventory, not independent state or execution validation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FacetStateMember {
    pub artifact_id: crate::ArtifactId,
    pub member_path: String,
    pub sha256: String,
    pub byte_length: u64,
}

fn raw_state_path(facet_id: &str, bytes: &[u8]) -> String {
    format!(
        "dagml/facets/{}/states/{}.bin",
        sha256(facet_id.as_bytes()),
        sha256(bytes)
    )
}

fn raw_model_payloads(
    proof: &ValidatedOwnerArtifact,
) -> FacetResult<&BTreeMap<crate::ArtifactId, Vec<u8>>> {
    if let Some((_, _, raw)) = model_closure(&proof.payload) {
        return Ok(raw);
    }
    if let OwnerPayload::InitialRefit(package) = &proof.payload {
        return Ok(&package.raw_artifact_payloads);
    }
    Err(unsupported_payload(proof.content_type.id.clone()))
}

/// Resolve each owner ArtifactId to a deterministic content-addressed path in
/// this facet's namespace. Equal raw bytes share a member only within a facet.
/// This inventory describes external physical closure for Model/FullRefit only.
/// An ID and proof cannot establish the actual facet kind: readers must call
/// [`validate_facet_members`] for descriptor/proof agreement and the actual kind's
/// closure policy. Inventory alone never authorizes admission.
pub fn expected_facet_state_members(
    facet_id: &str,
    proof: &ValidatedOwnerArtifact,
) -> FacetResult<Vec<FacetStateMember>> {
    nonempty(facet_id)?;
    raw_model_payloads(proof)?
        .iter()
        .map(|(id, bytes)| {
            Ok(FacetStateMember {
                artifact_id: id.clone(),
                member_path: raw_state_path(facet_id, bytes),
                sha256: sha256(bytes),
                byte_length: u64::try_from(bytes.len())
                    .map_err(|cause| error(cause.to_string()))?,
            })
        })
        .collect()
}

/// Verify external raw members against the actual validated root package,
/// including exact root bytes, declared membership, raw bytes and digests.
/// Only Model/FullRefit facets own external raw-state closure. Other proven
/// embedded kinds accept their root member only and return an empty inventory.
/// Stored inventory/diagnostic assertions are never used as proof.
pub fn validate_facet_state_members(
    artifact: &ArtifactFacet,
    proof: &ValidatedOwnerArtifact,
    members: &BTreeMap<String, Vec<u8>>,
) -> FacetResult<Vec<FacetStateMember>> {
    matches_proof(artifact, proof)?;
    let ArtifactFacetStorage::Embedded {
        root_member,
        member_paths,
    } = &artifact.storage
    else {
        return Err(unsupported_payload(
            "external state closure requires an embedded root",
        ));
    };
    if members.get(root_member).map(Vec::as_slice) != Some(proof.original_bytes()) {
        return Err(mismatch(
            "raw state closure root bytes differ from owner proof",
        ));
    }
    if !matches!(
        artifact.kind,
        ArtifactFacetKind::Model | ArtifactFacetKind::FullRefit
    ) {
        if member_paths.len() != 1 || member_paths.first() != Some(root_member) {
            return Err(mismatch(
                "proven non-state facet must contain exactly its root member",
            ));
        }
        return Ok(Vec::new());
    }
    let expected = expected_facet_state_members(&artifact.id, proof)?;
    let mut paths = BTreeSet::from([root_member.as_str()]);
    let raw = raw_model_payloads(proof)?;
    for state in &expected {
        paths.insert(state.member_path.as_str());
        if members.get(&state.member_path) != raw.get(&state.artifact_id) {
            return Err(mismatch(format!(
                "external raw member for `{}` differs from owner root",
                state.artifact_id
            )));
        }
    }
    if paths
        != member_paths
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
        || paths.len() != member_paths.len()
    {
        return Err(mismatch(
            "raw state member_paths do not exactly cover the root package closure",
        ));
    }
    Ok(expected)
}

/// Verify the identical physical-closure rule used by the direct exporter.
/// Every proven embedded facet retains exact original root bytes. Model and
/// FullRefit additionally require their exact content-addressed state closure;
/// every other kind requires member_paths to contain only root_member.
pub fn validate_facet_members(
    artifact: &ArtifactFacet,
    proof: &ValidatedOwnerArtifact,
    members: &BTreeMap<String, Vec<u8>>,
) -> FacetResult<Vec<FacetStateMember>> {
    validate_facet_state_members(artifact, proof, members)
}

/// Capture current owner objects directly after training/package export and
/// before archive assembly. Original supplied bytes use the byte variant.
pub fn build_training_artifact_facets(
    outcome: &TrainingOutcome,
    package: &PortablePredictorPackage,
) -> FacetResult<ArtifactFacetExport> {
    if !predictor_matches_outcome(outcome, package)? {
        return Err(mismatch(
            "package does not exactly cross-link current TrainingOutcome",
        ));
    }
    let outcome_proof = owner_object(FACET_TRAINING_OUTCOME, outcome.schema_version, outcome)?;
    let package_proof = owner_object(FACET_PREDICTOR_PACKAGE, package.schema_version, package)?;
    build_training_proofs(outcome_proof, package_proof)
}

/// Original outcome/package bytes are kept byte-for-byte, not reconstructed
/// from their parsed objects. Companions are serialized by their owner here.
pub fn build_training_artifact_facets_from_bytes(
    outcome_bytes: &[u8],
    package_bytes: &[u8],
) -> FacetResult<ArtifactFacetExport> {
    // Read versions only after strict token/shape checks; known parsers remain
    // the authority for supported schema versions and self-fingerprints.
    let outcome: TrainingOutcome = strict_decode(
        std::str::from_utf8(outcome_bytes).map_err(|cause| invalid_payload(cause.to_string()))?,
    )?;
    let package: PortablePredictorPackage = strict_decode(
        std::str::from_utf8(package_bytes).map_err(|cause| invalid_payload(cause.to_string()))?,
    )?;
    let outcome = validate_owner_facet_bytes(
        &FacetContentType::new(FACET_TRAINING_OUTCOME, outcome.schema_version),
        outcome_bytes,
    )?
    .ok_or_else(|| error("outcome unsupported"))?;
    let package = validate_owner_facet_bytes(
        &FacetContentType::new(FACET_PREDICTOR_PACKAGE, package.schema_version),
        package_bytes,
    )?
    .ok_or_else(|| error("package unsupported"))?;
    build_training_proofs(outcome, package)
}

fn build_training_proofs(
    outcome_proof: ValidatedOwnerArtifact,
    package_proof: ValidatedOwnerArtifact,
) -> FacetResult<ArtifactFacetExport> {
    let OwnerPayload::Outcome(outcome) = &outcome_proof.payload else {
        return Err(error("expected outcome"));
    };
    let OwnerPayload::Predictor(package) = &package_proof.payload else {
        return Err(error("expected package"));
    };
    if !predictor_matches_outcome(outcome, package)? {
        return Err(mismatch("package/outcome mismatch"));
    }
    let mut export = ArtifactFacetExport::new();
    use ArtifactFacetKind::*;
    use ArtifactFacetRelation::*;
    export.add(
        "recipe",
        Recipe,
        owner_object(FACET_GRAPH_SPEC, 1, &package.template.graph)?,
        Some("dagml/recipe.json"),
    )?;
    export.add(
        "bundle",
        Attachment,
        owner_object(
            FACET_EXECUTION_BUNDLE,
            outcome.execution_bundle.schema_version,
            &outcome.execution_bundle,
        )?,
        Some("dagml/run.json"),
    )?;
    export.add(
        "score",
        Score,
        owner_object(
            FACET_SCORE_SET,
            outcome.score_set.schema_version,
            &outcome.score_set,
        )?,
        Some("dagml/score.json"),
    )?;
    if let Some(caches) = &outcome.portable_prediction_caches {
        export.add(
            "prediction",
            Prediction,
            owner_object(FACET_PREDICTION_CACHES, caches.schema_version, caches)?,
            Some("dagml/prediction.json"),
        )?;
        export.link(ProducedBy, "prediction", "run");
    }
    let plan = owner_object(FACET_EXECUTION_PLAN, 1, &outcome.effective_plan)?;
    export.add("plan", Attachment, plan, Some("dagml/plan.json"))?;
    let raw = package.execution_bundle.raw_artifact_payloads.clone();
    // TrainingOutcome's owner fingerprint binds the run label. Use the
    // full outcome identity for the run reference; bundle stays supporting
    // closure and cannot independently authorize a producer relation.
    export.add("run", Run, outcome_proof.clone(), None)?;
    export.add(
        "result",
        Result,
        outcome_proof.clone(),
        Some("dagml/result.json"),
    )?;
    // Same owner identity, resolved through this live context. Never assign
    // the outcome member to both facets or duplicate/rewrite owner-fingerprinted JSON.
    export.add("evidence", TrainingEvidence, outcome_proof, None)?;
    export.add("model", Model, package_proof, Some("dagml/model.json"))?;
    export.add_raw_members("model", &raw)?;
    for target in ["run", "result", "evidence", "model"] {
        export.link(RecipeOf, "recipe", target);
    }
    for source in ["model", "result", "evidence", "score"] {
        export.link(ProducedBy, source, "run");
    }
    export.link(EvidenceFor, "evidence", "model");
    export.link(Evaluates, "score", "result");
    export.validate()?;
    Ok(export)
}

/// No CV parent or independent score is invented for an initial full refit.
pub fn build_initial_refit_artifact_facets(
    package: &InitialFullRefitPackage,
) -> FacetResult<ArtifactFacetExport> {
    package.validate()?;
    let mut export = ArtifactFacetExport::new();
    let proof = owner_object(FACET_INITIAL_REFIT, package.schema_version, package)?;
    export.add(
        "recipe",
        ArtifactFacetKind::Recipe,
        owner_object(
            FACET_GRAPH_SPEC,
            1,
            &package.effective_plan.graph_plan.graph,
        )?,
        Some("dagml/recipe.json"),
    )?;
    export.add(
        "full_refit",
        ArtifactFacetKind::FullRefit,
        proof.clone(),
        Some("dagml/full_refit.json"),
    )?;
    export.add("model", ArtifactFacetKind::Model, proof.clone(), None)?;
    export.add("run", ArtifactFacetKind::Run, proof, None)?;
    export.add_raw_members("full_refit", &package.raw_artifact_payloads)?;
    for target in ["full_refit", "model", "run"] {
        export.link(ArtifactFacetRelation::RecipeOf, "recipe", target);
    }
    for source in ["full_refit", "model"] {
        export.link(ArtifactFacetRelation::ProducedBy, source, "run");
    }
    export.validate()?;
    Ok(export)
}

/// Target-bound child closure. The actual source package and outcome must be
/// supplied; source scores remain source evidence and are never child scores.
pub fn build_refit_artifact_facets(
    child: &PortableRefitPackageV3,
    parent: &TrainingOutcome,
    source_package: &PortablePredictorPackage,
) -> FacetResult<ArtifactFacetExport> {
    child.validate()?;
    child
        .outcome
        .recipe
        .validate_against_source_package(source_package)?;
    if child.outcome.recipe.parent_outcome != parent.to_reference()? {
        return Err(mismatch("refit parent outcome mismatch"));
    }
    let mut export = build_training_artifact_facets(parent, source_package)?;
    let proof = owner_object(FACET_REFIT_V3, child.schema_version, child)?;
    export.add(
        "full_refit",
        ArtifactFacetKind::FullRefit,
        proof.clone(),
        Some("dagml/full_refit.json"),
    )?;
    export.add("refit_run", ArtifactFacetKind::Run, proof, None)?;
    export.add_raw_members(
        "full_refit",
        &child.outcome.execution_bundle.raw_artifact_payloads,
    )?;
    export.link(ArtifactFacetRelation::RecipeOf, "recipe", "full_refit");
    export.link(ArtifactFacetRelation::ProducedBy, "full_refit", "refit_run");
    export.link(ArtifactFacetRelation::RefitOf, "full_refit", "evidence");
    export.validate()?;
    Ok(export)
}

/// Check declared host/runtime capabilities before a caller considers any
/// hydration. Success is a prerequisite check, NOT executable admission.
/// There is no resolver/loader callback here and `load_with` is never called.
pub fn check_facet_hydration_requirements(
    proof: &ValidatedOwnerArtifact,
    available_capabilities: &BTreeSet<String>,
    trust_host_serializers: bool,
) -> FacetResult<()> {
    let OwnerPayload::Predictor(package) = &proof.payload else {
        return Err(unsupported_payload(proof.content_type.id.clone()));
    };
    for binding in &package.artifact_bindings {
        let record = package
            .execution_bundle
            .refit_artifacts
            .iter()
            .find(|record| record.artifact.id == binding.artifact_id)
            .ok_or_else(|| error("artifact binding has no record"))?;
        let executable_serializer = matches!(
            record.artifact.backend,
            Some(
                crate::ArtifactBackend::Joblib
                    | crate::ArtifactBackend::Rds
                    | crate::ArtifactBackend::Mat
                    | crate::ArtifactBackend::Torch
                    | crate::ArtifactBackend::Tensorflow
            )
        );
        if binding.load_mode != ArtifactLoadMode::NativePortable || executable_serializer {
            if !trust_host_serializers {
                return Err(FacetRefusal::MissingTrust);
            }
            let backend = record
                .artifact
                .backend
                .ok_or_else(|| error("host serializer backend is undeclared"))?;
            for capability in [
                format!("controller:{}", record.controller_id),
                format!("serializer:{backend:?}").to_lowercase(),
            ] {
                if !available_capabilities.contains(&capability) {
                    return Err(FacetRefusal::MissingCapability { capability });
                }
            }
            if let Some(plugin) = &record.artifact.plugin {
                let version = record
                    .artifact
                    .plugin_version
                    .as_deref()
                    .ok_or_else(|| error("host runtime plugin version is undeclared"))?;
                let capability = format!("plugin:{plugin}:{version}");
                if !available_capabilities.contains(&capability) {
                    return Err(FacetRefusal::MissingCapability { capability });
                }
            }
        } else if !available_capabilities.contains("dagml.native-artifact-runtime.v1") {
            return Err(FacetRefusal::MissingCapability {
                capability: "dagml.native-artifact-runtime.v1".into(),
            });
        }
    }
    Ok(())
}

/// Caller-supplied IO descriptor vocabulary; it grants no owner admission.
#[derive(Clone, Copy, Debug)]
pub struct FacetDatasetContract<'a> {
    pub owner: &'a str,
    pub content_types: &'a [(&'a str, u32)],
}

/// Structural entrypoint roles, not proof of executable admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FacetEntrypointRole {
    StoredPredict {
        root_id: String,
        input_ids: Vec<String>,
    },
    ExternalPredict {
        root_id: String,
        schema_anchor: Option<String>,
    },
    ExternalCalibrate {
        model_id: String,
        outcome_id: String,
    },
}

/// Resolve only declared descriptor roles. IO validates dataset bytes separately.
pub fn resolve_facet_entrypoint(
    entrypoint_owner: &str,
    operation: &str,
    root_artifact: &str,
    input_artifacts: &[String],
    requirements: Option<&str>,
    graph: &ArtifactFacetGraph,
    dataset: &FacetDatasetContract<'_>,
) -> FacetResult<FacetEntrypointRole> {
    graph.validate_structure()?;
    if dataset.owner.trim().is_empty()
        || dataset.owner == "dag-ml"
        || dataset.content_types.is_empty()
        || dataset
            .content_types
            .iter()
            .any(|(id, version)| id.trim().is_empty() || *version == 0)
    {
        return Err(error("invalid dataset contract"));
    }
    if entrypoint_owner != "dag-ml" {
        return Err(mismatch("entrypoint must be DAG-owned"));
    }
    if !matches!(
        operation,
        "predict" | "predict-external" | "calibrate-external"
    ) {
        return Err(FacetRefusal::UnsupportedOperation {
            operation: operation.into(),
        });
    }
    let find = |id: &str| {
        graph
            .artifacts
            .iter()
            .find(|artifact| artifact.id == id)
            .ok_or_else(|| error("undeclared operation artifact"))
    };
    let root = find(root_artifact)?;
    let mut seen = BTreeSet::new();
    for id in input_artifacts {
        if !seen.insert(id) {
            return Err(error("duplicate operation artifact"));
        }
        find(id)?;
    }
    if let Some(id) = requirements {
        find(id)?;
    }
    let root_kind = if operation == "calibrate-external" {
        root.kind == ArtifactFacetKind::Model
    } else {
        matches!(
            root.kind,
            ArtifactFacetKind::Model | ArtifactFacetKind::FullRefit
        )
    };
    if root.owner != "dag-ml" || !root_kind {
        return Err(mismatch("invalid operation root"));
    }
    let embedded = |artifact: &ArtifactFacet| -> FacetResult<()> {
        if matches!(artifact.storage, ArtifactFacetStorage::Reference { .. }) {
            return Err(FacetRefusal::UnresolvedOwnerContent {
                artifact_id: artifact.id.clone(),
            });
        }
        Ok(())
    };
    embedded(root)?;
    let check_dataset = |artifact: &ArtifactFacet| -> FacetResult<()> {
        if artifact.kind != ArtifactFacetKind::Dataset
            || artifact.owner != dataset.owner
            || !dataset.content_types.contains(&(
                artifact.content_type.id.as_str(),
                artifact.content_type.version,
            ))
        {
            return Err(mismatch("dataset descriptor does not match contract"));
        }
        embedded(artifact)
    };
    match operation {
        "predict" => {
            if input_artifacts.len() != 1 {
                return Err(FacetRefusal::UnsupportedOperation {
                    operation: operation.into(),
                });
            }
            if requirements.is_some() {
                return Err(mismatch("stored predict requirements must be null"));
            }
            check_dataset(find(&input_artifacts[0])?)?;
            Ok(FacetEntrypointRole::StoredPredict {
                root_id: root.id.clone(),
                input_ids: input_artifacts.to_vec(),
            })
        }
        "predict-external" => {
            if !input_artifacts.is_empty() {
                return Err(mismatch("external operation inputs must be empty"));
            }
            if let Some(id) = requirements {
                check_dataset(find(id)?)?;
            }
            Ok(FacetEntrypointRole::ExternalPredict {
                root_id: root.id.clone(),
                schema_anchor: requirements.map(str::to_owned),
            })
        }
        "calibrate-external" => {
            if !input_artifacts.is_empty() {
                return Err(mismatch("external operation inputs must be empty"));
            }
            let id = requirements.ok_or_else(|| mismatch("calibration outcome is required"))?;
            let outcome = find(id)?;
            if outcome.owner != "dag-ml"
                || !matches!(
                    outcome.kind,
                    ArtifactFacetKind::Result | ArtifactFacetKind::TrainingEvidence
                )
            {
                return Err(mismatch("invalid calibration outcome"));
            }
            Ok(FacetEntrypointRole::ExternalCalibrate {
                model_id: root.id.clone(),
                outcome_id: outcome.id.clone(),
            })
        }
        _ => unreachable!(),
    }
}

/// Inspect-only admission from actual validated content. Execution operations
/// need their own live owner execution validator and are explicitly refused.
/// Container SHA/entrypoint binding belongs to Core, outside this API.
pub fn require_facet_operation(
    operation: &str,
    root_artifact: &str,
    input_artifacts: &[String],
    graph: &ArtifactFacetGraph,
    context: &FacetValidationContext<'_>,
) -> FacetResult<()> {
    validate_artifact_facet_links(graph, context)?;
    for id in std::iter::once(root_artifact).chain(input_artifacts.iter().map(String::as_str)) {
        if !graph.artifacts.iter().any(|artifact| artifact.id == id) {
            return Err(error("undeclared operation artifact"));
        }
        if !context.owner_proofs.contains_key(id) {
            return Err(FacetRefusal::UnresolvedOwnerContent {
                artifact_id: id.into(),
            });
        }
    }
    let proof = context.owner_proofs.get(root_artifact).ok_or_else(|| {
        FacetRefusal::UnresolvedOwnerContent {
            artifact_id: root_artifact.into(),
        }
    })?;
    match operation {
        "inspect" => Ok(()),
        "score-inspect"
            if matches!(proof.payload, OwnerPayload::Scores(_))
                && graph.artifacts.iter().any(|artifact| {
                    artifact.id == root_artifact && artifact.kind == ArtifactFacetKind::Score
                }) =>
        {
            Ok(())
        }
        _ => Err(FacetRefusal::UnsupportedOperation {
            operation: operation.into(),
        }),
    }
}

// Shared DAG-owned declarative RolePipeline validation. Methods state stays
// opaque here; wrapper size is limited by the host allocator.
pub(crate) mod role_payload {
    use super::*;
    use crate::Result;
    use crate::{ArtifactBackend, RefitArtifactRecord};
    use serde_json::Value;
    fn role_refuse<T>(message: &str) -> Result<T> {
        Err(DagMlError::RuntimeValidation(message.into()))
    }
    #[derive(Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct MethodsRolePipelinePayload {
        schema: String,
        node_id: String,
        params_fingerprint: String,
        target_names: Vec<String>,
        steps: Vec<MethodsRolePipelineStep>,
        feature_names: Vec<String>,
        states: Vec<Vec<u8>>,
    }

    #[derive(Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct MethodsRolePipelineStep {
        #[serde(default)]
        class: Option<String>,
        #[serde(default, rename = "methodId")]
        method_id: Option<String>,
        #[serde(default)]
        params: serde_json::Map<String, Value>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct MethodsRolePipelinePayloadV2 {
        schema: String,
        controller_version: String,
        recipe_profile: String,
        node_id: String,
        params_fingerprint: String,
        target_names: Vec<String>,
        steps: Value,
        feature_names: Vec<String>,
        states: Vec<Vec<u8>>,
    }

    fn is_wasm_pipeline_steps_v2(record: &RefitArtifactRecord) -> bool {
        record.artifact.plugin.as_deref() == Some("dagml.methods.wasm.regression")
            && record.artifact.plugin_version.as_deref() == Some("1.1.0")
            && record.artifact.controller_id.as_str() == "controller:methods.wasm.regression"
            && record.controller_id.as_str() == "controller:methods.wasm.regression"
    }

    fn pipeline_steps_v1(value: &Value) -> Result<Vec<MethodsRolePipelineStep>> {
        let steps = value
            .as_array()
            .filter(|steps| !steps.is_empty())
            .ok_or_else(|| {
                DagMlError::RuntimeValidation("pipeline_steps_v1 needs a nonempty recipe".into())
            })?;
        for step in steps {
            let Some(step) = step.as_object() else {
                return role_refuse("pipeline_steps_v1 step must be an object");
            };
            let nonempty_text = |key| {
                step.get(key)
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.is_empty())
            };
            let has_exclusive_declaration = (nonempty_text("class")
                && !step.contains_key("methodId"))
                || (nonempty_text("methodId") && !step.contains_key("class"));
            if step
                .keys()
                .any(|key| !matches!(key.as_str(), "class" | "methodId" | "params"))
                || !has_exclusive_declaration
                || step.get("params").is_some_and(|params| {
                    params
                        .as_object()
                        .is_none_or(|params| params.contains_key("pipeline_steps_v1"))
                })
            {
                return role_refuse("pipeline_steps_v1 steps need one nonempty class/methodId and optional params object without the reserved recipe key");
            }
        }
        Ok(serde_json::from_value(value.clone())?)
    }

    fn decode_wasm_pipeline_steps_v2(bytes: &[u8]) -> Result<MethodsRolePipelinePayloadV2> {
        let saved: MethodsRolePipelinePayloadV2 = serde_json::from_slice(bytes)?;
        if saved.schema != "dagml.methods.regression.v2"
            || saved.controller_version != "1.1.0"
            || saved.recipe_profile != "pipeline_steps.v1"
        {
            return role_refuse(
                "RolePipeline v2 requires the exact WASM 1.1.0 pipeline_steps.v1 profile",
            );
        }
        Ok(saved)
    }

    pub(crate) fn validate_methods_role_pipeline_recipe(
        record: &RefitArtifactRecord,
        bytes: &[u8],
        plan: &ExecutionPlan,
    ) -> Result<()> {
        let node = plan.node_plans.get(&record.node_id).ok_or_else(|| {
            DagMlError::RuntimeValidation("RolePipeline has no effective node plan".into())
        })?;
        if record.controller_id.as_str() == crate::METHODS_NATIVE_REGRESSION_CONTROLLER {
            if record.artifact.plugin.as_deref() != Some(crate::METHODS_NATIVE_REGRESSION_PLUGIN) {
                return role_refuse("native PLS wrapper must retain its native owner identity");
            }
            crate::methods_phase_controls::validate_native_pls_phase_plan(plan)?;
            let expected =
                crate::NativePlsRoleParams::from_params(&node.params)?.recipe(crate::Phase::Refit);
            let saved: MethodsRolePipelinePayload = serde_json::from_slice(bytes)?;
            let expected: Vec<MethodsRolePipelineStep> =
                serde_json::from_value(serde_json::to_value(expected)?)?;
            if saved.steps != expected || record.params_fingerprint != node.params_fingerprint {
                return role_refuse(
                    "native PLS RAW recipe differs from the validated effective REFIT phase",
                );
            }
            return Ok(());
        }
        let graph_node = plan
            .graph_plan
            .graph
            .nodes
            .iter()
            .find(|node| node.id == record.node_id)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation("RolePipeline has no effective graph operator".into())
            })?;
        let operator = graph_node.operator.as_ref().ok_or_else(|| {
            DagMlError::RuntimeValidation("RolePipeline has no explicit graph operator".into())
        })?;
        let kind = operator
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "RolePipeline graph operator has no native type".into(),
                )
            })?;
        if is_wasm_pipeline_steps_v2(record) {
            if node.controller_id != record.controller_id
                || node.controller_version != "1.1.0"
                || kind != "N4mRolePipeline"
            {
                return role_refuse("RolePipeline v2 requires its exact effective WASM 1.1.0 controller and N4mRolePipeline operator");
            }
            let base = node.params.get("pipeline_steps_v1").ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "RolePipeline v2 requires explicit node.params.pipeline_steps_v1".into(),
                )
            })?;
            let mut expected = pipeline_steps_v1(base)?;
            if operator.get("steps").is_some_and(|steps| steps != base) {
                return role_refuse(
                    "RolePipeline operator.steps differs from its base pipeline_steps_v1 recipe",
                );
            }
            let last = expected.last_mut().expect("nonempty recipe checked above");
            last.params.extend(
                node.params
                    .iter()
                    .filter(|(key, _)| key.as_str() != "pipeline_steps_v1")
                    .map(|(key, value)| (key.clone(), value.clone())),
            );
            let saved = decode_wasm_pipeline_steps_v2(bytes)?;
            if pipeline_steps_v1(&saved.steps)? != expected
                || record.params_fingerprint != node.params_fingerprint
            {
                return role_refuse("RolePipeline saved recipe differs from its effective graph and selected parameters");
            }
            return Ok(());
        }
        let mut expected: Vec<MethodsRolePipelineStep> = if kind == "N4mRolePipeline" {
            serde_json::from_value(operator.get("steps").cloned().ok_or_else(|| {
                DagMlError::RuntimeValidation("RolePipeline graph operator has no steps".into())
            })?)?
        } else if kind.strip_prefix("n4m:").is_some_and(|id| !id.is_empty()) {
            vec![MethodsRolePipelineStep {
                class: Some(kind.into()),
                method_id: None,
                params: serde_json::Map::new(),
            }]
        } else {
            return role_refuse(
                "RolePipeline graph operator is outside the native Methods recipe profile",
            );
        };
        let last = expected.last_mut().ok_or_else(|| {
            DagMlError::RuntimeValidation("RolePipeline planned recipe is empty".into())
        })?;
        // This is the controller's declarative rule, not a numerical lowerer:
        // selected effective node parameters override only the final recipe step.
        last.params.extend(
            node.params
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        let saved: MethodsRolePipelinePayload = serde_json::from_slice(bytes)?;
        if saved.steps != expected || record.params_fingerprint != node.params_fingerprint {
            return role_refuse(
            "RolePipeline saved recipe differs from its effective graph and selected parameters",
        );
        }
        Ok(())
    }

    /// Validate the declarative wrapper without interpreting Methods estimator bytes.
    /// Model hydration and numerical validation remain controller/Methods-owned.
    pub(crate) fn validate_role_payload(record: &RefitArtifactRecord, bytes: &[u8]) -> Result<()> {
        let artifact = &record.artifact;
        artifact.validate_portable()?;
        let wasm_v2 = is_wasm_pipeline_steps_v2(record);
        let legacy_v1 = matches!(
            artifact.plugin.as_deref(),
            Some(
                "dagml.methods.wasm.regression"
                    | "dagml.methods.python.regression"
                    | "dagml.methods.r.regression"
                    | "dagml.methods.octave.regression"
                    | "dagml.methods.native.regression"
            )
        ) && artifact.plugin_version.as_deref() == Some("1.0.0");
        if artifact.kind != "methods_role_pipeline"
            || artifact.backend != Some(ArtifactBackend::Raw)
            || !(wasm_v2 || legacy_v1)
            || artifact.native_predictor_descriptor.is_some()
            || artifact.native_estimator_descriptor.is_some()
            || bytes.is_empty()
        {
            return role_refuse("Archive V2 requires the trusted Methods RolePipeline RAW codec");
        }
        if artifact.plugin.as_deref() == Some(crate::METHODS_NATIVE_REGRESSION_PLUGIN)
            && (artifact.controller_id.as_str() != crate::METHODS_NATIVE_REGRESSION_CONTROLLER
                || record.controller_id.as_str() != crate::METHODS_NATIVE_REGRESSION_CONTROLLER)
        {
            return role_refuse("native PLS RAW plugin requires its exact native controller owner");
        }
        if artifact.plugin.as_deref() == Some("dagml.methods.octave.regression")
            && (artifact.controller_id.as_str() != "controller:methods.octave.regression"
                || record.controller_id.as_str() != "controller:methods.octave.regression")
        {
            return role_refuse(
                "Octave Methods RAW plugin requires its exact Octave controller owner",
            );
        }
        if artifact.plugin.as_deref() == Some("dagml.methods.python.regression")
            && (artifact.controller_id.as_str() != "controller:methods.python.regression"
                || record.controller_id.as_str() != "controller:methods.python.regression")
        {
            return role_refuse(
                "Python Methods RAW plugin requires its exact Python controller owner",
            );
        }
        let raw = sha256(bytes);
        if artifact.content_fingerprint.as_deref() != Some(raw.as_str())
            || artifact.size_bytes != Some(bytes.len() as u64)
            || artifact.uri.as_deref() != Some(format!("artifacts/{raw}.json").as_str())
        {
            return role_refuse(
                "Archive V2 RolePipeline URI, size and raw SHA-256 must exactly bind its bytes",
            );
        }
        let text = std::str::from_utf8(bytes).map_err(|error| {
            DagMlError::RuntimeValidation(format!("RolePipeline payload is not UTF-8: {error}"))
        })?;
        crate::canonical::parse_typed_json(text).map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "RolePipeline payload is outside strict TCV1: {error}"
            ))
        })?;
        let payload: MethodsRolePipelinePayload = if wasm_v2 {
            let saved = decode_wasm_pipeline_steps_v2(bytes)?;
            MethodsRolePipelinePayload {
                schema: saved.schema,
                node_id: saved.node_id,
                params_fingerprint: saved.params_fingerprint,
                target_names: saved.target_names,
                steps: pipeline_steps_v1(&saved.steps)?,
                feature_names: saved.feature_names,
                states: saved.states,
            }
        } else {
            serde_json::from_str(text)?
        };
        let unique_text = |values: &[String]| {
            !values.is_empty()
                && values.iter().all(|value| !value.is_empty())
                && values.iter().collect::<BTreeSet<_>>().len() == values.len()
        };
        if (!wasm_v2 && payload.schema != "dagml.methods.regression.v1")
            || payload.node_id != record.node_id.as_str()
            || payload.params_fingerprint != record.params_fingerprint
            || !unique_text(&payload.target_names)
            || !unique_text(&payload.feature_names)
            || payload.steps.is_empty()
            // n4m_role_pipeline_create/import_states use int32_t counts;
            // the separate N4MP wire format's 256-step cap does not apply.
            || i32::try_from(payload.steps.len()).is_err()
            || payload.steps.len() != payload.states.len()
            || payload
                .states
                .iter()
                .any(|state| !state.starts_with(b"N4ME"))
        {
            return role_refuse("Archive V2 RolePipeline wrapper does not bind a complete native recipe and state sequence");
        }
        for step in &payload.steps {
            // Each state is N4ME, whose method-id reader limits names to
            // 256 bytes. Parameter counts instead belong to each MethodSpec.
            let valid = match (&step.class, &step.method_id) {
                (Some(class), None) => class
                    .strip_prefix("n4m:")
                    .is_some_and(|id| !id.is_empty() && id.len() <= 256),
                (None, Some(id)) => !id.is_empty() && id.len() <= 256,
                _ => false,
            };
            if !valid {
                return role_refuse(
                    "Archive V2 RolePipeline steps must be explicit native Methods declarations",
                );
            }
        }
        Ok(())
    }
}

type OwnerModelClosure<'a> = (
    &'a ExecutionPlan,
    &'a [crate::RefitArtifactRecord],
    &'a BTreeMap<crate::ArtifactId, Vec<u8>>,
);

fn model_closure(payload: &OwnerPayload) -> Option<OwnerModelClosure<'_>> {
    match payload {
        OwnerPayload::Outcome(value) => Some((
            &value.effective_plan,
            &value.execution_bundle.refit_artifacts,
            &value.execution_bundle.raw_artifact_payloads,
        )),
        OwnerPayload::Predictor(value) => Some((
            &value.effective_plan,
            &value.execution_bundle.refit_artifacts,
            &value.execution_bundle.raw_artifact_payloads,
        )),
        OwnerPayload::ChildRefit(value) => Some((
            &value.outcome.effective_plan,
            &value.outcome.execution_bundle.refit_artifacts,
            &value.outcome.execution_bundle.raw_artifact_payloads,
        )),
        _ => None,
    }
}

fn validate_record_composite(
    record: &crate::RefitArtifactRecord,
    bytes: Option<&Vec<u8>>,
    plan: &ExecutionPlan,
) -> FacetResult<()> {
    let needs_wrapper = matches!(
        record.artifact.kind.as_str(),
        "methods_role_pipeline"
            | "methods_multimodal_pipeline"
            | "methods_role_classifier_pipeline"
            | "methods_multimodal_classifier_pipeline"
    );
    if needs_wrapper {
        let bytes =
            bytes.ok_or_else(|| error("owner composite has no complete raw wrapper bytes"))?;
        match record.artifact.kind.as_str() {
            "methods_role_pipeline" => {
                role_payload::validate_role_payload(record, bytes)?;
                role_payload::validate_methods_role_pipeline_recipe(record, bytes, plan)?;
            }
            "methods_multimodal_pipeline" => {
                crate::validate_methods_multimodal_pipeline_payload(record, bytes)?;
                crate::validate_methods_multimodal_pipeline_recipe(record, bytes, plan)?;
            }
            _ => crate::validate_methods_classifier_payload(record, bytes, plan)?,
        }
    }
    Ok(())
}

fn validate_owner_composites(payload: &OwnerPayload) -> FacetResult<()> {
    if let Some((plan, records, raw)) = model_closure(payload) {
        for record in records {
            validate_record_composite(record, raw.get(&record.artifact.id), plan)?;
        }
    }
    if let OwnerPayload::InitialRefit(value) = payload {
        for artifact in &value.artifacts {
            validate_record_composite(
                &artifact.record,
                value
                    .raw_artifact_payloads
                    .get(&artifact.record.artifact.id),
                &value.effective_plan,
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FacetModelStateInspection {
    pub artifact_id: String,
    pub verdict: FacetLinkVerdict,
    pub native_descriptor: Option<crate::NativePredictorDescriptorV1>,
}

/// Payload validation proves owner-fingerprinted control closure, not opaque fitted-state
/// semantics. Without a live owner runtime those state claims are unverified.
pub fn inspect_facet_model_states(
    proof: &ValidatedOwnerArtifact,
) -> FacetResult<Vec<FacetModelStateInspection>> {
    let records = model_records(&proof.payload)?;
    Ok(records
        .into_iter()
        .map(|record| FacetModelStateInspection {
            artifact_id: record.artifact.id.to_string(),
            verdict: FacetLinkVerdict::unverified(
                "actual owner runtime inspection absent; model state remains opaque",
            ),
            native_descriptor: None,
        })
        .collect())
}

fn model_records(payload: &OwnerPayload) -> FacetResult<Vec<&crate::RefitArtifactRecord>> {
    if let Some((_, records, _)) = model_closure(payload) {
        return Ok(records.iter().collect());
    }
    if let OwnerPayload::InitialRefit(package) = payload {
        return Ok(package
            .artifacts
            .iter()
            .map(|artifact| &artifact.record)
            .collect());
    }
    Err(unsupported_payload("model state closure"))
}

/// Explicit optional native inspection. A configured actual MethodsRuntime is
/// required; the existing inspector calls inspect_n4mm and never hydrates.
#[cfg(feature = "methods-optimizer")]
pub fn inspect_facet_model_states_with_methods(
    proof: &ValidatedOwnerArtifact,
    runtime: &crate::MethodsRuntime,
) -> FacetResult<Vec<FacetModelStateInspection>> {
    n4m::configure_library(runtime.library_path()).map_err(|source| {
        FacetRefusal::RuntimeUnavailable {
            source: Box::new(source),
        }
    })?;
    let mut report = inspect_facet_model_states(proof)?;
    let raw = if let Some((_, _, raw)) = model_closure(&proof.payload) {
        raw
    } else if let OwnerPayload::InitialRefit(package) = &proof.payload {
        &package.raw_artifact_payloads
    } else {
        return Err(unsupported_payload("raw model closure"));
    };
    for (item, record) in report.iter_mut().zip(model_records(&proof.payload)?) {
        if record.artifact.kind != "n4m_model" {
            item.verdict=FacetLinkVerdict::unverified("native facet inspector does not support this owner state family; bytes remain opaque");
            continue;
        }
        let Some(bytes) = raw.get(&record.artifact.id) else {
            item.verdict = FacetLinkVerdict::unverified(
                "native state bytes are unresolved; metadata alone cannot establish state facts",
            );
            continue;
        };
        let descriptor =
            crate::inspect_methods_native_predictor_descriptor_v1(&record.controller_id, bytes)?;
        if record
            .artifact
            .native_predictor_descriptor
            .as_ref()
            .is_some_and(|declared| declared != &descriptor)
        {
            return Err(mismatch(
                "actual Methods state inspection mismatches supplied descriptor",
            ));
        }
        item.native_descriptor = Some(descriptor);
        item.verdict = FacetLinkVerdict::comparison(
            true,
            "native state facts reported by actual Methods inspect_n4mm through the configured immutable process library (no hydration)",
        );
    }
    Ok(report)
}

/// Configure an actual owner runtime, preserving availability failures as a
/// distinct typed refusal. This does not hydrate or inspect model state.
#[cfg(feature = "methods-optimizer")]
pub fn configure_facet_methods_runtime(
    path: impl AsRef<std::path::Path>,
) -> FacetResult<crate::MethodsRuntime> {
    crate::MethodsRuntime::configure(path).map_err(|source| FacetRefusal::RuntimeUnavailable {
        source: Box::new(source),
    })
}

/// Borrowed projection of existing schema2 owner state, never persisted admission
/// or an additional calibration artifact. Owner validation checks the retained
/// replay and the pre-conformal anchor, rather than the changed outcome hash.
#[derive(Debug, Serialize)]
pub struct FacetCalibrationView<'a> {
    pub binding_id: &'a str,
    pub target_names: &'a [String],
    pub calibration_fingerprint: &'a str,
    pub pre_conformal_outcome_fingerprint: &'a str,
    pub bundle_reference: &'a crate::ConformalCalibrationRef,
    pub calibration_replay_outcome_fingerprint: &'a str,
}

pub fn facet_calibration_view(
    proof: &ValidatedOwnerArtifact,
) -> FacetResult<Option<FacetCalibrationView<'_>>> {
    let (calibration, replay, reference) = match &proof.payload {
        OwnerPayload::Outcome(outcome) => {
            outcome.validate()?;
            (
                &outcome.conformal_calibration,
                &outcome.conformal_calibration_replay,
                &outcome.execution_bundle.conformal_calibration,
            )
        }
        OwnerPayload::Predictor(package) => {
            package.validate()?;
            (
                &package.conformal_calibration,
                &package.conformal_calibration_replay,
                &package.execution_bundle.conformal_calibration,
            )
        }
        _ => return Err(unsupported_payload(proof.content_type.id.clone())),
    };
    match (calibration, replay, reference) {
        (None, None, None) => Ok(None),
        (Some(calibration), Some(replay), Some(reference)) => Ok(Some(FacetCalibrationView {
            binding_id: &calibration.binding_id,
            target_names: &calibration.target_names,
            calibration_fingerprint: &calibration.calibration_fingerprint,
            pre_conformal_outcome_fingerprint: &calibration
                .context
                .source_training_outcome_fingerprint,
            bundle_reference: reference,
            calibration_replay_outcome_fingerprint: &replay.outcome_fingerprint,
        })),
        _ => Err(mismatch("incomplete owner calibration closure")),
    }
}

/// Actual replay/truth inputs to an explicit calibration action. The caller
/// supplies no calibration context or provenance hashes; the owner derives them.
pub struct FacetCalibrationRequest<'a> {
    pub replay: &'a crate::TrainingReplayOutcome,
    pub binding_id: &'a str,
    pub relations: &'a crate::SampleRelationSet,
    pub truth: crate::ConformalCalibrationTruth,
    pub coverages: Vec<f64>,
    pub multi_target_policy: crate::ConformalMultiTargetPolicy,
    pub small_sample_policy: crate::ConformalSmallSamplePolicy,
}

struct CalibrationPrefix<'a> {
    source: &'a TrainingOutcome,
    package: &'a PortablePredictorPackage,
}

fn calibration_prefix<'a>(
    graph: &ArtifactFacetGraph,
    context: &'a FacetValidationContext<'_>,
    members: &BTreeMap<String, Vec<u8>>,
    outcome_id: &str,
    package_id: &str,
    binding_id: &str,
) -> FacetResult<CalibrationPrefix<'a>> {
    validate_facet_export_members(graph, context, members)?;
    let outcome_artifact = graph
        .artifacts
        .iter()
        .find(|artifact| artifact.id == outcome_id)
        .ok_or_else(|| error("calibration outcome is undeclared"))?;
    let package_artifact = graph
        .artifacts
        .iter()
        .find(|artifact| artifact.id == package_id)
        .ok_or_else(|| error("calibration model is undeclared"))?;
    if !matches!(
        outcome_artifact.kind,
        ArtifactFacetKind::Result | ArtifactFacetKind::TrainingEvidence
    ) || package_artifact.kind != ArtifactFacetKind::Model
    {
        return Err(mismatch(
            "calibration requires declared result/evidence and model facets",
        ));
    }
    let outcome = context.owner_proofs.get(outcome_id).ok_or_else(|| {
        FacetRefusal::UnresolvedOwnerContent {
            artifact_id: outcome_id.into(),
        }
    })?;
    let package = context.owner_proofs.get(package_id).ok_or_else(|| {
        FacetRefusal::UnresolvedOwnerContent {
            artifact_id: package_id.into(),
        }
    })?;
    matches_proof(outcome_artifact, outcome)?;
    validate_facet_members(package_artifact, package, members)?;
    let (OwnerPayload::Outcome(source), OwnerPayload::Predictor(package)) =
        (&outcome.payload, &package.payload)
    else {
        return Err(unsupported_payload(
            "calibration requires outcome and predictor package",
        ));
    };
    if !predictor_matches_outcome(source, package)? {
        return Err(mismatch("calibration source package/outcome mismatch"));
    }
    if source.conformal_calibration.is_some() {
        return Err(FacetRefusal::UnsupportedOperation {
            operation: "recalibrate an already calibrated outcome".into(),
        });
    }
    let binding = package
        .output_bindings
        .iter()
        .find(|binding| binding.binding_id == binding_id)
        .ok_or_else(|| {
            mismatch("requested calibration binding is not a validated package output binding")
        })?;
    if binding.prediction_kind != crate::PredictionKind::RegressionPoint {
        return Err(FacetRefusal::UnsupportedOperation {
            operation: "calibrate a non-regression-point output binding".into(),
        });
    }
    Ok(CalibrationPrefix { source, package })
}

/// Validate the static calibration closure without invoking replay or runtime.
pub fn preflight_facet_calibration(
    graph: &ArtifactFacetGraph,
    context: &FacetValidationContext<'_>,
    members: &BTreeMap<String, Vec<u8>>,
    outcome_id: &str,
    package_id: &str,
    binding_id: &str,
) -> FacetResult<()> {
    calibration_prefix(graph, context, members, outcome_id, package_id, binding_id).map(|_| ())
}

/// Return a complete newly calibrated owner set. Does not patch a container,
/// claim preservation of unrelated attachments, train a model, or confer an
/// independent evaluation claim. Raw state and source scores remain unchanged.
pub fn calibrate_training_artifact_facets(
    graph: &ArtifactFacetGraph,
    context: &FacetValidationContext<'_>,
    members: &BTreeMap<String, Vec<u8>>,
    outcome_id: &str,
    package_id: &str,
    request: FacetCalibrationRequest<'_>,
) -> FacetResult<ArtifactFacetExport> {
    let CalibrationPrefix { source, package } = calibration_prefix(
        graph,
        context,
        members,
        outcome_id,
        package_id,
        request.binding_id,
    )?;
    let mut calibrated = source.clone();
    crate::calibrate_attached_training_replay_with_derived_context(
        &mut calibrated,
        request.replay,
        request.binding_id,
        request.relations,
        request.truth,
        request.coverages,
        request.multi_target_policy,
        request.small_sample_policy,
    )?;
    let mut derived = calibrated.to_portable_predictor_package(
        &package.package_id,
        package.fitted_artifact_mode,
        if package
            .artifact_bindings
            .iter()
            .any(|binding| binding.load_mode == ArtifactLoadMode::HostSidecar)
        {
            ArtifactLoadMode::HostSidecar
        } else {
            ArtifactLoadMode::NativePortable
        },
    )?;
    // Preserve each actual owner binding's mode, including mixed host/native
    // sets. Calibration itself performs no hydration of any serializer family.
    for binding in &mut derived.artifact_bindings {
        let original = package
            .artifact_bindings
            .iter()
            .find(|candidate| candidate.artifact_id == binding.artifact_id)
            .ok_or_else(|| mismatch("calibration changed artifact binding closure"))?;
        binding.load_mode = original.load_mode;
    }
    derived.package_fingerprint = derived.compute_fingerprint()?;
    derived.validate()?;
    build_training_artifact_facets(&calibrated, &derived)
}

/// Trusted in-process owner adapter contract for a frozen concrete cohort.
/// Implementations retain an immutable snapshot and its exact owner-derived
/// envelopes. This trait is not authentication, a numeric hash verifier, or a
/// defense against malicious Rust code/interior mutability. Host JSON cannot
/// construct an admitted request. Platform executors accept only their concrete
/// frozen cohort type, not an arbitrary implementation or later provider.
pub trait FacetReplayCohort {
    fn envelopes(&self) -> &BTreeMap<String, crate::ExternalDataPlanEnvelope>;
}

/// Explicit owner requests; kinds and capability strings never infer an action.
#[derive(Debug)]
pub enum FacetPredictRequest {
    Replay(crate::TrainingReplayRequest),
    InitialRefit { output_ids: Vec<String> },
}

/// Invocation-local owner preconditions plus the moved concrete cohort.
/// No Deserialize, Clone, public fields, token constructor, or buffer setters.
///
/// ```compile_fail
/// # use dag_ml_core::*;
/// let forged: AdmittedFacetReplay<()> = serde_json::from_str("{}").unwrap();
/// ```
pub struct AdmittedFacetReplay<C> {
    root: ValidatedOwnerArtifact,
    request: FacetPredictRequest,
    cohort: C,
    envelopes: BTreeMap<String, crate::ExternalDataPlanEnvelope>,
}

impl<C> AdmittedFacetReplay<C> {
    pub fn root_identity(&self) -> &OwnerIdentity {
        self.root.identity()
    }
    pub fn request(&self) -> &FacetPredictRequest {
        &self.request
    }
    /// Borrow only. No replacement input can be passed to the native executor.
    pub fn cohort(&self) -> &C {
        &self.cohort
    }
    pub fn envelopes(&self) -> &BTreeMap<String, crate::ExternalDataPlanEnvelope> {
        &self.envelopes
    }
}

/// Validate all supplied proven physical members and live semantic links before
/// owner request preflight. No native library, controller, hydration or numerical
/// callbacks occur. Unknown opaque content remains unverified; it cannot be the
/// selected executable root. C's separate dataset/evaluates/residual status stays.
pub fn admit_facet_predict<C: FacetReplayCohort>(
    graph: &ArtifactFacetGraph,
    context: &FacetValidationContext<'_>,
    members: &BTreeMap<String, Vec<u8>>,
    root_id: &str,
    request: FacetPredictRequest,
    cohort: C,
) -> FacetResult<AdmittedFacetReplay<C>> {
    // Use the identical C exporter/reader closure, including all proven roots.
    validate_facet_export_members(graph, context, members)?;
    let artifact = graph
        .artifacts
        .iter()
        .find(|artifact| artifact.id == root_id)
        .ok_or_else(|| error("selected replay root is undeclared"))?;
    if !matches!(
        artifact.kind,
        ArtifactFacetKind::Model | ArtifactFacetKind::FullRefit
    ) {
        return Err(mismatch("replay root must be a model or full_refit facet"));
    }
    let root =
        context
            .owner_proofs
            .get(root_id)
            .ok_or_else(|| FacetRefusal::UnresolvedOwnerContent {
                artifact_id: root_id.into(),
            })?;
    matches_proof(artifact, root)?;
    // Operation roots require exact supplied physical closure, not a reference
    // declaration, so no external resolution or later raw-state substitution.
    validate_facet_members(artifact, root, members)?;
    let envelopes = cohort.envelopes().clone();
    if envelopes.is_empty() {
        return Err(error("replay requires a concrete owner cohort"));
    }
    for envelope in envelopes.values() {
        envelope.validate()?;
    }
    match (&root.payload, &request) {
        (OwnerPayload::Predictor(package), FacetPredictRequest::Replay(request)) => {
            if request.phase != crate::Phase::Predict {
                return Err(FacetRefusal::UnsupportedOperation {
                    operation: format!("facet {:?}", request.phase),
                });
            }
            check_replay_envelope_keys(request, &envelopes)?;
            check_replay_cohort_targets(
                &package.effective_plan,
                &package.output_bindings,
                request,
                &envelopes,
            )?;
            crate::preflight_loaded_predictor_replay(package, request, &envelopes)?;
        }
        (OwnerPayload::ChildRefit(package), FacetPredictRequest::Replay(request)) => {
            if request.phase != crate::Phase::Predict {
                return Err(FacetRefusal::UnsupportedOperation {
                    operation: format!("facet {:?}", request.phase),
                });
            }
            check_replay_envelope_keys(request, &envelopes)?;
            check_replay_cohort_targets(
                &package.outcome.effective_plan,
                &package.outcome.output_bindings,
                request,
                &envelopes,
            )?;
            crate::preflight_loaded_portable_refit_replay_v3(package, request, &envelopes)?;
        }
        (OwnerPayload::InitialRefit(package), FacetPredictRequest::InitialRefit { output_ids }) => {
            let keys = package
                .effective_plan
                .node_plans
                .values()
                .flat_map(|node| &node.data_bindings)
                .map(|binding| {
                    crate::data_binding_requirement_key(&binding.node_id, &binding.input_name)
                })
                .collect::<BTreeSet<_>>();
            if keys.iter().collect::<BTreeSet<_>>() != envelopes.keys().collect() {
                return Err(mismatch(
                    "initial refit and frozen cohort binding keys differ",
                ));
            }
            let mut values = envelopes.values();
            let envelope = values.next().expect("nonempty checked");
            if values.any(|other| other != envelope) {
                return Err(mismatch(
                    "initial refit requires one actual shared cohort envelope",
                ));
            }
            crate::preflight_initial_full_refit_prediction(package, envelope, output_ids)?;
        }
        _ => return Err(unsupported_payload("owner root/request combination")),
    }
    Ok(AdmittedFacetReplay {
        root: root.clone(),
        request,
        cohort,
        envelopes,
    })
}

fn check_replay_envelope_keys(
    request: &crate::TrainingReplayRequest,
    envelopes: &BTreeMap<String, crate::ExternalDataPlanEnvelope>,
) -> FacetResult<()> {
    if request.data_envelope_keys.iter().collect::<BTreeSet<_>>() != envelopes.keys().collect() {
        return Err(mismatch(
            "replay request and frozen cohort binding keys differ",
        ));
    }
    Ok(())
}

fn check_replay_cohort_targets(
    plan: &ExecutionPlan,
    bindings: &[crate::OutputBinding],
    request: &crate::TrainingReplayRequest,
    envelopes: &BTreeMap<String, crate::ExternalDataPlanEnvelope>,
) -> FacetResult<()> {
    crate::replay::validate_replay_cohort_target_claims(
        plan,
        bindings.iter(),
        request,
        envelopes,
        true,
    )
    .map_err(crate::replay::ReplayCohortTargetError::into_facet_refusal)
}

#[cfg(feature = "methods-optimizer")]
mod facet_methods {
    use super::*;
    use crate::runtime::{MethodsPlsDataset, RuntimeControllerRegistry};

    /// One trusted owner snapshot projection. Both fields MUST come from the
    /// same actual IO/DMD snapshot, with owner projection/feature order checked
    /// by its Rust adapter. Separate arbitrary maps or declared hashes alone
    /// do not establish this association. Public fields are adapter inputs,
    /// not executable admission; the frozen/admitted types have private fields.
    pub struct MethodsFacetBinding {
        pub envelope: crate::ExternalDataPlanEnvelope,
        pub dataset: MethodsPlsDataset,
    }

    /// Consuming trusted Rust owner boundary. Implement this on the actual
    /// IO/DMD snapshot adapter, not on deserialized assertions. The adapter is
    /// responsible for actual numeric projection and identity from ONE snapshot.
    /// Generic DAG cannot authenticate that owner or recompute its IO hashes.
    pub trait MethodsFacetCohortOwner {
        fn into_bindings(self) -> FacetResult<BTreeMap<String, MethodsFacetBinding>>;
    }

    /// Concrete immutable invocation-local numerical cohort. No Clone/serde,
    /// public fields, setters, loose-map constructor, or post-admission provider.
    pub struct FrozenMethodsFacetCohort {
        envelopes: BTreeMap<String, crate::ExternalDataPlanEnvelope>,
        datasets: BTreeMap<String, MethodsPlsDataset>,
    }

    impl FacetReplayCohort for FrozenMethodsFacetCohort {
        fn envelopes(&self) -> &BTreeMap<String, crate::ExternalDataPlanEnvelope> {
            &self.envelopes
        }
    }

    /// Freeze the owner projection once. Structural checks below complement,
    /// but do not replace, the actual trusted adapter's value/identity binding.
    pub fn freeze_methods_facet_cohort(
        owner: impl MethodsFacetCohortOwner,
    ) -> FacetResult<FrozenMethodsFacetCohort> {
        let bindings = owner.into_bindings()?;
        if bindings.is_empty() {
            return Err(error("empty owner cohort projection"));
        }
        let mut envelopes = BTreeMap::new();
        let mut datasets = BTreeMap::new();
        for (key, binding) in bindings {
            nonempty(&key)?;
            binding.envelope.validate()?;
            binding
                .dataset
                .validate("frozen facet owner projection", false)?;
            let cohort = binding.envelope.predict_cohort.as_ref().ok_or_else(|| {
                FacetRefusal::UnverifiedEvidence {
                    detail: "frozen native input requires the actual V2 PREDICT cohort".into(),
                }
            })?;
            if binding.dataset.sample_ids.iter().collect::<BTreeSet<_>>()
                != cohort.physical_sample_ids.iter().collect::<BTreeSet<_>>()
                || binding.dataset.target_names != cohort.target_names
            {
                return Err(mismatch(
                    "actual owner projection sample/target identities differ from cohort",
                ));
            }
            if binding.dataset.y.is_some() && cohort.target_content_fingerprint.is_none() {
                return Err(mismatch(
                    "target-bearing owner projection has no target content identity",
                ));
            }
            envelopes.insert(key.clone(), binding.envelope);
            datasets.insert(key, binding.dataset);
        }
        Ok(FrozenMethodsFacetCohort {
            envelopes,
            datasets,
        })
    }

    #[derive(Debug)]
    pub enum FacetMethodsReplayOutcome {
        Predictor(Box<crate::TrainingReplayOutcome>),
        ChildRefit(Box<crate::PortableRefitReplayOutcomeV3>),
        InitialRefit(Box<crate::InitialRefitReplayOutcome>),
    }

    /// Explicit native execution consumes the private admitted value. Runtime
    /// and output labels are the only late arguments: no data, envelopes,
    /// providers, controller callbacks or capability assertions may replace the
    /// concrete frozen cohort. Existing owner schedulers own all mathematics.
    pub fn execute_admitted_methods_facet_replay(
        admitted: AdmittedFacetReplay<FrozenMethodsFacetCohort>,
        runtime: crate::MethodsRuntime,
        outcome_id: String,
        run_id: crate::RunId,
    ) -> FacetResult<FacetMethodsReplayOutcome> {
        // n4m's immutable process loader refuses a different configured path.
        n4m::configure_library(runtime.library_path()).map_err(|source| {
            FacetRefusal::RuntimeUnavailable {
                source: Box::new(source),
            }
        })?;
        let native = crate::hpo::methods_native_controller_ids();
        let plan = match &admitted.root.payload {
            OwnerPayload::Predictor(package) => {
                if package
                    .artifact_bindings
                    .iter()
                    .any(|binding| binding.load_mode != ArtifactLoadMode::NativePortable)
                {
                    return Err(FacetRefusal::MissingTrust);
                }
                &package.effective_plan
            }
            OwnerPayload::ChildRefit(package) => &package.outcome.effective_plan,
            OwnerPayload::InitialRefit(package) => &package.effective_plan,
            _ => return Err(unsupported_payload("native PREDICT root")),
        };
        for node in plan.node_plans.values() {
            if !native.contains(&node.controller_id) {
                return Err(unsupported_payload(format!(
                    "native controller {}",
                    node.controller_id
                )));
            }
        }
        match (&admitted.root.payload, &admitted.request) {
            (OwnerPayload::Predictor(package), FacetPredictRequest::Replay(request)) => {
                Ok(FacetMethodsReplayOutcome::Predictor(Box::new(
                    crate::execute_loaded_methods_predictor_replay(
                        crate::MethodsPortablePredictorReplayInput {
                            package,
                            request,
                            data_envelopes: &admitted.envelopes,
                            methods_inputs: &admitted.cohort.datasets,
                            runtime,
                            outcome_id,
                            run_id,
                            warnings: Vec::new(),
                            diagnostics: BTreeMap::new(),
                        },
                    )?,
                )))
            }
            (OwnerPayload::ChildRefit(package), FacetPredictRequest::Replay(request)) => {
                Ok(FacetMethodsReplayOutcome::ChildRefit(Box::new(
                    crate::execute_loaded_methods_portable_refit_replay_v3(
                        crate::MethodsPortableRefitReplayInputV3 {
                            package,
                            request,
                            data_envelopes: &admitted.envelopes,
                            methods_inputs: &admitted.cohort.datasets,
                            runtime,
                            outcome_id,
                            run_id,
                            supplemental_controllers: RuntimeControllerRegistry::new(),
                            warnings: Vec::new(),
                            diagnostics: BTreeMap::new(),
                        },
                    )?,
                )))
            }
            (
                OwnerPayload::InitialRefit(package),
                FacetPredictRequest::InitialRefit { output_ids },
            ) => {
                let provider = crate::runtime::MethodsPortableReplayProvider::new(
                    admitted.envelopes.clone(),
                    admitted.cohort.datasets,
                )?;
                let mut controllers = RuntimeControllerRegistry::new();
                crate::hpo::register_methods_native_controllers(&mut controllers, runtime)?;
                let store = crate::runtime::InMemoryArtifactStore::new();
                Ok(FacetMethodsReplayOutcome::InitialRefit(Box::new(
                    crate::execute_initial_full_refit_prediction(crate::InitialRefitReplayInput {
                        package,
                        envelope: admitted.envelopes.values().next().expect("admitted cohort"),
                        output_ids,
                        run_id,
                        controllers: &controllers,
                        data_provider: &provider,
                        artifact_store: &store,
                    })?
                    .outcome,
                )))
            }
            _ => Err(unsupported_payload("admitted native request combination")),
        }
    }
}

#[cfg(feature = "methods-optimizer")]
pub use facet_methods::*;

#[cfg(test)]
mod wrapper_limits_tests {
    use super::*;
    use crate::RefitArtifactRecord;
    use serde_json::{json, Value};

    const RAW: &[u8] = include_bytes!(
        "../tests/fixtures/artifact_facets/native_role/source_role_pipeline_0.raw.json"
    );

    fn fixture() -> (RefitArtifactRecord, ExecutionPlan) {
        let outcome: TrainingOutcome = serde_json::from_str(include_str!(
            "../tests/fixtures/artifact_facets/native_role/training_outcome.json"
        ))
        .unwrap();
        (
            outcome.execution_bundle.refit_artifacts[0].clone(),
            outcome.effective_plan,
        )
    }

    fn bind_bytes(record: &mut RefitArtifactRecord, bytes: &[u8]) {
        let digest = sha256(bytes);
        record.artifact.uri = Some(format!("artifacts/{digest}.json"));
        record.artifact.content_fingerprint = Some(digest);
        record.artifact.size_bytes = Some(bytes.len() as u64);
    }

    #[test]
    fn more_than_four_million_nodes_reach_closed_wrapper_validation_through_facets() {
        let (mut record, plan) = fixture();
        // Deliberately invalid closed envelope, not fabricated fitted state.
        let mut bytes = Vec::with_capacity(8_000_100);
        bytes.extend_from_slice(b"{\"oversized_probe\":[");
        for _ in 0..4_000_000 {
            bytes.extend_from_slice(b"0,");
        }
        bytes.extend_from_slice(b"0]}");
        bind_bytes(&mut record, &bytes);
        let error = validate_record_composite(&record, Some(&bytes), &plan).unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `oversized_probe`"));
    }

    #[test]
    fn long_names_preserve_genuine_state_and_composite_recipe_binding() {
        let (mut record, plan) = fixture();
        let mut saved: Value = serde_json::from_slice(RAW).unwrap();
        saved["target_names"][0] = json!("t".repeat(4097));
        saved["feature_names"][0] = json!("f".repeat(4097));
        let bytes = serde_json::to_vec(&saved).unwrap();
        bind_bytes(&mut record, &bytes);
        validate_record_composite(&record, Some(&bytes), &plan).unwrap();
        saved["feature_names"][1] = saved["feature_names"][0].clone();
        let bytes = serde_json::to_vec(&saved).unwrap();
        bind_bytes(&mut record, &bytes);
        assert!(validate_record_composite(&record, Some(&bytes), &plan).is_err());
    }

    #[test]
    fn large_declarations_still_require_the_exact_selected_recipe() {
        for large_steps in [true, false] {
            let (mut record, plan) = fixture();
            let mut saved: Value = serde_json::from_slice(RAW).unwrap();
            // These are intentionally invalid recipes using unchanged genuine
            // state bytes. They must reach signed recipe refusal, never import.
            if large_steps {
                saved["steps"] = json!(vec![saved["steps"][0].clone(); 257]);
                saved["states"] = json!(vec![saved["states"][0].clone(); 257]);
            } else {
                for index in 0..257 {
                    saved["steps"][0]["params"][format!("extra_{index}")] = json!(index);
                }
            }
            let bytes = serde_json::to_vec(&saved).unwrap();
            bind_bytes(&mut record, &bytes);
            let error = validate_record_composite(&record, Some(&bytes), &plan).unwrap_err();
            assert!(error.to_string().contains(
                "native PLS RAW recipe differs from the validated effective REFIT phase"
            ));
        }
    }

    #[test]
    fn strict_tcv1_and_n4me_identity_guards_remain_on_the_composite_path() {
        let (mut record, plan) = fixture();
        let raw = std::str::from_utf8(RAW).unwrap();
        for invalid in [
            format!("{{\"schema\":null,{}", &raw[1..]),
            "{\"probe\":18446744073709551616}".into(),
            "{\"probe\":1e999}".into(),
            format!("{}0{}", "[".repeat(129), "]".repeat(129)),
        ] {
            let bytes = invalid.into_bytes();
            bind_bytes(&mut record, &bytes);
            let error = validate_record_composite(&record, Some(&bytes), &plan).unwrap_err();
            assert!(error.to_string().contains("outside strict TCV1"));
        }
        for mutation in ["state_count", "prefix", "id", "empty_names"] {
            let mut saved: Value = serde_json::from_slice(RAW).unwrap();
            match mutation {
                "state_count" => saved["states"] = json!([]),
                "prefix" => saved["states"][0][0] = json!(0),
                "id" => saved["steps"][0]["methodId"] = json!("m".repeat(257)),
                "empty_names" => saved["feature_names"] = json!([]),
                _ => unreachable!(),
            }
            let bytes = serde_json::to_vec(&saved).unwrap();
            bind_bytes(&mut record, &bytes);
            assert!(validate_record_composite(&record, Some(&bytes), &plan).is_err());
        }
    }

    #[test]
    #[ignore = "explicit 128 MiB materialized-input regression; genuine opaque state unchanged"]
    fn wrapper_above_128_mib_keeps_composite_and_byte_identity_checks() {
        let (mut record, plan) = fixture();
        let mut bytes = Vec::with_capacity(128 * 1024 * 1024 + 1);
        bytes.extend_from_slice(RAW);
        bytes.resize(128 * 1024 * 1024 + 1, b' ');
        bind_bytes(&mut record, &bytes);
        validate_record_composite(&record, Some(&bytes), &plan).unwrap();
        record.artifact.size_bytes = Some(bytes.len() as u64 - 1);
        assert!(validate_record_composite(&record, Some(&bytes), &plan).is_err());
    }
}
