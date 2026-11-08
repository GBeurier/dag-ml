//! Unwired V1 source/realization evidence. Checksums are not authentication.
//! Owner checkpoint, bridge, representation and lane gates remain prerequisites.
use crate::canonical::{parse_bounded_typed_json, tcv1_preimage, Tcv1Error, TypedCanonicalValue};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    cmp::Ordering,
    fmt,
    io::{self, Write},
    mem::size_of,
};
pub const PROFILE: &str = "dagml-realization-evidence-v1";
pub const MAX_EVIDENCE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;
const FRAME_DOMAIN: &[u8] = b"DAGML-REALIZATION-FRAME-V1\0";
const COMPLETION_DOMAIN: &[u8] = b"DAGML-REALIZATION-COMPLETE-V1\0";
const SCOPE_DOMAIN: &[u8] = b"DAGML-REALIZATION-SCOPES-V1\0";
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub memory_bytes: usize,
    pub chunk_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            memory_bytes: MAX_EVIDENCE_BYTES,
            chunk_bytes: 65536,
        }
    }
}
#[derive(Debug)]
pub enum CaptureError {
    Invalid(&'static str),
    EvidenceBudgetExceeded {
        limit_bytes: usize,
        charged_bytes: usize,
        requested_bytes: usize,
        stage: &'static str,
    },
    Allocation,
    Sink(io::Error),
}
impl CaptureError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Invalid(c) => c,
            Self::EvidenceBudgetExceeded { .. } => "evidence_budget_exceeded",
            Self::Allocation => "allocation",
            Self::Sink(_) => "sink",
        }
    }
}
impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}
impl std::error::Error for CaptureError {}
type Result<T> = std::result::Result<T, CaptureError>;
fn invalid<T>(code: &'static str) -> Result<T> {
    Err(CaptureError::Invalid(code))
}
fn require(ok: bool, code: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        invalid(code)
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or(CaptureError::Invalid("size_overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or(CaptureError::Invalid("size_overflow"))
}
fn identifier(s: &str) -> Result<()> {
    require(
        !s.is_empty()
            && s.len() <= 128
            && s.bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b':' | b'-')),
        "identifier",
    )
}
fn hex(s: &str) -> Result<()> {
    require(
        s.len() % 2 == 0
            && s.bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
        "hex",
    )
}
fn fingerprint(s: &str) -> Result<()> {
    hex(s)?;
    require(s.len() == 64, "fingerprint")
}
fn encode_hex(raw: &[u8]) -> String {
    const D: &[u8] = b"0123456789abcdef";
    let mut out = String::with_capacity(raw.len() * 2);
    for b in raw {
        out.push(D[(b >> 4) as usize] as char);
        out.push(D[(b & 15) as usize] as char);
    }
    out
}
fn digest(raw: &[u8]) -> String {
    encode_hex(&Sha256::digest(raw))
}
fn required_option<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::deserialize(d)
}
struct Budget {
    limit: usize,
    used: usize,
    peak: usize,
}
impl Budget {
    fn new(l: Limits) -> Result<Self> {
        require(
            (4096..=MAX_EVIDENCE_BYTES).contains(&l.memory_bytes)
                && l.chunk_bytes > 0
                && l.chunk_bytes <= (MAX_FRAME_BYTES - 2048) / 2,
            "memory_limit",
        )?;
        Ok(Self {
            limit: l.memory_bytes,
            used: 0,
            peak: 0,
        })
    }
    fn charge(&mut self, n: usize, stage: &'static str) -> Result<()> {
        if n > self.limit.saturating_sub(self.used) {
            return Err(CaptureError::EvidenceBudgetExceeded {
                limit_bytes: self.limit,
                charged_bytes: self.used,
                requested_bytes: n,
                stage,
            });
        }
        self.used += n;
        self.peak = self.peak.max(self.used);
        Ok(())
    }
    fn release(&mut self, n: usize) {
        debug_assert!(n <= self.used);
        self.used -= n;
    }
}
// Generic decoder preflight: no DOM, strings, integer conversion or owner calls.
// G(n,c,s)=(4n+8c)s includes minimum capacity and overlapping reallocations.
// These are conservative logical reservations; target allocator/RSS conformance
// is a separate compiler-admission gate. No Serde budget hook is asserted.
#[derive(Clone, Copy, Debug)]
pub struct DecodeAdmission {
    pub nodes: usize,
    pub input_bytes: usize,
    pub reservation_bytes: usize,
}
pub fn decoder_admission(raw: &str) -> Result<DecodeAdmission> {
    let mut quoted = false;
    let mut escaped = false;
    let mut depth = 0usize;
    let mut containers = 0usize;
    let mut nodes = 0usize;
    let mut strings = 0usize;
    let mut start = 0usize;
    let mut bytes = 0usize;
    let mut unicode_keys = 0usize;
    for (i, c) in raw.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                quoted = false;
                let token = &raw[start..i];
                bytes = add(bytes, token.len())?;
                if raw[i + 1..]
                    .trim_start_matches([' ', '\t', '\r', '\n'])
                    .starts_with(':')
                    && (!token.is_ascii() || token.contains("\\u"))
                {
                    unicode_keys = add(unicode_keys, token.len())?;
                }
            }
        } else if c == '"' {
            quoted = true;
            start = i + 1;
            strings = add(strings, 1)?;
            nodes = add(nodes, 1)?;
        } else if c == '[' || c == '{' {
            depth = add(depth, 1)?;
            require(depth <= 128, "parser_limit")?;
            containers = add(containers, 1)?;
            nodes = add(nodes, 1)?;
        } else if c == ']' || c == '}' {
            require(depth > 0, "json_syntax")?;
            depth -= 1;
        } else if c.is_ascii_digit() || matches!(c, '-' | 't' | 'f' | 'n') {
            nodes = add(nodes, 1)?;
        }
    }
    require(!quoted && depth == 0, "json_syntax")?;
    let slots = add(mul(4, nodes)?, mul(8, containers)?)?;
    let canonical = add(
        mul(
            slots,
            add(
                size_of::<TypedCanonicalValue>(),
                size_of::<(String, TypedCanonicalValue)>(),
            )?,
        )?,
        mul(
            add(mul(12, nodes)?, mul(16, containers)?)?,
            size_of::<(String, usize)>() + 16,
        )?,
    )?;
    let string_bound = add(mul(4, bytes)?, mul(16, strings)?)?;
    // Unicode table maximum full canonical decomposition=4 scalars. Includes
    // decomposition/recomposition buffers, sorting, UTF8 output and growth.
    let normalization = add(
        mul(
            unicode_keys,
            4 * 16
                + 16 * size_of::<(u8, char)>()
                + 16 * size_of::<char>()
                + 4 * size_of::<(u8, char)>(),
        )?,
        mul(32, strings)?,
    )?;
    // Serde Content largest payload fits four words plus tag/alignment envelope.
    // Includes internally tagged Event/nested unions AND untagged WireRecord
    // buffered branch replay; containers+1 conservatively bounds nested copies.
    let content = add(mul(slots, 18 * size_of::<usize>())?, string_bound)?;
    let serde = add(
        mul(containers.min(128) + 1, content)?,
        add(mul(slots, 256)?, string_bound)?,
    )?;
    Ok(DecodeAdmission {
        nodes,
        input_bytes: raw.len(),
        reservation_bytes: add(
            4096,
            add(canonical, add(string_bound, add(normalization, serde)?)?)?,
        )?,
    })
}
fn record_text(raw: &str) -> Result<&str> {
    let mut text = raw.strip_suffix('\n').unwrap_or(raw);
    if raw.ends_with('\n') {
        text = text.strip_suffix('\r').unwrap_or(text);
    }
    let mut q = false;
    let mut escape = false;
    for (i, c) in text.char_indices() {
        if q {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                q = false;
            }
        } else if c == '"' {
            q = true;
        } else if matches!(
            c,
            '\r' | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        ) {
            return invalid("record_separator");
        } else if c == '-' {
            let s = &text.as_bytes()[i..];
            if s.get(1) == Some(&b'0')
                && (s.len() == 2 || matches!(s[2], b' ' | b'\t' | b',' | b'}' | b']'))
            {
                return invalid("negative_zero");
            }
        }
    }
    Ok(text)
}
fn strict(raw: &str, nodes: usize) -> Result<TypedCanonicalValue> {
    parse_bounded_typed_json(record_text(raw)?, nodes).map_err(|e| {
        CaptureError::Invalid(match e {
            Tcv1Error::DuplicateObjectKey { .. } => "duplicate_key",
            Tcv1Error::NfcKeyCollision { .. } => "nfc_key_collision",
            Tcv1Error::IntegerOutOfRange { .. } => "integer_range",
            Tcv1Error::Binary64OutOfRange { .. } | Tcv1Error::NonFiniteBinary64 => "nonfinite",
            Tcv1Error::NestingTooDeep | Tcv1Error::LengthOverflow => "parser_limit",
            _ => "json_syntax",
        })
    })
}
fn decode<T: DeserializeOwned>(raw: &str, b: &mut Budget, stage: &'static str) -> Result<T> {
    let c = decoder_admission(raw)?;
    b.charge(c.reservation_bytes, stage)?;
    let out = (|| {
        let value = strict(raw, c.nodes)?;
        drop(value);
        serde_json::from_str(raw).map_err(|_| CaptureError::Invalid("event_schema"))
    })();
    b.release(c.reservation_bytes);
    out
}
struct CountWriter(usize);
impl Write for CountWriter {
    fn write(&mut self, s: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .checked_add(s.len())
            .ok_or_else(|| io::Error::other("size overflow"))?;
        Ok(s.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn encoded_len<T: Serialize>(v: &T) -> Result<usize> {
    let mut w = CountWriter(0);
    serde_json::to_writer(&mut w, v).map_err(|_| CaptureError::Invalid("serialization"))?;
    Ok(w.0)
}
fn owned(s: &str, b: &mut Budget) -> Result<String> {
    b.charge(s.len(), "owned string")?;
    let mut out = String::new();
    out.try_reserve_exact(s.len())
        .map_err(|_| CaptureError::Allocation)?;
    out.push_str(s);
    Ok(out)
}
// AVL index: fixed-size nodes use target size_of layout and explicitly
// charged Vec capacity. Lookup is logarithmic; closed-scope indices are freed.
struct IndexNode<T> {
    kind: u8,
    id: String,
    value: T,
    left: Option<usize>,
    right: Option<usize>,
    height: i16,
}
struct Index<T> {
    nodes: Vec<IndexNode<T>>,
    root: Option<usize>,
    work: Cell<u64>,
}
impl<T> Index<T> {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
            work: Cell::new(0),
        }
    }
    fn find(&self, k: u8, s: &str) -> Option<usize> {
        let mut at = self.root;
        while let Some(i) = at {
            self.work.set(self.work.get() + 1);
            let n = &self.nodes[i];
            match (k, s).cmp(&(n.kind, n.id.as_str())) {
                Ordering::Equal => return Some(i),
                Ordering::Less => at = n.left,
                Ordering::Greater => at = n.right,
            }
        }
        None
    }
    fn get(&self, k: u8, s: &str) -> Option<&T> {
        self.find(k, s).map(|i| &self.nodes[i].value)
    }
    fn get_mut(&mut self, k: u8, s: &str) -> Option<&mut T> {
        let i = self.find(k, s)?;
        Some(&mut self.nodes[i].value)
    }
    fn height(&self, x: Option<usize>) -> i16 {
        x.map_or(0, |i| self.nodes[i].height)
    }
    fn update(&mut self, i: usize) {
        self.nodes[i].height = 1 + self
            .height(self.nodes[i].left)
            .max(self.height(self.nodes[i].right));
    }
    fn rotate_right(&mut self, y: usize) -> usize {
        let x = self.nodes[y].left.expect("AVL left");
        let t = self.nodes[x].right;
        self.nodes[y].left = t;
        self.nodes[x].right = Some(y);
        self.update(y);
        self.update(x);
        x
    }
    fn rotate_left(&mut self, x: usize) -> usize {
        let y = self.nodes[x].right.expect("AVL right");
        let t = self.nodes[y].left;
        self.nodes[x].right = t;
        self.nodes[y].left = Some(x);
        self.update(x);
        self.update(y);
        y
    }
    fn attach(&mut self, at: Option<usize>, n: usize) -> usize {
        let Some(i) = at else { return n };
        let less = (self.nodes[n].kind, self.nodes[n].id.as_str())
            < (self.nodes[i].kind, self.nodes[i].id.as_str());
        if less {
            let left = self.attach(self.nodes[i].left, n);
            self.nodes[i].left = Some(left);
        } else {
            let right = self.attach(self.nodes[i].right, n);
            self.nodes[i].right = Some(right);
        }
        self.update(i);
        let balance = self.height(self.nodes[i].left) - self.height(self.nodes[i].right);
        if balance > 1 {
            let l = self.nodes[i].left.unwrap();
            if self.height(self.nodes[l].left) < self.height(self.nodes[l].right) {
                let v = self.rotate_left(l);
                self.nodes[i].left = Some(v);
            }
            return self.rotate_right(i);
        }
        if balance < -1 {
            let r = self.nodes[i].right.unwrap();
            if self.height(self.nodes[r].right) < self.height(self.nodes[r].left) {
                let v = self.rotate_right(r);
                self.nodes[i].right = Some(v);
            }
            return self.rotate_left(i);
        }
        i
    }
    fn insert(&mut self, k: u8, id: &str, value: T, b: &mut Budget) -> Result<()> {
        require(self.find(k, id).is_none(), "duplicate_occurrence")?;
        if self.nodes.len() == self.nodes.capacity() {
            let old = self.nodes.capacity();
            let new = old.saturating_mul(2).max(4);
            let charge = mul(new, size_of::<IndexNode<T>>())?;
            b.charge(charge, "reference index growth")?;
            self.nodes
                .try_reserve_exact(new - old)
                .map_err(|_| CaptureError::Allocation)?;
            b.release(mul(old, size_of::<IndexNode<T>>())?);
        }
        let id = owned(id, b)?;
        let i = self.nodes.len();
        self.nodes.push(IndexNode {
            kind: k,
            id,
            value,
            left: None,
            right: None,
            height: 1,
        });
        self.root = Some(self.attach(self.root, i));
        Ok(())
    }
    fn heap_bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<IndexNode<T>>()
            + self.nodes.iter().map(|n| n.id.capacity()).sum::<usize>()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeKind {
    Declaration,
    Search,
    Proposal,
    Terminal,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    New,
    Restored,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionKind {
    Explicit,
    Generated,
    Derived,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueSlot {
    Requested,
    Effective,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChoiceSemantics {
    Deterministic,
    Stochastic,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    Running,
    Completed,
    Failed,
    Pruned,
    Cancelled,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeMode {
    Transition,
    RestoredObservation,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    Mutex,
    Requires,
    Exclude,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Exhausted,
    OwnerExhausted,
    CountLimit,
    Cancelled,
    Failed,
    AdaptiveStop,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopBasisKind {
    OwnerStatus,
    OperationError,
    OwnerDecision,
    EnumerationComplete,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    Known,
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ownership {
    Confirmed,
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupState {
    NotAttempted,
    Confirmed,
    Failed,
    Unknown,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureStage {
    Ask,
    ParameterResolution,
    ProposalConversion,
    CandidatePlanning,
    RunContextCreation,
    ProviderCreation,
    ControllerCreation,
    Evaluation,
    Cleanup,
    OrphanRecovery,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Consumption {
    None,
    Observed,
    Unknown,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ref {
    pub scope_id: String,
    pub id: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclarationRef {
    pub scope_id: String,
    pub source_digest: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub observation: Origin,
    #[serde(deserialize_with = "required_option")]
    pub original_capture_id: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub original_scope_id: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub original_id: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub checkpoint_ref: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub id: String,
    pub version: String,
    pub context_hex: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParentSeed {
    pub seed_ref: Ref,
    pub value_slot: ValueSlot,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub availability: Availability,
    #[serde(deserialize_with = "required_option")]
    pub value_hex: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerFailure {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub status: TrialStatus,
    pub mode: OutcomeMode,
    pub stage: Observation,
    pub reason: Observation,
    #[serde(deserialize_with = "required_option")]
    pub failure: Option<OwnerFailure>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixRef {
    pub owner_ref_hex: String,
    pub checkpoint_fingerprint: String,
    #[serde(deserialize_with = "required_option")]
    pub pair_fingerprint: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedAttempt {
    #[serde(deserialize_with = "required_option")]
    pub owner_ref_hex: Option<String>,
    pub ownership: Ownership,
    #[serde(deserialize_with = "required_option")]
    pub status: Option<TrialStatus>,
    pub cleanup: CleanupState,
    #[serde(deserialize_with = "required_option")]
    pub diagnostic_hex: Option<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailedAttempt {
    #[serde(deserialize_with = "required_option")]
    pub failed_capture_id: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub failed_scope_id: Option<String>,
    #[serde(deserialize_with = "required_option")]
    pub failure_stage: Option<FailureStage>,
    pub causal_diagnostic_hex: String,
    pub owned_attempts: Vec<OwnedAttempt>,
    #[serde(deserialize_with = "required_option")]
    pub trusted_prefix_ref: Option<PrefixRef>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub scope_count: u64,
    pub visited_count: u64,
    pub candidate_count: u64,
    pub rejected_count: u64,
    pub terminal_without_candidate_count: u64,
    pub trial_count: u64,
    pub new_trial_count: u64,
    pub restored_trial_count: u64,
    pub resume_observation_count: u64,
    pub completed_count: u64,
    pub failed_count: u64,
    pub pruned_count: u64,
    pub cancelled_count: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SeedValue {
    U64 {
        value: u64,
    },
    OwnerEncoded {
        profile: String,
        version: String,
        value_hex: String,
    },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Declaration {
        scope_id: String,
        scope_kind: ScopeKind,
        #[serde(deserialize_with = "required_option")]
        parent_scope_id: Option<String>,
        declaration_text: String,
        source_digest: String,
        node_ids: Vec<String>,
    },
    ScopeOpen {
        scope_id: String,
        scope_kind: ScopeKind,
        parent_scope_id: String,
        declaration_ref: DeclarationRef,
    },
    CaptureProvenance {
        scope_id: String,
        failed_attempts: Vec<FailedAttempt>,
    },
    SeedDeclared {
        scope_id: String,
        seed_id: String,
        #[serde(deserialize_with = "required_option")]
        requested: Option<SeedValue>,
        provenance: Provenance,
    },
    EffectiveSeedResolved {
        scope_id: String,
        seed_id: String,
        declared_ref: Ref,
        effective: SeedValue,
        resolution: ResolutionKind,
        #[serde(deserialize_with = "required_option")]
        parent: Option<ParentSeed>,
        #[serde(deserialize_with = "required_option")]
        policy: Option<Policy>,
        first_actual_use: Ref,
        provenance: Provenance,
    },
    Choice {
        scope_id: String,
        choice_id: String,
        node_id: String,
        source_path_hex: Vec<String>,
        value_hex: String,
        semantics: ChoiceSemantics,
        #[serde(deserialize_with = "required_option")]
        seed_ref: Option<Ref>,
        provenance: Provenance,
    },
    SeedUse {
        scope_id: String,
        use_id: String,
        seed_ref: Ref,
        operation: Policy,
        provenance: Provenance,
    },
    Clone {
        scope_id: String,
        origin_node_id: String,
        clone_node_id: String,
        partition_digest: String,
        shared_choices: bool,
        choice_ids: Vec<String>,
    },
    Visit {
        scope_id: String,
        visit_id: String,
        choice_ids: Vec<String>,
        provenance: Provenance,
    },
    Reject {
        scope_id: String,
        visit_id: String,
        reason: RejectReason,
    },
    Candidate {
        scope_id: String,
        visit_id: String,
        owner_variant_id: String,
        owner_fingerprint: String,
        active_node_ids: Vec<String>,
        parameters_hex: String,
        provenance: Provenance,
    },
    TrialVisit {
        scope_id: String,
        trial_id: String,
        visit_ref: Ref,
        owner_trial_ref_hex: String,
        origin: Origin,
        #[serde(deserialize_with = "required_option")]
        candidate_ref: Option<Ref>,
        #[serde(deserialize_with = "required_option")]
        owner_status: Option<TrialStatus>,
        owner_evidence_hex: String,
        provenance: Provenance,
    },
    TrialBind {
        scope_id: String,
        trial_ref: Ref,
        candidate_ref: Ref,
    },
    TrialOutcome {
        scope_id: String,
        trial_ref: Ref,
        outcome: Outcome,
        owner_evidence_hex: String,
    },
    TrialResume {
        scope_id: String,
        trial_ref: Ref,
        #[serde(deserialize_with = "required_option")]
        observed_status: Option<TrialStatus>,
        owner_evidence_hex: String,
        provenance: Provenance,
    },
    OperationFailed {
        scope_id: String,
        operation_id: String,
        stage: FailureStage,
        diagnostic_hex: String,
        #[serde(deserialize_with = "required_option")]
        owner_ref_hex: Option<String>,
        ownership: Ownership,
        consumption: Consumption,
    },
    SearchClose {
        scope_id: String,
        reason: StopReason,
        #[serde(deserialize_with = "required_option")]
        unvisited: Option<bool>,
        basis: StopBasisKind,
        evidence: Observation,
    },
    ScopeClose {
        scope_id: String,
    },
}
impl Event {
    pub fn scope_id(&self) -> &str {
        match self {
            Self::Declaration { scope_id, .. } => scope_id,
            Self::ScopeOpen { scope_id, .. } => scope_id,
            Self::CaptureProvenance { scope_id, .. } => scope_id,
            Self::SeedDeclared { scope_id, .. } => scope_id,
            Self::EffectiveSeedResolved { scope_id, .. } => scope_id,
            Self::Choice { scope_id, .. } => scope_id,
            Self::SeedUse { scope_id, .. } => scope_id,
            Self::Clone { scope_id, .. } => scope_id,
            Self::Visit { scope_id, .. } => scope_id,
            Self::Reject { scope_id, .. } => scope_id,
            Self::Candidate { scope_id, .. } => scope_id,
            Self::TrialVisit { scope_id, .. } => scope_id,
            Self::TrialBind { scope_id, .. } => scope_id,
            Self::TrialOutcome { scope_id, .. } => scope_id,
            Self::TrialResume { scope_id, .. } => scope_id,
            Self::OperationFailed { scope_id, .. } => scope_id,
            Self::SearchClose { scope_id, .. } => scope_id,
            Self::ScopeClose { scope_id, .. } => scope_id,
        }
    }
}
impl SeedValue {
    fn validate(&self) -> Result<()> {
        if let Self::OwnerEncoded { value_hex, .. } = self {
            hex(value_hex)?;
        }
        Ok(())
    }
}
impl Ref {
    fn validate(&self) -> Result<()> {
        identifier(&self.scope_id)?;
        identifier(&self.id)?;
        Ok(())
    }
}
impl DeclarationRef {
    fn validate(&self) -> Result<()> {
        identifier(&self.scope_id)?;
        fingerprint(&self.source_digest)?;
        Ok(())
    }
}
impl Provenance {
    fn validate(&self) -> Result<()> {
        if let Some(v) = &self.original_capture_id {
            identifier(v)?;
        }
        if let Some(v) = &self.original_scope_id {
            identifier(v)?;
        }
        if let Some(v) = &self.original_id {
            identifier(v)?;
        }
        if let Some(v) = &self.checkpoint_ref {
            hex(v)?;
        }
        Ok(())
    }
}
impl Policy {
    fn validate(&self) -> Result<()> {
        identifier(&self.id)?;
        hex(&self.context_hex)?;
        Ok(())
    }
}
impl ParentSeed {
    fn validate(&self) -> Result<()> {
        self.seed_ref.validate()?;
        Ok(())
    }
}
impl Observation {
    fn validate(&self) -> Result<()> {
        if let Some(v) = &self.value_hex {
            hex(v)?;
            if self.availability == Availability::Known {
                require(!v.is_empty(), "observation")?;
            }
        }
        Ok(())
    }
}
impl Outcome {
    fn validate(&self) -> Result<()> {
        self.stage.validate()?;
        self.reason.validate()?;
        Ok(())
    }
}
impl PrefixRef {
    fn validate(&self) -> Result<()> {
        hex(&self.owner_ref_hex)?;
        require(!self.owner_ref_hex.is_empty(), "owner_prefix_ref")?;
        fingerprint(&self.checkpoint_fingerprint)?;
        if let Some(v) = &self.pair_fingerprint {
            fingerprint(v)?;
        }
        Ok(())
    }
}
impl OwnedAttempt {
    fn validate(&self) -> Result<()> {
        if let Some(v) = &self.owner_ref_hex {
            hex(v)?;
        }
        if let Some(v) = &self.diagnostic_hex {
            hex(v)?;
        }
        Ok(())
    }
}
impl FailedAttempt {
    fn validate(&self) -> Result<()> {
        if let Some(v) = &self.failed_capture_id {
            identifier(v)?;
        }
        if let Some(v) = &self.failed_scope_id {
            identifier(v)?;
        }
        hex(&self.causal_diagnostic_hex)?;
        for v in &self.owned_attempts {
            v.validate()?;
        }
        if let Some(v) = &self.trusted_prefix_ref {
            v.validate()?;
        }
        Ok(())
    }
}
impl Event {
    fn validate(&self) -> Result<()> {
        identifier(self.scope_id())?;
        match self {
            Self::Declaration {
                parent_scope_id,
                source_digest,
                node_ids,
                ..
            } => {
                if let Some(v) = parent_scope_id {
                    identifier(v)?;
                }
                fingerprint(source_digest)?;
                for v in node_ids {
                    identifier(v)?;
                }
            }
            Self::ScopeOpen {
                parent_scope_id,
                declaration_ref,
                ..
            } => {
                identifier(parent_scope_id)?;
                declaration_ref.validate()?;
            }
            Self::CaptureProvenance {
                failed_attempts, ..
            } => {
                for v in failed_attempts {
                    v.validate()?;
                }
            }
            Self::SeedDeclared {
                seed_id,
                requested,
                provenance,
                ..
            } => {
                identifier(seed_id)?;
                if let Some(v) = requested {
                    v.validate()?;
                }
                provenance.validate()?;
            }
            Self::EffectiveSeedResolved {
                seed_id,
                declared_ref,
                effective,
                parent,
                policy,
                first_actual_use,
                provenance,
                ..
            } => {
                identifier(seed_id)?;
                declared_ref.validate()?;
                effective.validate()?;
                if let Some(v) = parent {
                    v.validate()?;
                }
                if let Some(v) = policy {
                    v.validate()?;
                }
                first_actual_use.validate()?;
                provenance.validate()?;
            }
            Self::Choice {
                choice_id,
                node_id,
                source_path_hex,
                value_hex,
                seed_ref,
                provenance,
                ..
            } => {
                identifier(choice_id)?;
                identifier(node_id)?;
                for v in source_path_hex {
                    hex(v)?;
                }
                hex(value_hex)?;
                if let Some(v) = seed_ref {
                    v.validate()?;
                }
                provenance.validate()?;
            }
            Self::SeedUse {
                use_id,
                seed_ref,
                operation,
                provenance,
                ..
            } => {
                identifier(use_id)?;
                seed_ref.validate()?;
                operation.validate()?;
                provenance.validate()?;
            }
            Self::Clone {
                origin_node_id,
                clone_node_id,
                partition_digest,
                choice_ids,
                ..
            } => {
                identifier(origin_node_id)?;
                identifier(clone_node_id)?;
                fingerprint(partition_digest)?;
                for v in choice_ids {
                    identifier(v)?;
                }
            }
            Self::Visit {
                visit_id,
                choice_ids,
                provenance,
                ..
            } => {
                identifier(visit_id)?;
                for v in choice_ids {
                    identifier(v)?;
                }
                provenance.validate()?;
            }
            Self::Reject { visit_id, .. } => {
                identifier(visit_id)?;
            }
            Self::Candidate {
                visit_id,
                owner_variant_id,
                owner_fingerprint,
                active_node_ids,
                parameters_hex,
                provenance,
                ..
            } => {
                identifier(visit_id)?;
                identifier(owner_variant_id)?;
                fingerprint(owner_fingerprint)?;
                for v in active_node_ids {
                    identifier(v)?;
                }
                hex(parameters_hex)?;
                provenance.validate()?;
            }
            Self::TrialVisit {
                trial_id,
                visit_ref,
                owner_trial_ref_hex,
                candidate_ref,
                owner_evidence_hex,
                provenance,
                ..
            } => {
                identifier(trial_id)?;
                visit_ref.validate()?;
                hex(owner_trial_ref_hex)?;
                if let Some(v) = candidate_ref {
                    v.validate()?;
                }
                hex(owner_evidence_hex)?;
                provenance.validate()?;
            }
            Self::TrialBind {
                trial_ref,
                candidate_ref,
                ..
            } => {
                trial_ref.validate()?;
                candidate_ref.validate()?;
            }
            Self::TrialOutcome {
                trial_ref,
                outcome,
                owner_evidence_hex,
                ..
            } => {
                trial_ref.validate()?;
                outcome.validate()?;
                hex(owner_evidence_hex)?;
            }
            Self::TrialResume {
                trial_ref,
                owner_evidence_hex,
                provenance,
                ..
            } => {
                trial_ref.validate()?;
                hex(owner_evidence_hex)?;
                provenance.validate()?;
            }
            Self::OperationFailed {
                operation_id,
                diagnostic_hex,
                owner_ref_hex,
                ..
            } => {
                identifier(operation_id)?;
                hex(diagnostic_hex)?;
                if let Some(v) = owner_ref_hex {
                    hex(v)?;
                }
            }
            Self::SearchClose { evidence, .. } => {
                evidence.validate()?;
            }
            Self::ScopeClose { .. } => {}
        }
        Ok(())
    }
}
const NODE: u8 = 0;
const DECLARED: u8 = 1;
const RESOLUTION: u8 = 2;
const CHOICE: u8 = 3;
const USE: u8 = 4;
const VISIT: u8 = 5;
const VARIANT: u8 = 6;
const TRIAL: u8 = 7;
const OWNER_TRIAL: u8 = 8;
// One resolution per scoped declaration; independent seeds declare independent identities.
const DECLARED_EFFECTIVE: u8 = 9;
fn retained_value_bound(v: &Value) -> Result<usize> {
    match v {
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(0),
        Value::String(s) => {
            if s.is_empty() {
                Ok(0)
            } else {
                add(mul(4, s.len())?, 8)
            }
        }
        Value::Array(a) => {
            let mut n = if a.is_empty() {
                0
            } else {
                mul(add(mul(4, a.len())?, 8)?, size_of::<Value>())?
            };
            for value in a {
                n = add(n, retained_value_bound(value)?)?;
            }
            Ok(n)
        }
        Value::Object(o) => {
            let mut n = if o.is_empty() {
                0
            } else {
                add(
                    mul(add(mul(4, o.len())?, 8)?, size_of::<(String, Value)>())?,
                    mul(add(mul(12, o.len())?, 16)?, size_of::<usize>() + 1)?,
                )?
            };
            for (key, value) in o {
                n = add(
                    n,
                    add(add(mul(4, key.len())?, 8)?, retained_value_bound(value)?)?,
                )?;
            }
            Ok(n)
        }
    }
}
struct Fact {
    data: Value,
    status: u8,
    a: u64,
    b: u64,
    flag: bool,
    lease: usize,
}
impl Fact {
    fn empty() -> Self {
        Self {
            data: Value::Null,
            status: 0,
            a: 0,
            b: 0,
            flag: false,
            lease: 0,
        }
    }
}
struct Scope {
    role: ScopeKind,
    parent: Option<String>,
    declaration: String,
    digest: String,
    facts: Index<Fact>,
    children: u64,
    pending: u64,
}
impl Scope {
    fn heap_bytes(&self) -> usize {
        self.parent.as_ref().map_or(0, String::capacity)
            + self.declaration.capacity()
            + self.digest.capacity()
            + self.facts.heap_bytes()
            + self
                .facts
                .nodes
                .iter()
                .map(|n| n.value.lease)
                .sum::<usize>()
    }
}
fn text<'a>(v: &'a Value, k: &str) -> Result<&'a str> {
    v.get(k)
        .and_then(Value::as_str)
        .ok_or(CaptureError::Invalid("event_schema"))
}
fn array<'a>(v: &'a Value, k: &str) -> Result<&'a Vec<Value>> {
    v.get(k)
        .and_then(Value::as_array)
        .ok_or(CaptureError::Invalid("event_schema"))
}
fn opt_text<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(Value::as_str)
}
fn check_provenance(p: &Value) -> Result<()> {
    let old = text(p, "observation")? == "restored";
    for k in [
        "original_capture_id",
        "original_scope_id",
        "original_id",
        "checkpoint_ref",
    ] {
        require(p.get(k).is_some_and(|v| !v.is_null()) == old, "provenance")?;
    }
    if old {
        require(!text(p, "checkpoint_ref")?.is_empty(), "provenance")?;
    }
    Ok(())
}
fn bound_provenance(a: &Value, b: &Value) -> Result<()> {
    require(compatible(a, b), "provenance_binding")
}
fn compatible(a: &Value, b: &Value) -> bool {
    a["observation"] == b["observation"]
        && (a["observation"] == "new"
            || (a["checkpoint_ref"] == b["checkpoint_ref"]
                && a["original_capture_id"] == b["original_capture_id"]))
}
fn unique(values: &[Value]) -> Result<()> {
    let mut refs: Vec<&str> = Vec::with_capacity(values.len());
    for v in values {
        refs.push(v.as_str().ok_or(CaptureError::Invalid("event_schema"))?);
    }
    refs.sort_unstable();
    require(
        !refs.windows(2).any(|w| w[0] == w[1]),
        "duplicate_reference",
    )
}
impl Counts {
    fn zero() -> Self {
        Self {
            scope_count: 0,
            visited_count: 0,
            candidate_count: 0,
            rejected_count: 0,
            terminal_without_candidate_count: 0,
            trial_count: 0,
            new_trial_count: 0,
            restored_trial_count: 0,
            resume_observation_count: 0,
            completed_count: 0,
            failed_count: 0,
            pruned_count: 0,
            cancelled_count: 0,
        }
    }
}
struct State {
    scopes: Index<Option<Scope>>,
    counts: Counts,
    closures: Sha256,
    closed: u64,
    provenance_seen: bool,
    incomplete: bool,
    released_work: u64,
}
impl State {
    fn new() -> Self {
        let mut closures = Sha256::new();
        closures.update(SCOPE_DOMAIN);
        Self {
            scopes: Index::new(),
            counts: Counts::zero(),
            closures,
            closed: 0,
            provenance_seen: false,
            incomplete: false,
            released_work: 0,
        }
    }
    fn scope(&self, s: &str) -> Result<&Scope> {
        self.scopes
            .get(0, s)
            .ok_or(CaptureError::Invalid("unresolved_reference"))?
            .as_ref()
            .ok_or(CaptureError::Invalid("scope_closed"))
    }
    fn scope_mut(&mut self, s: &str) -> Result<&mut Scope> {
        self.scopes
            .get_mut(0, s)
            .ok_or(CaptureError::Invalid("unresolved_reference"))?
            .as_mut()
            .ok_or(CaptureError::Invalid("scope_closed"))
    }
    fn ancestor(&self, target: &str, scope: &str) -> Result<bool> {
        let mut current = Some(scope);
        while let Some(s) = current {
            if s == target {
                return Ok(true);
            }
            current = self.scope(s)?.parent.as_deref();
        }
        Ok(false)
    }
    // A Proposal immediately under a Declaration is standalone, even when that
    // Declaration is nested inside another Search. Do not cross that boundary.
    fn trial_owner_scope<'a>(&'a self, proposal: &'a str) -> Result<&'a str> {
        if let Some(parent) = self.scope(proposal)?.parent.as_deref() {
            if self.scope(parent)?.role == ScopeKind::Search {
                return Ok(parent);
            }
        }
        Ok(proposal)
    }
    fn fact(&self, s: &str, k: u8, id: &str) -> Result<&Fact> {
        self.scope(s)?
            .facts
            .get(k, id)
            .ok_or(CaptureError::Invalid("unresolved_reference"))
    }
    fn fact_mut(&mut self, s: &str, k: u8, id: &str) -> Result<&mut Fact> {
        self.scope_mut(s)?
            .facts
            .get_mut(k, id)
            .ok_or(CaptureError::Invalid("unresolved_reference"))
    }
    fn reference<'a>(
        &'a self,
        k: u8,
        r: &Value,
        current: &str,
        ancestors: bool,
    ) -> Result<&'a Fact> {
        let scope = text(r, "scope_id")?;
        require(
            scope == current || (ancestors && self.ancestor(scope, current)?),
            "scope_binding",
        )?;
        self.fact(scope, k, text(r, "id")?)
    }
    fn node_root(&self, s: &str, node: &str) -> Result<String> {
        let scope = self.scope(s)?;
        let d = self.scope(&scope.declaration)?;
        let mut current = node.to_owned();
        loop {
            if d.facts.get(NODE, &current).is_some() {
                return Ok(current);
            }
            let f = self.fact(s, NODE, &current)?;
            if f.data.is_null() || f.data["shared_choices"] == false {
                return Ok(current);
            }
            current = text(&f.data, "origin_node_id")?.to_owned();
        }
    }
    fn put(&mut self, s: &str, k: u8, id: &str, value: &Value, b: &mut Budget) -> Result<()> {
        require(
            self.scope(s)?.facts.get(k, id).is_none(),
            "duplicate_occurrence",
        )?;
        let lease = retained_value_bound(value)?;
        b.charge(lease, "retained fact")?;
        let data = value.clone();
        self.scope_mut(s)?.facts.insert(
            k,
            id,
            Fact {
                data,
                status: 0,
                a: 0,
                b: 0,
                flag: false,
                lease,
            },
            b,
        )
    }
    fn seed_used(&mut self, r: &Value, s: &str, id: &str, b: &mut Budget) -> Result<()> {
        // Fixed closed parent/reference shape: bound temporary owned IDs/maps;
        // iterative traversal prevents a long derivation chain growing stack.
        b.charge(8192, "seed ancestry scratch")?;
        let result = (|| {
            let mut current = r.clone();
            loop {
                let target = text(&current, "scope_id")?.to_owned();
                let rid = text(&current, "id")?.to_owned();
                let fact = self.reference(RESOLUTION, &current, s, true)?;
                if fact.flag {
                    break;
                }
                require(
                    text(&fact.data["first_actual_use"], "scope_id")? == s
                        && text(&fact.data["first_actual_use"], "id")? == id,
                    "seed_use_binding",
                )?;
                let next = if fact.data["parent"]["value_slot"] == "effective" {
                    Some(fact.data["parent"]["seed_ref"].clone())
                } else {
                    None
                };
                self.fact_mut(&target, RESOLUTION, &rid)?.flag = true;
                match next {
                    Some(parent) => current = parent,
                    None => break,
                }
            }
            Ok(())
        })();
        b.release(8192);
        result
    }
    fn accept(&mut self, event: &Event, b: &mut Budget) -> Result<()> {
        event.validate().map_err(|e| match e {
            CaptureError::Invalid("identifier" | "hex" | "fingerprint") => {
                CaptureError::Invalid("event_schema")
            }
            other => other,
        })?;
        let length = encoded_len(event)?;
        let serialization = add(mul(4, length)?, 4096)?;
        b.charge(serialization, "semantic serialization")?;
        let result = (|| {
            let raw =
                serde_json::to_string(event).map_err(|_| CaptureError::Invalid("serialization"))?;
            let reserve = decoder_admission(&raw)?.reservation_bytes;
            b.charge(reserve, "semantic value scratch")?;
            let result = (|| {
                let v: Value = serde_json::from_str(&raw)
                    .map_err(|_| CaptureError::Invalid("event_schema"))?;
                self.accept_value(&v, b)
            })();
            b.release(reserve);
            result
        })();
        b.release(serialization);
        result
    }
    fn accept_value(&mut self, e: &Value, b: &mut Budget) -> Result<()> {
        let s = text(e, "scope_id")?;
        let kind = text(e, "kind")?;
        if kind == "declaration" || kind == "scope_open" {
            require(self.scopes.get(0, s).is_none(), "duplicate_occurrence")?;
            let parent = opt_text(e, "parent_scope_id");
            let role: ScopeKind = serde_json::from_value(e["scope_kind"].clone())
                .map_err(|_| CaptureError::Invalid("scope_role"))?;
            let declaration;
            let source_digest;
            if kind == "declaration" {
                require(role == ScopeKind::Declaration, "scope_role")?;
                if let Some(p) = parent {
                    require(self.scope(p)?.role == ScopeKind::Proposal, "scope_role")?;
                }
                require(
                    digest(text(e, "declaration_text")?.as_bytes()) == text(e, "source_digest")?,
                    "source_digest",
                )?;
                unique(array(e, "node_ids")?)?;
                declaration = s.to_owned();
                source_digest = text(e, "source_digest")?.to_owned();
            } else {
                let p = parent.ok_or(CaptureError::Invalid("scope_role"))?;
                let pv = self.scope(p)?;
                require(
                    matches!(
                        (role, pv.role),
                        (ScopeKind::Search, ScopeKind::Declaration)
                            | (
                                ScopeKind::Proposal,
                                ScopeKind::Search | ScopeKind::Declaration
                            )
                            | (ScopeKind::Terminal, ScopeKind::Proposal)
                    ),
                    "scope_role",
                )?;
                declaration = if pv.role == ScopeKind::Declaration {
                    p.to_owned()
                } else {
                    pv.declaration.clone()
                };
                require(
                    text(&e["declaration_ref"], "scope_id")? == declaration
                        && text(&e["declaration_ref"], "source_digest")?
                            == self.scope(&declaration)?.digest,
                    "foreign_declaration",
                )?;
                source_digest = String::new();
            }
            let scope = Scope {
                role,
                parent: parent.map(|x| owned(x, b)).transpose()?,
                declaration: owned(&declaration, b)?,
                digest: owned(&source_digest, b)?,
                facts: Index::new(),
                children: 0,
                pending: 0,
            };
            self.scopes.insert(0, s, Some(scope), b)?;
            if let Some(p) = parent {
                self.scope_mut(p)?.children += 1;
            }
            if kind == "declaration" {
                for node in array(e, "node_ids")? {
                    self.scope_mut(s)?.facts.insert(
                        NODE,
                        node.as_str().unwrap(),
                        Fact::empty(),
                        b,
                    )?;
                }
            }
            self.counts.scope_count += 1;
            return Ok(());
        }
        let role = self.scope(s)?.role;
        match kind {
            "capture_provenance" => {
                require(
                    role == ScopeKind::Declaration
                        && self.scope(s)?.parent.is_none()
                        && !self.provenance_seen,
                    "provenance_carrier",
                )?;
                for f in array(e, "failed_attempts")? {
                    for a in array(f, "owned_attempts")? {
                        require(
                            a["ownership"] != "confirmed"
                                || opt_text(a, "owner_ref_hex")
                                    .is_some_and(|owner| !owner.is_empty()),
                            "ownership",
                        )?;
                        require(
                            a["cleanup"] != "confirmed"
                                || matches!(
                                    opt_text(a, "status"),
                                    Some("completed" | "failed" | "pruned" | "cancelled")
                                ),
                            "cleanup_observation",
                        )?;
                    }
                }
                self.provenance_seen = true;
            }
            "seed_declared" => {
                require(role != ScopeKind::Terminal, "scope_role")?;
                check_provenance(&e["provenance"])?;
                self.put(s, DECLARED, text(e, "seed_id")?, e, b)?;
            }
            "effective_seed_resolved" => {
                require(role != ScopeKind::Terminal, "scope_role")?;
                check_provenance(&e["provenance"])?;
                let decl = &self.reference(DECLARED, &e["declared_ref"], s, true)?.data;
                match text(e, "resolution")? {
                    "explicit" => require(
                        !decl["requested"].is_null()
                            && decl["requested"] == e["effective"]
                            && e["parent"].is_null()
                            && e["policy"].is_null(),
                        "seed_resolution",
                    )?,
                    "generated" => require(
                        decl["requested"].is_null()
                            && e["parent"].is_null()
                            && !e["policy"].is_null(),
                        "seed_resolution",
                    )?,
                    "derived" => {
                        require(
                            !e["parent"].is_null() && !e["policy"].is_null(),
                            "seed_resolution",
                        )?;
                        let p = &e["parent"];
                        let requested = p["value_slot"] == "requested";
                        let parent = &self
                            .reference(
                                if requested { DECLARED } else { RESOLUTION },
                                &p["seed_ref"],
                                s,
                                true,
                            )?
                            .data;
                        require(
                            !parent[if requested { "requested" } else { "effective" }].is_null(),
                            "effective_seed",
                        )?;
                    }
                    _ => return invalid("seed_resolution"),
                }
                let declared_scope = text(&e["declared_ref"], "scope_id")?;
                let declared_id = text(&e["declared_ref"], "id")?;
                require(
                    self.scope(declared_scope)?
                        .facts
                        .get(DECLARED_EFFECTIVE, declared_id)
                        .is_none(),
                    "seed_resolution_conflict",
                )?;
                self.put(
                    declared_scope,
                    DECLARED_EFFECTIVE,
                    declared_id,
                    &Value::Null,
                    b,
                )?;
                self.put(s, RESOLUTION, text(e, "seed_id")?, e, b)?;
            }
            "choice" | "seed_use" => {
                require(
                    matches!(role, ScopeKind::Proposal | ScopeKind::Search),
                    "scope_role",
                )?;
                check_provenance(&e["provenance"])?;
                let choice = kind == "choice";
                let id = text(e, if choice { "choice_id" } else { "use_id" })?;
                // Ref deliberately has no kind: first_actual_use must be unambiguous.
                require(
                    self.scope(s)?.facts.get(CHOICE, id).is_none()
                        && self.scope(s)?.facts.get(USE, id).is_none(),
                    "duplicate_occurrence",
                )?;
                let stochastic = if choice {
                    let node = text(e, "node_id")?;
                    self.node_root(s, node)?;
                    let declaration = &self.scope(s)?.declaration;
                    if self.scope(declaration)?.facts.get(NODE, node).is_none() {
                        require(
                            self.fact(s, NODE, node)?.data["shared_choices"] == false,
                            "clone_choice_binding",
                        )?;
                    }
                    let random = e["semantics"] == "stochastic";
                    require(random != e["seed_ref"].is_null(), "choice_seed")?;
                    random
                } else {
                    true
                };
                if stochastic {
                    let resolution = self.reference(RESOLUTION, &e["seed_ref"], s, true)?;
                    require(
                        compatible(&resolution.data["provenance"], &e["provenance"]),
                        "seed_provenance",
                    )?;
                    if !resolution.flag {
                        self.seed_used(&e["seed_ref"], s, id, b)?;
                    }
                }
                self.put(s, if choice { CHOICE } else { USE }, id, e, b)?;
            }
            "clone" => {
                require(role == ScopeKind::Proposal, "scope_role")?;
                let root = self.node_root(s, text(e, "origin_node_id")?)?;
                let node = text(e, "clone_node_id")?;
                let d = self.scope(s)?.declaration.as_str();
                require(
                    self.scope(d)?.facts.get(NODE, node).is_none(),
                    "duplicate_occurrence",
                )?;
                let choices = array(e, "choice_ids")?;
                require(
                    e["shared_choices"] == true || choices.is_empty(),
                    "clone_choice_binding",
                )?;
                unique(choices)?;
                for c in choices {
                    let choice = &self.fact(s, CHOICE, c.as_str().unwrap())?.data;
                    require(
                        self.node_root(s, text(choice, "node_id")?)? == root,
                        "clone_choice_binding",
                    )?;
                }
                self.put(s, NODE, node, e, b)?;
            }
            "visit" => {
                require(role == ScopeKind::Proposal, "scope_role")?;
                check_provenance(&e["provenance"])?;
                unique(array(e, "choice_ids")?)?;
                for id in array(e, "choice_ids")? {
                    let choice = self.fact(s, CHOICE, id.as_str().unwrap())?;
                    bound_provenance(&choice.data["provenance"], &e["provenance"])?;
                }
                self.put(s, VISIT, text(e, "visit_id")?, e, b)?;
                self.counts.visited_count += 1;
            }
            "candidate" | "reject" => {
                require(role == ScopeKind::Proposal, "scope_role")?;
                let visit = text(e, "visit_id")?;
                require(self.fact(s, VISIT, visit)?.status == 0, "visit_disposition")?;
                if kind == "candidate" {
                    check_provenance(&e["provenance"])?;
                    bound_provenance(
                        &self.fact(s, VISIT, visit)?.data["provenance"],
                        &e["provenance"],
                    )?;
                    unique(array(e, "active_node_ids")?)?;
                    for node in array(e, "active_node_ids")? {
                        self.node_root(s, node.as_str().unwrap())?;
                    }
                    self.put(s, VARIANT, text(e, "owner_variant_id")?, e, b)?;
                    self.counts.candidate_count += 1;
                    self.fact_mut(s, VISIT, visit)?.status = 1;
                } else {
                    self.counts.rejected_count += 1;
                    self.fact_mut(s, VISIT, visit)?.status = 2;
                }
            }
            "trial_visit" => {
                require(role == ScopeKind::Proposal, "scope_role")?;
                check_provenance(&e["provenance"])?;
                require(e["origin"] == e["provenance"]["observation"], "provenance")?;
                let visit = self.reference(VISIT, &e["visit_ref"], s, false)?;
                bound_provenance(&visit.data["provenance"], &e["provenance"])?;
                require(
                    !text(e, "owner_trial_ref_hex")?.is_empty(),
                    "owner_trial_ref",
                )?;
                require(
                    e["origin"] == "restored"
                        || e["owner_status"].is_null()
                        || e["owner_status"] == "running",
                    "trial_status",
                )?;
                let bound = !e["candidate_ref"].is_null();
                if bound {
                    require(
                        e["candidate_ref"] == e["visit_ref"]
                            && self.reference(VISIT, &e["candidate_ref"], s, false)?.status == 1,
                        "trial_binding",
                    )?;
                }
                let owner_scope = self.trial_owner_scope(s)?.to_owned();
                self.put(
                    &owner_scope,
                    OWNER_TRIAL,
                    text(e, "owner_trial_ref_hex")?,
                    &Value::Null,
                    b,
                )?;
                let id = text(e, "trial_id")?;
                self.put(s, TRIAL, id, e, b)?;
                self.fact_mut(s, TRIAL, id)?.flag = bound;
                self.fact_mut(s, VISIT, text(&e["visit_ref"], "id")?)?.a += 1;
                self.scope_mut(s)?.pending += 1;
                self.counts.trial_count += 1;
                if e["origin"] == "restored" {
                    self.counts.restored_trial_count += 1;
                } else {
                    self.counts.new_trial_count += 1;
                }
            }
            "trial_bind" => {
                require(role == ScopeKind::Proposal, "scope_role")?;
                let trial = self.reference(TRIAL, &e["trial_ref"], s, false)?;
                require(trial.status == 0 && !trial.flag, "trial_terminal")?;
                require(
                    e["candidate_ref"] == trial.data["visit_ref"]
                        && self.reference(VISIT, &e["candidate_ref"], s, false)?.status == 1,
                    "trial_binding",
                )?;
                self.fact_mut(s, TRIAL, text(&e["trial_ref"], "id")?)?.flag = true;
            }
            "trial_resume" => {
                require(role == ScopeKind::Proposal, "scope_role")?;
                check_provenance(&e["provenance"])?;
                require(e["provenance"]["observation"] == "restored", "provenance")?;
                let trial = self.reference(TRIAL, &e["trial_ref"], s, false)?;
                bound_provenance(&trial.data["provenance"], &e["provenance"])?;
                let status = match trial.status {
                    0 => trial.data["owner_status"].clone(),
                    1 => Value::from("completed"),
                    2 => Value::from("failed"),
                    3 => Value::from("pruned"),
                    _ => Value::from("cancelled"),
                };
                require(e["observed_status"] == status, "trial_status")?;
                self.counts.resume_observation_count += 1;
            }
            "trial_outcome" => {
                require(role == ScopeKind::Terminal, "scope_role")?;
                let p = self.scope(s)?.parent.clone().unwrap();
                require(text(&e["trial_ref"], "scope_id")? == p, "scope_binding")?;
                let trial = self.reference(TRIAL, &e["trial_ref"], &p, false)?;
                let o = &e["outcome"];
                let status = text(o, "status")?;
                require(trial.status == 0 && status != "running", "trial_terminal")?;
                let initial = opt_text(&trial.data, "owner_status");
                let restored = trial.data["origin"] == "restored"
                    && !matches!(initial, None | Some("running"));
                require(
                    (o["mode"] == "restored_observation") == restored
                        && (!restored || initial == Some(status)),
                    "trial_status",
                )?;
                for key in ["stage", "reason"] {
                    require(
                        (o[key]["availability"] == "known") != o[key]["value_hex"].is_null(),
                        "observation",
                    )?;
                }
                let visit = trial.data["visit_ref"].clone();
                require(
                    status != "completed"
                        || trial.flag
                        || self.reference(VISIT, &visit, &p, false)?.status == 2,
                    "trial_binding",
                )?;
                require(
                    status != "completed" || o["failure"].is_null(),
                    "trial_status",
                )?;
                let code = match status {
                    "completed" => 1,
                    "failed" => 2,
                    "pruned" => 3,
                    "cancelled" => 4,
                    _ => return invalid("trial_status"),
                };
                self.fact_mut(&p, TRIAL, text(&e["trial_ref"], "id")?)?
                    .status = code;
                let vf = self.fact_mut(&p, VISIT, text(&visit, "id")?)?;
                vf.b += 1;
                vf.flag |= code == 1;
                self.scope_mut(&p)?.pending -= 1;
                match code {
                    1 => self.counts.completed_count += 1,
                    2 => self.counts.failed_count += 1,
                    3 => self.counts.pruned_count += 1,
                    _ => self.counts.cancelled_count += 1,
                }
            }
            "operation_failed" => {
                require(
                    e["ownership"] != "confirmed"
                        || opt_text(e, "owner_ref_hex").is_some_and(|owner| !owner.is_empty()),
                    "ownership",
                )?;
                self.incomplete = true;
            }
            "scope_close" | "search_close" => {
                let v = self.scope(s)?;
                let search = v.role == ScopeKind::Search
                    || (v.role == ScopeKind::Proposal
                        && self.scope(v.parent.as_deref().unwrap())?.role
                            == ScopeKind::Declaration);
                require((kind == "search_close") == search, "closure_role")?;
                require(v.children == 0, "open_children")?;
                require(v.pending == 0, "pending_trial")?;
                if search {
                    require(
                        e["basis"] != "enumeration_complete" || e["reason"] == "exhausted",
                        "closure_reason",
                    )?;
                    require(
                        e["reason"] != "exhausted"
                            || (e["unvisited"] == false
                                && e["basis"] == "enumeration_complete"
                                && e["evidence"]["availability"] == "known"),
                        "closure_reason",
                    )?;
                    require(
                        e["basis"] != "operation_error" || self.incomplete,
                        "closure_reason",
                    )?;
                    require(
                        (e["evidence"]["availability"] == "known")
                            != e["evidence"]["value_hex"].is_null(),
                        "observation",
                    )?;
                    require(
                        e["unvisited"].is_null() || e["evidence"]["availability"] == "known",
                        "unvisited_evidence",
                    )?;
                }
                let mut without = 0;
                for n in &v.facts.nodes {
                    if n.kind == RESOLUTION {
                        require(n.value.flag, "unused_effective_seed")?;
                    }
                    if n.kind == VISIT && n.value.status == 0 {
                        require(
                            n.value.a > 0 && n.value.b == n.value.a && !n.value.flag,
                            "pending_visit",
                        )?;
                        without += 1;
                    }
                }
                self.counts.terminal_without_candidate_count += without;
                let v = self.scopes.get_mut(0, s).unwrap().take().unwrap();
                if let Some(parent) = &v.parent {
                    self.scope_mut(parent)?.children -= 1;
                }
                self.released_work += v.facts.work.get();
                b.release(v.heap_bytes());
                drop(v);
                self.closures.update((s.len() as u64).to_be_bytes());
                self.closures.update(s.as_bytes());
                self.closed += 1;
            }
            _ => return invalid("event_schema"),
        }
        Ok(())
    }
    fn finish(&self) -> Result<()> {
        require(self.provenance_seen, "provenance_carrier")?;
        require(!self.incomplete, "incomplete_error")?;
        require(
            self.counts.scope_count > 0 && self.closed == self.counts.scope_count,
            "empty_or_unclosed",
        )?;
        require(
            self.counts.visited_count
                == self.counts.candidate_count
                    + self.counts.rejected_count
                    + self.counts.terminal_without_candidate_count,
            "completion_counts",
        )
    }
    fn work(&self) -> u64 {
        self.scopes.work.get()
            + self.released_work
            + self
                .scopes
                .nodes
                .iter()
                .filter_map(|n| n.value.as_ref())
                .map(|s| s.facts.work.get())
                .sum::<u64>()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    pub schema_version: u64,
    pub profile: String,
    pub record_type: String,
    pub capture_id: String,
    pub sequence: u64,
    #[serde(deserialize_with = "required_option")]
    pub previous_digest: Option<String>,
    pub byte_length: u64,
    pub payload_hex: String,
    pub frame_digest: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Completion {
    pub schema_version: u64,
    pub profile: String,
    pub record_type: String,
    pub capture_id: String,
    pub frame_count: u64,
    pub last_frame_digest: String,
    pub closed_scopes_digest: String,
    pub counts: Counts,
    pub completion_digest: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WireRecord {
    Frame(Frame),
    Completion(Completion),
}
fn preimage_raw<T: Serialize>(value: &T, field: &str, domain: &[u8]) -> Result<Vec<u8>> {
    let raw = serde_json::to_string(value).map_err(|_| CaptureError::Invalid("serialization"))?;
    require(
        raw.is_ascii() && raw.len() <= MAX_FRAME_BYTES,
        "wire_size_ascii",
    )?;
    let mut typed = strict(&raw, 256)?;
    let TypedCanonicalValue::Object(entries) = &mut typed else {
        return invalid("wire_schema");
    };
    entries.retain(|(key, _)| key != field);
    let encoded = tcv1_preimage(&typed).map_err(|_| CaptureError::Invalid("canonical"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(domain.len() + encoded.len())
        .map_err(|_| CaptureError::Allocation)?;
    out.extend_from_slice(domain);
    out.extend_from_slice(&encoded);
    Ok(out)
}
fn preimage_reserve<T: Serialize>(v: &T) -> Result<usize> {
    // Four live/growing string/byte buffers: serde JSON, parsed ASCII strings,
    // canonical bytes and domain-joined bytes, each with <=4x growth envelope.
    // Fixed Frame/Completion+Counts layouts have fewer than256 value/key tokens.
    add(
        mul(16, encoded_len(v)?)?,
        256 * (size_of::<TypedCanonicalValue>() + size_of::<(String, TypedCanonicalValue)>() + 128),
    )
}
fn hash_record<T: Serialize>(v: &T, field: &str, domain: &[u8], b: &mut Budget) -> Result<String> {
    let n = preimage_reserve(v)?;
    b.charge(n, "wire canonicalization")?;
    let r = preimage_raw(v, field, domain).map(|raw| digest(&raw));
    b.release(n);
    r
}
/// Caller-owned bounded utility; does not admit an event or prove completeness.
pub fn frame_preimage(v: &Frame) -> Result<Vec<u8>> {
    let mut b = Budget::new(Limits::default())?;
    b.charge(preimage_reserve(v)?, "wire canonicalization")?;
    preimage_raw(v, "frame_digest", FRAME_DOMAIN)
}
/// Caller-owned bounded utility; no trusted expected context is inferred.
pub fn completion_preimage(v: &Completion) -> Result<Vec<u8>> {
    let mut b = Budget::new(Limits::default())?;
    b.charge(preimage_reserve(v)?, "wire canonicalization")?;
    preimage_raw(v, "completion_digest", COMPLETION_DOMAIN)
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExpectedCohortCommitment {
    capture_id: String,
    profile: String,
    schema_version: u64,
    completion_digest: String,
}
impl ExpectedCohortCommitment {
    pub fn completion_digest(&self) -> &str {
        &self.completion_digest
    }
    pub fn capture_id(&self) -> &str {
        &self.capture_id
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedWire {
    capture_id: String,
    profile: String,
    schema_version: u64,
    completion_digest: String,
}
/// Decode a caller-supplied commitment. Callers should source it independently
/// of incoming evidence; this API cannot enforce that or authenticate its origin.
/// No constructor accepts a decoded Completion. Checksums are not signatures.
pub fn decode_expected_context(
    raw: Option<&str>,
    limits: Limits,
) -> Result<ExpectedCohortCommitment> {
    let raw = raw.ok_or(CaptureError::Invalid("expected_required"))?;
    require(raw.trim() != "null", "expected_required")?;
    let mut b = Budget::new(limits)?;
    let v: ExpectedWire = match decode(raw, &mut b, "expected context decode") {
        Ok(v) => v,
        Err(e @ CaptureError::EvidenceBudgetExceeded { .. }) => return Err(e),
        Err(_) => return invalid("expected_shape"),
    };
    require(
        identifier(&v.capture_id).is_ok()
            && fingerprint(&v.completion_digest).is_ok()
            && v.profile == PROFILE
            && v.schema_version == 1,
        "expected_invalid",
    )?;
    Ok(ExpectedCohortCommitment {
        capture_id: v.capture_id,
        profile: v.profile,
        schema_version: v.schema_version,
        completion_digest: v.completion_digest,
    })
}
/// Untrusted structural inspection: self-consistency without a caller commitment.
#[derive(Debug)]
pub struct StructuralInspection {
    pub completion: Completion,
}
/// Constructed only through [`verify_stream`] with a caller-supplied commitment.
/// This proves agreement with that commitment, not authentication or origin.
/// Fields cannot be constructed or replaced by callers; there is no serde decode.
///
/// ```compile_fail
/// use dag_ml_core::realization_evidence::VerifiedCapture;
/// let forged = serde_json::from_str::<VerifiedCapture>("{}");
/// ```
///
/// ```compile_fail
/// use dag_ml_core::realization_evidence::{Completion, VerifiedCapture};
/// fn forge(completion: Completion) -> VerifiedCapture {
///     VerifiedCapture { completion }
/// }
/// ```
///
/// ```
/// use dag_ml_core::realization_evidence::{
///     verify_stream, Completion, ExpectedCohortCommitment, Limits, VerifiedCapture,
/// };
/// fn check(wire: &str, expected: &ExpectedCohortCommitment) {
///     let verified: VerifiedCapture = verify_stream(wire, expected, Limits::default()).unwrap();
///     let _: &Completion = verified.completion();
/// }
/// ```
#[derive(Debug)]
pub struct VerifiedCapture {
    completion: Completion,
}
impl VerifiedCapture {
    pub fn completion(&self) -> &Completion {
        &self.completion
    }
}
/// Returned only by a successfully flushed [`Recorder::finish`].
/// No untrusted deserialization constructor is provided.
///
/// ```compile_fail
/// use dag_ml_core::realization_evidence::FinishedCapture;
/// let forged = serde_json::from_str::<FinishedCapture>("{}");
/// ```
///
/// ```compile_fail
/// use dag_ml_core::realization_evidence::{Completion, ExpectedCohortCommitment, FinishedCapture};
/// fn forge(completion: Completion, expected_commitment: ExpectedCohortCommitment) -> FinishedCapture {
///     FinishedCapture { completion, expected_commitment }
/// }
/// ```
///
/// ```
/// use dag_ml_core::realization_evidence::{Completion, ExpectedCohortCommitment, FinishedCapture};
/// fn read(finished: &FinishedCapture) {
///     let _: &Completion = finished.completion();
///     let _: &ExpectedCohortCommitment = finished.expected_commitment();
/// }
/// ```
#[derive(Debug)]
pub struct FinishedCapture {
    completion: Completion,
    expected_commitment: ExpectedCohortCommitment,
}

impl FinishedCapture {
    pub fn completion(&self) -> &Completion {
        &self.completion
    }
    pub fn expected_commitment(&self) -> &ExpectedCohortCommitment {
        &self.expected_commitment
    }
}
struct Decoder {
    state: State,
    pending: Vec<u8>,
    previous: Option<String>,
    sequence: u64,
    capture: Option<String>,
    completion: Option<Completion>,
}
impl Decoder {
    fn new() -> Self {
        Self {
            state: State::new(),
            pending: Vec::new(),
            previous: None,
            sequence: 0,
            capture: None,
            completion: None,
        }
    }
    fn logical(&mut self, b: &mut Budget) -> Result<()> {
        let raw = std::str::from_utf8(&self.pending)
            .map_err(|_| CaptureError::Invalid("logical_utf8"))?;
        let text = record_text(raw)?;
        let admission = decoder_admission(text)?;
        b.charge(admission.reservation_bytes, "logical decoder")?;
        let result = (|| {
            let strict_value = strict(text, admission.nodes)?;
            drop(strict_value);
            let event: Event =
                serde_json::from_str(text).map_err(|_| CaptureError::Invalid("event_schema"))?;
            self.state.accept(&event, b)
        })();
        b.release(admission.reservation_bytes);
        result?;
        self.pending.clear();
        Ok(())
    }
    fn feed(&mut self, line: &str, b: &mut Budget) -> Result<()> {
        require(self.completion.is_none(), "after_completion")?;
        let line = record_text(line)?;
        require(
            !line.is_empty() && line.is_ascii() && line.len() <= MAX_FRAME_BYTES,
            "wire_size_ascii",
        )?;
        let c = decoder_admission(line)?;
        b.charge(c.reservation_bytes, "wire decoder")?;
        let r = (|| {
            let typed = strict(line, c.nodes)?;
            drop(typed);
            let wire: WireRecord =
                serde_json::from_str(line).map_err(|_| CaptureError::Invalid("wire_schema"))?;
            match wire {
                WireRecord::Frame(f) => {
                    require(
                        f.schema_version == 1 && f.profile == PROFILE && f.record_type == "frame",
                        "wire_schema",
                    )?;
                    require(
                        identifier(&f.capture_id).is_ok()
                            && fingerprint(&f.frame_digest).is_ok()
                            && hex(&f.payload_hex).is_ok()
                            && f.previous_digest
                                .as_deref()
                                .is_none_or(|v| fingerprint(v).is_ok()),
                        "wire_schema",
                    )?;
                    require(
                        f.sequence == self.sequence && f.previous_digest == self.previous,
                        "frame_chain",
                    )?;
                    if let Some(id) = &self.capture {
                        require(id == &f.capture_id, "capture_id")?;
                    } else {
                        self.capture = Some(owned(&f.capture_id, b)?);
                    }
                    require(
                        f.byte_length > 0 && f.byte_length as usize == f.payload_hex.len() / 2,
                        "byte_length",
                    )?;
                    require(
                        hash_record(&f, "frame_digest", FRAME_DOMAIN, b)? == f.frame_digest,
                        "frame_digest",
                    )?;
                    for pair in f.payload_hex.as_bytes().chunks_exact(2) {
                        let byte = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16)
                            .map_err(|_| CaptureError::Invalid("hex"))?;
                        if self.pending.len() == self.pending.capacity() {
                            let old = self.pending.capacity();
                            let new = old.saturating_mul(2).max(1024);
                            b.charge(new, "reassembly growth")?;
                            self.pending
                                .try_reserve_exact(new - old)
                                .map_err(|_| CaptureError::Allocation)?;
                            b.release(old);
                        }
                        self.pending.push(byte);
                        if byte == b'\n' {
                            self.logical(b)?;
                        }
                    }
                    let previous = owned(&f.frame_digest, b)?;
                    if let Some(old) = self.previous.replace(previous) {
                        b.release(old.capacity());
                    }
                    self.sequence += 1;
                }
                WireRecord::Completion(c) => {
                    require(
                        c.schema_version == 1
                            && c.profile == PROFILE
                            && c.record_type == "completion",
                        "wire_schema",
                    )?;
                    require(
                        identifier(&c.capture_id).is_ok()
                            && fingerprint(&c.last_frame_digest).is_ok()
                            && fingerprint(&c.closed_scopes_digest).is_ok()
                            && fingerprint(&c.completion_digest).is_ok(),
                        "wire_schema",
                    )?;
                    self.state.finish()?;
                    require(
                        self.capture.as_deref() == Some(c.capture_id.as_str()),
                        "capture_id",
                    )?;
                    require(
                        self.pending.is_empty()
                            && self.sequence > 0
                            && c.frame_count == self.sequence
                            && self.previous.as_deref() == Some(c.last_frame_digest.as_str()),
                        "completion_chain",
                    )?;
                    require(
                        c.counts == self.state.counts
                            && c.closed_scopes_digest
                                == encode_hex(&self.state.closures.clone().finalize()),
                        "completion_counts",
                    )?;
                    require(
                        hash_record(&c, "completion_digest", COMPLETION_DOMAIN, b)?
                            == c.completion_digest,
                        "completion_digest",
                    )?;
                    b.charge(
                        c.capture_id.capacity()
                            + c.profile.capacity()
                            + c.record_type.capacity()
                            + c.last_frame_digest.capacity()
                            + c.closed_scopes_digest.capacity()
                            + c.completion_digest.capacity(),
                        "completion ownership",
                    )?;
                    self.completion = Some(c);
                }
            }
            Ok(())
        })();
        b.release(c.reservation_bytes);
        r
    }
}
/// Self-consistency only. Does not produce a qualified cohort claim.
pub fn inspect_stream(input: &str, limits: Limits) -> Result<StructuralInspection> {
    let mut budget = Budget::new(limits)?;
    let mut d = Decoder::new();
    for line in input.split_inclusive('\n') {
        require(line.len() <= MAX_FRAME_BYTES + 2, "wire_size_ascii")?;
        d.feed(line, &mut budget)?;
    }
    Ok(StructuralInspection {
        completion: d
            .completion
            .ok_or(CaptureError::Invalid("missing_completion"))?,
    })
}
/// Compare against a caller-supplied commitment. Its origin is the caller's
/// responsibility; checksums and successful self-anchoring do not authenticate it.
pub fn verify_stream(
    input: &str,
    expected: &ExpectedCohortCommitment,
    limits: Limits,
) -> Result<VerifiedCapture> {
    let inspection = inspect_stream(input, limits)?;
    let c = inspection.completion;
    require(
        c.capture_id == expected.capture_id
            && c.profile == expected.profile
            && c.schema_version == expected.schema_version,
        "expected_context",
    )?;
    require(
        c.completion_digest == expected.completion_digest,
        "expected_completion_mismatch",
    )?;
    Ok(VerifiedCapture { completion: c })
}
pub struct Recorder<W: Write> {
    sink: W,
    id: String,
    limits: Limits,
    budget: Budget,
    state: State,
    shadow: Decoder,
    raw: Vec<u8>,
    sequence: u64,
    previous: Option<String>,
    poisoned: bool,
    completed: bool,
    refusal_flush: RefusalFlushStatus,
}
/// Secondary diagnostics from the single safe refusal flush attempt.
/// A failed flush never replaces the primary error returned by record/finish.
#[derive(Debug)]
pub enum RefusalFlushStatus {
    NotAttempted,
    Flushed,
    Failed(CaptureError),
}
impl<W: Write> Recorder<W> {
    pub fn new(sink: W, capture_id: &str, limits: Limits) -> Result<Self> {
        identifier(capture_id)?;
        let mut budget = Budget::new(limits)?;
        budget.charge(limits.chunk_bytes, "recorder buffer")?;
        let mut raw = Vec::new();
        raw.try_reserve_exact(limits.chunk_bytes)
            .map_err(|_| CaptureError::Allocation)?;
        let id = owned(capture_id, &mut budget)?;
        Ok(Self {
            sink,
            id,
            limits,
            budget,
            state: State::new(),
            shadow: Decoder::new(),
            raw,
            sequence: 0,
            previous: None,
            poisoned: false,
            completed: false,
            refusal_flush: RefusalFlushStatus::NotAttempted,
        })
    }
    pub fn logical_peak_bytes(&self) -> usize {
        self.budget.peak
    }
    pub fn lookup_comparisons(&self) -> u64 {
        self.state.work() + self.shadow.state.work()
    }
    pub fn refusal_flush_status(&self) -> &RefusalFlushStatus {
        &self.refusal_flush
    }
    // Called only immediately after accept/finish refuses, before any bytes of
    // the refused event enter raw. The shadow has never failed. Poison remains
    // set, so neither a retry nor another record can emit a second attempt.
    fn flush_refused_prefix(&mut self, primary: &CaptureError) {
        if !matches!(
            primary,
            CaptureError::Invalid(_) | CaptureError::EvidenceBudgetExceeded { .. }
        ) {
            return;
        }
        self.refusal_flush = match self
            .emit_frame()
            .and_then(|()| self.sink.flush().map_err(CaptureError::Sink))
        {
            Ok(()) => RefusalFlushStatus::Flushed,
            Err(error) => RefusalFlushStatus::Failed(error),
        };
    }
    pub fn record(&mut self, event: &Event) -> Result<()> {
        require(!self.poisoned && !self.completed, "capture_terminal")?;
        self.poisoned = true;
        if let Err(primary) = self.state.accept(event, &mut self.budget) {
            self.flush_refused_prefix(&primary);
            return Err(primary);
        }
        let n = add(mul(4, encoded_len(event)?)?, 8)?;
        self.budget.charge(n, "logical serialization")?;
        let result = (|| {
            let data =
                serde_json::to_vec(event).map_err(|_| CaptureError::Invalid("serialization"))?;
            for byte in data.into_iter().chain(std::iter::once(b'\n')) {
                self.raw.push(byte);
                if self.raw.len() == self.limits.chunk_bytes {
                    self.emit_frame()?;
                }
            }
            Ok(())
        })();
        self.budget.release(n);
        result?;
        if matches!(event, Event::OperationFailed { .. }) {
            self.emit_frame()?;
            self.sink.flush().map_err(CaptureError::Sink)?;
        }
        self.poisoned = false;
        Ok(())
    }
    fn emit_frame(&mut self) -> Result<()> {
        if self.raw.is_empty() {
            return Ok(());
        }
        let buffers = add(mul(10, self.raw.len())?, 20480)?;
        self.budget.charge(buffers, "frame buffers")?;
        let result = (|| {
            let mut f = Frame {
                schema_version: 1,
                profile: PROFILE.into(),
                record_type: "frame".into(),
                capture_id: self.id.clone(),
                sequence: self.sequence,
                previous_digest: self.previous.clone(),
                byte_length: self.raw.len() as u64,
                payload_hex: encode_hex(&self.raw),
                frame_digest: String::new(),
            };
            f.frame_digest = hash_record(&f, "frame_digest", FRAME_DOMAIN, &mut self.budget)?;
            let line =
                serde_json::to_string(&f).map_err(|_| CaptureError::Invalid("serialization"))?;
            require(line.len() <= MAX_FRAME_BYTES, "wire_size_ascii")?;
            self.shadow.feed(&line, &mut self.budget)?;
            self.sink
                .write_all(line.as_bytes())
                .and_then(|()| self.sink.write_all(b"\n"))
                .map_err(CaptureError::Sink)?;
            let next = owned(&f.frame_digest, &mut self.budget)?;
            if let Some(old) = self.previous.replace(next) {
                self.budget.release(old.capacity());
            }
            self.sequence += 1;
            self.raw.clear();
            Ok(())
        })();
        self.budget.release(buffers);
        result
    }
    pub fn finish(&mut self) -> Result<FinishedCapture> {
        require(!self.poisoned && !self.completed, "capture_terminal")?;
        self.poisoned = true;
        if let Err(primary) = self.state.finish() {
            self.flush_refused_prefix(&primary);
            return Err(primary);
        }
        self.emit_frame()?;
        // Counts and all strings are bounded fixed-layout data. Reserve before every
        // completion allocation; shadow verifier shares this same Limits budget.
        self.budget.charge(16384, "finish buffers")?;
        let result = (|| {
            let mut c = Completion {
                schema_version: 1,
                profile: PROFILE.into(),
                record_type: "completion".into(),
                capture_id: self.id.clone(),
                frame_count: self.sequence,
                last_frame_digest: self
                    .previous
                    .clone()
                    .ok_or(CaptureError::Invalid("missing_frame"))?,
                closed_scopes_digest: encode_hex(&self.state.closures.clone().finalize()),
                counts: self.state.counts.clone(),
                completion_digest: String::new(),
            };
            c.completion_digest =
                hash_record(&c, "completion_digest", COMPLETION_DOMAIN, &mut self.budget)?;
            let line =
                serde_json::to_string(&c).map_err(|_| CaptureError::Invalid("serialization"))?;
            require(line.len() <= MAX_FRAME_BYTES, "wire_size_ascii")?;
            self.shadow.feed(&line, &mut self.budget)?;
            self.sink
                .write_all(line.as_bytes())
                .and_then(|()| self.sink.write_all(b"\n"))
                .and_then(|()| self.sink.flush())
                .map_err(CaptureError::Sink)?;
            let expected = ExpectedCohortCommitment {
                capture_id: c.capture_id.clone(),
                profile: c.profile.clone(),
                schema_version: 1,
                completion_digest: c.completion_digest.clone(),
            };
            self.completed = true;
            Ok(FinishedCapture {
                completion: c,
                expected_commitment: expected,
            })
        })();
        self.budget.release(16384);
        result
    }
}

#[cfg(test)]
#[path = "realization_evidence_budget_tests.rs"]
mod integrity_budget_tests;
