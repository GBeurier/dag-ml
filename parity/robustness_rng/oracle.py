"""Independent test oracle for ``dagml-robustness-counter.v1``.

The profile maps one robustness draw identity to one Philox4x32-10 block:

* ``seed`` is a portable exact JSON integer in ``0..2**53-1``.  Its low and
  high 32-bit limbs, in that order, are the two Philox key words.
* The counter identity is the exact TCV1 value
  ``[scenario_fingerprint, severity_binary64, unit_id, target_kind,
  target_id_or_empty, draw_index]``.  Text is NFC-normalized by TCV1 and
  binary64 negative zero is canonicalized to positive zero.
* SHA-256 is applied to the domain-separated TCV1 preimage.  The first 16
  digest bytes form the counter and are decoded as four big-endian u32 words.
* Random123 Philox4x32-10 transforms that counter with the derived key.

This module is deliberately Python-standard-library-only and imports no DAG-ML
production or parity implementation.  It is a test oracle, never a runtime RNG.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
from pathlib import Path
from typing import Any, Sequence

PROFILE_NAME = "dagml-robustness-counter.v1"
PHILOX_ALGORITHM = "philox4x32-10"
TCV1_PREFIX = b"DAGML-TCV1\0"
PORTABLE_INTEGER_MAX = 2**53 - 1
TCV1_INTEGER_MIN = -(2**63)
TCV1_INTEGER_MAX = 2**64 - 1
UINT32_MASK = 2**32 - 1

PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
PHILOX_ROUNDS = 10

TARGET_KINDS = frozenset({"source", "node", "global"})
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


class OracleError(ValueError):
    """The input is outside the frozen robustness-counter profile."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def load_json(path: Path) -> Any:
    """Load strict UTF-8 JSON, refusing duplicate keys and non-finite numbers."""

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OracleError(f"{path} contains duplicate key {key!r}")
            result[key] = value
        return result

    def no_nonfinite(token: str) -> None:
        raise OracleError(f"{path} contains non-finite JSON number {token}")

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            no_nonfinite(token)
        return value

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_nonfinite,
            parse_float=finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleError(f"{path} is not strict UTF-8 JSON: {exc}") from exc


def _nfc_text(value: Any, label: str) -> tuple[str, bytes]:
    _require(isinstance(value, str), f"{label} must be text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise OracleError(f"{label} contains a surrogate code point") from exc
    normalized = unicodedata.normalize("NFC", value)
    return normalized, normalized.encode("utf-8")


def _u64(value: int) -> bytes:
    _require(0 <= value <= 2**64 - 1, "TCV1 length exceeds u64")
    return struct.pack(">Q", value)


def tcv1_encode(value: Any, label: str = "value") -> bytes:
    """Encode strict TCV1, NFC-normalizing every string before encoding."""

    if value is None:
        return b"N"
    if value is False:
        return b"F"
    if value is True:
        return b"T"
    if isinstance(value, int):
        _require(
            TCV1_INTEGER_MIN <= value <= TCV1_INTEGER_MAX,
            f"{label} integer is outside the TCV1 range",
        )
        payload = str(value).encode("ascii")
        return b"I" + _u64(len(payload)) + payload
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} must be finite binary64")
        return b"D" + struct.pack(">d", 0.0 if value == 0.0 else value)
    if isinstance(value, str):
        _normalized, payload = _nfc_text(value, label)
        return b"S" + _u64(len(payload)) + payload
    if isinstance(value, list):
        members = b"".join(
            tcv1_encode(member, f"{label}[{index}]")
            for index, member in enumerate(value)
        )
        return b"A" + _u64(len(value)) + members
    if isinstance(value, dict):
        items: list[tuple[bytes, str, Any]] = []
        normalized_keys: set[bytes] = set()
        for key, member in value.items():
            normalized, encoded = _nfc_text(key, f"{label} key")
            _require(encoded not in normalized_keys, f"{label} has NFC-colliding keys")
            normalized_keys.add(encoded)
            items.append((encoded, normalized, member))
        items.sort(key=lambda item: item[0])
        encoded_items = b"".join(
            tcv1_encode(key, f"{label} key") + tcv1_encode(member, f"{label}.{key}")
            for _encoded, key, member in items
        )
        return b"O" + _u64(len(items)) + encoded_items
    raise OracleError(f"{label} contains non-JSON type {type(value).__name__}")


def tcv1_preimage(value: Any) -> bytes:
    """Return the domain-separated TCV1 preimage."""

    return TCV1_PREFIX + tcv1_encode(value)


def _portable_integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    _require(
        0 <= value <= PORTABLE_INTEGER_MAX,
        f"{label} must be in 0..2^53-1",
    )
    return value


def derive_key_words(seed: Any) -> tuple[int, int]:
    """Split one portable seed into Philox key words, low limb first."""

    portable_seed = _portable_integer(seed, "seed")
    return portable_seed & UINT32_MASK, (portable_seed >> 32) & UINT32_MASK


def _normalized_counter_payload(
    *,
    scenario_fingerprint: Any,
    severity: Any,
    unit_id: Any,
    target_kind: Any,
    target_id: Any,
    draw_index: Any,
) -> list[Any]:
    _require(
        isinstance(scenario_fingerprint, str)
        and _FINGERPRINT.fullmatch(scenario_fingerprint) is not None,
        "scenario_fingerprint must be 64 lowercase hexadecimal characters",
    )
    _require(
        isinstance(severity, float),
        "severity must be represented as binary64, not an integer",
    )
    _require(math.isfinite(severity), "severity must be finite binary64")
    _require(severity >= 0.0, "severity must be non-negative")
    normalized_severity = 0.0 if severity == 0.0 else severity

    normalized_unit, _encoded_unit = _nfc_text(unit_id, "unit_id")
    _require(bool(normalized_unit), "unit_id must be non-empty")

    _require(
        isinstance(target_kind, str) and target_kind in TARGET_KINDS,
        "target_kind must be source, node, or global",
    )
    normalized_target, _encoded_target = _nfc_text(target_id, "target_id")
    if target_kind == "global":
        _require(normalized_target == "", "global target_id must be empty")
    else:
        _require(bool(normalized_target), f"{target_kind} target_id must be non-empty")

    normalized_draw_index = _portable_integer(draw_index, "draw_index")
    return [
        scenario_fingerprint,
        normalized_severity,
        normalized_unit,
        target_kind,
        normalized_target,
        normalized_draw_index,
    ]


def derive_counter(
    *,
    scenario_fingerprint: Any,
    severity: Any,
    unit_id: Any,
    target_kind: Any,
    target_id: Any,
    draw_index: Any,
) -> tuple[list[Any], bytes, bytes, tuple[int, int, int, int]]:
    """Derive the normalized payload, TCV1 bytes, digest, and Philox counter."""

    payload = _normalized_counter_payload(
        scenario_fingerprint=scenario_fingerprint,
        severity=severity,
        unit_id=unit_id,
        target_kind=target_kind,
        target_id=target_id,
        draw_index=draw_index,
    )
    preimage = tcv1_preimage(payload)
    digest = hashlib.sha256(preimage).digest()
    counter = struct.unpack(">4I", digest[:16])
    return payload, preimage, digest, counter


def _u32_words(values: Sequence[Any], size: int, label: str) -> tuple[int, ...]:
    _require(
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes, bytearray)),
        f"{label} must be a word sequence",
    )
    _require(len(values) == size, f"{label} must contain exactly {size} words")
    words: list[int] = []
    for index, value in enumerate(values):
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"{label}[{index}] must be an integer",
        )
        _require(0 <= value <= UINT32_MASK, f"{label}[{index}] must be a u32")
        words.append(value)
    return tuple(words)


def _mulhilo32(left: int, right: int) -> tuple[int, int]:
    product = left * right
    return (product >> 32) & UINT32_MASK, product & UINT32_MASK


def philox4x32_10(counter: Sequence[Any], key: Sequence[Any]) -> tuple[int, ...]:
    """Return one Random123-compatible Philox4x32-10 output block."""

    c0, c1, c2, c3 = _u32_words(counter, 4, "counter")
    k0, k1 = _u32_words(key, 2, "key")
    for round_index in range(PHILOX_ROUNDS):
        hi0, lo0 = _mulhilo32(PHILOX_M0, c0)
        hi1, lo1 = _mulhilo32(PHILOX_M1, c2)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & UINT32_MASK,
            lo1,
            (hi0 ^ c3 ^ k1) & UINT32_MASK,
            lo0,
        )
        if round_index + 1 != PHILOX_ROUNDS:
            k0 = (k0 + PHILOX_W0) & UINT32_MASK
            k1 = (k1 + PHILOX_W1) & UINT32_MASK
    return c0, c1, c2, c3


def _hex_words(words: Sequence[int]) -> list[str]:
    return [f"{word:08x}" for word in words]


def derive_robustness_block(
    *,
    seed: Any,
    scenario_fingerprint: Any,
    severity: Any,
    unit_id: Any,
    target_kind: Any,
    target_id: Any,
    draw_index: Any,
) -> dict[str, Any]:
    """Derive one fully inspectable robustness counter and Philox block."""

    key = derive_key_words(seed)
    payload, preimage, digest, counter = derive_counter(
        scenario_fingerprint=scenario_fingerprint,
        severity=severity,
        unit_id=unit_id,
        target_kind=target_kind,
        target_id=target_id,
        draw_index=draw_index,
    )
    output = philox4x32_10(counter, key)
    return {
        "profile": PROFILE_NAME,
        "normalized_payload": payload,
        "tcv1_preimage_hex": preimage.hex(),
        "counter_sha256": digest.hex(),
        "counter_128_hex": digest[:16].hex(),
        "counter_words": list(counter),
        "counter_words_hex": _hex_words(counter),
        "key_words": list(key),
        "key_words_hex": _hex_words(key),
        "output_words": list(output),
        "output_words_hex": _hex_words(output),
        "output_128_hex": "".join(_hex_words(output)),
    }
