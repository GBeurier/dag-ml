#!/usr/bin/env python3
"""Check native Archive V2/V3 C ABI assembly in a fresh host process.

Generate inputs with the two native Methods integration tests using
DAGML_ARCHIVE_CAPI_FIXTURES_DIR, then run this script against a freshly built
libdag_ml_capi shared library. No process-local fitted handle crosses this test.
"""

import ctypes
import hashlib
import json
import sys
from pathlib import Path


class BytesView(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_uint8)), ("len", ctypes.c_size_t)]


class OwnedBytes(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.POINTER(ctypes.c_uint8)),
        ("len", ctypes.c_size_t),
        ("capacity", ctypes.c_size_t),
    ]


class DagMlString(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("len", ctypes.c_size_t)]


def raw_buffer(data):
    return (ctypes.c_uint8 * len(data)).from_buffer_copy(data)


def assemble(library, version, archive_id, documents):
    name = f"dagml_archive_v{version}_native_{'portable' if version == 2 else 'refit'}_payloads_json"
    function = getattr(library, name)
    function.restype = ctypes.c_uint32
    function.argtypes = (
        [BytesView, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        + ([ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t] if version == 2 else [])
        + [ctypes.POINTER(OwnedBytes), ctypes.POINTER(DagMlString)]
    )
    encoded_id = raw_buffer(archive_id.encode())
    buffers = [raw_buffer(document) for document in documents]
    args = [BytesView(encoded_id, len(encoded_id))]
    for buffer in buffers:
        args.extend((buffer, len(buffer)))
    output = OwnedBytes()
    error = DagMlString()
    status = function(*args, ctypes.byref(output), ctypes.byref(error))
    if status:
        message = ctypes.string_at(error.ptr, error.len).decode() if error.ptr else ""
        library.dagml_string_free(error)
        raise AssertionError(f"{name} returned {status}: {message}")
    try:
        result = json.loads(ctypes.string_at(output.ptr, output.len))
    finally:
        library.dagml_owned_bytes_free(output)
    return result


def check(version, payload, expected_package):
    manifest = payload["manifest"]
    members = {path: bytes(data) for path, data in payload["members"].items()}
    assert manifest["schema_version"] == version
    assert manifest["profile"] == f"nirs4all.archive_workspace.v{version}"
    assert len(members) == len(manifest["member_inventory"])
    for record in manifest["member_inventory"]:
        data = members[record["path"]]
        assert len(data) == record["uncompressed_size_bytes"]
        assert hashlib.sha256(data).hexdigest() == record["raw_sha256"]
    package_path = (
        "dagml/portable_predictor_package.json"
        if version == 2
        else "dagml/portable_refit_package.json"
    )
    assert json.loads(members[package_path]) == json.loads(expected_package)
    assert any(path.startswith("methods/") and path.endswith(".n4mm") for path in members)
    return len(members)


def main():
    library = ctypes.CDLL(sys.argv[1])
    library.dagml_owned_bytes_free.argtypes = [OwnedBytes]
    library.dagml_string_free.argtypes = [DagMlString]
    fixture_dir = Path(sys.argv[2])
    outcome = (fixture_dir / "outcome_v2.json").read_bytes()
    package_v2 = (fixture_dir / "package_v2.json").read_bytes()
    package_v3 = (fixture_dir / "package_v3.json").read_bytes()
    v2 = assemble(library, 2, "archive:capi.process.v2", [outcome, package_v2])
    v3 = assemble(library, 3, "archive:capi.process.v3", [package_v3])
    print(f"fresh-process Archive V2: {check(2, v2, package_v2)} exact members")
    print(f"fresh-process Archive V3: {check(3, v3, package_v3)} exact members")


if __name__ == "__main__":
    main()
