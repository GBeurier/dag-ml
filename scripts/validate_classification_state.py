"""Independent bounded classifier wire/graph validator; no Methods or DAG imports."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct

METHOD = "models.classification.pls_logistic"
RAW_KIND = "methods_multimodal_classifier_pipeline"
META_KIND = "methods_role_classifier_pipeline"
RAW_PROFILE = "dagml_methods_multimodal_classifier_pipeline_raw_sha256"
META_PROFILE = "dagml_methods_role_classifier_pipeline_raw_sha256"


def check(condition, message="Classifier state differs from its signed bounded profile"):
    if not condition:
        raise ValueError(message)


def vocabulary(value):
    check(isinstance(value, dict) and set(value) == {"schema_version", "class_labels", "label_names"} and type(value["schema_version"]) is int and value["schema_version"] == 1)
    classes, names = value["class_labels"], value["label_names"]
    check(isinstance(classes, list) and 2 <= len(classes) <= 65_536 and all(type(x) is int for x in classes) and classes == list(range(len(classes))) and isinstance(names, list) and len(names) == len(classes))
    strings = all(type(x) is str and len(x.encode()) <= 1_048_576 for x in names)
    integers = all(type(x) is int and -(1 << 63) <= x < (1 << 63) for x in names)
    check((strings or integers) and names == sorted(names) and len(set(names)) == len(names))
    return classes


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, count):
        check(type(count) is int and 0 <= count <= len(self.data) - self.pos)
        value = self.data[self.pos:self.pos + count]
        self.pos += count
        return value

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def u32(self):
        return self.number("I")

    def u64(self):
        return self.number("Q")

    def f64(self):
        value = self.number("d")
        check(math.isfinite(value))
        return value

    def block(self):
        count = self.u64()
        check(count <= 67_108_864)
        return self.take(count)

    def text(self, short=False):
        count = self.u32() if short else self.u64()
        check(count <= (4096 if short else 1_048_576))
        return self.take(count).decode("utf-8", errors="strict")

    def end(self):
        check(self.pos == len(self.data))


def checked_reader(data, magic):
    check(28 <= len(data) <= 67_108_864 and data[:4] == magic)
    hashed = 0xcbf29ce484222325
    for byte in data[:-8]:
        hashed = ((hashed ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    check(hashed == struct.unpack("<Q", data[-8:])[0])
    value = Reader(data[:-8])
    check(value.take(4) == magic and value.u32() == 1 and [value.u32() for _ in range(3)] == [2, 17, 0])
    return value


def estimator(data, method):
    reader = checked_reader(data, b"N4ME")
    check(reader.text(True) == method)
    count = reader.u32()
    check(count <= 256)
    params = {}
    for _ in range(count):
        name, kind, length = reader.text(True), reader.u32(), reader.u64()
        check(name not in params and 1 <= kind <= 4 and length == 1)
        params[name] = (kind, reader.u64())
    caps, width, outputs = reader.u64(), reader.u64(), reader.u64()
    check(0 < width <= 1_048_576 and 0 <= outputs <= 1_048_576 and not caps & 512)
    count = reader.u32()
    check(0 < count <= 256)
    blocks = {}
    for _ in range(count):
        tag = reader.u32()
        check(tag not in blocks)
        blocks[tag] = reader.block()
    reader.end()
    return params, caps, width, outputs, blocks


def latent(data, width, classes, components):
    reader = checked_reader(data, b"N4MM")
    check([reader.u32() for _ in range(3)] == [0, 1, 0])
    check(reader.u64() > components and [reader.u64() for _ in range(3)] == [width, classes, components])
    check([reader.u32() for _ in range(5)] == [1, 0, 1, 0, 0] and reader.f64() > 0 and reader.u32() > 0)
    for count in [width, width, classes, classes, width * classes, width * components, width * components, classes * components, width * components, 0, 0]:
        check(reader.u64() == count)
        for _ in range(count):
            reader.f64()
    reader.end()


def classifier(data, parameters, width, classes):
    check(isinstance(parameters, dict) and set(parameters) == {"n_components", "max_iter"} and all(type(x) is int and 0 < x <= 2**31-1 for x in parameters.values()))
    params, caps, actual_width, outputs, blocks = estimator(data, METHOD)
    check(params == {name: (1, value) for name, value in parameters.items()} and caps == 156 and actual_width == width and outputs == len(classes) and set(blocks) == {0x31534c43})
    reader = Reader(blocks[0x31534c43])
    check(reader.number("q") == width and reader.u64() == len(classes) and [reader.number("q") for _ in classes] == classes)
    count = reader.number("q")
    check(28 <= count <= 67_108_864)
    packed = reader.u64()
    check(packed == (count + 7) // 8)
    model = reader.take(packed * 8)
    check(not any(model[count:]))
    latent(model[:count], width, len(classes), parameters["n_components"])
    for count in [len(classes)-1, (len(classes)-1) * parameters["n_components"]]:
        check(reader.u64() == count)
        for _ in range(count):
            reader.f64()
    reader.end()


def encoder(data, declaration, width):
    pca = declaration["kind"] == "tensor_pca"
    params, caps, actual, outputs, blocks = estimator(data, "preprocessing.feature_selection.flexible_pca" if pca else "preprocessing.scaling.standard_scale")
    expected = {"n_components": (2, struct.unpack("<Q", struct.pack("<d", float(declaration["n_components"])))[0])} if pca else {"with_mean": (3, 1), "with_std": (3, 1)}
    count = declaration["n_components"] if pca else width
    check(actual == width and outputs == 0 and caps == 129 and params == expected and set(blocks) == {0x31545354})
    reader = Reader(blocks[0x31545354])
    check(reader.number("q") == width)
    if pca:
        check(reader.u64() == width)
        for _ in range(width):
            reader.f64()
        check(reader.number("q") == count and reader.u64() == width * count)
        for _ in range(width * count):
            reader.f64()
    else:
        check([reader.number("q"), reader.number("q")] == [1, 1])
        for positive in (False, True):
            check(reader.u64() == width)
            for _ in range(width):
                value = reader.f64()
                check(not positive or value > 0)
    reader.end()
    return count


def canonical_recipe(recipe, schemas):
    result = bytearray()
    def text(value):
        encoded = value.encode()
        result.extend(struct.pack("<Q", len(encoded)))
        result.extend(encoded)
    text(METHOD)
    result.extend(struct.pack("<QQI", recipe["model"]["params"]["n_components"], recipe["model"]["params"]["max_iter"], len(recipe["source_order"])))
    for source in recipe["source_order"]:
        schema, decl = schemas[source], recipe["encoders"][source]
        for value in [source, schema["representation_id"], schema["dtype"], schema["identity"]]:
            text(value)
        result.extend(struct.pack("<I", len(schema["input_shape"])))
        for dimension in schema["input_shape"]:
            result.extend(struct.pack("<Q", dimension))
        kind = {"standard_scaler": 1, "tensor_pca": 2, "column_transformer": 3}[decl["kind"]]
        result.extend(struct.pack("<IdQQqqIIII", kind, recipe["source_weights"][source], decl.get("n_components", 0), decl.get("random_state", 0), 0 if kind == 3 else -1, 1 if kind == 3 else -1, int(kind != 2), int(kind != 2), 0, int(kind == 3)))
    return bytes(result)


def raw_classifier(data, recipe, schemas, classes):
    reader = checked_reader(data, b"N4MC")
    check(reader.block() == canonical_recipe(recipe, schemas))
    width = 0
    for source in recipe["source_order"]:
        shape = 1 if source == "metadata" else math.prod(schemas[source]["input_shape"])
        width += encoder(reader.block(), recipe["encoders"][source], shape)
        count = reader.u64()
        check(count <= 65_536 and (count > 0 if source == "metadata" else count == 0))
        values = [reader.text() for _ in range(count)]
        check(values == sorted(values) and len(set(values)) == count)
        width += count
    check(reader.u64() == len(classes) and [reader.number("q") for _ in classes] == classes)
    classifier(reader.block(), recipe["model"]["params"], width, classes)
    reader.end()


def validate_payload(record, payload, plan, load_json, validate_raw_recipe):
    """Validate strict wrapper, signed graph/parameter identity and native state."""
    artifact = record["artifact"]
    raw = artifact["kind"] == RAW_KIND
    check(raw or artifact["kind"] == META_KIND)
    owner = "controller:methods.python.multimodal.classification" if raw else "controller:methods.python.classification"
    plugin = "dagml.methods.python.multimodal.classification" if raw else "dagml.methods.python.classification"
    digest = hashlib.sha256(payload).hexdigest()
    check(0 < len(payload) <= 134_217_728 and record["controller_id"] == artifact["controller_id"] == owner and artifact["backend"] == "raw" and artifact.get("plugin") == plugin and artifact.get("plugin_version") == "1.0.0" and artifact.get("native_predictor_descriptor") is None and artifact.get("native_estimator_descriptor") is None and artifact["size_bytes"] == len(payload) and artifact["content_fingerprint"] == digest and artifact["uri"] == f"artifacts/{digest}.json")
    saved = load_json(payload, artifact["uri"])
    fields = {"schema", "node_id", "params_fingerprint", "target_names", "classification"} | ({"recipe", "source_schemas", "state"} if raw else {"steps", "source_order", "feature_names", "states"})
    check(set(saved) == fields and saved["schema"] == ("dagml.methods.multimodal.classification.v1" if raw else "dagml.methods.classification.v1") and saved["node_id"] == record["node_id"] and saved["params_fingerprint"] == record["params_fingerprint"] and saved["target_names"] == ["y"])
    classes = vocabulary(saved["classification"])
    graph = plan["graph_plan"]["graph"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    operator, node = nodes[record["node_id"]]["operator"], plan["node_plans"][record["node_id"]]
    check(node["controller_id"] == owner and node["params_fingerprint"] == record["params_fingerprint"] and operator["type"] == ("N4mMultimodalClassifierPipeline" if raw else "N4mRoleClassifierPipeline") and operator["classification"] == saved["classification"])
    check(set(operator) == ({"type", "recipe", "source_schemas", "classification"} if raw else {"type", "steps", "source_order", "classification"}))
    params = node["params"]
    allowed = {"recipe", "source_schemas", "classification", "model__n_components", "model__max_iter"} if raw else {"n_components", "max_iter"}
    check(set(params) <= allowed)
    expected = copy.deepcopy(operator["recipe"] if raw else operator["steps"])
    for name, value in params.items():
        if name in {"recipe", "source_schemas", "classification"}:
            check(operator[name] == value)
        elif raw:
            expected["model"]["params"][name.removeprefix("model__")] = value
        else:
            expected[0]["params"][name] = value
    if raw:
        check(saved["recipe"] == expected and saved["source_schemas"] == operator["source_schemas"] and expected["model"]["method_id"] == METHOD)
        validate_raw_recipe(expected, operator["source_schemas"])
        state = saved["state"]
        check(isinstance(state, list) and all(type(byte) is int and 0 <= byte <= 255 for byte in state))
        raw_classifier(bytes(state), expected, saved["source_schemas"], classes)
    else:
        check(saved["steps"] == expected and len(expected) == 1 and set(expected[0]) == {"methodId", "params"} and expected[0]["methodId"] == METHOD and saved["source_order"] == operator["source_order"] and not node.get("data_bindings"))
        order = operator["source_order"]
        check(2 <= len(order) <= 4 and len(set(order)) == len(order))
        incoming = [edge for edge in graph["edges"] if edge["target"]["node_id"] == record["node_id"]]
        check(len(incoming) == len(order))
        producers = {}
        for edge in incoming:
            base = nodes[edge["source"]["node_id"]]["operator"]
            check(edge["source"]["port_name"] == "probabilities" and edge["contract"]["requires_oof"] is True and base["type"] == "N4mMultimodalClassifierPipeline" and base["classification"] == saved["classification"] and len(base["recipe"]["source_order"]) == 1)
            source = base["recipe"]["source_order"][0]
            check(source not in producers)
            producers[source] = edge["source"]["node_id"]
        check(set(producers) == set(order))
        names = [f"{producers[source]}/class:{class_id}" for source in order for class_id in classes]
        check(saved["feature_names"] == names and isinstance(saved["states"], list) and len(saved["states"]) == 1 and isinstance(saved["states"][0], list) and all(type(byte) is int and 0 <= byte <= 255 for byte in saved["states"][0]))
        classifier(bytes(saved["states"][0]), expected[0]["params"], len(names), classes)
