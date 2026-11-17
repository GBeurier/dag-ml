"""Export a genuine N4MC state for the closed DAG-ML multimodal classifier recipe.

Usage: python export_states.py OUTPUT.json, with nirs4all-methods, NumPy and
scikit-learn importable. The data are seeded synthetic values; nothing is
measured or licensed.
"""
import json
import struct
import sys

import n4m
import numpy as np

METHOD = "models.classification.pls_logistic"
rng = np.random.default_rng(7103)
arrays = {
    "nir": rng.normal(size=(14, 4)),
    "image": rng.normal(size=(14, 2, 2, 1)),
    "series": rng.normal(size=(14, 2, 2)),
    "metadata": np.array([[float(i), f"category:{i % 3}"] for i in range(14)], dtype=object),
}
representations = ("signal_1d", "rgb_image", "series_mv", "tabular_mixed")
schemas = {
    name: {
        "representation_id": representation,
        "input_shape": list(values.shape[1:]),
        "dtype": "float64" if name != "metadata" else "object",
        "identity": json.dumps({"source": name}),
    }
    for (name, values), representation in zip(arrays.items(), representations, strict=True)
}
pca = {"kind": "tensor_pca", "n_components": 1, "whiten": False, "random_state": 17}
encoders = {
    "nir": {"kind": "standard_scaler", "with_mean": True, "with_std": True},
    "image": pca,
    "series": pca,
    "metadata": {
        "kind": "column_transformer", "numeric_columns": [0], "categorical_columns": [1],
        "with_mean": True, "with_std": True, "handle_unknown": "ignore",
        "sparse_output": False, "drop": None,
    },
}
recipe = {
    "schema_version": 1, "fusion": "early", "source_order": list(arrays),
    "encoders": encoders, "source_weights": {name: 1.0 for name in arrays},
    "model": {"method_id": METHOD, "params": {"n_components": 1, "max_iter": 200}},
}
model = n4m.MultimodalClassifierPipeline(recipe, schemas)
model.fit(arrays, np.arange(14, dtype=np.int64) % 3)
state = bytes(model.export_state())
fixture = {
    "producer": "nirs4all-methods " + n4m.version(),
    "writer_abi": list(struct.unpack_from("<III", state, 8)),
    "recipe": recipe, "source_schemas": schemas, "class_labels": [0, 1, 2],
    "state_hex": state.hex(),
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(fixture, stream, indent=1, sort_keys=True)
    stream.write("\n")
