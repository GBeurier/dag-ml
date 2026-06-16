from __future__ import annotations

project = "dag-ml"
author = "G. Beurier"
copyright = "2026, G. Beurier"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
    "sphinxext.opengraph",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

# Pages kept out of the published navigation: build scratch space, the bulky
# source design corpus (kept for traceability, not user-facing), the contract
# leaf schemas (browsed from contracts/README), and narrow internal backlogs
# that would only add orphan pages to the public site.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "design/source/*",
    "design/DSL_NIRS4ALL_PARITY.md",
    "HOST_ADAPTER_BACKLOG.md",
    "STUDIO_LITE_WASM_GAPS.md",
    "HETEROGENEOUS_MULTISOURCE_REPETITIONS_ROADMAP.md",
]

# autosectionlabel makes every section heading a cross-reference target; the
# document prefix keeps labels unique across the many normative docs.
autosectionlabel_prefix_document = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
    "attrs_inline",
    "dollarmath",
]
myst_heading_anchors = 3

html_theme = "alabaster"
html_title = "dag-ml"
html_static_path = ["_static"]
html_logo = "_static/brand/stacked.png"
html_favicon = "_static/brand/favicon.ico"

ogp_site_url = "https://dag-ml.readthedocs.io/en/latest/"
ogp_image = "https://dag-ml.readthedocs.io/en/latest/_static/brand/og.png"
