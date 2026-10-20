"""Sphinx configuration for the swarm-arena docs."""

import importlib.metadata

project = "swarm-arena"
author = "Shreyas Vinaya"
copyright = "2026, Shreyas Vinaya"
release = importlib.metadata.version("swarm-arena")

extensions = ["sphinx.ext.autodoc", "numpydoc", "myst_parser"]
html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build", "superpowers"]
myst_heading_anchors = 3

numpydoc_show_class_members = False
autodoc_member_order = "bysource"
autodoc_default_options = {
    "exclude-members": "model_config,model_post_init,model_fields,model_computed_fields",
}
# Model parameters are documented in the numpydoc Parameters sections. Generated
# Annotated signatures otherwise expose Pydantic's private constraint metadata.
autodoc_typehints = "none"
