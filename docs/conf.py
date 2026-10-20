"""Sphinx configuration for the swarm-arena docs."""

import importlib.metadata
from inspect import Parameter, Signature
from inspect import signature as object_signature
from typing import Any

from sphinx.application import Sphinx

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


def _skip_default_input_alias(
    app: Sphinx, what: str, name: str, obj: Any, skip: bool, options: Any
) -> bool | None:
    """Hide private default schemas; configured schemas use get_input_model."""
    if name == "input_model" and isinstance(obj, type) and obj.__name__.startswith("_"):
        return True
    return None


def _public_signature(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, str | None] | None:
    """Render Part's omitted-data default without exposing its sentinel type."""
    if name != "swarm_arena.protocol.Part":
        return None
    original = object_signature(obj)
    parameters = [
        param.replace(
            annotation=Parameter.empty,
            default=Ellipsis if param.name == "data" else param.default,
        )
        for param in original.parameters.values()
    ]
    public = original.replace(parameters=parameters, return_annotation=Signature.empty)
    return str(public).replace("data=Ellipsis", "data=..."), return_annotation


def setup(app: Sphinx) -> None:
    """
    Register public API rendering rules.

    Parameters
    ----------
    app : Sphinx
        Documentation application receiving the autodoc callbacks.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> app = Mock(spec=Sphinx)
    >>> setup(app)
    >>> app.connect.call_count
    2
    """
    app.connect("autodoc-skip-member", _skip_default_input_alias)
    app.connect("autodoc-process-signature", _public_signature)
