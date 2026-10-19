"""Sphinx configuration for the swarm-arena docs."""

import importlib.metadata

project = "swarm-arena"
author = "Shreyas Vinaya"
copyright = "2026, Shreyas Vinaya"
release = importlib.metadata.version("swarm-arena")

extensions = ["sphinx.ext.autodoc", "numpydoc", "myst_parser"]
html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build", "superpowers"]

numpydoc_show_class_members = False
autodoc_member_order = "bysource"
