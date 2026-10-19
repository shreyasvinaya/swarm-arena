"""The installed package version, read from the distribution metadata."""

import importlib.metadata

__version__: str = importlib.metadata.version("swarm-arena")
