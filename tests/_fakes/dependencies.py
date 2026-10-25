"""Distinguish optional runtimes that are absent from installations that are broken."""

import importlib
import importlib.metadata
from types import ModuleType

import pytest


def ray_dependency() -> ModuleType:
    """Skip an absent Ray distribution; fail any installed-runtime import error."""
    try:
        importlib.metadata.version("ray")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Ray tests require the ray extra", allow_module_level=True)
    try:
        return importlib.import_module("swarm_arena.ray")
    except Exception as error:
        error_type = type(error).__name__
    pytest.fail(f"installed Ray runtime failed to import: {error_type}", pytrace=False)
