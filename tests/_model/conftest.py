"""Shared model-test fixtures."""

from types import MappingProxyType

import pytest

from swarm_arena._model import migrate


@pytest.fixture
def isolated_migrations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace both registry views for one test and restore them at teardown."""
    functions: dict[tuple[str, int], migrate.MigrationFn] = {}
    monkeypatch.setattr(migrate, "_migration_functions", functions)
    monkeypatch.setattr(migrate, "_MIGRATIONS", MappingProxyType(functions))
