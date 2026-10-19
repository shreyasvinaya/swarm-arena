from typing import Any

import pytest

from swarm_arena._model import migrate
from swarm_arena._model.migrate import migration, upgrade


@pytest.fixture(autouse=True)
def isolated_migrations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(migrate, "_MIGRATIONS", {})


def test_current_version_is_identity() -> None:
    assert upgrade("plan", {"schema_version": 1, "x": 1}) == {"schema_version": 1, "x": 1}


def test_missing_version_means_one() -> None:
    assert upgrade("plan", {"x": 1}) == {"schema_version": 1, "x": 1}


def test_newer_version_refused() -> None:
    with pytest.raises(
        ValueError,
        match=r"plan schema_version 2 is newer than this swarm-arena supports \(1\)",
    ):
        upgrade("plan", {"schema_version": 2})


def test_migrations_apply_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(migrate.CURRENT, "demo", 3)

    @migration("demo", 1)
    def one(raw: dict[str, Any]) -> dict[str, Any]:
        return {("b" if k == "a" else k): v for k, v in raw.items()}

    @migration("demo", 2)
    def two(raw: dict[str, Any]) -> dict[str, Any]:
        return {**raw, "c": raw["b"] * 2}

    assert upgrade("demo", {"a": 2}) == {"schema_version": 3, "b": 2, "c": 4}


def test_unknown_kind_missing_step_and_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unknown stored-format kind 'nope'"):
        upgrade("nope", {})
    monkeypatch.setitem(migrate.CURRENT, "demo", 2)
    with pytest.raises(RuntimeError, match="no migration for demo 1"):
        upgrade("demo", {})

    def step(raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    assert migration("demo", 1)(step) is step
    with pytest.raises(ValueError, match="already registered"):
        migration("demo", 1)(step)


def test_upgrade_copies_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(migrate.CURRENT, "demo", 2)

    @migration("demo", 1)
    def step(raw: dict[str, Any]) -> dict[str, Any]:
        raw["x"] = 2
        return raw

    raw = {"schema_version": 1, "x": 1}
    assert upgrade("demo", raw) == {"schema_version": 2, "x": 2}
    assert raw == {"schema_version": 1, "x": 1}
    assert upgrade("plan", raw) is not raw


def test_current_kinds() -> None:
    assert migrate.CURRENT == {
        "candidate": 1,
        "learning": 1,
        "llm_call": 1,
        "plan": 1,
        "run_config": 1,
        "job_started": 1,
        "request": 1,
        "candidate_request": 1,
        "candidate_output": 1,
        "tool_result": 1,
    }
