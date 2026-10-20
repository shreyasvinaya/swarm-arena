"""Exercise immutable run pointers and shared key boundaries."""

import json
import traceback
from pathlib import Path

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena._board.pointers import (
    RunPointer,
    _validate_namespace,
    list_pointers,
    pointer_key,
    write_pointer,
)
from swarm_arena._errors import StorageError, _rebuild_core_signal
from swarm_arena._store.local import LocalStore


def test_pointer_written_once_and_idempotent(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    p = RunPointer(run_id="r1", run_root=str(tmp_path / "runs"))
    write_pointer(arena, "toy", p)
    write_pointer(arena, "toy", p)
    assert json.loads(arena.read("toy/runs/r1.json").data) == {
        "schema_version": 1,
        "run_id": "r1",
        "run_root": str(tmp_path / "runs"),
    }


def test_pointer_conflict_raises(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    write_pointer(arena, "toy", RunPointer(run_id="r1", run_root="/runs"))
    secret = "SENTINEL-private-root"
    with pytest.raises(StorageError) as caught:
        write_pointer(arena, "toy", RunPointer(run_id="r1", run_root=f"/{secret}"))
    error = caught.value
    assert secret not in str(error) + repr(error) + repr(error.args) + "".join(
        traceback.format_exception(error)
    )
    assert _rebuild_core_signal(error) is not None
    assert (
        arena.read("toy/runs/r1.json").data
        == b'{"run_id":"r1","run_root":"/runs","schema_version":1}'
    )


def test_pointer_rejects_relative_run_root() -> None:
    with pytest.raises(ValueError, match="absolute"):
        RunPointer(run_id="r1", run_root="runs")
    with pytest.raises(ValidationError, match="run_root must be absolute or s3://") as caught:
        RunPointer(run_id="r1", run_root="SENTINEL-private-relative-root")
    assert_secret_absent(caught.value, "SENTINEL-private-relative-root", sanitized=True)


def test_list_pointers_skips_corrupt_and_newer(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    write_pointer(arena, "toy", RunPointer(run_id="r1", run_root="s3://bucket/runs"))
    arena.create("toy/runs/r2.json", b"{not json SENTINEL-private")
    arena.create("toy/runs/r3.json", b'{"schema_version":99,"secret":"SENTINEL-private"}')
    arena.create("toy/runs/r4.json", b'{"run_id":"other","run_root":"/runs"}')
    arena.create("toy/runs/nested/r5.json", b"{}")
    pointers, warnings = list_pointers(arena, "toy")
    assert [p.run_id for p in pointers] == ["r1"]
    assert len(warnings) == 4
    assert all("SENTINEL-private" not in warning for warning in warnings)


@pytest.mark.parametrize("component", ["", ".", "..", "../escape", "/tmp", "a/b", "a\\b", "a\x00b"])
def test_invalid_components(component: str) -> None:
    with pytest.raises(ValueError, match="namespace"):
        _validate_namespace(component)
    with pytest.raises(ValueError, match="run_id"):
        pointer_key("toy", component)
    with pytest.raises(ValidationError, match="run_id"):
        RunPointer(run_id=component, run_root="/runs")


def test_pointer_keys_and_sorted_listing(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    for run_id in ("r2", "r1"):
        write_pointer(arena, "toy", RunPointer(run_id=run_id, run_root="/runs"))
    assert pointer_key("toy", "r1") == "toy/runs/r1.json"
    pointers, warnings = list_pointers(arena, "toy")
    assert [p.run_id for p in pointers] == ["r1", "r2"]
    assert warnings == []


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":99}',
        '{"schema_version":true}',
        '{"run_id":"SENTINEL-private","run_root":NaN}',
    ],
)
def test_pointer_safe_validation(raw: str) -> None:
    with pytest.raises(ValidationError) as caught:
        RunPointer.model_validate_json(raw)
    error = caught.value
    assert_secret_absent(error, "SENTINEL-private", sanitized=True)


def test_listing_applies_pointer_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any

    from swarm_arena._model import migrate

    arena = LocalStore(tmp_path)
    arena.create("toy/runs/r1.json", b'{"run_id":"r1","run_root":"/old"}')
    monkeypatch.setattr(migrate, "CURRENT", {**migrate.CURRENT, "run_pointer": 2})
    calls: list[dict[str, Any]] = []

    def step(raw: dict[str, Any]) -> dict[str, Any]:
        calls.append(raw.copy())
        return {**raw, "run_root": "/new"}

    monkeypatch.setitem(migrate._migration_functions, ("run_pointer", 1), step)
    pointers, warnings = list_pointers(arena, "toy")
    assert warnings == []
    assert [(p.run_root, p.schema_version) for p in pointers] == [("/new", 2)]
    assert calls == [{"run_id": "r1", "run_root": "/old"}]
