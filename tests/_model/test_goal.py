from typing import Any

import pytest
from pydantic import ValidationError

from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.goal import FileSet, Goal, GoalKey

H = "a" * 64


def test_fileset_from_glob_list() -> None:
    fs = FileSet.model_validate(["problem/*.phyk"])
    assert fs.globs == ["problem/*.phyk"]
    assert not fs.snapshotted
    assert FileSet().snapshotted
    assert FileSet(globs=["x/*"], manifest={"x/a": H}).snapshotted


@pytest.mark.parametrize(
    "bad",
    [
        {"globs": ["/abs/*"]},
        {"globs": ["../x"]},
        {"manifest": {"swarm/x": H}},
        {"manifest": {"a": "nothex"}},
        {"manifest": {"./a": H}},
    ],
)
def test_fileset_rejects(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FileSet.model_validate(bad)


def test_goal_slugs() -> None:
    assert Goal(id="ns-viscosity", namespace="physika").title == ""
    for bad in ({"id": "NS Visc", "namespace": "physika"}, {"id": "a", "namespace": "-x"}):
        with pytest.raises(ValidationError):
            Goal.model_validate(bad)


def test_goal_key_values() -> None:
    g = Goal(
        id="g",
        namespace="ns",
        inputs={"b": 1, "a": 2},
        files=FileSet(globs=["p/*"], manifest={"p/x": H}),
    )
    assert (
        GoalKey.of(g)
        == g.key()
        == GoalKey(
            namespace="ns",
            id="g",
            inputs_sha256=sha256_hex(b'{"a":2,"b":1}'),
            files_sha256=sha256_hex(canonical_json({"p/x": H})),
        )
    )
    other = g.model_copy(update={"files": FileSet(globs=["p/*"], manifest={"p/x": "b" * 64})})
    assert other.key().files_sha256 != g.key().files_sha256


def test_goal_key_requires_snapshot() -> None:
    with pytest.raises(ValueError, match="snapshot"):
        Goal(id="g", namespace="n", files=FileSet(globs=["a/*"])).key()


def test_goal_key_canonical_identity() -> None:
    goal = Goal(
        id="g",
        namespace="ns",
        inputs={"z": [1, {"b": True, "a": "λ"}], "a": False},
        files=FileSet(manifest={"b": H, "a": "b" * 64}),
    )
    reordered = Goal(
        id="g",
        namespace="ns",
        inputs={"a": False, "z": [1, {"a": "λ", "b": True}]},
        files=FileSet(globs=["*"], manifest={"a": "b" * 64, "b": H}),
        title="Different title",
        description="Different description",
        metadata={"note": 1},
    )
    assert goal.key() == reordered.key()
    assert goal.model_copy(update={"inputs": {"a": 1}}).key() != goal.key()
    assert goal.model_copy(update={"namespace": "other"}).key() != goal.key()
    assert goal.model_copy(update={"id": "other"}).key() != goal.key()


@pytest.mark.parametrize("bad", ["", "Upper", "a/b", "x\n", ".", "-x", "x y"])
def test_goal_slug_boundaries(bad: str) -> None:
    for field in ("id", "namespace"):
        with pytest.raises(ValidationError):
            Goal.model_validate({"id": "g", "namespace": "n", field: bad})


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "a" * 65, H + "\n"])
def test_fileset_requires_exact_lowercase_digest(digest: str) -> None:
    with pytest.raises(ValidationError):
        FileSet(manifest={"a": digest})
