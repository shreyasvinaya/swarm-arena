"""Exercise human object payloads, deterministic persistence, and ordering."""

import re
import traceback
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena._board.human_objects import (
    HumanObject,
    fold_curation,
    human_object_key,
    human_object_name,
    human_object_order,
    list_human_objects,
    read_human_object,
    write_human_objects,
)
from swarm_arena._errors import AlreadyExists
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.results import Learning, Score
from swarm_arena._store.local import LocalStore

NOW = datetime(2026, 10, 18, 13, tzinfo=UTC)
NAME = "20261018T130000.000Z-0a1b2c3d.jsonl"


def _learning() -> Learning:
    return Learning(
        id="L-1",
        namespace="toy",
        run_id="r1",
        goal_id="g",
        round=1,
        author="human:a",
        title="Title",
        text="Body",
    )


def test_human_object_name_format() -> None:
    name = human_object_name(NOW)
    assert re.fullmatch(r"20261018T130000\.000Z-[0-9a-f]{8}\.jsonl", name)
    local = NOW.astimezone(timezone(timedelta(hours=5, minutes=30))).replace(microsecond=123999)
    assert human_object_name(local).startswith("20261018T130000.123Z-")
    with pytest.raises(ValueError, match="timezone"):
        human_object_name(datetime(2026, 10, 18))


def test_order_is_timestamp_then_name() -> None:
    a = "toy/curation/20261018T130001.000Z-aaaa0000.jsonl"
    b = "toy/reviews/bob/20261018T130000.000Z-ffff0000.jsonl"
    c = "toy/curation/20261018T130000.000Z-0000ffff.jsonl"
    assert sorted([a, b, c], key=human_object_order) == [c, b, a]


def test_later_pin_or_hide_wins() -> None:
    hide = HumanObject(kind="hide", learning_id="L-1", author="human:a")
    pin = hide.model_copy(update={"kind": "pin"})
    assert fold_curation([("n1", 0, hide), ("n2", 0, pin)]) == {"L-1": (True, False)}
    assert fold_curation([("n1", 0, pin), ("n2", 0, hide)]) == {"L-1": (False, True)}
    name = human_object_key("toy", "curation", NAME)
    assert fold_curation([(name, 1, hide), (name, 0, pin)]) == {"L-1": (False, True)}
    assert (
        fold_curation([(name, 0, HumanObject(kind="add", learning=_learning(), author="human:a"))])
        == {}
    )


def test_write_uses_given_name(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    obj = HumanObject(kind="review", score=Score(key=(1.0,)), author="human:alice")
    key = write_human_objects(arena, "toy", "reviews", [obj], now=NOW, reviewer="alice", name=NAME)
    assert key == f"toy/reviews/alice/{NAME}"
    assert arena.read(key).data == canonical_json(obj.model_dump(mode="json")) + b"\n"
    with pytest.raises(AlreadyExists):
        write_human_objects(arena, "toy", "reviews", [obj], now=NOW, reviewer="alice", name=NAME)


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    review = HumanObject(kind="review", score=Score(key=(1.0,)), author="human:a")
    curation = [
        HumanObject(kind="add", learning=_learning(), author="human:a"),
        HumanObject(kind="pin", learning_id="L-1", author="human:a"),
    ]
    reviews = write_human_objects(arena, "toy", "reviews", [review], now=NOW, reviewer="alice")
    curated = write_human_objects(arena, "toy", "curation", curation, now=NOW)
    assert read_human_object(arena, reviews) == [review]
    assert read_human_object(arena, curated) == curation
    assert list_human_objects(arena, ["toy", "toy"]) == ([reviews], [curated])


def test_reviewer_rules(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    for reviewer in (None, "Alice Smith", "../a", "a/b"):
        with pytest.raises(ValueError, match="reviewer"):
            write_human_objects(arena, "toy", "reviews", [], now=NOW, reviewer=reviewer)
    with pytest.raises(ValueError, match="curation"):
        write_human_objects(arena, "toy", "curation", [], now=NOW, reviewer="alice")
    assert arena.list("") == []


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "pin"},
        {"kind": "hide", "learning_id": ""},
        {"kind": "pin", "learning_id": "L-1", "score": Score(key=(1.0,))},
        {"kind": "add"},
        {"kind": "add", "learning": _learning(), "learning_id": "L-1"},
        {"kind": "review"},
        {"kind": "review", "score": Score(key=(1.0,)), "learning": _learning()},
    ],
)
def test_kind_payload_invariants(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        HumanObject(author="human:a", **payload)


@pytest.mark.parametrize(
    "basename",
    [
        "../x",
        "/x",
        "x",
        "20261318T130000.000Z-0a1b2c3d.jsonl",
        "20261018T250000.000Z-0a1b2c3d.jsonl",
        "20261018T130000.000Z-ABCDEF12.jsonl",
    ],
)
def test_invalid_basename(basename: str) -> None:
    with pytest.raises(ValueError, match="basename"):
        human_object_key("toy", "curation", basename)


def test_reserved_locations_and_folder_payloads(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    for key in (f"toy/reviews/{NAME}", f"toy/curation/nested/{NAME}", f"toy/other/{NAME}"):
        with pytest.raises(ValueError, match="location"):
            read_human_object(arena, key)
    review = HumanObject(kind="review", score=Score(key=(1.0,)), author="human:a")
    with pytest.raises(ValueError, match="folder"):
        write_human_objects(arena, "toy", "curation", [review], now=NOW)
    assert arena.list("") == []


@pytest.mark.parametrize(
    "raw",
    [
        b"{not-json SENTINEL-private",
        b'{"kind":"SENTINEL-private"}',
        b'{"schema_version":99,"author":"SENTINEL-private"}',
        b'{"kind":"review","author":"SENTINEL-private","score":{"key":[NaN]}}',
    ],
)
def test_read_rejects_unsafe_input(tmp_path: Path, raw: bytes) -> None:
    arena = LocalStore(tmp_path)
    key = human_object_key("toy", "reviews", NAME, reviewer="alice")
    arena.create(key, raw + b"\n")
    with pytest.raises(ValidationError) as caught:
        read_human_object(arena, key)
    error = caught.value
    assert_secret_absent(error, "SENTINEL-private", sanitized=True)
    surfaces = (
        str(error) + repr(error) + repr(error.args) + "".join(traceback.format_exception(error))
    )
    assert "SENTINEL-private" not in surfaces


def test_read_applies_human_object_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._model import migrate

    arena = LocalStore(tmp_path)
    key = human_object_key("toy", "curation", NAME)
    arena.create(key, b'{"kind":"pin","learning_id":"L-1","author":"human:a"}\n')
    monkeypatch.setattr(migrate, "CURRENT", {**migrate.CURRENT, "human_object": 2})
    calls: list[dict[str, Any]] = []

    def step(raw: dict[str, Any]) -> dict[str, Any]:
        calls.append(raw.copy())
        return {**raw, "kind": "hide"}

    monkeypatch.setitem(migrate._migration_functions, ("human_object", 1), step)
    loaded = read_human_object(arena, key)
    assert [(obj.kind, obj.schema_version) for obj in loaded] == [("hide", 2)]
    assert len(calls) == 1


def test_lists_are_globally_ordered_and_empty_files_roundtrip(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    names = [
        human_object_key("z", "curation", "20261018T120000.000Z-ffffffff.jsonl"),
        human_object_key("a", "curation", "20261018T130000.000Z-00000000.jsonl"),
        human_object_key("z", "reviews", NAME, reviewer="alice"),
        human_object_key("a", "reviews", NAME, reviewer="bob"),
    ]
    for key in names:
        arena.create(key, b"")
    assert list_human_objects(arena, ["z", "a"]) == (sorted(names[2:]), names[:2])
    assert read_human_object(arena, names[0]) == []
    arena.create("z/curation/unexpected.jsonl", b"")
    with pytest.raises(ValueError, match="basename"):
        list_human_objects(arena, ["z"])


def test_read_rejects_curation_score(tmp_path: Path) -> None:
    arena = LocalStore(tmp_path)
    key = human_object_key("toy", "curation", NAME)
    obj = HumanObject(kind="review", score=Score(key=(1.0,)), author="human:a")
    arena.create(key, canonical_json(obj.model_dump(mode="json")) + b"\n")
    with pytest.raises(ValueError, match="folder"):
        read_human_object(arena, key)


@pytest.mark.parametrize("length", [65, 256])
def test_review_key_rejects_overlong_reviewer(length: int) -> None:
    with pytest.raises(ValueError, match="reviewer"):
        human_object_key("toy", "reviews", NAME, reviewer="a" * length)
    assert "/" + "b" * 64 + "/" in human_object_key("toy", "reviews", NAME, reviewer="b" * 64)
