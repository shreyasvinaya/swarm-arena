"""Human review preserves committed identity, authority and exact receipt bytes."""

import csv
import io
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from _fakes.finished import FinishedRun
from _fakes.journals import FakeRun
from _fakes.secrecy import assert_secret_absent

import swarm_arena._scoring.human_review as human_review
from swarm_arena import RunConfig
from swarm_arena._board.human_objects import read_human_object
from swarm_arena._errors import ConditionalWritesUnsupported, Fenced, StorageError, _core_error
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.lease import Lease, read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._model.ids import run_tag
from swarm_arena._scoring.human_review import (
    CSV_HEADER,
    HumanReview,
    parse_review_csv,
    render_review,
)
from swarm_arena._store.base import Store, open_store


@pytest.fixture
def review_run(run_config: RunConfig) -> FakeRun:
    run = FakeRun(run_config.run_root, "review-run")
    run.commit(
        EventDraft(
            type=EventType.RUN_CONFIG, payload={"config": run_config.model_dump(mode="json")}
        ),
        run.plan(),
        *[run.candidate(f"c-{i}") for i in range(1, 5)],
        *[run.score(f"s-{i}", f"c-{i}", 5 - i) for i in range(1, 5)],
    )
    run._lease.release()
    return run


def _events(store: Store) -> list[Event]:
    owner = read_owner(store)
    assert owner is not None
    assert owner[0].head is not None
    return [e for _, batch in read_chain(store, owner[0].head) for e in batch]


@pytest.fixture
def state(review_run: FakeRun) -> RunState:
    return RunState.from_events(_events(review_run.store))


def _csv(*rows: dict[str, str], header: tuple[str, ...] = CSV_HEADER) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _file(tmp_path: Path, data: bytes | None = None) -> Path:
    path = tmp_path / "review.csv"
    path.write_bytes(
        data if data is not None else _csv({"id": "c-1", "verdict": "accept", "score": "7"})
    )
    return path


def _reviews(config: RunConfig) -> list[str]:
    return [k for k in open_store(config.arena_root).list("") if "/reviews/" in k]


async def test_export_writes_csv_json_and_html(finished_run: FinishedRun, tmp_path: Path) -> None:
    out = await HumanReview("alice").export(
        finished_run.run_id, config=finished_run.config, out_dir=tmp_path / "rev"
    )
    rows = list(csv.reader((out / "review.csv").read_text().splitlines()))
    assert tuple(rows[0]) == CSV_HEADER
    assert 1 < len(rows) <= 21
    assert all((out / "candidates" / f"{r[0]}.json").exists() for r in rows[1:])
    assert rows[1][0] in (out / "index.html").read_text()


async def test_export_top_and_goal_filter(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path
) -> None:
    out = await HumanReview("alice").export(
        review_run.run_id, config=run_config, out_dir=tmp_path / "rev", top=1
    )
    assert len(list(csv.reader((out / "review.csv").read_text().splitlines()))) == 2
    with pytest.raises(ValueError, match="nope"):
        await HumanReview("alice").export(
            review_run.run_id, config=run_config, out_dir=tmp_path / "bad", goal="nope"
        )
    assert not (tmp_path / "bad").exists()


def test_export_escapes_html(state: RunState) -> None:
    state.scores["s-1"] = state.scores["s-1"].model_copy(
        update={"notes": "<script>alert(1)</script>"}
    )
    page = render_review(state)["index.html"]
    assert b"&lt;script&gt;" in page
    assert b"<script>alert" not in page


def test_render_is_deterministic(state: RunState) -> None:
    assert render_review(state) == render_review(state)
    rows = list(csv.DictReader(io.StringIO(render_review(state)["review.csv"].decode())))
    assert rows[0]["scores"] == "j=1"
    assert all(not r[k] for r in rows for k in ("verdict", "score", "notes"))


async def test_import_roundtrip(finished_run: FinishedRun, tmp_path: Path) -> None:
    config, run_id = finished_run.config, finished_run.run_id
    review = HumanReview("alice")
    out = await review.export(run_id, config=config, out_dir=tmp_path / "rev")
    rows = list(csv.DictReader(io.StringIO((out / "review.csv").read_text())))
    rows = rows[:2]
    rows[0].update(verdict="accept", score="7")
    rows[1].update(verdict="reject")
    scores = await review.import_csv(run_id, _file(tmp_path, _csv(*rows)), config=config)
    assert [s.key for s in scores] == [(1.0, 7.0), (-1.0, 0.0)]
    assert {s.judge for s in scores} == {"human:alice"}
    events = _events(open_store(config.run_root).child(run_id))
    receipt = events[-1]
    assert receipt.type == EventType.REVIEW_IMPORTED
    assert events[-2].payload == {"force": False}
    keys = _reviews(config)
    assert len(keys) == 1
    assert receipt.payload["objects"] == keys
    assert {s.batch_id for s in scores} == set(keys)
    assert [
        obj.score for obj in read_human_object(open_store(config.arena_root), keys[0])
    ] == scores


async def test_import_run_without_commit_raises(run_config: RunConfig, tmp_path: Path) -> None:
    run = FakeRun(run_config.run_root, "empty")
    before = run.store.read("control/owner.json").data
    with pytest.raises(ValueError, match="no committed events"):
        await HumanReview("alice").import_csv("empty", _file(tmp_path), config=run_config)
    assert run.store.read("control/owner.json").data == before


async def test_import_accepts_spreadsheet_csv(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path
) -> None:
    data = b"\xef\xbb\xbf" + _csv(
        {"id": "c-1", "verdict": "Accept", "notes": "line one\r\nline two"}
    )
    scores = await HumanReview("alice").import_csv(
        review_run.run_id, _file(tmp_path, data), config=run_config
    )
    assert scores[0].key == (1.0, 0.0)
    assert scores[0].notes == "line one\r\nline two"


async def test_import_rejects_bad_rows_and_writes_nothing(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(human_review, "check_storage", lambda cfg: calls.append(1))
    data = _csv(
        {"id": "c-nope", "score": "1"},
        {"id": "c-1", "verdict": "maybe"},
        {"id": "c-2", "score": "abc"},
        {"id": "c-3", "score": "nan"},
    )
    before = review_run.store.read("control/owner.json").data
    with pytest.raises(ValueError, match="row 2:") as err:
        await HumanReview("alice").import_csv(
            review_run.run_id, _file(tmp_path, data), config=run_config
        )
    assert all(f"row {n}:" in str(err.value) for n in (2, 3, 4, 5))
    assert review_run.store.read("control/owner.json").data == before
    assert not calls
    assert not _reviews(run_config)


async def test_import_checks_storage_before_takeover(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fail(config: RunConfig) -> None:
        calls.append(1)
        raise ConditionalWritesUnsupported("conditional writes")

    monkeypatch.setattr(human_review, "check_storage", fail)
    before = review_run.store.read("control/owner.json").data
    with pytest.raises(ConditionalWritesUnsupported):
        await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    assert calls == [1]
    assert review_run.store.read("control/owner.json").data == before
    assert not _reviews(run_config)


async def test_import_fails_while_owner_live(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path
) -> None:
    review_run._lease.renew()
    before = review_run.store.read("control/owner.json").data
    with pytest.raises(RuntimeError, match="live owner"):
        await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    assert review_run.store.read("control/owner.json").data == before
    assert not _reviews(run_config)


async def test_import_retries_takeover_and_uses_final_head(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Lease.take_over.__func__
    seen: list[int] = []

    def race(cls: type[Lease], store: Store, **kwargs: Any) -> Lease:
        seen.append(1)
        if len(seen) == 1:
            review_run.commit(review_run.score(f"s-{run_tag(review_run.run_id)}-e1-9999", "c-1", 9))
            review_run._lease.release()
            raise Fenced()
        return real(cls, store, **kwargs)

    monkeypatch.setattr(Lease, "take_over", classmethod(race))
    scores = await HumanReview("alice").import_csv(
        review_run.run_id, _file(tmp_path), config=run_config
    )
    events = _events(review_run.store)
    assert len(seen) == 2
    assert events[-2].seq == review_run.heads[-1].seq + 1
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert scores[0].id.endswith("-10000")


async def test_import_loses_to_another_owner(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Lease.take_over.__func__
    seen: list[dict[str, Any]] = []

    def race(cls: type[Lease], store: Store, **kwargs: Any) -> Lease:
        seen.append(kwargs)
        other: dict[str, Any] = {**kwargs, "owner_id": "other", "force": True}
        real(cls, store, **other)
        return real(cls, store, **kwargs)

    monkeypatch.setattr(Lease, "take_over", classmethod(race))
    with pytest.raises(Fenced, match="another owner took the run"):
        await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    assert seen[0]["expect_epoch"] == 1
    owner = read_owner(review_run.store)
    assert owner is not None
    assert (owner[0].owner_id, owner[0].epoch) == ("other", 2)
    assert not _reviews(run_config)
    assert not any(e.type == EventType.REVIEW_IMPORTED for e in _events(review_run.store))


async def test_import_ids_use_new_epoch(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path
) -> None:
    scores = await HumanReview("alice").import_csv(
        review_run.run_id, _file(tmp_path), config=run_config
    )
    assert re.fullmatch(rf"s-{run_tag(review_run.run_id)}-e2-\d{{4}}", scores[0].id)


def test_reviewer_slug() -> None:
    with pytest.raises(ValueError, match="reviewer"):
        HumanReview("Alice Smith")
    assert HumanReview("alice").judge_name == "human:alice"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\xff",
        _csv({"id": "c-1"}, header=("id",)),
        _csv({"id": "c-1", "score": "1"}, {"id": "c-1"}),
        _csv({"id": "c-1", "score": "inf"}),
    ],
)
def test_parse_rejects_invalid_csv(data: bytes, state: RunState) -> None:
    with pytest.raises(ValueError, match="row "):
        parse_review_csv(data, state, judge="human:alice")


def test_parse_error_never_exposes_values(state: RunState) -> None:
    sentinel = "SECRET_REVIEW_CELL"
    with pytest.raises(ValueError, match="row 2:") as error:
        parse_review_csv(
            _csv({"id": "c-1", "verdict": sentinel, "score": sentinel}), state, judge="human:alice"
        )
    assert_secret_absent(error.value, sentinel)


def test_parse_blank_reviews_do_not_allocate_ids(state: RunState) -> None:
    before = dict(state.id_counters)
    assert parse_review_csv(render_review(state)["review.csv"], state, judge="human:alice") == []
    assert state.id_counters == before
    scores = parse_review_csv(_csv({"id": "c-1", "score": "-2.5"}), state, judge="human:alice")
    assert scores[0].key == (0.0, -2.5)
    assert scores[0].id == ""


async def test_committed_receipt_repaired_before_next_import(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = human_review.write_human_objects

    def fail(*args: Any, **kwargs: Any) -> str:
        raise _core_error(StorageError, "publication interrupted")

    monkeypatch.setattr(human_review, "write_human_objects", fail)
    with pytest.raises(StorageError, match="publication interrupted"):
        await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    receipt = _events(review_run.store)[-1]
    assert receipt.type == EventType.REVIEW_IMPORTED
    assert not _reviews(run_config)
    monkeypatch.setattr(human_review, "write_human_objects", real)
    await HumanReview("bob").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    key = receipt.payload["objects"][0]
    objects = read_human_object(open_store(run_config.arena_root), key)
    assert [obj.score.model_dump(mode="json") for obj in objects if obj.score] == receipt.payload[
        "scores"
    ]
    assert len(_reviews(run_config)) == 2


async def test_import_final_reparse_failure_releases_without_events(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = human_review.parse_review_csv
    parsed: list[RunState] = []
    before = _events(review_run.store)

    def parse(data: bytes, state: RunState, *, judge: str) -> list[human_review.Score]:
        parsed.append(state)
        if len(parsed) == 2:
            raise ValueError("row 2: final state rejected")
        return original(data, state, judge=judge)

    monkeypatch.setattr(human_review, "parse_review_csv", parse)
    with pytest.raises(ValueError, match="final state rejected"):
        await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    assert len(parsed) == 2
    assert _events(review_run.store) == before
    owner = read_owner(review_run.store)
    assert owner is not None
    assert owner[0].epoch == 2
    assert owner[0].expires_at < human_review.utc_now()
    assert not _reviews(run_config)


async def test_import_missing_owner_on_retry(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine import resume

    monkeypatch.setattr(resume, "read_owner", lambda store: None)
    before = review_run.store.read("control/owner.json").data
    with pytest.raises(FileNotFoundError, match=r"control/owner\.json"):
        await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    assert review_run.store.read("control/owner.json").data == before
    assert not _reviews(run_config)


async def test_import_uses_no_engine_services_or_clock(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._engine import run

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("standalone review import constructed engine services")

    monkeypatch.setattr(run, "_start_executor", forbidden)
    monkeypatch.setattr(run, "_start_board", forbidden)
    monkeypatch.setattr(run, "RunClock", forbidden)
    await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    assert all("run_elapsed_s" not in e.payload for e in _events(review_run.store)[-2:])


async def test_empty_review_import_is_valid(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path
) -> None:
    assert (
        await HumanReview("alice").import_csv(
            review_run.run_id, _file(tmp_path, _csv()), config=run_config
        )
        == []
    )
    keys = _reviews(run_config)
    assert len(keys) == 1
    assert open_store(run_config.arena_root).read(keys[0]).data == b""


def _add_namespace(run: FakeRun) -> None:
    from swarm_arena import Goal

    run.goal = Goal(id="g2", namespace="second")
    run.commit(run.plan(), run.candidate("c-second"), run.score("s-second", "c-second", 2))
    run._lease.release()


async def test_partial_namespace_publication_repairs_exact_bytes(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_namespace(review_run)
    original = human_review.write_human_objects
    seen: list[tuple[str, bytes]] = []
    arena = open_store(run_config.arena_root)

    def fail_second(*args: Any, **kwargs: Any) -> str:
        if len(seen) == 1:
            raise _core_error(StorageError, "interrupted second namespace")
        key = original(*args, **kwargs)
        seen.append((key, arena.read(key).data))
        return key

    monkeypatch.setattr(human_review, "write_human_objects", fail_second)
    path = _file(
        tmp_path, _csv({"id": "c-1", "score": "7"}, {"id": "c-second", "verdict": "accept"})
    )
    with pytest.raises(StorageError, match="second namespace"):
        await HumanReview("alice").import_csv(review_run.run_id, path, config=run_config)
    receipt = _events(review_run.store)[-1]
    assert len(receipt.payload["objects"]) == 2
    assert len(_reviews(run_config)) == 1
    monkeypatch.setattr(human_review, "write_human_objects", original)
    await HumanReview("bob").import_csv(
        review_run.run_id, _file(tmp_path, _csv()), config=run_config
    )
    assert arena.read(seen[0][0]).data == seen[0][1]
    for key in receipt.payload["objects"]:
        objects = read_human_object(arena, key)
        assert [o.score.model_dump(mode="json") for o in objects if o.score] == [
            s for s in receipt.payload["scores"] if s["batch_id"] == key
        ]
    assert len([k for k in _reviews(run_config) if "/alice/" in k]) == 2


async def test_receipt_conflicting_bytes_stop_before_new_import(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path
) -> None:
    await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    before = _events(review_run.store)
    key = _reviews(run_config)[0]
    arena = open_store(run_config.arena_root)
    arena.replace(key, b"SECRET_OBJECT_BYTES", if_match=arena.read(key).etag)
    with pytest.raises(StorageError, match="bytes differ") as error:
        await HumanReview("bob").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    assert_secret_absent(error.value, "SECRET_OBJECT_BYTES")
    assert _events(review_run.store) == before
    assert _reviews(run_config) == [key]
    owner = read_owner(review_run.store)
    assert owner is not None
    assert owner[0].expires_at < human_review.utc_now()


async def test_repair_stops_on_fencing_before_publication(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = human_review._repair_review_objects
    calls: list[int] = []
    written: list[int] = []

    async def fence(
        state: RunState,
        arena: Store,
        store: Store,
        lease: Lease,
        writer: human_review.JournalWriter,
    ) -> None:
        calls.append(1)
        if len(calls) == 2:
            Lease.take_over(
                store,
                owner_id="other",
                host="local",
                ttl_s=60,
                force=True,
                expect_epoch=lease.doc.epoch,
            )
        await real(state, arena, store, lease, writer)

    monkeypatch.setattr(human_review, "_repair_review_objects", fence)
    monkeypatch.setattr(human_review, "write_human_objects", lambda *a, **kw: written.append(1))
    with pytest.raises(Fenced):
        await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    assert not written
    owner = read_owner(review_run.store)
    assert owner is not None
    assert owner[0].owner_id == "other"
    assert _events(review_run.store)[-1].type == EventType.REVIEW_IMPORTED


async def test_failed_journal_commit_never_publishes_objects(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._store.local import LocalStore

    original = LocalStore.create

    def fail(self: LocalStore, key: str, data: bytes) -> str:
        if key.startswith("journal/"):
            raise Fenced()
        return original(self, key, data)

    monkeypatch.setattr(LocalStore, "create", fail)
    with pytest.raises(Fenced):
        await HumanReview("alice").import_csv(review_run.run_id, _file(tmp_path), config=run_config)
    assert not _reviews(run_config)
    assert not any(e.type == EventType.REVIEW_IMPORTED for e in _events(review_run.store))


def test_render_filters_replaced_part_identity(review_run: FakeRun) -> None:
    from swarm_arena import Goal, PartSpec, Plan

    review_run.commit(
        review_run.plan(revision=1, parts=["p"], first_round=1),
        review_run.candidate("c-old", part="p"),
        review_run.score("s-old", "c-old", 0),
    )
    old = RunState.from_events(_events(review_run.store)).plans["g1"][-1][0]
    replacement = PartSpec(name="p", goal=Goal(id="g1-p", namespace="toy", inputs={"new": True}))
    plan: Plan = old.model_copy(update={"revision": 2, "parts": [replacement]})
    review_run.commit(
        EventDraft(
            type=EventType.PLAN_REVISION,
            goal_id="g1",
            payload={"plan": plan.model_dump(mode="json"), "origin": "human", "first_round": 2},
        ),
        review_run.candidate("c-new", part="p", round=2),
        review_run.score("s-new", "c-new", 2),
    )
    files = render_review(RunState.from_events(_events(review_run.store)))
    assert "candidates/c-old.json" not in files
    assert "candidates/c-new.json" in files


def test_render_top_is_per_goal(review_run: FakeRun) -> None:
    _add_namespace(review_run)
    files = render_review(RunState.from_events(_events(review_run.store)), top=1)
    rows = list(csv.DictReader(io.StringIO(files["review.csv"].decode())))
    assert {r["goal"] for r in rows} == {"g1", "g2"}
    assert len(rows) == 2


async def test_review_export_create_once(state: RunState, review_run: FakeRun) -> None:
    from swarm_arena._engine.run import write_review_export

    head = review_run.heads[-1]
    await write_review_export(state, review_run.store, head)
    await write_review_export(state, review_run.store, head)
    key = f"review/e{head.epoch}-{head.seq}/review.csv"
    review_run.store.replace(key, b"SECRET_EXPORT_BYTES", if_match=review_run.store.read(key).etag)
    with pytest.raises(StorageError, match="bytes differ") as error:
        await write_review_export(state, review_run.store, head)
    assert_secret_absent(error.value, "SECRET_EXPORT_BYTES")


@pytest.mark.parametrize("operation", ["resume", "import"])
@pytest.mark.parametrize("interrupt", ["cancel", "fence"])
@pytest.mark.parametrize("site", ["authority", "create"])
async def test_stalled_receipt_io_ends_owner_before_worker_returns(
    review_run: FakeRun,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    interrupt: str,
    site: str,
) -> None:
    import asyncio
    import threading

    import anyio

    from swarm_arena import Swarm
    from swarm_arena._exec.threads import run_sync_daemon
    from swarm_arena._model.canonical import utc_now
    from swarm_arena._store.local import LocalStore

    _add_namespace(review_run)
    config = run_config.model_copy(update={"settle_timeout_s": 0.2, "lease_ttl_s": 0.3})
    path = _file(tmp_path)

    def fail(*args: Any, **kwargs: Any) -> str:
        raise _core_error(StorageError, "interrupted publication")

    with monkeypatch.context() as patch:
        patch.setattr(human_review, "write_human_objects", fail)
        with pytest.raises(StorageError, match="interrupted publication"):
            await HumanReview("alice").import_csv(review_run.run_id, path, config=config)
    receipt = _events(review_run.store)[-1]
    keys = receipt.payload["objects"]
    assert len(keys) == 2
    entered, release, returned, caller_done = (threading.Event() for _ in range(4))
    workers: list[threading.Thread] = []
    renewals_after_return: list[int] = []
    creates: list[str] = []
    scopes: list[anyio.CancelScope] = []
    failures: list[BaseException] = []
    repairing = False
    real_repair = human_review._repair_review_objects
    real_read, real_create, real_renew = human_review.read_owner, LocalStore.create, Lease.renew

    async def repair(*args: Any, **kwargs: Any) -> None:
        nonlocal repairing
        repairing = True
        await real_repair(*args, **kwargs)

    def block() -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(10), "worker release deadline"
        returned.set()

    def authority(*args: Any, **kwargs: Any) -> Any:
        result = real_read(*args, **kwargs)
        if repairing and site == "authority":
            block()
        return result

    def create(self: LocalStore, key: str, data: bytes) -> str:
        if key in keys:
            creates.append(key)
            if site == "create" and len(creates) == 1:
                block()
        return real_create(self, key, data)

    def renew(self: Lease) -> None:
        if caller_done.is_set():
            renewals_after_return.append(self.doc.epoch)
        real_renew(self)

    async def caller() -> None:
        try:
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                if operation == "resume":
                    await Swarm(config).resume(review_run.run_id)
                else:
                    await HumanReview("bob").import_csv(review_run.run_id, path, config=config)
        except BaseException as error:
            failures.append(error)
        finally:
            caller_done.set()

    monkeypatch.setattr(human_review, "_repair_review_objects", repair)
    monkeypatch.setattr(human_review, "read_owner", authority)
    monkeypatch.setattr(LocalStore, "create", create)
    monkeypatch.setattr(Lease, "renew", renew)
    task = asyncio.create_task(caller())
    services: list[asyncio.Task[Any]] = []
    successor: Lease | None = None
    try:
        assert await run_sync_daemon(entered.wait, 5)
        services = [t for t in asyncio.all_tasks() if t.get_name() == f"resume-{review_run.run_id}"]
        started = anyio.current_time()
        if interrupt == "cancel":
            scopes[0].cancel()
        else:
            successor = Lease.take_over(
                review_run.store, owner_id="successor", host="local", ttl_s=60, force=True
            )
        done, _ = await asyncio.wait([task], timeout=3 * config.settle_timeout_s)
        assert done, "caller retained by stalled receipt I/O"
        assert anyio.current_time() - started < 3 * config.settle_timeout_s
        owner = read_owner(review_run.store)
        assert owner is not None
        if interrupt == "cancel":
            assert owner[0].expires_at <= utc_now(), "cancelled owner still holds its lease"
            assert not failures
        else:
            assert owner[0].owner_id == "successor"
            assert len(failures) == 1
            assert isinstance(failures[0], Fenced)
        assert all(t.done() for t in services), "resume service outlived its caller"
        assert not returned.is_set()
        release.set()
        assert await run_sync_daemon(returned.wait, 5)
        for worker in workers:
            await run_sync_daemon(worker.join, 5)
            assert not worker.is_alive()
        await asyncio.gather(task, *services)
        assert not renewals_after_return
        assert read_owner(review_run.store) == owner
        assert creates == ([] if site == "authority" else [keys[0]])
        assert not any(e.payload.get("reviewer") == "bob" for e in _events(review_run.store))
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(task, *services, return_exceptions=True), 5)
        for worker in workers:
            await run_sync_daemon(worker.join, 5)
            assert not worker.is_alive()
        if successor is not None:
            successor.release()


@pytest.mark.parametrize("operation", ["resume", "import"])
@pytest.mark.parametrize("native", [True, False])
async def test_cancelled_takeover_releases_undelivered_lease(
    review_run: FakeRun,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    native: bool,
) -> None:
    import asyncio
    import threading

    import anyio

    from swarm_arena._engine.resume import resume_run
    from swarm_arena._exec.threads import run_sync_daemon
    from swarm_arena._model.canonical import utc_now

    acquired, release, returned, cleaned = (threading.Event() for _ in range(4))
    original = Lease.take_over.__func__
    original_release = Lease.release
    before = _events(review_run.store)
    scopes: list[anyio.CancelScope] = []
    leases: list[Lease] = []
    path = _file(tmp_path)

    def take_over(cls: type[Lease], *args: Any, **kwargs: Any) -> Lease:
        lease = original(cls, *args, **kwargs)
        leases.append(lease)
        acquired.set()
        assert release.wait(5), "takeover worker release deadline"
        returned.set()
        return lease

    def release_lease(self: Lease) -> None:
        original_release(self)
        if self in leases:
            cleaned.set()

    async def caller() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            if operation == "resume":
                runner = await resume_run(
                    run_config, review_run.run_id, force=False, replanner=None
                )
                await runner.aclose()
            else:
                await HumanReview("alice").import_csv(review_run.run_id, path, config=run_config)

    monkeypatch.setattr(Lease, "take_over", classmethod(take_over))
    monkeypatch.setattr(Lease, "release", release_lease)
    task = asyncio.create_task(caller())
    try:
        assert await run_sync_daemon(acquired.wait, 5)
        if native:
            task.cancel()
            done, _ = await asyncio.wait([task], timeout=0.5)
            assert done, "native cancellation waits for takeover delivery"
        else:
            scopes[0].cancel()
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
        assert await run_sync_daemon(returned.wait, 5)
        assert await run_sync_daemon(cleaned.wait, 0.5), "undelivered lease was not released"
        owner = read_owner(review_run.store)
        assert owner is not None
        assert owner[0].expires_at <= utc_now()
        assert _events(review_run.store) == before
        monkeypatch.setattr(Lease, "take_over", classmethod(original))
        # An ordinary later import succeeds without forcing or waiting for expiry.
        await HumanReview("bob").import_csv(review_run.run_id, path, config=run_config)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
        assert await run_sync_daemon(returned.wait, 5)
        for lease in leases:
            with anyio.CancelScope(shield=True):
                with suppress(Fenced):
                    await run_sync_daemon(lease.release)


@pytest.mark.parametrize("operation", ["import", "export"])
@pytest.mark.parametrize("site", ["create", "read"])
@pytest.mark.parametrize("kind", ["raw", "registered", "fenced"])
async def test_review_storage_errors_rebuild_safe_diagnostics(
    review_run: FakeRun,
    run_config: RunConfig,
    state: RunState,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    site: str,
    kind: str,
) -> None:
    from uuid import uuid4

    from swarm_arena._engine.run import write_review_export
    from swarm_arena._store.local import LocalStore

    path = _file(tmp_path)
    head = review_run.heads[-1]
    if site == "read":
        if operation == "import":
            await HumanReview("alice").import_csv(review_run.run_id, path, config=run_config)
        else:
            await write_review_export(state, review_run.store, head)
    secret = uuid4().hex
    expected_type = Fenced if kind == "fenced" else StorageError
    error = (
        StorageError(secret)
        if kind == "raw"
        else _core_error(expected_type, "safe review storage failure")
    )
    error.args = (secret,)
    error.add_note(secret)
    error.__cause__ = ValueError(secret)
    original = getattr(LocalStore, site)

    def fail(self: LocalStore, key: str, *args: Any, **kwargs: Any) -> Any:
        if "/reviews/" in key or key.startswith("review/"):
            raise error
        return original(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, site, fail)
    call = (
        HumanReview("bob").import_csv(review_run.run_id, path, config=run_config)
        if operation == "import"
        else write_review_export(state, review_run.store, head)
    )
    with pytest.raises(expected_type) as caught:
        await call
    assert caught.value is not error
    assert_secret_absent(caught.value, secret, sanitized=True)
    assert str(caught.value) == (
        "review storage failed: StorageError" if kind == "raw" else "safe review storage failure"
    )


@pytest.mark.parametrize("length", [65, 256])
async def test_unsupported_reviewer_leaves_owner_and_recovery_unchanged(
    review_run: FakeRun, run_config: RunConfig, tmp_path: Path, length: int
) -> None:
    from swarm_arena._engine.resume import resume_run

    before = read_owner(review_run.store)
    path = _file(tmp_path)
    with pytest.raises(ValueError, match="reviewer"):
        await HumanReview("a" * length).import_csv(review_run.run_id, path, config=run_config)
    assert read_owner(review_run.store) == before
    assert not _reviews(run_config)
    await HumanReview("b" * 64).import_csv(review_run.run_id, path, config=run_config)
    runner = await resume_run(run_config, review_run.run_id, force=False, replanner=None)
    await runner.aclose()


@pytest.mark.parametrize("location", ["local_component", "local_path", "s3_key"])
async def test_invalid_review_location_fails_before_probe_or_takeover(
    review_run: FakeRun,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    from swarm_arena._engine.resume import resume_run

    root = {
        "local_component": str(tmp_path / ("a" * 256)),
        "local_path": str(tmp_path / Path(*["a" * 100] * 45)),
        "s3_key": "s3://bucket/" + "é" * 512,
    }[location]
    config = run_config.model_copy(update={"arena_root": root})
    before = read_owner(review_run.store)
    path = _file(tmp_path)
    probes: list[int] = []

    def probe(config: RunConfig) -> None:
        probes.append(1)
        raise AssertionError("invalid location reached storage probe")

    with monkeypatch.context() as patch:
        patch.setattr(human_review, "check_storage", probe)
        with pytest.raises(ValueError, match=r"review.*location"):
            await HumanReview("alice").import_csv(review_run.run_id, path, config=config)
    assert not probes
    assert read_owner(review_run.store) == before
    assert not _reviews(run_config)
    await HumanReview("bob").import_csv(review_run.run_id, path, config=run_config)
    runner = await resume_run(run_config, review_run.run_id, force=False, replanner=None)
    await runner.aclose()


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t", "\r"])
def test_review_csv_neutralizes_text_and_preserves_canonical_json(
    state: RunState, prefix: str
) -> None:
    from swarm_arena._model.canonical import canonical_json

    text = prefix + 'HYPERLINK("https://example.invalid/","click")'
    candidate = state.candidates["c-1"].model_copy(update={"team": text, "agent": text})
    state.candidates[candidate.id] = candidate
    original = state.scores["s-1"]
    score = original.model_copy(update={"id": "s-0", "judge": text})
    state.scores[score.id] = score
    state.positions[score.id] = (state.positions[original.id][0], 0)
    files = render_review(state)
    row = next(
        row
        for row in csv.DictReader(io.StringIO(files["review.csv"].decode(), newline=""))
        if row["id"] == candidate.id
    )
    assert row["team"] == "'" + text
    assert row["agent"] == "'" + text
    assert row["scores"] == "'" + text + "=4; j=4"
    assert files[f"candidates/{candidate.id}.json"] == canonical_json(
        {
            "candidate": candidate.model_dump(mode="json"),
            "scores": [score.model_dump(mode="json"), original.model_dump(mode="json")],
        }
    )
    row.update(goal="display only", verdict="Accept", score="7")
    parsed = parse_review_csv(_csv(row), state, judge="human:alice")
    assert len(parsed) == 1
    assert parsed[0].candidate_id == candidate.id
    assert parsed[0].key == (1.0, 7.0)


def test_judge_name_is_read_only_and_parser_requires_human_judge(state: RunState) -> None:
    review = HumanReview("alice")
    with pytest.raises(AttributeError):
        setattr(review, "judge_name", "program:forged")  # noqa: B010 - exercise read-only property
    assert review.judge_name == "human:alice"
    with pytest.raises(ValueError, match="judge"):
        parse_review_csv(_csv({"id": "c-1", "score": "7"}), state, judge="program:forged")


@pytest.mark.parametrize("kind", ["judge", "score", "duplicate_id"])
async def test_complete_proposed_receipt_is_validated_before_admission(
    review_run: FakeRun,
    run_config: RunConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    original = human_review.parse_review_csv
    parses = 0
    before = _events(review_run.store)
    path = _file(tmp_path, _csv({"id": "c-1", "score": "7"}, {"id": "c-2", "score": "8"}))

    def parse(data: bytes, state: RunState, *, judge: str) -> list[human_review.Score]:
        nonlocal parses
        parses += 1
        scores = original(data, state, judge=judge)
        if parses == 2:
            change = {"judge": "program:forged"} if kind == "judge" else {"value": float("nan")}
            if kind != "duplicate_id":
                scores[0] = scores[0].model_copy(update=change)
        return scores

    with monkeypatch.context() as patch:
        patch.setattr(human_review, "parse_review_csv", parse)
        if kind == "duplicate_id":
            patch.setattr(human_review.IdGen, "next", lambda self, prefix: "s-duplicate")
        with pytest.raises(StorageError, match="invalid review receipt"):
            await HumanReview("alice").import_csv(review_run.run_id, path, config=run_config)
    assert _events(review_run.store) == before
    assert not _reviews(run_config)
    owner = read_owner(review_run.store)
    assert owner is not None
    assert owner[0].expires_at <= human_review.utc_now()
    await HumanReview("bob").import_csv(review_run.run_id, path, config=run_config)
