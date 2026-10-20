import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _fakes.tools import cand

from swarm_arena import Goal, Plan, Ref
from swarm_arena._budget.amount import Amount
from swarm_arena._engine import control as control_mod
from swarm_arena._engine.control import (
    GoalStatus,
    StatusReport,
    format_status,
    read_status,
    request_stop,
    stop_key,
)
from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.lease import Head, Lease, OwnerDoc, read_owner
from swarm_arena._journal.segment import encode_segment
from swarm_arena._model.results import Score
from swarm_arena._store.base import Store, open_store


def test_stop_key() -> None:  # tests/_engine/test_control.py
    assert (stop_key(2, now=False), stop_key(2, now=True)) == (
        "control/stop-e2",
        "control/stop-now-e2",
    )


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _live_run(run_root: Path, run_id: str, *, epoch: int = 1) -> Store:
    store = open_store(str(run_root)).child(run_id)
    Lease.acquire_new(store, owner_id="o1", host="h", ttl_s=60)
    for _ in range(1, epoch):
        Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
    return store


def _released_run(run_root: Path, run_id: str) -> None:
    store = _live_run(run_root, run_id)
    owner = read_owner(store)
    assert owner is not None
    Lease(store, *owner, ttl_s=60).release()


def _event(seq: int, type: str, payload: dict[str, Any], *, run_id: str = "r1") -> Event:
    return Event(
        seq=seq,
        epoch=1,
        ts=datetime(2026, 10, 20, tzinfo=UTC),
        run_id=run_id,
        context_id=run_id,
        goal_id="g",
        type=EventType(type),
        payload=payload,
    )


def _commit(store: Store, events: list[Event]) -> None:
    owner = read_owner(store)
    assert owner is not None
    key = f"journal/e1-{len(events)}.jsonl"
    store.create(key, encode_segment(None, events))
    Lease(store, *owner, ttl_s=60).advance(Head(epoch=1, seq=len(events), segment=key))


def _events() -> list[Event]:
    plan = Plan(
        goal=Goal(id="g", namespace="n"), teams=[], judges=[Ref(path="must_not_import:Judge")]
    )
    candidate = cand(id="c1", run_id="r1")
    return [
        _event(1, "plan_revision", {"plan": plan.model_dump(mode="json"), "origin": "human"}),
        _event(2, "goal_started", {}),
        _event(3, "round_started", {"round": 1}),
        _event(4, "tool_end", {"candidate": candidate.model_dump(mode="json")}),
        _event(
            5,
            "score",
            {
                "score": Score(
                    id="s1", candidate_id="c1", judge="stamped", key=(-2.0,), value=2
                ).model_dump(mode="json")
            },
        ),
    ]


def _run_with_uncommitted_tail(run_root: Path) -> str:
    store = _live_run(run_root, "r1")
    _commit(store, _events())
    store.create(
        "journal/e1-6.jsonl",
        encode_segment(
            "journal/e1-5.jsonl",
            [
                _event(
                    6, "tool_end", {"candidate": cand(id="c2", run_id="r1").model_dump(mode="json")}
                )
            ],
        ),
    )
    store.create("control/stop-now-e1", b"")
    store.create("manifest.json", b"stale or malformed view")
    return "r1"


async def test_request_stop_needs_live_owner(run_root: Path) -> None:
    _released_run(run_root, "r1")
    with pytest.raises(RuntimeError, match="no live owner"):
        await request_stop(str(run_root), "r1", now=False)


async def test_request_stop_twice_is_ok_and_returns_epoch(run_root: Path) -> None:
    _live_run(run_root, "r1", epoch=1)
    assert await request_stop(str(run_root), "r1", now=False) == 1
    assert await request_stop(str(run_root), "r1", now=False) == 1
    assert await request_stop(str(run_root), "r1", now=True) == 1
    store = open_store(str(run_root)).child("r1")
    assert store.exists("control/stop-e1")
    assert store.exists("control/stop-now-e1")


async def test_request_stop_fails_when_epoch_changes_meanwhile(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _live_run(run_root, "r1", epoch=1)
    real = control_mod.read_owner
    calls: list[int] = []

    def read_then_take_over(s: Store) -> tuple[OwnerDoc, str] | None:
        calls.append(1)
        if len(calls) == 2:
            Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
        return real(s)

    monkeypatch.setattr(control_mod, "read_owner", read_then_take_over)
    with pytest.raises(RuntimeError, match="epoch changed from 1 to 2"):
        await request_stop(str(run_root), "r1", now=False)
    assert store.exists("control/stop-e1")
    assert not store.exists("control/stop-e2")


async def test_status_reads_committed_journal_only(run_root: Path) -> None:
    rep = await read_status(str(run_root), _run_with_uncommitted_tail(run_root))
    assert rep.head_seq == 5
    assert rep.goals["g"].candidates == 1
    assert (
        format_status(rep).splitlines()[0]
        == f"run {rep.run_id}  epoch 1  owner live  head 5  stop none"
    )
    assert rep.goals["g"].best == 2.0
    assert format_status(rep).splitlines()[1:] == [
        "goal g  running  round 1/3  candidates 1  best 2.0",
        "spent 0 tokens, $0.00  lost 0",
    ]


async def test_status_no_owner_and_empty_expired_owner(run_root: Path) -> None:
    with pytest.raises(RuntimeError, match="no run"):
        await read_status(str(run_root), "missing")
    with pytest.raises(RuntimeError, match="no live owner"):
        await request_stop(str(run_root), "missing", now=False)
    _released_run(run_root, "r1")
    rep = await read_status(str(run_root), "r1")
    assert rep.epoch == 1
    assert rep.head_seq == 0
    assert not rep.owner_live
    assert rep.goals == {}
    assert "owner expired" in format_status(rep)


async def test_status_preserves_one_identity_fidelity_and_tuple_value(run_root: Path) -> None:
    store = _live_run(run_root, "r1")
    events = _events()
    for id, part, round, fidelity, judge, rubric, key, value in [
        ("old", None, 1, "old", "stamped", "", (999.0,), 999),
        ("new", None, 2, "new", "stamped", "new", (4.0, 2.0), 7),
        ("part", "p", 2, "new", "part-judge", "", (1000.0,), 1000),
    ]:
        candidate = cand(id=id, run_id="r1", part=part, round=round, fidelity=fidelity)
        events.append(
            _event(len(events) + 1, "tool_end", {"candidate": candidate.model_dump(mode="json")})
        )
        score = Score(id=id, candidate_id=id, judge=judge, rubric_hash=rubric, key=key, value=value)
        events.append(_event(len(events) + 1, "score", {"score": score.model_dump(mode="json")}))
    _commit(store, events)
    assert (await read_status(str(run_root), "r1")).goals["g"].best == 7


@pytest.mark.parametrize(("part", "key", "value"), [(None, (1.0, 2.0), None), ("p", (10.0,), 10.0)])
async def test_status_non_scalar_or_parts_only_has_no_best(
    run_root: Path,
    part: str | None,
    key: tuple[float, ...],
    value: float | None,
) -> None:
    store = _live_run(run_root, "r1")
    events = _events()
    events[3] = _event(
        4, "tool_end", {"candidate": cand(run_id="r1", part=part).model_dump(mode="json")}
    )
    events[4] = _event(
        5,
        "score",
        {
            "score": Score(
                id="s1", candidate_id="c1", judge="stamped", key=key, value=value
            ).model_dump(mode="json")
        },
    )
    _commit(store, events)
    assert (await read_status(str(run_root), "r1")).goals["g"].best is None


async def test_status_spent_lost_and_stop_from_replay(run_root: Path) -> None:
    store = _live_run(run_root, "r1")
    events = _events()
    for id, state, keep, overflow in [
        ("open", "open", False, False),
        ("settled", "settled", False, False),
        ("estimated", "estimated", False, False),
        ("kept", "settled", True, False),
        ("overflow", "settled", True, True),
    ]:
        events.append(
            _event(
                len(events) + 1,
                "reserve",
                {
                    "reservation": {
                        "id": id,
                        "amount": {"tokens": 10, "usd": 1.0},
                        "keep_charged": keep,
                    }
                },
            )
        )
        if state != "open":
            events.append(
                _event(
                    len(events) + 1,
                    "settle",
                    {
                        "id": id,
                        "estimated": state == "estimated",
                        "actual": {"tokens": 2, "usd": 0.2},
                        "usage_overflow": overflow,
                    },
                )
            )
    events.append(_event(len(events) + 1, "run_lost", {"candidate_id": "lost-1"}))
    events.append(_event(len(events) + 1, "stop_requested", {"now": True}))
    _commit(store, events)
    rep = await read_status(str(run_root), "r1")
    assert rep.spent == Amount(tokens=34, usd=3.4)
    assert rep.lost == 1
    assert rep.stop == "stop-now"


async def test_storage_and_lease_work_off_event_loop(
    run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_run(run_root, "r1")
    loop_thread = threading.get_ident()
    real = control_mod.read_owner

    def read(s: Store) -> tuple[OwnerDoc, str] | None:
        assert threading.get_ident() != loop_thread
        return real(s)

    monkeypatch.setattr(control_mod, "read_owner", read)
    await request_stop(str(run_root), "r1", now=False)
    assert (await read_status(str(run_root), "r1")).epoch == 1


def test_status_models_frozen_and_format_missing_best() -> None:
    goal = GoalStatus(status="queued", round=0, rounds_total=3, candidates=0, best=None)
    rep = StatusReport(
        run_id="r",
        epoch=2,
        owner_live=False,
        head_seq=0,
        stop="stop",
        goals={"g": goal},
        spent=Amount(tokens=42, usd=1.256),
        lost=2,
    )
    assert format_status(rep).splitlines()[1:] == [
        "goal g  queued  round 0/3  candidates 0  best -",
        "spent 42 tokens, $1.26  lost 2",
    ]
    field_name = "round"
    with pytest.raises(ValueError, match="frozen"):
        setattr(goal, field_name, 1)
    field_name = "epoch"
    with pytest.raises(ValueError, match="frozen"):
        setattr(rep, field_name, 3)


async def test_status_epoch_comes_from_authority_and_old_stops_do_not_latch(run_root: Path) -> None:
    store = _live_run(run_root, "r1")
    _commit(store, _events())
    store.create("control/stop-e1", b"")
    Lease.take_over(store, owner_id="o2", host="h", ttl_s=60, force=True)
    rep = await read_status(str(run_root), "r1")
    assert rep.epoch == 2
    assert rep.head_seq == 5
    assert rep.stop == "none"
    assert await request_stop(str(run_root), "r1", now=False) == 2
    assert store.exists("control/stop-e2")


async def test_status_explicit_ranking_excludes_auxiliary_scores(run_root: Path) -> None:
    store = _live_run(run_root, "r1")
    events = _events()
    plan = Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[],
        judges=[
            Ref(path="must_not_import:Rank", name="stamped"),
            Ref(path="must_not_import:Aux", name="aux"),
        ],
        ranking="stamped",
    )
    events[0] = _event(
        1, "plan_revision", {"plan": plan.model_dump(mode="json"), "origin": "human"}
    )
    events.append(
        _event(
            6,
            "score",
            {
                "score": Score(
                    id="aux", candidate_id="c1", judge="aux", key=(999.0,), value=999
                ).model_dump(mode="json")
            },
        )
    )
    _commit(store, events)
    rep = await read_status(str(run_root), "r1")
    assert rep.goals["g"].best == 2


async def test_status_flattens_multiple_committed_segments(run_root: Path) -> None:
    store = _live_run(run_root, "r1")
    events = _events()
    owner = read_owner(store)
    assert owner is not None
    store.create("journal/e1-3.jsonl", encode_segment(None, events[:3]))
    store.create("journal/e1-5.jsonl", encode_segment("journal/e1-3.jsonl", events[3:]))
    Lease(store, *owner, ttl_s=60).advance(Head(epoch=1, seq=5, segment="journal/e1-5.jsonl"))
    rep = await read_status(str(run_root), "r1")
    assert rep.head_seq == 5
    assert rep.goals["g"].candidates == 1
    assert rep.goals["g"].best == 2


async def test_status_unjoined_parts_have_no_combined_main_best(run_root: Path) -> None:
    store = _live_run(run_root, "r1")
    events = _events()
    events.append(_event(6, "final_order", {"candidate_ids": [], "parts": {"p": None}}))
    _commit(store, events)
    assert (await read_status(str(run_root), "r1")).goals["g"].best is None
