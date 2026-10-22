from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Goal, Limits, Plan, RunConfig, Score
from swarm_arena._errors import ChainError
from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.replay import GoalState, ReservationState, RunState
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.planning_types import Decision
from swarm_arena._model.results import AgentStatus, Learning
from swarm_arena.executor import Job
from swarm_arena.testing import make_candidate
from swarm_arena.tool import RunStatus, ToolOutput


def _plan() -> Plan:
    return Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[])


def _ev(
    seq: int,
    type: str,
    payload: dict[str, Any],
    *,
    goal_id: str = "",
    epoch: int = 1,
    team: str = "",
    agent: str = "",
    task_id: str = "",
) -> Event:
    return Event(
        seq=seq,
        type=EventType(type),
        payload=payload,
        goal_id=goal_id,
        epoch=epoch,
        team=team,
        agent=agent,
        task_id=task_id,
        run_id="r",
        context_id="r",
        ts=datetime(2026, 10, 18, tzinfo=UTC),
    )


_PRE = [
    _ev(1, "run_config", {"config": RunConfig().model_dump(mode="json")}),
    _ev(
        2,
        "plan_revision",
        {"plan": _plan().model_dump(mode="json"), "origin": "human"},
        goal_id="g",
    ),
]


def _reserve(seq: int, id: str) -> Event:
    return _ev(
        seq,
        "reserve",
        {
            "reservation": {
                "id": id,
                "holder": "g/t/a",
                "kind": "tool",
                "amount": {"tokens": 0, "usd": 1.0},
                "goal_id": "g",
            }
        },
    )


def _settle(seq: int, id: str, *, estimated: bool = False) -> Event:
    return _ev(
        seq, "settle", {"id": id, "actual": {"tokens": 0, "usd": 0.5}, "estimated": estimated}
    )


def _start(seq: int, id: str) -> Event:
    return _ev(
        seq,
        "tool_start",
        {
            "candidate_id": id,
            "started": Job.Started(kind="process", job_id=id).model_dump(mode="json"),
            "reservation_id": "r-1",
        },
    )


def _end(seq: int, id: str) -> Event:
    candidate = make_candidate(
        ToolOutput(status=RunStatus.OK), id=id, run_id="r", goal_id="g", overlay={}
    )
    return _ev(
        seq, "tool_end", {"candidate": candidate.model_dump(mode="json"), "a2a": {}}, goal_id="g"
    )


@pytest.mark.parametrize("missing_plan", [False, True])
def test_candidate_requires_a_declared_producing_goal(missing_plan: bool) -> None:
    end = _end(3, "candidate")
    end.payload["candidate"]["part"] = "UNDECLARED_PART_SECRET"
    with pytest.raises(ValueError, match="invalid journal replay payload") as caught:
        RunState.from_events([*(_PRE if not missing_plan else []), end])
    assert_secret_absent(caught.value, "UNDECLARED_PART_SECRET")


def _cmd_start(seq: int, job_id: str) -> Event:
    return _ev(
        seq,
        "command_start",
        {
            "job_id": job_id,
            "started": Job.Started(kind="process", job_id=job_id).model_dump(mode="json"),
            "candidate_id": "c-1",
            "component": "check",
        },
    )


def _learning() -> Learning:
    return Learning(
        id="L-1",
        namespace="n",
        run_id="r",
        goal_id="g",
        round=1,
        author="g/t/a",
        title="Hint",
        text="Try this",
        evidence=["c-1"],
    )


def test_llm_call_events_fold_into_llm_calls() -> None:
    from swarm_arena._model.results import LLMCall

    call = LLMCall(id="llm-1", run_id="r", goal_id="g", role="manager", model="m", provider="p")
    event = _ev(3, "llm_call", {"call": call.model_dump(mode="json"), "request_sha256": "a" * 64})
    state = RunState.from_events([*_PRE, event])
    assert state.llm_calls[call.id] == call
    event.payload["call"]["tool_calls"].append({"id": "mutated", "name": "tool"})
    assert state.llm_calls[call.id].tool_calls == []


def test_llm_call_events_fold_request_sha256() -> None:
    from swarm_arena._model.results import LLMCall

    call = LLMCall(id="llm-1", run_id="r", goal_id="g", role="manager", model="m", provider="p")
    event = _ev(
        3,
        "llm_call",
        {
            "call": call.model_dump(mode="json"),
            "request_sha256": "a" * 64,
            "cache_identity": "b" * 64,
        },
    )
    state = RunState.from_events([*_PRE, event])
    assert state.llm_request_sha256 == {call.id: "a" * 64}
    assert state.llm_cache_identity == {call.id: "b" * 64}


@pytest.mark.parametrize("field", ["request_sha256", "cache_identity"])
def test_llm_call_hash_fields_reject_secret_payloads(field: str) -> None:
    from swarm_arena._model.results import LLMCall

    call = LLMCall(id="llm-1", run_id="r", goal_id="g", role="manager", model="m", provider="p")
    event = _ev(3, "llm_call", {"call": call.model_dump(mode="json"), field: "HASH_SECRET"})
    with pytest.raises(ValueError, match="invalid journal replay payload") as caught:
        RunState.from_events([*_PRE, event])
    assert_secret_absent(caught.value, "HASH_SECRET")


def test_origin_comes_from_journal_not_plan_body() -> None:
    plan = _plan().model_copy(update={"origin": "human"})
    state = RunState.from_events(
        [
            _PRE[0],
            _ev(
                2,
                "plan_revision",
                {"plan": plan.model_dump(mode="json"), "origin": "llm"},
                goal_id="g",
            ),
        ]
    )
    assert state.plans["g"][0][1] == "llm"


def test_reservation_without_settle_is_open_and_settled_is_settled() -> None:
    state = RunState.from_events([*_PRE, _reserve(3, "r-1"), _reserve(4, "r-2"), _settle(5, "r-2")])
    assert state.reservations["r-1"].state == "open"
    assert state.reservations["r-2"].state == "settled"
    assert state.reservations["r-1"].actual is None
    assert state.reservations["r-2"].actual == {"tokens": 0, "usd": 0.5}
    assert isinstance(state.reservations["r-1"], ReservationState)


def test_estimated_settlement() -> None:
    state = RunState.from_events([*_PRE, _reserve(3, "r-1"), _settle(4, "r-1", estimated=True)])
    assert state.reservations["r-1"].state == "estimated"


def test_started_without_end_is_in_flight() -> None:
    state = RunState.from_events([*_PRE, _start(3, "c-1"), _start(4, "c-2"), _end(5, "c-2")])
    assert set(state.started) == {"c-1"}
    assert "c-2" in state.candidates
    assert state.positions["c-2"] == (1, 5)
    assert isinstance(state.started["c-1"], Job.Started)


def test_id_counters_are_max_per_prefix() -> None:
    state = RunState.from_events(
        [*_PRE, _end(3, "c-3f9a1c2b5d6e7f80-e1-0007"), _reserve(4, "r-3f9a1c2b5d6e7f80-e1-0002")]
    )
    assert state.id_counters == {"c": 7, "r": 2}


def test_all_id_prefixes_fullmatch_across_tags_and_epochs() -> None:
    ids = [f"{p}-0123456789abcdef-e2-10000" for p in ("c", "L", "s", "r", "llm", "x", "k")]
    ids += [
        "c-fedcba9876543210-e1-0012",
        "L-h-0123456789abcdef",
        "c-0123456789abcdef-e2-999",
        "c-0123456789abcdef-e2-99999-suffix",
        "prefix-c-0123456789abcdef-e2-99999",
        "c-0123456789abcde-e2-99999",
        "z-0123456789abcdef-e2-99999",
    ]
    events = [_ev(i + 3, "judge_error", {}, task_id=id) for i, id in enumerate(ids)]
    state = RunState.from_events([*_PRE, *events])
    assert state.id_counters == dict.fromkeys(("c", "L", "s", "r", "llm", "x", "k"), 10000)


def test_goal_status_from_goal_finished_reason() -> None:
    def status(*evs: Event) -> str:
        return (
            RunState.from_events([*_PRE, _ev(3, "goal_started", {}, goal_id="g"), *evs])
            .goals["g"]
            .status
        )

    assert status(_ev(4, "goal_finished", {"reason": "stop"}, goal_id="g")) == "stopped"
    assert status(_ev(4, "goal_finished", {"reason": None}, goal_id="g")) == "finished"
    assert (
        status(
            _ev(4, "no_result", {"reason": "x"}, goal_id="g"),
            _ev(5, "goal_finished", {"reason": None}, goal_id="g"),
        )
        == "no_result"
    )
    for reason in ("stop --now", "run_timeout_s", "run budget", "goal budget", "reserve_exhausted"):
        assert status(_ev(4, "goal_finished", {"reason": reason}, goal_id="g")) == "stopped"
    assert status(_ev(4, "goal_finished", {"reason": "finished"}, goal_id="g")) == "finished"


def test_round_started_after_final_order_resets_converge_state() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _ev(3, "goal_started", {}, goal_id="g"),
            _ev(4, "converge_started", {"select_only": True}, goal_id="g"),
            _ev(5, "final_order", {"candidate_ids": ["c-1"], "parts": {"p": "c-1"}}, goal_id="g"),
            _ev(6, "round_started", {"round": 2, "run_elapsed_s": 1.0}, goal_id="g"),
        ]
    )
    g = state.goals["g"]
    assert (g.converge_started, g.select_only, g.final_order, g.final_parts) == (
        False,
        False,
        [],
        None,
    )


def test_converge_round_preserves_main_and_final_state() -> None:
    main, converge = Frontier(reviews=["main"]), Frontier(reviews=["converge"])
    events = [
        *_PRE,
        _ev(3, "round_started", {"round": 1, "run_elapsed_s": 0.0}, goal_id="g"),
        _ev(4, "frontier", {"round": 1, "frontier": main.model_dump(mode="json")}, goal_id="g"),
        _ev(5, "round_closed", {"round": 1, "run_elapsed_s": 1.0}, goal_id="g"),
        _ev(6, "converge_started", {"select_only": True}, goal_id="g"),
        _ev(7, "final_order", {"candidate_ids": ["c-1"], "parts": {"p": "c-1"}}, goal_id="g"),
        _ev(8, "frontier", {"round": 2, "frontier": converge.model_dump(mode="json")}, goal_id="g"),
        _ev(
            9,
            "round_started",
            {"round": 2, "run_elapsed_s": 2.0, "converge": True, "shares": {}},
            goal_id="g",
        ),
    ]
    state = RunState.from_events(events)
    g = state.goals["g"]
    assert (g.round, g.rounds_closed, g.converge_round) == (1, 1, 2)
    assert (g.converge_started, g.select_only, g.final_order, g.final_parts) == (
        True,
        True,
        ["c-1"],
        {"p": "c-1"},
    )
    assert g.frontiers == {1: main, 2: converge}
    assert state.ledger_events[-1] == events[-1]


def test_goal_started_after_goal_finished_reopens() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _ev(3, "goal_started", {}, goal_id="g"),
            _ev(4, "converge_started", {"select_only": False}, goal_id="g"),
            _ev(5, "final_order", {"candidate_ids": ["c-1"], "parts": None}, goal_id="g"),
            _ev(6, "goal_finished", {"reason": "stop"}, goal_id="g"),
            _ev(7, "goal_started", {"pool": {"tokens": 0, "usd": 15.5}}, goal_id="g", epoch=2),
        ]
    )
    g = state.goals["g"]
    assert (g.status, g.end_reason, g.converge_started, g.final_order) == (
        "running",
        None,
        False,
        [],
    )


def test_limits_changed_folds_into_config_and_ledger_events() -> None:
    old, new = (
        Limits(max_usd=100.0).model_dump(mode="json"),
        Limits(max_usd=200.0).model_dump(mode="json"),
    )
    state = RunState.from_events([*_PRE, _ev(3, "limits_changed", {"old": old, "new": new})])
    assert state.config.limits.max_usd == 200.0
    assert [e.type for e in state.ledger_events] == ["run_config", "limits_changed"]


def test_agent_counters_folded() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _ev(3, "round_started", {"round": 1, "run_elapsed_s": 0.0}, goal_id="g"),
            _ev(
                4,
                "agent_counters",
                {"turns": 1, "board_reads": 0},
                goal_id="g",
                team="t",
                agent="a",
            ),
            _ev(
                5,
                "agent_counters",
                {"turns": 3, "board_reads": 2},
                goal_id="g",
                team="t",
                agent="a",
            ),
        ]
    )
    assert state.goals["g"].counters == {"t/a": {"turns": 3, "board_reads": 2}}


@pytest.mark.parametrize("transition", ["round_started", "goal_started"])
def test_counters_reset_on_new_round_or_lifecycle(transition: str) -> None:
    events = [
        *_PRE,
        _ev(3, "agent_counters", {"turns": 3, "board_reads": 2}, goal_id="g", team="t", agent="a"),
    ]
    if transition == "goal_started":
        events.append(_ev(4, "goal_finished", {"reason": "stop"}, goal_id="g"))
    events.append(_ev(5, transition, {"round": 2, "run_elapsed_s": 1.0}, goal_id="g"))
    assert RunState.from_events(events).goals["g"].counters == {}


def test_open_command_starts_folded() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _cmd_start(3, "x-1"),
            _cmd_start(4, "x-2"),
            _ev(5, "command_end", {"job_id": "x-2", "status": "ok"}),
        ]
    )
    assert set(state.started_commands) == {"x-1"}
    assert isinstance(state.started_commands["x-1"], Job.Started)


def test_ledger_events_keep_budget_events_in_seq_order() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _reserve(3, "r-1"),
            _ev(4, "learning", {"learning": _learning().model_dump(mode="json")}),
            _settle(5, "r-1"),
        ]
    )
    assert [e.seq for e in state.ledger_events if e.type in {"reserve", "settle"}] == [3, 5]
    assert all(e.type != "learning" for e in state.ledger_events)


def test_takeover_resets_stop_and_sets_epoch() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _ev(3, "stop_requested", {"now": False}),
            _ev(4, "takeover", {"force": True}, epoch=2),
        ]
    )
    assert state.stop == "none"
    assert state.epoch == 2


@pytest.mark.parametrize(("now", "expected"), [(False, "stop"), (True, "stop-now")])
def test_stop_requested(now: bool, expected: str) -> None:
    assert RunState.from_events([*_PRE, _ev(3, "stop_requested", {"now": now})]).stop == expected


def test_stop_now_cannot_be_downgraded_in_an_epoch() -> None:
    assert (
        RunState.from_events(
            [
                *_PRE,
                _ev(3, "stop_requested", {"now": True}),
                _ev(4, "stop_requested", {"now": False}),
            ]
        ).stop
        == "stop-now"
    )


def test_run_elapsed_is_max_recorded() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _ev(3, "round_closed", {"run_elapsed_s": 12.5}, goal_id="g"),
            _ev(4, "round_started", {"round": 2, "run_elapsed_s": 1.0}, goal_id="g"),
        ]
    )
    assert state.run_elapsed_s == 12.5


def _full_run_events() -> list[Event]:
    return [
        *_PRE,
        _ev(3, "goal_queued", {}, goal_id="g"),
        _ev(4, "goal_started", {}, goal_id="g"),
        _reserve(5, "r-1"),
        _start(6, "c-1"),
        _end(7, "c-1"),
        _settle(8, "r-1"),
        _ev(
            9,
            "score",
            {"score": Score(id="s-1", candidate_id="c-1", key=(1.0,)).model_dump(mode="json")},
        ),
        _ev(10, "learning", {"learning": _learning().model_dump(mode="json")}),
        _ev(
            11,
            "decision",
            {"round": 1, "decision": Decision(prune={"t/a": "done"}).model_dump(mode="json")},
            goal_id="g",
        ),
        _ev(
            12,
            "agent_status",
            {"status": "finished", "reason": None},
            goal_id="g",
            team="t",
            agent="a",
        ),
        _ev(13, "notes_saved", {"text": "next"}, goal_id="g", team="t", agent="a"),
        _ev(14, "goal_finished", {"reason": None}, goal_id="g"),
    ]


def test_from_events_is_pure() -> None:
    events = _full_run_events()
    before = deepcopy(events)
    state = RunState.from_events(events)
    assert state == RunState.from_events(list(events)) == RunState.from_events(reversed(events))
    assert events == before
    state.reservations["r-1"].reservation["amount"]["usd"] = 9
    state.ledger_events[0].payload["config"]["run_root"] = "changed"
    assert events == before


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("index", [1, 4, 10], ids=["plan_revision", "reserve", "decision"])
def test_duplicate_seq_is_rejected(index: int, *, reverse: bool) -> None:
    events = _full_run_events()
    duplicate = deepcopy(events[index])
    events.append(duplicate)
    before = deepcopy(events)
    with pytest.raises(ChainError, match=rf"^duplicate journal seq {duplicate.seq}$"):
        RunState.from_events(reversed(events) if reverse else iter(events))
    assert events == before


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_seq_conflicting_converge_is_rejected(*, reverse: bool) -> None:
    secret = "DUPLICATE_REPLAY_SECRET"
    first = _ev(3, "converge_started", {"select_only": True, "extra": secret}, goal_id="g")
    second = first.model_copy(update={"payload": {"select_only": False, "extra": secret}})
    events = [*_PRE, *([second, first] if reverse else [first, second])]
    with pytest.raises(ChainError, match=r"^duplicate journal seq 3$") as caught:
        RunState.from_events(events)
    assert_secret_absent(caught.value, secret)


def test_duplicate_seq_is_rejected_before_applying_any_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_apply(self: RunState, event: Event) -> None:
        pytest.fail("applied an event before validating sequence uniqueness")

    monkeypatch.setattr(RunState, "_apply", unexpected_apply)
    duplicate = _reserve(3, "r-1")
    with pytest.raises(ChainError, match=r"^duplicate journal seq 3$"):
        RunState.from_events([*_PRE, duplicate, duplicate])


def test_reviewer_replay_edge_cases() -> None:
    prefix = [
        _PRE[0],
        _ev(2, "goal_started", {}, goal_id="g"),
        _ev(3, "round_started", {"round": 1}, goal_id="g"),
        _ev(4, "round_closed", {"round": 1}, goal_id="g"),
        _ev(5, "converge_started", {"select_only": True}, goal_id="g"),
        _ev(6, "final_order", {"candidate_ids": ["c-1"], "parts": {"p": "c-1"}}, goal_id="g"),
        _ev(7, "round_started", {"round": 2, "converge": True, "shares": {}}, goal_id="g"),
    ]
    prefix = [
        event.model_copy(update={"task_id": "g/t/a#r1" if event.goal_id else "r"})
        for event in prefix
    ]
    state = RunState.from_events(prefix)
    goal = state.goals["g"]
    assert (goal.status, goal.round, goal.rounds_closed, goal.converge_round) == (
        "running",
        1,
        1,
        2,
    )
    assert (goal.converge_started, goal.select_only, goal.final_order, goal.final_parts) == (
        True,
        True,
        ["c-1"],
        {"p": "c-1"},
    )
    assert state == RunState.from_events(reversed(prefix))
    assert RunState.from_events([]) == RunState.from_events(iter([]))
    assert all(event.context_id == event.run_id and event.task_id for event in prefix)
    with pytest.raises(ValueError, match="EventType"):
        _ev(8, "unknown_future_type", {})
    unknown = prefix[-1].model_copy(update={"seq": 8, "type": "unknown_future_type", "payload": {}})
    assert RunState.from_events([*prefix, unknown]) == state


def test_folded_records_and_positions() -> None:
    state = RunState.from_events(_full_run_events())
    assert state.run_id == "r"
    assert state.scores["s-1"].candidate_id == "c-1"
    assert state.learnings["L-1"] == _learning()
    assert state.positions == {"c-1": (1, 7), "s-1": (1, 9), "L-1": (1, 10)}
    assert state.goals["g"].decisions == [(1, Decision(prune={"t/a": "done"}))]
    assert state.goals["g"].agents == {"t/a": AgentStatus.FINISHED}
    assert state.goals["g"].notes == {"t/a": "next"}
    assert isinstance(state.goals["g"], FrozenModel)
    assert state.llm_request_sha256 == {}


def test_lost_start_is_removed_and_reservation_remains_open() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _reserve(3, "r-1"),
            _start(4, "c-1"),
            _ev(5, "run_lost", {"candidate_id": "c-1", "reservation_id": "r-1"}),
        ]
    )
    assert state.lost == {"c-1"}
    assert state.started == {}
    assert state.reservations["r-1"].state == "open"


def test_plan_base_dir_reused_without_importing_components(tmp_path: Path) -> None:
    plan = _plan().model_dump(mode="json")
    plan["judges"] = [
        {"path": "module_that_does_not_exist:Judge", "config": {"nested": {"value": 1}}}
    ]
    first = _ev(
        2,
        "plan_revision",
        {"plan": plan, "origin": "human", "base_dir": str(tmp_path), "planner_call_ids": []},
        goal_id="g",
    )
    second = _ev(3, "plan_revision", {"plan": plan, "origin": "llm"}, goal_id="g")
    state = RunState.from_events([_PRE[0], first, second])
    assert all(p.base_dir == tmp_path for p, _ in state.plans["g"])
    assert state.plans["g"][0][0] == Plan._from_data(plan, base_dir=tmp_path)
    assert state == RunState.from_events([_PRE[0], first, second])


def test_latest_converge_and_final_order_win() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _ev(3, "converge_started", {"select_only": True}, goal_id="g"),
            _ev(4, "converge_started", {"select_only": False}, goal_id="g"),
            _ev(5, "final_order", {"candidate_ids": ["c-1"], "parts": {"p": "c-1"}}, goal_id="g"),
            _ev(6, "final_order", {"candidate_ids": ["c-2"], "parts": None}, goal_id="g"),
        ]
    )
    g = state.goals["g"]
    assert (g.converge_started, g.select_only, g.final_order, g.final_parts) == (
        True,
        False,
        ["c-2"],
        None,
    )
    assert GoalState().repeat_keys == {}
    assert GoalState().join_attempts == 0


@pytest.mark.parametrize(
    ("type", "payload"),
    [
        ("agent_status", {"status": "REPLAY_SECRET", "reason": None}),
        ("plan_revision", {"plan": _plan().model_dump(mode="json"), "origin": "REPLAY_SECRET"}),
        ("limits_changed", {"new": {"max_usd": "REPLAY_SECRET"}}),
    ],
)
def test_invalid_replay_payloads_hide_values(type: str, payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="invalid journal replay payload") as caught:
        RunState.from_events([*_PRE, _ev(3, type, payload, goal_id="g", team="t", agent="a")])
    assert_secret_absent(caught.value, "REPLAY_SECRET")


def test_continuation_resets_terminal_state_and_keeps_history() -> None:
    events = [
        *_PRE,
        _ev(3, "round_started", {"round": 1, "run_elapsed_s": 0.0}, goal_id="g"),
        _ev(4, "round_closed", {"round": 1, "run_elapsed_s": 1.0}, goal_id="g"),
        _ev(5, "converge_started", {"select_only": True}, goal_id="g"),
        _ev(6, "round_started", {"round": 2, "converge": True}, goal_id="g"),
        _ev(7, "agent_counters", {"turns": 3, "board_reads": 2}, goal_id="g", team="t", agent="a"),
        _ev(8, "final_order", {"candidate_ids": ["c-1"], "parts": {"p": None}}, goal_id="g"),
        _ev(9, "no_result", {"reason": "empty"}, goal_id="g"),
        _ev(10, "goal_finished", {"reason": "stop"}, goal_id="g"),
        _ev(11, "goal_started", {"pool": {"tokens": 0, "usd": 15.5}}, goal_id="g", epoch=2),
    ]
    g = RunState.from_events(events).goals["g"]
    assert (g.status, g.end_reason, g.round, g.rounds_closed) == ("running", None, 1, 1)
    assert (g.converge_started, g.converge_round, g.select_only) == (False, None, False)
    assert (g.final_order, g.final_parts, g.counters) == ([], None, {})
    assert events[7].payload["candidate_ids"] == ["c-1"]
    assert events[9].payload["reason"] == "stop"


def test_full_ledger_history_keeps_future_keys_and_sequence_order() -> None:
    events = [
        *_PRE,
        _ev(3, "run_started", {"final_slices": {"g": {"tokens": 0, "usd": 1.0}}}),
        _ev(4, "goal_started", {"pool": {"tokens": 0, "usd": 1.0}}, goal_id="g"),
        _ev(
            5,
            "round_started",
            {"round": 1, "shares": {"g/t/a": {"tokens": 0, "usd": 1.0}}},
            goal_id="g",
        ),
        _reserve(6, "r-1"),
        _settle(7, "r-1"),
        _ev(8, "goal_finished", {"reason": None}, goal_id="g"),
        _ev(9, "limits_changed", {"old": {}, "new": Limits(max_usd=2.0).model_dump(mode="json")}),
    ]
    state = RunState.from_events(reversed(events))
    assert state.ledger_events == [events[0], *events[2:]]


def test_rejected_plan_does_not_replace_revisions() -> None:
    state = RunState.from_events(
        [*_PRE, _ev(3, "plan_rejected", {"errors": ["teams"], "origin": "llm"}, goal_id="g")]
    )
    assert state.plans["g"] == [(_plan(), "human")]


def test_reserve_exhaustion_enters_select_only() -> None:
    state = RunState.from_events([*_PRE, _ev(3, "reserve_exhausted", {}, goal_id="g")])
    assert state.goals["g"].select_only
    assert state.goals["g"].pending_reason is None


def test_plan_replay_refuses_null_in_nested_config() -> None:
    plan = _plan().model_dump(mode="json")
    plan["judges"] = [
        {"path": "module_that_does_not_exist:Judge", "config": {"REPLAY_SECRET": None}}
    ]
    with pytest.raises(ValueError, match="invalid journal replay payload") as caught:
        RunState.from_events(
            [
                _PRE[0],
                _ev(
                    2,
                    "plan_revision",
                    {"plan": plan, "origin": "human", "base_dir": "/plans"},
                    goal_id="g",
                ),
            ]
        )
    assert_secret_absent(caught.value, "REPLAY_SECRET")


def test_goal_default_containers_are_independent() -> None:
    first, second = GoalState(), GoalState()
    first.repeat_keys["c-1"] = "key"
    first.final_order.append("c-1")
    assert second.repeat_keys == {}
    assert second.final_order == []


@pytest.mark.parametrize("converge", [False, True])
def test_round_started_preserves_recorded_agent_statuses(converge: bool) -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _ev(
                3,
                "agent_status",
                {"status": "pruned", "reason": "done"},
                goal_id="g",
                team="t",
                agent="a",
            ),
            _ev(4, "round_started", {"round": 2, "converge": converge}, goal_id="g"),
        ]
    )
    assert state.goals["g"].agents == {"t/a": AgentStatus.PRUNED}


def test_final_order_and_manifest_folded() -> None:
    events = [
        *_PRE,
        _ev(3, "run_started", {"estimate": {}, "warnings": [], "manifest": {"run_id": "r"}}),
        _ev(9, "final_order", {"candidate_ids": [], "parts": {"a": "c-1", "b": None}}, goal_id="g"),
    ]
    state = RunState.from_events(events)
    assert (state.goals["g"].final_order, state.goals["g"].final_parts) == (
        [],
        {"a": "c-1", "b": None},
    )
    assert state.manifest == {"run_id": "r"}
    state.manifest["run_id"] = "changed"
    assert events[2].payload["manifest"] == {"run_id": "r"}


def test_no_result_reason_is_distinct_from_terminal_reason_and_resets() -> None:
    events = [
        *_PRE,
        _ev(3, "no_result", {"reason": "no passing candidate"}, goal_id="g"),
        _ev(4, "goal_finished", {"reason": "stop"}, goal_id="g"),
    ]
    goal = RunState.from_events(events).goals["g"]
    assert (goal.no_result_reason, goal.end_reason) == ("no passing candidate", "stop")
    continued = RunState.from_events([*events, _ev(5, "goal_started", {}, goal_id="g")])
    assert continued.goals["g"].no_result_reason is None


def test_repeat_keys_folded() -> None:
    end = _end(3, "c-1")
    end = end.model_copy(update={"payload": {**end.payload, "repeat_key": "k1"}})
    state = RunState.from_events([*_PRE, end])
    assert state.goals["g"].repeat_keys == {"c-1": "k1"}


def test_converge_state_folded() -> None:
    events = [
        *_PRE,
        _ev(7, "part_kept", {"part": "b", "candidate_id": "c-b1"}, goal_id="g"),
        _ev(8, "join_failed", {"attempt": 1, "reasons": ["x"]}, goal_id="g"),
        _ev(9, "converge_started", {"select_only": True}, goal_id="g"),
        _ev(10, "join_failed", {"attempt": 2, "reasons": ["y"]}, goal_id="g"),
    ]
    state = RunState.from_events(events)
    g = state.goals["g"]
    assert (g.kept_parts, g.join_attempts, g.select_only, g.converge_started) == (
        {"b": "c-b1"},
        2,
        True,
        True,
    )
    assert g.join_failures == ["x", "y"]
    events += [
        _ev(11, "goal_finished", {"reason": "stop"}, goal_id="g"),
        _ev(12, "goal_started", {}, goal_id="g"),
    ]
    continued = RunState.from_events(events).goals["g"]
    assert (continued.kept_parts, continued.join_attempts, continued.join_failures) == ({}, 0, [])


def test_tool_fingerprints_folded_with_legacy_null() -> None:
    first = _end(3, "c-1")
    first = first.model_copy(update={"payload": {**first.payload, "fingerprint": "same"}})
    state = RunState.from_events([*_PRE, first, _end(4, "c-2")])
    assert state.fingerprints == {"c-1": "same", "c-2": None}


def test_join_history_survives_round_and_resets_with_partition() -> None:
    from swarm_arena import PartSpec

    events = [
        *_PRE,
        _ev(3, "part_kept", {"part": "b", "candidate_id": "c-b1"}, goal_id="g"),
        _ev(4, "join_failed", {"attempt": 1, "reasons": ["failed"]}, goal_id="g"),
        _ev(5, "round_started", {"round": 2}, goal_id="g"),
    ]
    old = RunState.from_events(events).goals["g"]
    assert (old.join_attempts, old.join_failures, old.kept_parts) == (1, ["failed"], {"b": "c-b1"})
    plan = _plan().model_copy(
        update={"parts": [PartSpec(name="a", goal=Goal(id="a", namespace="n"))], "revision": 1}
    )
    events.append(
        _ev(
            6,
            "plan_revision",
            {"plan": plan.model_dump(mode="json"), "origin": "autonomous"},
            goal_id="g",
        )
    )
    new = RunState.from_events(events).goals["g"]
    assert (new.join_attempts, new.join_failures, new.kept_parts) == (0, [], {})


def test_goal_queue_lifecycle_and_stopped_no_result() -> None:
    events = [_ev(1, "goal_queued", {}, goal_id="g"), _ev(2, "goal_started", {}, goal_id="g")]
    assert RunState.from_events(events[:1]).goals["g"].status == "queued"
    assert RunState.from_events(events).goals["g"].status == "running"
    queued = [
        events[0],
        _ev(2, "no_result", {"reason": "stopped"}, goal_id="g"),
        _ev(3, "goal_finished", {"reason": "stopped"}, goal_id="g"),
    ]
    assert RunState.from_events(queued).goals["g"].status == "no_result"


@pytest.mark.parametrize(
    "reason", [None, "stop", "stop --now", "run_timeout_s", "reserve_exhausted"]
)
def test_pending_termination_reason_survives_takeover_and_clears_on_continuation(
    reason: str | None,
) -> None:
    events = [
        *_PRE,
        _ev(3, "goal_started", {}, goal_id="g"),
        _ev(4, "converge_started", {"select_only": True, "reason": reason}, goal_id="g"),
        _ev(5, "takeover", {"force": True}, epoch=2),
        _ev(
            6,
            "final_order",
            {"candidate_ids": [], "parts": {}, "reason": reason},
            goal_id="g",
            epoch=2,
        ),
    ]
    # These additive keys remain inside Event's existing finite-JSON payload.
    events = [Event.model_validate_json(e.model_dump_json()) for e in events]
    state = RunState.from_events(events)
    assert state.stop == "none"
    assert state.goals["g"].pending_reason == reason
    assert state.goals["g"].end_reason is None
    ended = _ev(7, "goal_finished", {"reason": reason}, goal_id="g", epoch=2)
    state = RunState.from_events([*events, ended, _ev(8, "goal_started", {}, goal_id="g", epoch=3)])
    assert state.goals["g"].pending_reason is None


@pytest.mark.parametrize("now", [False, True])
def test_absent_termination_reason_is_natural_despite_old_stop_flags(now: bool) -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _ev(3, "stop_requested", {"now": now}),
            _ev(4, "final_order", {"candidate_ids": [], "parts": None}, goal_id="g"),
            _ev(5, "takeover", {"force": True}, epoch=2),
        ]
    )
    assert state.goals["g"].pending_reason is None
    assert state.stop == "none"


@pytest.mark.parametrize("outcome", ["plan_revision", "plan_rejected"])
def test_round_closure_retains_pending_replan_until_committed_outcome(outcome: str) -> None:
    closed = _ev(3, "round_closed", {"round": 1, "replan": True}, goal_id="g")
    events = [*_PRE, closed]
    state = RunState.from_events(events)
    assert state.goals["g"].pending_replan == 2
    revision = _plan().model_copy(update={"revision": 1})
    payload = (
        {"plan": revision.model_dump(mode="json"), "origin": "autonomous", "first_round": 2}
        if outcome == "plan_revision"
        else {"errors": ["invalid proposal"], "origin": "autonomous", "first_round": 2}
    )
    event = _ev(4, outcome, payload, goal_id="g")
    state = RunState.from_events([*events, Event.model_validate_json(event.model_dump_json())])
    assert state.goals["g"].pending_replan is None
    assert state.plan_for_round("g", 1).revision == 0
    assert state.plan_for_round("g", 2).revision == (1 if outcome == "plan_revision" else 0)


def test_legacy_revision_before_closure_applies_to_successor_round() -> None:
    revision = _plan().model_copy(update={"revision": 1})
    state = RunState.from_events(
        [
            *_PRE,
            _ev(3, "round_started", {"round": 1}, goal_id="g"),
            _ev(
                4,
                "plan_revision",
                {"plan": revision.model_dump(mode="json"), "origin": "autonomous"},
                goal_id="g",
            ),
        ]
    )
    assert state.plan_for_round("g", 1).revision == 0
    assert state.plan_for_round("g", 2).revision == 1
    assert state.goals["g"].replanned_rounds == [1, 2]


def test_replay_accepts_curated() -> None:
    receipt = _ev(
        3,
        "curated",
        {
            "summaries": [],
            "supersedes": {},
            "errors": [],
            "decision_seq": 2,
            "revision": 0,
            "round": 1,
        },
        goal_id="g",
    )
    state = RunState.from_events([*_PRE, receipt])
    assert state.learnings == {}
    assert state.scores == {}
    assert state.goals["g"].curated_decisions == [2]
    assert state.goals["g"].join_phase == "converge"


def test_curated_preserves_retry_and_pending_replan() -> None:
    events = [
        *_PRE,
        _ev(3, "join_failed", {"attempt": 1, "reasons": ["retry"]}, goal_id="g"),
        _ev(4, "plan_rejected", {"first_round": 2, "join_retry": 1}, goal_id="g"),
        _ev(5, "round_closed", {"round": 2, "replan": True}, goal_id="g"),
    ]
    before = RunState.from_events(events).goals["g"]
    after = RunState.from_events(
        [
            *events,
            _ev(
                6,
                "curated",
                {
                    "summaries": [],
                    "supersedes": {},
                    "errors": [],
                    "decision_seq": 2,
                    "revision": 0,
                    "round": 2,
                },
                goal_id="g",
            ),
        ]
    ).goals["g"]
    assert after.model_copy(update={"curated_decisions": []}) == before


@pytest.mark.parametrize("receipt", [False, True])
def test_curator_refusal_replays_terminal_disposition(receipt: bool) -> None:
    events = [
        *_PRE,
        _ev(3, "round_started", {"round": 1}, goal_id="g"),
        _ev(4, "decision", {"round": 1, "decision": {}}, goal_id="g"),
        _ev(5, "reserve_exhausted", {"round": 1, "reason": "reserve_exhausted"}, goal_id="g"),
    ]
    if receipt:
        events.append(
            _ev(6, "curated", {"decision_seq": 4, "summaries": [], "errors": []}, goal_id="g")
        )
    events.append(_ev(7, "takeover", {"force": True}, epoch=2))
    state = RunState.from_events(events).goals["g"]
    assert state.pending_reason == "reserve_exhausted"
    assert state.select_only
    assert state.pending_replan is None
    assert state.curated_decisions == ([4] if receipt else [])


def test_closed_seq_per_round() -> None:
    state = RunState.from_events(
        [
            *_PRE,
            _ev(7, "round_closed", {"round": 1, "replan": True}, goal_id="g"),
            _ev(12, "round_closed", {"round": 2, "reason": "stop"}, goal_id="g"),
        ]
    )
    goal = state.goals["g"]
    assert goal.closed_seq == {1: 7, 2: 12}
    assert goal.pending_replan == 2
    assert goal.pending_reason == "stop"


def test_replay_accepts_review_imported() -> None:
    score = Score(
        id="s-0123456789abcdef-e2-0042", candidate_id="c", judge="human:alice", key=(1, 7)
    )
    state = RunState.from_events(
        [
            *_PRE,
            _ev(
                3,
                "review_imported",
                {
                    "reviewer": "alice",
                    "scores": [score.model_dump(mode="json")],
                    "objects": [],
                },
                epoch=2,
            ),
        ]
    )
    assert state.scores == {}
    assert state.id_counters["s"] == 42


def test_replay_restores_planner_provenance_and_base_dir(tmp_path: Path) -> None:
    from unittest.mock import Mock

    from swarm_arena._engine.goal import r0_revision_draft
    from swarm_arena._engine.run import mark_planning_session

    plan = mark_planning_session(_plan(), Mock(run_id="r-p", call_ids=["llm-1"]))
    draft = r0_revision_draft(plan, tmp_path)
    state = RunState.from_events([_ev(1, "plan_revision", draft.payload, goal_id="g")])
    restored = state.plans["g"][0][0]
    assert (restored.planning_session, restored.planner_call_ids) == ("r-p", ("llm-1",))
    assert restored.base_dir == tmp_path


@pytest.mark.parametrize(
    ("key", "value"), [("planning_session", "../secret-sentinel"), ("planner_call_ids", [123])]
)
def test_replay_refuses_invalid_planner_provenance(key: str, value: Any) -> None:
    payload = {"plan": _plan().model_dump(mode="json"), "origin": "human", key: value}
    with pytest.raises(ValueError, match="invalid journal replay payload") as caught:
        RunState.from_events([_ev(1, "plan_revision", payload, goal_id="g")])
    assert_secret_absent(caught.value, "secret-sentinel")
