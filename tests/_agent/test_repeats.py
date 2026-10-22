import traceback
from dataclasses import dataclass, field, replace
from typing import cast
from unittest.mock import Mock

import pytest

from swarm_arena import Goal, Limits, RoundSpec, ScriptTool
from swarm_arena._agent.manager import LimitReached
from swarm_arena._agent.repeats import COUNTED, RepeatIndex, RepeatOutcome, repeat_key
from swarm_arena._agent.runtime import AgentCounters, AgentRuntime
from swarm_arena._agent.watchdog import Watchdog
from swarm_arena._errors import SoftError
from swarm_arena._journal.event import EventDraft
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.plan import AgentSpec, TeamSpec
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.tooling import ParentRef, RunStatus, ToolOutput, ToolRequest
from swarm_arena.testing import make_candidate


@pytest.mark.parametrize("nested", [False, True])
def test_next_replicate_exit_terminal(
    rt: AgentRuntime, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    from swarm_arena._agent import repeats as module
    from swarm_arena._errors import _core_category, _SignalCategory, _stop_cleanup_error

    reached: list[bool] = []

    def fail(**kwargs: object) -> str:
        reached.append(True)
        error = _stop_cleanup_error()
        if nested:
            raise BaseExceptionGroup("repeat", [BaseExceptionGroup("key", [error])])
        raise error

    monkeypatch.setattr(module, "repeat_key", fail)
    with pytest.raises(RuntimeError) as caught:
        rt.repeats.next_replicate(rt, ToolRequest())
    assert reached == [True]
    assert _core_category(caught.value) is _SignalCategory.TERMINAL


@dataclass
class _Journal:
    events: list[EventDraft] = field(default_factory=list)

    def append(self, event: EventDraft) -> None:
        self.events.append(event)


class _RandomTool(ScriptTool):
    def is_deterministic(self, request: ToolRequest) -> bool:
        return False


@pytest.fixture
def rt() -> AgentRuntime:
    goal = Goal(id="g", namespace="n")
    ref = Ref(path="swarm_arena:ScriptTool", config={"command": ["true"]})
    spec = AgentSpec(name="a", rationale="Search", tool=ref, manager=Ref(path="m:Manager"))
    return AgentRuntime(
        address="g/t/a",
        holder="g/t/a",
        goal_id="g",
        goal=goal,
        goal_key=goal.key(),
        part=None,
        team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        spec=spec,
        round=1,
        round_spec=RoundSpec(),
        fidelity="",
        tool=ScriptTool(command=["true"]),
        tool_ref=ref,
        execution_ref=ref,
        tool_identity="identity",
        catalog_entry=None,
        checks=[],
        view=Mock(),
        own=[],
        digest="",
        notes="",
        ledger=Mock(),
        tranche=Mock(),
        limits=Limits(),
        counters=AgentCounters(),
        repeats=RepeatIndex("this-run"),
        watchdog=Watchdog(Limits()),
        journal=cast(JournalWriter, _Journal()),
        board=Mock(),
        executor=Mock(),
        artifacts=Mock(),
        ids=Mock(),
        config=RunConfig(),
        resumed=(),
    )


@pytest.fixture
def rt_nd(rt: AgentRuntime) -> AgentRuntime:
    return replace(rt, tool=_RandomTool(command=["true"]))


@pytest.fixture
def rt_nd2(rt_nd: AgentRuntime) -> AgentRuntime:
    return replace(rt_nd, spec=rt_nd.spec.model_copy(update={"replicates": 2}))


def _rt_for_part(rt: AgentRuntime, part: str) -> AgentRuntime:
    goal = Goal(id=part.lower(), namespace=rt.goal.namespace)
    return replace(rt, goal=goal, goal_key=goal.key(), part=part)


def _c(
    status: RunStatus,
    *,
    agent: str,
    id: str = "c1",
    run_id: str = "this-run",
    partial: bool = False,
) -> Candidate:
    team, name = agent.split("/")
    return make_candidate(
        ToolOutput(status=status, partial=partial),
        id=id,
        run_id=run_id,
        team=team,
        agent=name,
    )


def _entry(
    idx: RepeatIndex,
    rt: AgentRuntime,
    request: ToolRequest,
    *,
    replicate: int,
) -> tuple[str, Candidate]:
    request = request.model_copy(update={"replicate": replicate})
    candidate = make_candidate(
        ToolOutput(status=RunStatus.OK),
        request=request,
        id=f"c-{replicate}",
        run_id="this-run",
        goal=rt.goal,
        team=rt.team.name,
        agent=rt.spec.name,
        overlay={
            p: v.removeprefix("sha256:")
            for p, v in request.model_dump(mode="json")["edits"].items()
        },
    )
    return idx.key(rt, request), candidate


def test_key_is_canonical_and_covers_spec_fields(rt: AgentRuntime) -> None:
    idx = RepeatIndex("this-run")
    k = idx.key(rt, ToolRequest(inputs={"a": 1, "b": 2}))
    assert k == idx.key(rt, ToolRequest(inputs={"b": 2, "a": 1}))
    assert k != idx.key(rt, ToolRequest(inputs={"a": 1, "b": 2}, edits={"p.txt": "x"}))
    assert k != idx.key(rt, ToolRequest(inputs={"a": 1, "b": 2}, round_overrides={"epochs": 100}))
    assert k != idx.key(_rt_for_part(rt, "A"), ToolRequest(inputs={"a": 1, "b": 2}))


def test_nondeterministic_key_includes_replicate(rt_nd: AgentRuntime) -> None:
    idx = RepeatIndex("this-run")
    assert idx.key(rt_nd, ToolRequest(replicate=0)) != idx.key(rt_nd, ToolRequest(replicate=1))


@pytest.mark.parametrize(
    ("earlier", "agent", "resumed", "action"),
    [
        (_c(RunStatus.OK, agent="t/a"), "t/a", frozenset(), "refuse"),
        (_c(RunStatus.OK, agent="t/a", id="c-own"), "t/a", frozenset({"c-own"}), "return_own"),
        (_c(RunStatus.OK, agent="t/a", id="c-old"), "t/a", frozenset({"c-other"}), "refuse"),
        (_c(RunStatus.ERROR, agent="t/b"), "t/a", frozenset(), "link"),
        (_c(RunStatus.TIMEOUT, agent="t/a"), "t/a", frozenset(), "run"),
        (_c(RunStatus.OK, agent="t/a", partial=True), "t/a", frozenset(), "run"),
        (_c(RunStatus.OK, agent="t/b", run_id="past-run"), "t/a", frozenset(), "run"),
    ],
)
def test_lookup_rules(
    earlier: Candidate,
    agent: str,
    resumed: frozenset[str],
    action: str,
) -> None:
    out = RepeatIndex("this-run", [("k", earlier)]).lookup(
        "k",
        agent=agent,
        deterministic=True,
        resumed=resumed,
    )
    assert out.action == action
    if action in {"refuse", "return_own", "link"}:
        assert out.candidate == earlier
    else:
        assert out.candidate is None
    if earlier.run_id == "past-run":
        assert out.hint == earlier


def test_nondeterministic_always_runs() -> None:
    idx = RepeatIndex("this-run", [("k", _c(RunStatus.OK, agent="t/a"))])
    assert idx.lookup("k", agent="t/a", deterministic=False).action == "run"


def test_add_records_own_entries() -> None:
    idx = RepeatIndex("this-run")
    c = _c(RunStatus.OK, agent="t/a")
    idx.add("k2", c)
    assert idx.added == [("k2", c)]
    assert idx.lookup("k2", agent="t/a", deterministic=True).action == "refuse"


def test_next_replicate_in_order_and_exhaustion(rt_nd2: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest(inputs={"x": 1.0})
    assert idx.next_replicate(rt_nd2, req) == 0
    idx.add(*_entry(idx, rt_nd2, req, replicate=0))
    assert idx.next_replicate(rt_nd2, req) == 1
    idx.add(*_entry(idx, rt_nd2, req, replicate=1))
    with pytest.raises(SoftError, match="replicates used up"):
        idx.next_replicate(rt_nd2, req)


def test_explicit_replicate_out_of_range_or_used_rejected(rt_nd2: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest(inputs={"x": 1.0})
    with pytest.raises(SoftError):
        idx.next_replicate(rt_nd2, req, requested=2)
    idx.add(*_entry(idx, rt_nd2, req, replicate=0))
    with pytest.raises(SoftError):
        idx.next_replicate(rt_nd2, req, requested=0)
    assert idx.next_replicate(rt_nd2, req, requested=1) == 1


def test_next_replicate_counts_journal_loaded_and_earlier_round_entries(
    rt_nd2: AgentRuntime,
) -> None:
    idx = RepeatIndex("this-run")
    live = ToolRequest(inputs={"x": 1.0}, edits={"p.txt": "x"}, round=2)
    loaded = ToolRequest.model_validate({**live.model_dump(mode="json"), "round": 1})
    idx.add(*_entry(idx, rt_nd2, loaded, replicate=0))
    assert idx.next_replicate(rt_nd2, live) == 1


def test_next_replicate_counts_loaded_request_with_edits(rt_nd2: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest(inputs={"x": 1.0}, edits={"p/a.txt": "new"})
    loaded = ToolRequest.model_validate(
        req.model_copy(update={"replicate": 0}).model_dump(mode="json")
    )
    idx.add(*_entry(idx, rt_nd2, loaded, replicate=0))
    assert idx.next_replicate(rt_nd2, req) == 1


def test_key_uses_base_ref_and_ignores_round_and_deterministic_replicate(rt: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest(inputs={"x": 1})
    changed_execution = rt.execution_ref.model_copy(update={"config": {"command": ["false"]}})
    assert idx.key(rt, req) == idx.key(replace(rt, execution_ref=changed_execution), req)
    assert idx.key(rt, req) == idx.key(rt, req.model_copy(update={"round": 9, "replicate": 12}))
    assert idx.key(rt, req) != idx.key(replace(rt, tool_ref=changed_execution), req)
    renamed = rt.tool_ref.model_copy(update={"name": "another label"})
    assert idx.key(rt, req) == idx.key(replace(rt, tool_ref=renamed), req)
    other_path = rt.tool_ref.model_copy(update={"path": "other:Tool"})
    assert idx.key(rt, req) != idx.key(replace(rt, tool_ref=other_path), req)


def test_key_hashes_live_and_stored_edits_equally(rt: AgentRuntime) -> None:
    req = ToolRequest(edits={"p.txt": "sha256:hello", "other.txt": "text"})
    loaded = ToolRequest.model_validate_json(req.model_dump_json())
    assert RepeatIndex("this-run").key(rt, req) == RepeatIndex("this-run").key(rt, loaded)
    assert repeat_key(goal_key=rt.goal_key, tool_ref=rt.tool_ref, request=req, deterministic=True)


def test_key_covers_parent_ids_and_type_distinct_inputs(rt: AgentRuntime) -> None:
    idx = RepeatIndex("this-run")
    req = ToolRequest(parents=[ParentRef(candidate_id="c1", run_id="r", tool_identity="T")])
    other = req.model_copy(
        update={"parents": [req.parents[0].model_copy(update={"candidate_id": "c2"})]}
    )
    assert idx.key(rt, req) != idx.key(rt, other)
    assert len({idx.key(rt, ToolRequest(inputs={"v": v})) for v in (True, 1, 1.0, None)}) == 4
    assert idx.key(rt, req) == idx.key(
        rt,
        req.model_copy(
            update={
                "parents": [req.parents[0].model_copy(update={"data": {"ignored": True}})],
            }
        ),
    )


@pytest.mark.parametrize(
    "status", [s for s in RunStatus if s not in {RunStatus.OK, RunStatus.ERROR}]
)
def test_lookup_ignores_uncounted_statuses(status: RunStatus) -> None:
    idx = RepeatIndex("this-run", [("k", _c(status, agent="t/a"))])
    assert idx.lookup("k", agent="t/a", deterministic=True) == RepeatOutcome(action="run")
    assert frozenset({RunStatus.OK, RunStatus.ERROR}) == COUNTED


def test_lookup_prioritizes_own_then_same_run_then_past_hint() -> None:
    past = _c(RunStatus.OK, agent="t/a", id="past", run_id="past")
    other = _c(RunStatus.OK, agent="t/b", id="other")
    own = _c(RunStatus.OK, agent="t/a", id="own")
    idx = RepeatIndex("this-run", [("k", past), ("k", other), ("k", own)])
    assert idx.added == []
    assert idx.lookup("k", agent="t/a", deterministic=True).candidate == own
    assert idx.lookup("k", agent="t/c", deterministic=True).candidate == other
    assert idx.lookup("absent", agent="t/a", deterministic=True) == RepeatOutcome(action="run")


def test_candidates_of_deduplicates_in_insertion_order_and_filters_ownership() -> None:
    a = _c(RunStatus.TIMEOUT, agent="t/a", id="a")
    b = _c(RunStatus.OK, agent="t/a", id="b", partial=True)
    past = _c(RunStatus.OK, agent="t/a", id="past", run_id="past")
    other = _c(RunStatus.OK, agent="t/b", id="other")
    idx = RepeatIndex("this-run", [("k", a), ("k", other), ("k", past), ("k2", b), ("k3", a)])
    assert idx.candidates_of("t/a") == [a, b]
    idx.add("k4", _c(RunStatus.OK, agent="t/a", id="new"))
    assert [c.id for c in idx.candidates_of("t/a")] == ["a", "b", "new"]
    assert idx.candidates_of("missing/agent") == []


def test_pending_indices_release_and_request_replicate_is_ignored(rt_nd2: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest(replicate=99)
    assert idx.next_replicate(rt_nd2, req) == 0
    assert idx.next_replicate(rt_nd2, req) == 1
    with pytest.raises(SoftError, match="replicates used up"):
        idx.next_replicate(rt_nd2, req)
    idx.release(rt_nd2, req, 0)
    idx.release(rt_nd2, req, 0)
    assert idx.next_replicate(rt_nd2, req) == 0


def test_add_frees_pending_for_uncounted_or_partial_completion(rt_nd2: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest()
    assert idx.next_replicate(rt_nd2, req) == 0
    key, candidate = _entry(idx, rt_nd2, req, replicate=0)
    idx.add(key, candidate.model_copy(update={"output": ToolOutput(status=RunStatus.TIMEOUT)}))
    assert idx.next_replicate(rt_nd2, req) == 0
    idx.add(
        key, candidate.model_copy(update={"output": ToolOutput(status=RunStatus.OK, partial=True)})
    )
    assert idx.next_replicate(rt_nd2, req) == 0


def test_used_replicates_require_own_counted_complete_same_run_entries(
    rt_nd2: AgentRuntime,
) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest()
    key, candidate = _entry(idx, rt_nd2, req, replicate=0)
    for changed in (
        candidate.model_copy(update={"agent": "b"}),
        candidate.model_copy(update={"team": "other"}),
        candidate.model_copy(update={"run_id": "past-run"}),
        candidate.model_copy(update={"output": ToolOutput(status=RunStatus.ERROR, partial=True)}),
        candidate.model_copy(update={"output": ToolOutput(status=RunStatus.OOM)}),
    ):
        idx.add(key, changed)
    assert idx.next_replicate(rt_nd2, req) == 0
    idx.release(rt_nd2, req, 0)
    idx.add(key, candidate.model_copy(update={"output": ToolOutput(status=RunStatus.ERROR)}))
    assert idx.next_replicate(rt_nd2, req) == 1


def test_requested_negative_and_pending_replicates_are_rejected(rt_nd2: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest()
    with pytest.raises(SoftError):
        idx.next_replicate(rt_nd2, req, requested=-1)
    assert idx.next_replicate(rt_nd2, req, requested=1) == 1
    with pytest.raises(SoftError):
        idx.next_replicate(rt_nd2, req, requested=1)
    assert idx.next_replicate(rt_nd2, req) == 0


def test_exhaustion_journals_refusals_and_trips_watchdog_without_secrets(
    rt_nd: AgentRuntime,
) -> None:
    secret = "SENTINEL-repeat-input-edit-override"
    idx = RepeatIndex("this-run")
    req = ToolRequest(inputs={"v": secret}, edits={"p.txt": secret}, round_overrides={"v": secret})
    idx.add(*_entry(idx, rt_nd, req, replicate=0))
    for _ in range(3):
        with pytest.raises(
            SoftError, match=r"repeat refused: replicates used up \(1 of 1\)"
        ) as err:
            idx.next_replicate(rt_nd, req)
        assert secret not in "".join(traceback.format_exception(err.value))
    with pytest.raises(LimitReached) as stopped:
        idx.next_replicate(rt_nd, req)
    assert stopped.value.limit == "max_refused_repeats"
    journal = cast(_Journal, rt_nd.journal)
    assert len(journal.events) == 4
    for event in journal.events:
        assert event.type == "watchdog"
        assert (event.goal_id, event.team, event.agent, event.round) == ("g", "t", "a", 1)
        assert event.payload == {"refused": "c-0"}
    for error in (err.value, stopped.value):
        assert secret not in str(error) + repr(error) + repr(error.args)
        assert secret not in repr(getattr(error, "__notes__", None))
        assert error.__cause__ is None
        assert secret not in "".join(traceback.format_exception(error))
    assert (
        secret not in canonical_json([e.model_dump(mode="json") for e in journal.events]).decode()
    )


def test_pending_exhaustion_journals_null_match(rt_nd: AgentRuntime) -> None:
    idx, req = RepeatIndex("this-run"), ToolRequest()
    assert idx.next_replicate(rt_nd, req) == 0
    with pytest.raises(SoftError):
        idx.next_replicate(rt_nd, req)
    assert cast(_Journal, rt_nd.journal).events[-1].payload == {"refused": None}


def test_replicate_pools_are_per_key(rt_nd2: AgentRuntime) -> None:
    idx = RepeatIndex("this-run")
    assert idx.next_replicate(rt_nd2, ToolRequest(inputs={"x": 1})) == 0
    assert idx.next_replicate(rt_nd2, ToolRequest(inputs={"x": 2})) == 0


def test_release_uses_retained_allocation_without_reentering_hook(rt_nd2: AgentRuntime) -> None:
    from _fakes.secrecy import assert_secret_absent

    idx, request = RepeatIndex("this-run"), ToolRequest(inputs={"x": 1})
    index = idx.next_replicate(rt_nd2, request)
    secret = "PRIVATE_RELEASE_STATE"
    tool = Mock(is_deterministic=Mock(side_effect=RuntimeError(secret)))
    rt_nd2.tool = tool
    rt_nd2.goal_key = Goal(id="changed", namespace="n").key()
    request.inputs["x"] = 2
    try:
        idx.release(rt_nd2, request, index)
        idx.release(rt_nd2, request, index)
    except Exception as error:
        assert_secret_absent(error, secret)
        raise
    assert not idx._pending
    tool.is_deterministic.assert_not_called()


def test_part_replicate_refusal_routes_main_goal(rt_nd: AgentRuntime) -> None:
    rt_nd.goal = Goal(id="sub", namespace="n")
    rt_nd.goal_key = rt_nd.goal.key()
    rt_nd.part = "part"
    idx, request = RepeatIndex("this-run"), ToolRequest()
    idx.next_replicate(rt_nd, request)
    with pytest.raises(SoftError):
        idx.next_replicate(rt_nd, request)
    assert cast(_Journal, rt_nd.journal).events[-1].goal_id == "g"
