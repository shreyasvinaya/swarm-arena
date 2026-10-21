"""Round pages are frozen views of the complete round-close prefix."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from swarm_arena import Goal, Limits, PartSpec, Plan, Ref, RunConfig
from swarm_arena._board.pages import page_key, round_page, state_positions
from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.lease import Head
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.views import read_source_head
from swarm_arena._model.goal import FileSet
from swarm_arena._model.results import Candidate, Learning, Score
from swarm_arena._model.tooling import ToolOutput, ToolRequest

head = Head(epoch=1, seq=100, segment="journal/e1-100.jsonl")


def _event(seq: int, kind: str, **payload: Any) -> Event:
    return Event(
        type=EventType(kind),
        seq=seq,
        epoch=1,
        run_id="page-run",
        context_id="page-run",
        goal_id="ns-viscosity",
        ts=datetime(2026, 10, 18, tzinfo=UTC),
        payload=payload,
    )


def _plan(**changes: Any) -> Plan:
    return Plan(
        goal=Goal(id="ns-viscosity", namespace="physika"),
        teams=[],
        judges=[Ref(path="missing:Judge", name="rel-error")],
        ranking="rel-error",
    ).model_copy(update=changes)


def _candidate(cid: str, **changes: Any) -> Candidate:
    return Candidate(
        id=cid,
        run_id="page-run",
        goal_id="ns-viscosity",
        team="spectral",
        agent="author-b",
        round=2,
        request=ToolRequest(),
        output=ToolOutput(status="ok"),
    ).model_copy(update=changes)


def _score(cid: str, value: float, **changes: Any) -> Score:
    return Score(
        id=f"s-{cid}",
        candidate_id=cid,
        judge="rel-error",
        key=(-value,),
        value=value,
    ).model_copy(update=changes)


def _events() -> list[Event]:
    c1 = _candidate("c-0193", team="fd-solver", agent="tuner-1", parent_ids=["c-0112"])
    c2 = _candidate("c-0171")
    learning = Learning(
        id="L-0088",
        namespace="physika",
        run_id="page-run",
        goal_id="ns-viscosity",
        round=2,
        author="fd-solver/tuner-2",
        title="Fit log(nu)",
        text="First line.\nSecond line.",
        evidence=[c1.id],
        verified=True,
    )
    return [
        _event(
            1, "run_config", config=RunConfig(limits=Limits(max_rounds=3)).model_dump(mode="json")
        ),
        _event(
            2, "plan_revision", plan=_plan().model_dump(mode="json"), origin="human", first_round=1
        ),
        _event(3, "round_started", round=2),
        _event(4, "tool_end", candidate=c1.model_dump(mode="json")),
        _event(5, "tool_end", candidate=c2.model_dump(mode="json")),
        _event(6, "score", score=_score(c1.id, 0.0041).model_dump(mode="json")),
        _event(7, "score", score=_score(c2.id, 0.009).model_dump(mode="json")),
        _event(8, "learning", learning=learning.model_dump(mode="json")),
        _event(9, "decision", round=2, decision={"prune": {"surrogate": "slow"}}),
        _event(10, "round_closed", round=2),
    ]


@pytest.fixture
def state() -> RunState:
    return RunState.from_events(_events())


def _page(state: RunState) -> str:
    return round_page(state, "ns-viscosity", 2, namespace_learnings=412, source=head).decode()


def test_page_layout(state: RunState) -> None:
    lines = _page(state).splitlines()
    assert lines[1] == f"# physika / ns-viscosity · run {state.run_id} · round 2 of 3"
    assert "Leaderboard (rel-error, lower is better)" in lines
    assert "| 1 | c-0193 | fd-solver / tuner-1 | 0.0041 | c-0112 |" in lines
    assert "| 2 | c-0171 | spectral / author-b | 0.009 | — |" in lines
    assert (
        lines[-1] == f"Full board: `swarm-arena board serve {state.run_id}` · "
        "412 learnings in namespace `physika`."
    )


def test_page_key() -> None:
    assert page_key("ns-viscosity", 2) == "goals/ns-viscosity/rounds/round-02.md"
    assert page_key("g", 100) == "goals/g/rounds/round-100.md"


def test_top_10_only() -> None:
    events = _events()[:3]
    for i in range(15):
        candidate = _candidate(f"c-{i}")
        events.extend(
            [
                _event(len(events) + 1, "tool_end", candidate=candidate.model_dump(mode="json")),
                _event(
                    len(events) + 2,
                    "score",
                    score=_score(candidate.id, i + 1).model_dump(mode="json"),
                ),
            ]
        )
    events.append(_event(len(events) + 1, "round_closed", round=2))
    rows = [
        line for line in _page(RunState.from_events(events)).splitlines() if line.startswith("| ")
    ]
    assert len(rows) == 11  # Header plus ten candidates.


def test_ignores_later_rounds(state: RunState) -> None:
    events = _events()
    later = _plan(revision=1, ranking="later", goal=Goal(id="ns-viscosity", namespace="later"))
    learning = state.learnings["L-0088"].model_copy(update={"id": "L-later", "title": "Later"})
    events.extend(
        [
            _event(
                11,
                "plan_revision",
                plan=later.model_dump(mode="json"),
                origin="human",
                first_round=1,
            ),
            _event(12, "limits_changed", new=Limits(max_rounds=99).model_dump(mode="json")),
            _event(
                13, "tool_end", candidate=_candidate("c-later", round=3).model_dump(mode="json")
            ),
            _event(14, "score", score=_score("c-0193", 99, id="s-later").model_dump(mode="json")),
            _event(15, "learning", learning=learning.model_dump(mode="json")),
            _event(16, "decision", round=2, decision={"prune": {"later": "later"}}),
            _event(17, "round_closed", round=3),
        ]
    )
    assert _page(RunState.from_events(events)) == _page(state)


def test_new_learnings_and_pruned(state: RunState) -> None:
    page = _page(state)
    assert "New learnings (1)" in page
    assert "- L-0088 **Fit log(nu)** fd-solver/tuner-2 · verified" in page
    assert "  First line.\n  Second line. Evidence: c-0193 (0.0041)" in page
    assert "Pruned: team `surrogate` (slow)." in page


def test_table_cells_escape_pipes() -> None:
    events = _events()
    events[3].payload["candidate"].update(team="a|b", agent="first\nsecond", parent_ids=["p|q"])
    assert "a\\|b / first second | 0.0041 | p\\|q" in _page(RunState.from_events(events))


def test_part_rows_use_part_judge() -> None:
    events = _events()
    plan = _plan(parts=[PartSpec(name="a", goal=Goal(id="part", namespace="physika"), ranking="a")])
    events[1].payload["plan"] = plan.model_dump(mode="json")
    events[3].payload["candidate"]["part"] = "a"
    events[5].payload["score"]["judge"] = "a"
    assert "| 1 | c-0193 | fd-solver / tuner-1 | 0.0041 | c-0112 |" in _page(
        RunState.from_events(events)
    )


@pytest.mark.parametrize("change", ["inputs", "manifest", "removed"])
def test_part_identity_and_future_activation(change: str) -> None:
    events = _events()
    part = PartSpec(name="a", goal=Goal(id="part", namespace="physika"))
    plan = _plan(parts=[part])
    events[1].payload["plan"] = plan.model_dump(mode="json")
    events[3].payload["candidate"].update(part="a", round=1)
    changes = (
        {"files": FileSet(manifest={"x.txt": "a" * 64})}
        if change == "manifest"
        else {"inputs": {"x": 2}}
    )
    updated = part.model_copy(update={"goal": part.goal.model_copy(update=changes)})
    revision = plan.model_copy(
        update={"revision": 1, "parts": [] if change == "removed" else [updated]}
    )
    events.insert(
        4,
        _event(
            0, "plan_revision", plan=revision.model_dump(mode="json"), origin="human", first_round=2
        ),
    )
    events = [e.model_copy(update={"seq": i}) for i, e in enumerate(events, 1)]
    page = _page(RunState.from_events(events))
    assert "| 1 | c-0193 |" not in page
    events[4].payload["first_round"] = 3
    assert "| 1 | c-0193 |" in _page(RunState.from_events(events))


def test_unnamed_judge_does_not_import_plugins() -> None:
    events = _events()
    events[1].payload["plan"].update(ranking="", judges=[{"path": "missing:Judge"}])
    with patch.object(Ref, "component_name", side_effect=AssertionError("plugin import")):
        assert "Leaderboard (rel-error, lower is better)" in _page(RunState.from_events(events))


def test_source_head_is_owned_by_publication(state: RunState) -> None:
    assert (
        read_source_head(
            "md", round_page(state, "ns-viscosity", 2, namespace_learnings=0, source=head)
        )
        is None
    )


def test_state_positions(state: RunState) -> None:
    assert state_positions(state, reversed(list(state.candidates.values())), until_seq=4) == [
        (state.candidates["c-0193"], ("", state.run_id, 4)),
    ]
    assert [p[1][2] for p in state_positions(state, state.scores.values())] == [6, 7]


@pytest.mark.parametrize(
    ("author", "verified", "label"),
    [
        ("human:alice", False, "from a human"),
        ("team/agent", False, "unverified"),
    ],
)
def test_learning_status_and_unscored_evidence(author: str, verified: bool, label: str) -> None:
    events = _events()
    events[7].payload["learning"].update(author=author, verified=verified, evidence=["c-0171"])
    events = [event for event in events if event.seq != 7]
    events[7].payload["decision"]["prune"] = {"fd-solver/tuner-2": "done"}
    page = _page(RunState.from_events(events))
    assert f"{author} · {label}" in page
    assert "Evidence: c-0171 (ok)" in page
    assert "Pruned: agent `fd-solver/tuner-2` (done)." in page


def test_source_does_not_change_page_body(state: RunState) -> None:
    newer = Head(epoch=2, seq=200, segment="journal/e2-200.jsonl")
    page = round_page(state, "ns-viscosity", 2, namespace_learnings=412, source=newer)
    assert page == _page(state).encode()


def test_evidence_uses_formatter_for_a_score_without_value() -> None:
    events = _events()
    events[5].payload["score"]["value"] = None
    assert "Evidence: c-0193 (-)" in _page(RunState.from_events(events))
