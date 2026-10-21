"""Committed board records shared by digest and migration regressions."""

from collections.abc import Sequence
from pathlib import Path

from _fakes.journals import START, FakeRun, add_pointer

from swarm_arena import Goal, PartSpec, Plan, Score
from swarm_arena._board.arena import ArenaBoard
from swarm_arena._board.human_objects import HumanObject, write_human_objects
from swarm_arena._engine.goal import r0_revision_draft
from swarm_arena._journal.event import EventDraft
from swarm_arena._store.base import open_store
from swarm_arena.board import Candidate, Learning
from swarm_arena.tool import ToolOutput, ToolRequest

GOAL = Goal(id="g", namespace="n")
OTHER = Goal(id="g2", namespace="n")
PART = Goal(id="sub", namespace="n")
KEY, OTHER_KEY, PART_KEY = GOAL.key(), OTHER.key(), PART.key()


def candidate(id: str, **changes: object) -> Candidate:
    return Candidate.model_validate(
        dict(
            id=id,
            run_id="run-1",
            goal_id="g",
            team="fd-solver",
            agent="tuner-1",
            round=1,
            request=ToolRequest(),
            output=ToolOutput(status="ok"),
        )
        | changes
    )


def learning(id: str, **changes: object) -> Learning:
    return Learning.model_validate(
        dict(
            id=id,
            namespace="n",
            run_id="run-1",
            goal_id="g",
            round=1,
            author="human:alice",
            title="A hint",
            text="Try this",
        )
        | changes
    )


def score(id: str, candidate: str, score_value: float, **changes: object) -> Score:
    return Score.model_validate(
        dict(
            id=id,
            candidate_id=candidate,
            key=(score_value,),
            judge="program",
            rubric_hash="new",
            panel_hash="panel",
        )
        | changes
    )


def records(*items: Candidate | Learning | Score) -> list[EventDraft]:
    drafts = []
    for item in items:
        if isinstance(item, Candidate):
            kind, field, goal_id = "tool_end", "candidate", item.goal_id
        elif isinstance(item, Learning):
            kind, field, goal_id = "learning", "learning", item.goal_id
        else:
            kind, field, goal_id = "score", "score", "g"
        drafts.append(
            EventDraft(type=kind, goal_id=goal_id, payload={field: item.model_dump(mode="json")})
        )
    return drafts


def seed_board(
    root: Path,
    *,
    candidates: Sequence[Candidate] = (),
    learnings: Sequence[Learning] = (),
    scores: Sequence[Score] = (),
) -> ArenaBoard:
    """Create real journals for every producer and return their opened arena."""
    # A different namespace must have its own valid routed plan, not a forged learning.
    learnings = [
        item.model_copy(update={"goal_id": "elsewhere"})
        if item.namespace != GOAL.namespace
        else item
        for item in learnings
    ]
    by_run: dict[str, list[Candidate | Learning | Score]] = {"run-1": []}
    owners = {c.id: c.run_id for c in candidates}
    for item in [*candidates, *learnings, *scores]:
        run_id = owners[item.candidate_id] if isinstance(item, Score) else item.run_id
        by_run.setdefault(run_id, []).append(item)
    arena = str(root / "arena")
    for run_id, items in by_run.items():
        run = FakeRun(str(root / "runs"), run_id, goal=GOAL)
        plans = [
            Plan(goal=GOAL, teams=[], judges=[], parts=[PartSpec(name="p", goal=PART)]),
            Plan(goal=OTHER, teams=[], judges=[]),
        ]
        plans.extend(
            Plan(goal=Goal(id=item.goal_id, namespace=item.namespace), teams=[], judges=[])
            for item in items
            if isinstance(item, Learning) and item.namespace != GOAL.namespace
        )
        run.commit(*(r0_revision_draft(plan, plan.base_dir) for plan in plans), *records(*items))
        add_pointer(arena, run)
    operations = [
        HumanObject(kind=kind, learning_id=item.id, author="human:a")
        for item in learnings
        for kind, enabled in (("pin", item.pinned), ("hide", item.hidden))
        if enabled and item.namespace == GOAL.namespace
    ]
    if operations:
        write_human_objects(open_store(arena), "n", "curation", operations, now=START)
    board = ArenaBoard(arena_root=arena, cache_dir=str(root / "cache"))
    board.open("run-1", str(root / "runs"), ["n"])
    board.frontier()
    return board
