from pathlib import Path
from unittest.mock import patch

import pytest
from _fakes.journals import FakeRun

from swarm_arena import Goal, PartSpec, Plan, Ref
from swarm_arena._board.cache import ArenaCache, Position, goal_key_digest
from swarm_arena._board.ingest import ingest_run
from swarm_arena._board.leaderboard import (
    flatten,
    format_value,
    group_judges,
    group_rows,
    pick_identity,
)
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.goal import FileSet
from swarm_arena._model.results import Candidate, CheckResult, Score
from swarm_arena._model.tooling import ToolOutput, ToolRequest


def _candidate(
    cid: str,
    pos: int,
    *,
    run: str = "r1",
    part: str | None = None,
    round: int = 1,
    fidelity: str = "f1",
    partial: bool = False,
    passed: bool = True,
) -> tuple[Candidate, Position]:
    return Candidate(
        id=cid,
        run_id=run,
        goal_id="g",
        part=part,
        team="t",
        agent="a",
        round=round,
        fidelity=fidelity,
        request=ToolRequest(),
        output=ToolOutput(status="ok", partial=partial),
        checks=[CheckResult(check="gate", passed=passed)],
        parent_ids=["parent"],
    ), ("2026-10-18T00:00:00.000000+00:00", run, pos)


def _score(
    cid: str,
    value: float | None,
    pos: int,
    *,
    run: str = "r1",
    judge: str = "j",
    rubric: str = "rh",
    panel: str = "",
    key: tuple[float, ...] | None = None,
) -> tuple[Score, Position]:
    return Score(
        candidate_id=cid,
        judge=judge,
        rubric_hash=rubric,
        panel_hash=panel,
        value=value,
        key=key if key is not None else (-value if value is not None else 1.0,),
    ), ("2026-10-18T00:00:00.000000+00:00", run, pos)


def test_ties_go_to_earliest() -> None:
    cs = [_candidate("c-a-1", 1, run="r2", round=1), _candidate("c-z-9", 9, round=4)]
    ss = [_score(c.id, 1, i) for i, (c, _) in enumerate(cs)]
    assert [r.candidate_id for r in flatten(group_rows(cs, ss, judge="j"))] == ["c-z-9", "c-a-1"]


def test_min_improvement_keeps_old_leader() -> None:
    cs = [_candidate("c-early", 1), _candidate("c-late", 2)]
    ss = [_score("c-early", 0.0100, 3), _score("c-late", 0.0095, 4)]
    assert [
        r.candidate_id for r in flatten(group_rows(cs, ss, judge="j", min_improvement=0.001))
    ] == ["c-early", "c-late"]
    cs.append(_candidate("c-best", 5))
    ss.append(_score("c-best", 0.0080, 6))
    assert [
        r.candidate_id for r in flatten(group_rows(cs, ss, judge="j", min_improvement=0.001))
    ] == ["c-best", "c-late", "c-early"]


@pytest.mark.parametrize("replace", [False, True])
def test_min_improvement_rows_unique(replace: bool) -> None:
    cs = [_candidate("c-early", 1), _candidate("c-late", 2)]
    ss = [_score("c-early", 0.0100, 3), _score("c-late", 0.0080 if replace else 0.0095, 4)]
    rows = flatten(group_rows(cs, ss, judge="j", min_improvement=0.001))
    assert len(rows) == len({r.candidate_id for r in rows}) == 2


def test_margin_requires_better_full_key_and_first_component() -> None:
    cs = [_candidate("c-early", 1), _candidate("c-late", 2)]
    ss = [_score("c-early", None, 3, key=(1, 1)), _score("c-late", None, 4, key=(1, 2))]
    assert flatten(group_rows(cs, ss, judge="j", min_improvement=0.1))[0].candidate_id == "c-early"
    assert flatten(group_rows(cs, ss, judge="j", min_improvement=0))[0].candidate_id == "c-late"


def test_without_margin_best_key_leads() -> None:
    cs = [_candidate("c-early", 1), _candidate("c-late", 2)]
    ss = [_score("c-early", 0.0100, 3), _score("c-late", 0.0095, 4)]
    assert flatten(group_rows(cs, ss, judge="j"))[0].candidate_id == "c-late"


def test_parts_ranked_by_their_own_judges() -> None:
    cs = [_candidate("ca", 1, part="a"), _candidate("cb", 2, part="b"), _candidate("cm", 3)]
    ss = [
        _score(c.id, value, i, judge=j)
        for i, (c, _) in enumerate(cs)
        for j, value in [("a", 1), ("b", 2), ("y", 3)]
    ]
    groups = group_rows(cs, ss, judge={None: "y", "a": "a", "b": "b"})
    assert {r.judge for r in groups["a"]} == {"a"}
    assert {r.judge for r in groups["b"]} == {"b"}
    assert groups["a"][0].value == 1
    assert groups["b"][0].value == 2
    assert groups[None][0].value == 3
    assert set(group_rows(cs, ss, judge={"a": "a"})) == {"a"}


def test_group_judges_from_part_scoring() -> None:
    plan = Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[],
        judges=[Ref(path="absent:Judge", name="y")],
        ranking="y",
        parts=[
            PartSpec(name="a", goal=Goal(id="ga", namespace="n"), ranking="a"),
            PartSpec(name="b", goal=Goal(id="gb", namespace="n")),
        ],
    )
    assert group_judges(plan) == {None: "y", "a": "a", "b": "y"}


def test_recorded_unnamed_judges_are_import_free_and_group_local() -> None:
    plan = Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[],
        judges=[Ref(path="absent:Judge")],
        parts=[
            PartSpec(name="a", goal=Goal(id="ga", namespace="n")),
            PartSpec(
                name="b", goal=Goal(id="gb", namespace="n"), judges=[Ref(path="other_absent:Judge")]
            ),
        ],
    )
    cs = [_candidate("main", 1), _candidate("a", 2, part="a"), _candidate("b", 3, part="b")]
    ss = [
        _score("main", 1, 4, judge="main"),
        _score("a", 2, 5, judge="inherited"),
        _score("b", 3, 6, judge="other"),
        _score("a", None, 7, judge="human:alice"),
        _score("excluded", 1, 8, judge="pollution"),
    ]
    with (
        patch.object(Ref, "component_name", side_effect=AssertionError("plugin import")),
        patch(
            "swarm_arena._planning.catalog_check.find_entry",
            side_effect=AssertionError("catalog import"),
        ),
    ):
        names = group_judges(plan, candidates=[c for c, _ in cs], scores=[s for s, _ in ss])
        assert names == {None: "main", "a": "inherited", "b": "other"}
        assert len(flatten(group_rows(cs, ss, judge=names))) == 3
        assert group_judges(plan) == {}


def test_ambiguous_or_absent_recorded_judge_is_omitted() -> None:
    plan = Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[Ref(path="absent:Judge")])
    cs = [_candidate("c", 1)[0]]
    ss = [_score("c", 1, 2, judge="a")[0], _score("c", 2, 3, judge="b")[0]]
    assert group_judges(plan, candidates=cs, scores=ss) == {}
    assert (
        group_judges(plan, candidates=cs, scores=[_score("c", None, 4, judge="human:alice")[0]])
        == {}
    )
    assert group_judges(plan.model_copy(update={"judges": []}), candidates=cs, scores=ss[:1]) == {}
    assert (
        group_judges(
            plan.model_copy(update={"judges": [Ref(path="a:J"), Ref(path="b:J")]}),
            candidates=cs,
            scores=ss[:1],
        )
        == {}
    )


def test_format_value() -> None:
    assert format_value(0.0041) == "0.0041"
    assert format_value(None) == "-"
    assert format_value(12345) == "1.234e+04"


def test_one_judge_identity_only() -> None:
    cs = [_candidate("c1", 1), _candidate("c2", 2)]
    ss = [_score("c1", 1, 3, rubric="r1"), _score("c2", 2, 4, rubric="r2", panel="p2")]
    rows = flatten(group_rows(cs, list(reversed(ss)), judge="j"))
    assert [(r.candidate_id, r.rubric_hash, r.panel_hash) for r in rows] == [("c2", "r2", "p2")]


def test_run_identity_preferred() -> None:
    ss = [_score("c1", 1, 3, run="r1", rubric="old"), _score("c2", 2, 4, run="r2", rubric="new")]
    assert pick_identity(ss, "j", run_id="r1") == ("j", "old", "")
    assert pick_identity(ss, "j", run_id="absent") == ("j", "new", "")
    assert pick_identity(ss, "missing", run_id="r1") is None
    assert pick_identity([], "j", run_id=None) is None
    cs = [_candidate("c1", 1), _candidate("c2", 2, run="r2")]
    assert [r.candidate_id for r in flatten(group_rows(cs, ss, judge="j", identity_run="r1"))] == [
        "c1"
    ]


def test_run_filter_and_identity_run_are_separate() -> None:
    cs = [_candidate("c-1", 1), _candidate("c-2", 2, run="r2")]
    ss = [
        _score("c-1", 1, 3, rubric="new"),
        _score("c-2", 2, 4, run="r2", rubric="new"),
        _score("c-1", 3, 5, rubric="r1-latest"),
    ]
    assert sorted(
        r.candidate_id for r in flatten(group_rows(cs, ss, judge="j", identity_run="r2"))
    ) == ["c-1", "c-2"]
    assert [
        r.candidate_id
        for r in flatten(group_rows(cs, ss, judge="j", run_id="r1", identity_run="r2"))
    ] == ["c-1"]
    assert {r.rubric_hash for r in flatten(group_rows(cs, ss, judge="j", identity_run="r2"))} == {
        "new"
    }


def test_newest_fidelity_only() -> None:
    cs = [_candidate("old", 1, fidelity="e10"), _candidate("new", 2, round=2, fidelity="e100")]
    ss = [_score(c.id, 1, i) for i, (c, _) in enumerate(cs)]
    assert [r.candidate_id for r in flatten(group_rows(cs, ss, judge="j"))] == ["new"]
    assert [r.candidate_id for r in flatten(group_rows(cs, ss, judge="j", fidelity="e10"))] == [
        "old"
    ]


def test_human_ranks_all_fidelities() -> None:
    cs = [_candidate("old", 1, fidelity="e10"), _candidate("new", 2, round=2, fidelity="e100")]
    ss = [_score(c.id, None, i, judge="human:alice") for i, (c, _) in enumerate(cs)]
    rows = flatten(group_rows(cs, ss, judge="human:alice"))
    assert [r.candidate_id for r in rows] == ["old", "new"]
    assert all(r.value is None for r in rows)


def test_partial_and_gated_out_excluded() -> None:
    cs = [
        _candidate("partial", 1, partial=True),
        _candidate("gated", 2, passed=False),
        _candidate("ok", 3),
    ]
    ss = [_score(c.id, 1, i) for i, (c, _) in enumerate(cs)]
    assert [r.candidate_id for r in flatten(group_rows(cs, ss, judge="j"))] == ["ok"]


def test_groups_ranked_separately() -> None:
    cs = [
        _candidate("a", 1, part="a"),
        _candidate("m1", 2),
        _candidate("m2", 3),
        _candidate("b", 4, part="b"),
    ]
    ss = [_score(c.id, 1, i) for i, (c, _) in enumerate(cs)]
    groups = group_rows(cs, ss, judge="j")
    assert [r.rank for r in groups[None]] == [1, 2]
    assert [r.rank for r in groups["a"]] == [1]
    assert [r.part for r in flatten(dict(reversed(list(groups.items()))))] == [None, None, "a", "b"]
    assert group_rows(cs, [], judge="j") == {}
    assert group_rows([], ss, judge="j") == {}


def test_latest_score_per_candidate_wins() -> None:
    cs = [_candidate("a", 1), _candidate("b", 2)]
    ss = [_score("a", 3, 5), _score("b", 2, 4), _score("a", 1, 3)]
    rows = flatten(group_rows(cs, ss, judge="j"))
    assert [(r.candidate_id, r.value) for r in rows] == [("b", 2), ("a", 3)]


def test_row_fields() -> None:
    cs = [_candidate("c", 1, part="a", round=3)]
    ss = [_score("c", 0.1, 2, panel="panel")]
    row = flatten(group_rows(cs, ss, judge="j"))[0]
    assert row.model_dump() == dict(
        rank=1,
        candidate_id="c",
        run_id="r1",
        goal_id="g",
        part="a",
        team="t",
        agent="a",
        fidelity="f1",
        key=(-0.1,),
        value=0.1,
        judge="j",
        rubric_hash="rh",
        panel_hash="panel",
        built_on=["parent"],
        partial=False,
    )


@pytest.mark.parametrize("changed", ["inputs", "manifest"])
def test_producing_key_selection_precedes_identity_and_fidelity(
    tmp_path: Path, changed: str
) -> None:
    run = FakeRun(str(tmp_path / "runs"), "r1")
    old = run.plan(parts=["a"], first_round=1)
    old_plan = Plan.model_validate(old.payload["plan"])
    new = run.plan(revision=1, parts=["a"], first_round=2)
    new_plan = Plan.model_validate(new.payload["plan"])
    part_goal = new_plan.parts[0].goal.model_copy(
        update=(
            {"inputs": {"x": 2}}
            if changed == "inputs"
            else {"files": FileSet(manifest={"x.txt": "a" * 64})}
        )
    )
    new_plan = new_plan.model_copy(
        update={"parts": [new_plan.parts[0].model_copy(update={"goal": part_goal})]}
    )
    new.payload["plan"] = new_plan.model_dump(mode="json")
    head = run.commit(
        old,
        run.candidate("old", part="a", round=1, fidelity="obsolete"),
        new,
        run.candidate("current", part="a", round=2, fidelity="current"),
        run.score("s-current", "current", 2, rubric_hash="current"),
        run.score("s-old", "old", 1, rubric_hash="obsolete"),
    )
    cache = ArenaCache(tmp_path / "toy.sqlite")
    try:
        ingest_run(cache, run.store, "r1", run.run_root, head)
        frontier = Frontier(
            marks=[
                Frontier.Mark(
                    run_id="r1",
                    run_root=run.run_root,
                    epoch=head.epoch,
                    seq=head.seq,
                    segment=head.segment,
                )
            ]
        )
        scores = cache.scores(frontier)
        for plan, expected in [(new_plan, "current"), (old_plan, "old")]:
            selected = cache.candidates(
                frontier, part_key=goal_key_digest(plan.parts[0].goal.key())
            )
            rows = flatten(group_rows(selected, scores, judge="j"))
            assert [r.candidate_id for r in rows] == [expected]
            assert rows[0].fidelity == ("obsolete" if expected == "old" else "current")
            assert rows[0].rubric_hash == rows[0].fidelity
    finally:
        cache.close()


@pytest.mark.parametrize("mapping", [False, True])
def test_human_judges_rank_all_fidelities_without_plan_margin(mapping: bool) -> None:
    cs = [
        _candidate("c-cheap", 1, fidelity="cheap"),
        _candidate("c-full", 2, fidelity="full", round=2),
    ]
    ss = [
        _score("c-cheap", 9, 3, judge="human:alice", key=(1, 9)),
        _score("c-full", 1, 4, judge="human:alice", key=(1, 1)),
    ]
    judge = {None: "human:alice"} if mapping else "human:alice"
    rows = flatten(group_rows(cs, ss, judge=judge, min_improvement=0.5))
    assert [r.candidate_id for r in rows] == ["c-cheap", "c-full"]
    # With equal verdicts a later higher numeric human score must replace first place.
    ss[1] = _score("c-full", 10, 4, judge="human:alice", key=(1, 10))
    assert flatten(group_rows(cs, ss, judge=judge, min_improvement=0.5))[0].value == 10


def test_mixed_mapping_preserves_program_fidelity_and_margin() -> None:
    cs = [
        _candidate("c-human-cheap", 1, fidelity="cheap"),
        _candidate("c-human-full", 2, fidelity="full", round=2),
        _candidate("c-part-cheap", 3, part="a", fidelity="cheap"),
        _candidate("c-part-full", 4, part="a", fidelity="full", round=2),
    ]
    ss = [
        _score(
            c.id,
            10 if c.fidelity == "cheap" else 1,
            i + 5,
            judge="human:alice" if c.part is None else "j",
            key=(1,),
        )
        for i, (c, _) in enumerate(cs)
    ]
    ranked = group_rows(cs, ss, judge={None: "human:alice", "a": "j"})
    assert len(ranked[None]) == 2
    assert [r.candidate_id for r in ranked["a"]] == ["c-part-full"]
