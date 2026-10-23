"""Public autonomous runs from copied human script plans and fixed fake drafts."""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from _fakes.planners import SequencePlanner
from _fakes.planning_llm import planner_config, planner_output
from _fakes.run import events, run_dir

from swarm_arena import ConvergeSpec, Limits, PartSpec, Plan, Ref, RunConfig, Swarm
from swarm_arena.llm import LLMPlanner
from swarm_arena.planning import Planner
from swarm_arena.testing import FakeModel, FakeTurn

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "own_script"


def setup(root: Path) -> tuple[Plan, RunConfig]:
    source = root / "source"
    shutil.copytree(EXAMPLE, source)
    (source / "judge/hidden/extra.npz").write_bytes(b"private-judge-file")
    text = (
        (source / "plan.toml")
        .read_text()
        .replace('"judge/hidden/*.csv"', '"judge/hidden/*.csv", "judge/hidden/*.npz"')
    )
    (source / "plan.toml").write_text(text)
    plan = Plan.load(source / "plan.toml")
    config = planner_config(root)
    config = config.model_copy(
        update={
            "limits": Limits(
                max_agents_per_fingerprint=4,
                max_rounds=2,
                max_tokens=1_000_000,
                default_run_reservation={"tokens": 0, "usd": 0.0},
            )
        }
    )
    return plan, config


def output() -> dict[str, Any]:
    result = planner_output()
    result.update(teams=[], keep_teams=["fit"], rounds=[])
    return result


async def _assert_rejected(tmp_path: Path, bad: str) -> None:
    plan, config = setup(tmp_path)
    draft = output()
    if bad in {"unknown", "fixed"}:
        draft["teams"] = planner_output()["teams"]
        draft["teams"][0]["name"] = "new"
        tool = draft["teams"][0]["agents"][0]["tool"]
        if bad == "unknown":
            tool["entry"] = "evil-tool"
        else:
            tool["settings"] = {"command": ["not-allowed"]}
    if bad == "hidden":
        draft["parts"] = [{"name": "p", "files": ["judge/hidden/*.npz"]}]
    fake = FakeModel([FakeTurn(text=json.dumps(draft))])
    fake.register("planner")
    replanner: Planner = LLMPlanner("planner")
    if bad == "raw":
        sys.modules.pop("_fakes.import_canary", None)
        team = plan.teams[0]
        agent = team.agents[0].model_copy(update={"tool": Ref(path="_fakes.import_canary:Tool")})
        replanner = SequencePlanner(
            [plan.model_copy(update={"teams": [team.model_copy(update={"agents": [agent]})]})]
        )
    await Swarm(config, replanner=replanner).run(plan)
    assert [e.payload["plan"]["revision"] for e in events(config, "plan_revision")] == [0]
    assert events(config, "plan_rejected")
    assert (
        json.loads((run_dir(config) / "goals/line-fit/solution.json").read_text())["status"] == "ok"
    )
    if bad == "raw":
        assert "_fakes.import_canary" not in sys.modules


async def _assert_kept(tmp_path: Path, part: bool) -> None:
    plan, config = setup(tmp_path)
    draft = output()
    if part:
        component = PartSpec(
            name="p1", goal=plan.goal.model_copy(update={"id": "line-fit.p1"}), checks=plan.checks
        )
        plan = plan.model_copy(
            update={
                "parts": [component],
                "teams": [plan.teams[0].model_copy(update={"part": "p1"})],
                "converge": ConvergeSpec(
                    mode="team", team=plan.teams[0].model_copy(update={"name": "join"})
                ),
            }
        )
        draft["keep_parts"] = ["p1"]
        draft["teams"] = planner_output()["teams"]
        draft["teams"][0].update(name="catalog", part="p1")
    FakeModel([FakeTurn(text=json.dumps(draft))]).register("planner")
    await Swarm(config, replanner=LLMPlanner("planner")).run(plan)
    revisions = events(config, "plan_revision")
    assert [e.payload["plan"]["revision"] for e in revisions] == [0, 1]
    r0, r1 = [Plan.model_validate(e.payload["plan"]) for e in revisions]
    assert r1.origin == "autonomous"
    assert r1.judges == r0.judges
    assert r1.teams[0] == r0.teams[0]
    assert revisions[1].payload["planner_call_ids"]
    if part:
        assert r1.parts == r0.parts


async def test_replan_naming_a_non_catalog_entry_is_rejected(tmp_path: Path) -> None:
    await _assert_rejected(tmp_path, "unknown")


async def test_replan_with_settings_outside_settable_is_rejected(tmp_path: Path) -> None:
    await _assert_rejected(tmp_path, "fixed")


async def test_raw_non_catalog_ref_from_a_replanner_is_rejected(tmp_path: Path) -> None:
    await _assert_rejected(tmp_path, "raw")


async def test_replan_part_with_hidden_judge_files_is_rejected(tmp_path: Path) -> None:
    await _assert_rejected(tmp_path, "hidden")


async def test_replan_keeping_the_human_scriptjudge_is_accepted(tmp_path: Path) -> None:
    await _assert_kept(tmp_path, False)


async def test_replan_keeping_the_human_team_and_a_checked_part_is_accepted(tmp_path: Path) -> None:
    await _assert_kept(tmp_path, True)


async def test_autonomous_revision_clears_external_planning_session(tmp_path: Path) -> None:
    plan, config = setup(tmp_path)
    plan._set_planner_provenance("r-external", ("llm-external",))

    class CopyPlanner(Planner):
        async def plan(self, goal: Any, config: RunConfig, history: Any) -> Plan:
            return history.revisions[-1].model_copy()

    await Swarm(config, replanner=CopyPlanner()).run(plan)
    revisions = events(config, "plan_revision")
    assert [e.payload["plan"]["revision"] for e in revisions] == [0, 1]
    assert revisions[0].payload["planning_session"] == "r-external"
    assert revisions[1].payload["planning_session"] is None
    assert revisions[1].payload["planner_call_ids"] == []
