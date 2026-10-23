"""Catalog-only drafts, immutable human goals and owned planner accounting."""

import asyncio
import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.llm_components import LateFakeModel
from _fakes.llm_run import open_llm_run
from _fakes.planning_llm import planner_config, planner_output
from _fakes.run import all_events
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Goal, Limits, PartSpec, Plan, Ref, RoundSpec
from swarm_arena._budget.llm_scope import llm_scope
from swarm_arena._model.planning_types import History
from swarm_arena._model.tooling import Estimate
from swarm_arena.llm import Approve, CollaborativePlanner, Edited, Feedback, LLMPlanner
from swarm_arena.llm._planner import PLANNER_SCHEMA, plan_from_output
from swarm_arena.testing import FakeModel, FakeTurn

GOAL = Goal(id="g", namespace="n")
SECRET = "secret-sentinel-planner-value"


class _Reviews:
    def __init__(self, decisions: list[Any]) -> None:
        self.decisions = iter(decisions)

    async def review(self, plan: Plan) -> Any:
        decision = next(self.decisions)
        return decision(plan) if callable(decision) else decision


async def test_loops_until_approve_and_sends_feedback(tmp_path: Path) -> None:
    fake = FakeModel([FakeTurn(text=json.dumps(planner_output())) for _ in range(3)])
    fake.register("planner")
    config = planner_config(tmp_path)
    planner = CollaborativePlanner(
        LLMPlanner("planner"),
        _Reviews([Feedback("add a spectral team"), Feedback("retain it"), Approve()]),
    )
    assert planner.models() == ("planner",)
    result = await planner.plan(GOAL, config, None)
    assert len(fake.calls) == 3
    prompt = json.dumps(fake.calls[2]["input"])
    assert "add a spectral team" in prompt
    assert "retain it" in prompt
    assert "Explore the catalog" in prompt
    assert result.origin == "together"
    assert len(result.planner_call_ids) == 3
    assert result.planning_session
    assert len([e for e in all_events(config) if e.type == "run_started"]) == 1
    assert len([e for e in all_events(config) if e.type == "goal_finished"]) == 1


async def test_edited_plan_is_stamped_together(tmp_path: Path) -> None:
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")

    def edit(plan: Plan) -> Edited:
        plan._base_dir = tmp_path
        return Edited(plan.model_copy(update={"origin": "human", "rationale": "human edit"}))

    result = await CollaborativePlanner(LLMPlanner("planner"), _Reviews([edit])).plan(
        GOAL, planner_config(tmp_path), None
    )
    assert result.origin == "together"
    assert result.rationale == "human edit"
    assert result.base_dir == tmp_path
    assert result.planning_session


@pytest.mark.parametrize("edited", [False, True])
async def test_plan_at_preserves_non_cwd_source_directory(tmp_path: Path, edited: bool) -> None:
    from swarm_arena.tool import FileSet

    source = tmp_path / "source"
    await anyio.Path(source).mkdir()
    await anyio.Path(source / "input.txt").write_text("human source bytes")
    assert source != Path.cwd()
    goal = GOAL.model_copy(update={"files": FileSet(globs=["input.txt"])})
    goal_file = source / "goal.toml"
    await anyio.Path(goal_file).write_text(Plan(goal=goal, teams=[], judges=[]).dumps())
    loaded = await anyio.to_thread.run_sync(Plan.load, goal_file)
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")

    def review(plan: Plan) -> Approve | Edited:
        assert plan.base_dir == source
        assert plan.goal.files.snapshotted
        if not edited:
            return Approve()
        proposed = Plan.loads(plan.dumps(), base_dir=source).model_copy(
            update={"rationale": "edited away from cwd"}
        )
        return Edited(proposed)

    result = await CollaborativePlanner(LLMPlanner("planner"), _Reviews([review]))._plan_at(
        loaded.goal, planner_config(tmp_path), None, base_dir=loaded.base_dir
    )
    assert result.base_dir == source
    assert result.goal.files.snapshotted
    assert result.origin == "together"
    assert result.planning_session
    if edited:
        assert result.rationale == "edited away from cwd"


@pytest.mark.parametrize(
    "field",
    ["id", "namespace", "title", "description", "metadata", "inputs", "files", "file_globs", "ref"],
)
async def test_edited_plan_still_meets_the_catalog_boundary(
    tmp_path: Path, field: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from swarm_arena.tool import FileSet

    fake = FakeModel([FakeTurn(text=json.dumps(planner_output()))])
    fake.register("planner")

    def edit(plan: Plan) -> Edited:
        if field == "ref":
            plan.teams[0].agents[0] = (
                plan.teams[0].agents[0].model_copy(update={"tool": Ref(path="untrusted:Tool")})
            )
            return Edited(plan)
        if field in {"files", "file_globs"}:
            value: Any = (
                FileSet(manifest={"other": "a" * 64})
                if field == "files"
                else FileSet(globs=["changed"])
            )
            field_name = "files"
        else:
            value = {"changed": SECRET} if field in {"inputs", "metadata"} else "changed"
            field_name = field
        return Edited(
            plan.model_copy(update={"goal": plan.goal.model_copy(update={field_name: value})})
        )

    config = planner_config(tmp_path)
    result = await CollaborativePlanner(LLMPlanner("planner"), _Reviews([edit, Approve()])).plan(
        GOAL, config, None
    )
    assert result.goal == GOAL
    assert len(fake.calls) == 1
    printed = capsys.readouterr().out
    assert "edited plan admission refused" in printed
    assert SECRET not in printed
    assert len([e for e in all_events(config) if e.type == "goal_finished"]) == 1


@pytest.mark.parametrize("limit", [1, 2, 5])
async def test_no_approval_after_max_iterations(tmp_path: Path, limit: int) -> None:
    fake = FakeModel([FakeTurn(text=json.dumps(planner_output())) for _ in range(limit)])
    fake.register("planner")
    with pytest.raises(
        RuntimeError, match=f"plan not approved after {limit} review" + ("s" if limit != 1 else "$")
    ):
        await CollaborativePlanner(
            LLMPlanner("planner"), _Reviews([Feedback("again")] * limit), max_iterations=limit
        ).plan(GOAL, planner_config(tmp_path), None)
    assert len(fake.calls) == limit


async def test_rejected_edits_still_exhaust_the_review_limit(tmp_path: Path) -> None:
    fake = FakeModel([FakeTurn(text=json.dumps(planner_output()))])
    fake.register("planner")

    def edit(plan: Plan) -> Edited:
        return Edited(
            plan.model_copy(update={"goal": plan.goal.model_copy(update={"title": SECRET})})
        )

    with pytest.raises(RuntimeError, match="plan not approved after 2 reviews"):
        await CollaborativePlanner(
            LLMPlanner("planner"), _Reviews([edit, edit]), max_iterations=2
        ).plan(GOAL, planner_config(tmp_path), None)
    assert len(fake.calls) == 1


@pytest.mark.parametrize("site", ["detach", "admission"])
async def test_human_failure_labels_are_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    from swarm_arena.llm import _planner

    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    if site == "detach":

        def detach(decision: object) -> Any:
            raise ValueError(SECRET)

        monkeypatch.setattr(_planner, "_detach_decision", detach)
        label = "review decision detach failed: ValueError"
    else:
        original_admit = _planner._admit_draft

        async def admit(*args: Any, **kwargs: Any) -> Plan:
            if kwargs.get("origin") == "together":
                raise RuntimeError(SECRET)
            return await original_admit(*args, **kwargs)

        monkeypatch.setattr(_planner, "_admit_draft", admit)
        label = "plan admission failed: RuntimeError"
    with pytest.raises(RuntimeError, match=label) as caught:
        await CollaborativePlanner(LLMPlanner("planner"), _Reviews([Approve()])).plan(
            GOAL, planner_config(tmp_path), None
        )
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("site", ["admission", "diagnostic"])
@pytest.mark.parametrize("kind", ["fenced", "terminal"])
async def test_admission_and_diagnostic_controls_never_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str, kind: str
) -> None:
    from swarm_arena._errors import Fenced, _core_error, _stop_cleanup_error
    from swarm_arena.llm import _planner

    error = _core_error(Fenced, "run ownership lost") if kind == "fenced" else _stop_cleanup_error()
    original_admit = _planner._admit_draft

    async def admit(*args: Any, **kwargs: Any) -> Plan:
        if kwargs.get("origin") == "together":
            raise error
        return await original_admit(*args, **kwargs)

    def report(*args: Any) -> None:
        raise error

    def edit(plan: Plan) -> Edited:
        return Edited(
            plan.model_copy(update={"goal": plan.goal.model_copy(update={"title": "changed"})})
        )

    monkeypatch.setattr(
        _planner,
        "_admit_draft" if site == "admission" else "_report_rejection",
        admit if site == "admission" else report,
    )
    fake = FakeModel([FakeTurn(text=json.dumps(planner_output()))])
    fake.register("planner")
    decisions = [Approve(), Approve()] if site == "admission" else [edit, Approve()]
    with pytest.raises(Fenced if kind == "fenced" else RuntimeError):
        await CollaborativePlanner(LLMPlanner("planner"), _Reviews(decisions)).plan(
            GOAL, planner_config(tmp_path), None
        )
    assert len(fake.calls) == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_review_limit_is_a_positive_integer(limit: Any) -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        CollaborativePlanner(LLMPlanner("planner"), _Reviews([]), max_iterations=limit)


async def test_reviewer_mutation_cannot_change_approved_draft(tmp_path: Path) -> None:
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")

    def change(plan: Plan) -> Approve:
        plan.goal.inputs["secret"] = SECRET
        return Approve()

    result = await CollaborativePlanner(LLMPlanner("planner"), _Reviews([change])).plan(
        GOAL, planner_config(tmp_path), None
    )
    assert not result.goal.inputs


async def test_reviewer_returned_values_are_detached_off_loop(tmp_path: Path) -> None:
    import threading

    class WorkerFeedback(Feedback):
        def __getattribute__(self, name: str) -> Any:
            if name == "text":
                assert threading.current_thread() is not threading.main_thread()
            return super().__getattribute__(name)

    FakeModel([FakeTurn(text=json.dumps(planner_output())) for _ in range(2)]).register("planner")
    result = await CollaborativePlanner(
        LLMPlanner("planner"), _Reviews([WorkerFeedback("again"), Approve()])
    ).plan(GOAL, planner_config(tmp_path), None)
    assert result.origin == "together"


@pytest.mark.parametrize("kind", ["terminal", "fenced", "forged_cancel", "failure"])
async def test_reviewer_failure_closes_session(tmp_path: Path, kind: str) -> None:
    from swarm_arena._errors import Fenced, _core_error, _stop_cleanup_error
    from swarm_arena._journal import llm_record

    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")

    def fail(plan: Plan) -> None:
        raise {
            "terminal": _stop_cleanup_error(),
            "fenced": _core_error(Fenced, "run ownership lost"),
            "forged_cancel": asyncio.CancelledError(SECRET),
            "failure": ValueError(SECRET),
        }[kind]

    config = planner_config(tmp_path)
    with pytest.raises(Fenced if kind == "fenced" else RuntimeError) as caught:
        await CollaborativePlanner(LLMPlanner("planner"), _Reviews([fail])).plan(GOAL, config, None)
    assert_secret_absent(caught.value, SECRET)
    assert len([e for e in all_events(config) if e.type == "goal_finished"]) == (kind != "fenced")
    sessions = await anyio.to_thread.run_sync(lambda: list(Path(config.run_root).iterdir()))
    assert not any(p.name in llm_record._bindings for p in sessions)


@pytest.mark.parametrize("mode", ["cancel", "deadline", "eof"])
async def test_terminal_session_exits_without_resurrection(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import os

    from swarm_arena._engine import run as run_module
    from swarm_arena._engine.stops import RunClock
    from swarm_arena._journal import llm_record
    from swarm_arena.llm._terminal_reviewer import TerminalReviewer

    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    config = planner_config(tmp_path)
    expired = anyio.Event()

    class ControlledClock:
        def __getattr__(self, name: str) -> Any:
            return getattr(anyio, name)

        async def sleep(self, seconds: float) -> None:
            if seconds == 123456.0:
                await expired.wait()
            else:
                await anyio.sleep(seconds)

    if mode == "deadline":
        monkeypatch.setattr(RunClock, "left", lambda self, limit: 123456.0)
        monkeypatch.setattr(run_module, "anyio", ControlledClock())
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd) as stdin:
        reviewer = TerminalReviewer(stdin=stdin, stdout=io.StringIO())
        task = asyncio.create_task(
            CollaborativePlanner(LLMPlanner("planner"), reviewer).plan(GOAL, config, None)
        )
        try:
            with anyio.fail_after(15):
                while os.get_blocking(read_fd) and not task.done():
                    await anyio.lowlevel.checkpoint()
                assert not task.done()
                if mode == "cancel":
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                else:
                    if mode == "deadline":
                        expired.set()
                    if mode == "eof":
                        os.close(write_fd)
                        write_fd = -1
                    with pytest.raises(
                        RuntimeError, match=r"terminal input ended|planning session stopped"
                    ):
                        await task
            assert os.get_blocking(read_fd)
            assert reviewer._edit_bytes == b""
            assert reviewer._draft_text == ""
            assert not reviewer._retry_edit
            events = all_events(config)
            assert len([e for e in events if e.type == "goal_finished"]) == 1
            if mode == "deadline":
                assert [e.payload["reason"] for e in events if e.type == "goal_finished"] == [
                    "run_timeout_s"
                ]
            sessions = await anyio.to_thread.run_sync(lambda: list(Path(config.run_root).iterdir()))
            assert not any(p.name in llm_record._bindings for p in sessions)
            if write_fd >= 0:
                os.write(write_fd, b"a\n")
            assert all_events(config) == events
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if write_fd >= 0:
                os.close(write_fd)


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(v) for v in value))
    return set()


def test_output_schema_has_no_provenance_fields() -> None:
    assert not {"origin", "revision", "parent_revision"} & _keys(PLANNER_SCHEMA)


def test_models() -> None:
    assert LLMPlanner("planner").models() == ("planner",)


def _convert(
    tmp_path: Path, output: dict[str, Any], current: Plan | None = None, rounds_done: int = 0
) -> Plan:
    return plan_from_output(
        output,
        goal=GOAL,
        catalog=planner_config(tmp_path).planner_catalog,
        current=current,
        rounds_done=rounds_done,
        limits=Limits(max_rounds=3),
    )


async def test_prompt_lists_catalog_entries_with_settable_schemas(tmp_path: Path) -> None:
    fake = FakeModel([FakeTurn(text=json.dumps(planner_output()))])
    fake.register("planner")
    await LLMPlanner("planner").plan(GOAL, planner_config(tmp_path), None)
    prompt = json.dumps(fake.calls[0], default=str)
    assert all(
        s in prompt
        for s in (
            "sdk",
            "Tune the optimizer",
            "instructions",
            "string",
            "opus",
            "manager",
            "max_tokens",
        )
    )


async def test_builds_refs_from_entries_and_settable_values(tmp_path: Path) -> None:
    output = planner_output()
    output["teams"][0]["agents"][0]["manager"] = {
        "entry": "sdk",
        "settings": {"instructions": "Tune lr."},
    }
    FakeModel([FakeTurn(text=json.dumps(output))]).register("planner")
    plan = await LLMPlanner("planner").plan(GOAL, planner_config(tmp_path), None)
    assert plan.teams[0].agents[0].manager == Ref(
        path="swarm_arena.llm:AgentsSDKManager",
        config={"model": "opus", "instructions": "Tune lr."},
    )


async def _assert_stamped(tmp_path: Path, spoof: bool) -> None:
    output = planner_output()
    if spoof:
        output.update(origin="human", revision=99, parent_revision=98)
    FakeModel([FakeTurn(text=json.dumps(output))]).register("planner")
    plan = await LLMPlanner("planner").plan(GOAL, planner_config(tmp_path), None)
    assert (plan.origin, plan.revision, plan.parent_revision) == ("llm", 0, None)


async def test_plan_outside_a_swarm_opens_a_planning_session(tmp_path: Path) -> None:
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    config = planner_config(tmp_path)
    plan = await LLMPlanner("planner").plan(GOAL, config, None)
    (session,) = await anyio.to_thread.run_sync(lambda: list(Path(config.run_root).iterdir()))
    events = all_events(config)
    calls = [e.payload["call"] for e in events if e.type == "llm_call"]
    assert [c["role"] for c in calls] == ["planner"]
    assert (plan.planning_session, plan.planner_call_ids) == (session.name, (calls[0]["id"],))


def test_unknown_entry_raises_value_error(tmp_path: Path) -> None:
    output = planner_output()
    output["teams"][0]["agents"][0]["tool"]["entry"] = "evil-tool"
    with pytest.raises(ValueError, match="evil-tool"):
        _convert(tmp_path, output)


@pytest.mark.parametrize(
    "bad",
    [
        {"entry": "judge", "settings": {}},
        {"entry": "tool", "settings": {"command": [SECRET]}},
        {"entry": "tool", "settings": []},
    ],
)
def test_entry_kind_and_fixed_settings_are_checked_before_ref(tmp_path: Path, bad: Any) -> None:
    output = planner_output()
    output["teams"][0]["agents"][0]["tool"] = bad
    with pytest.raises(ValueError, match=r"catalog|invalid|planner") as caught:
        _convert(tmp_path, output)
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("teams", [True]),
        ("parts", [{}]),
        ("rounds", [False]),
        ("keep_teams", [True]),
        ("converge", {"mode": "select", "top_n": True}),
    ],
)
def test_nested_output_shapes_are_strict(tmp_path: Path, key: str, bad: Any) -> None:
    output = planner_output()
    output[key] = bad
    with pytest.raises(ValueError, match=r"catalog|invalid|planner"):
        _convert(tmp_path, output)


def test_settable_values_are_validated_without_values_in_errors(tmp_path: Path) -> None:
    output = planner_output()
    output["teams"][0]["agents"][0]["manager"] = {
        "entry": "sdk",
        "settings": {"instructions": {"secret": SECRET}},
    }
    with pytest.raises(ValueError, match=r"catalog|invalid|planner") as caught:
        _convert(tmp_path, output)
    assert_secret_absent(caught.value, SECRET)


def test_replan_keeps_plan_level_checks_judges_ranking_and_policy(tmp_path: Path) -> None:
    current = _convert(tmp_path, planner_output()).model_copy(
        update={
            "checks": [Ref(path="human:Check")],
            "judges": [Ref(path="human:Judge")],
            "ranking": "human",
            "policy": Ref(path="human:Policy"),
        }
    )
    output = planner_output()
    output.update(
        checks=[{"entry": "check", "settings": {}}],
        judges=[{"entry": "ignored", "settings": {}}],
        policy={"entry": "ignored", "settings": {}},
    )
    result = _convert(tmp_path, output, current)
    assert result.checks == [*current.checks, Ref(path="_fakes.tools:AlwaysPass")]
    assert (result.judges, result.ranking, result.policy) == (
        current.judges,
        current.ranking,
        current.policy,
    )


def test_replan_keeps_listed_teams_and_parts_unchanged(tmp_path: Path) -> None:
    current = _convert(tmp_path, planner_output())
    part = PartSpec(name="p1", goal=Goal(id="p", namespace="n"), checks=[Ref(path="human:Check")])
    team = current.teams[0].model_copy(update={"part": "p1"})
    current = current.model_copy(update={"teams": [team], "parts": [part]})
    output = planner_output()
    output.update(keep_teams=["fit"], keep_parts=["p1"], teams=[])
    result = _convert(tmp_path, output, current)
    assert result.teams[0] is team
    assert result.parts[0] is part


@pytest.mark.parametrize("rounds_done", [1, 2])
def test_replan_from_implicit_rounds_keeps_completed_rounds(
    tmp_path: Path, rounds_done: int
) -> None:
    current = _convert(tmp_path, planner_output()).model_copy(update={"rounds": []})
    result = _convert(tmp_path, planner_output(), current, rounds_done)
    assert result.rounds == [RoundSpec()] * rounds_done + [RoundSpec(fidelity="full")]
    assert result.rounds_total(Limits(max_rounds=3)) == rounds_done + 1


def test_replan_without_output_rounds_keeps_the_schedule(tmp_path: Path) -> None:
    current = _convert(tmp_path, planner_output())
    output = planner_output()
    output["rounds"] = []
    assert _convert(tmp_path, output, current, 1).rounds == current.rounds


@pytest.mark.parametrize(
    "bad",
    [
        {"keep_teams": ["nope"]},
        {"keep_parts": ["p1"]},
        {"keep_teams": ["fit"]},
        {"keep_teams": ["fit", "fit"], "teams": []},
    ],
)
def test_bad_keep_lists_raise(tmp_path: Path, bad: dict[str, Any]) -> None:
    current = _convert(tmp_path, planner_output())
    output = planner_output() | bad
    with pytest.raises(ValueError, match=r"catalog|invalid|planner"):
        _convert(tmp_path, output, current)
    with pytest.raises(ValueError, match=r"catalog|invalid|planner"):
        _convert(tmp_path, output)


async def test_replan_prompt_includes_join_failures(tmp_path: Path) -> None:
    current = _convert(tmp_path, planner_output())
    current._base_dir = tmp_path
    history = History(
        goal_id="g",
        revisions=[current],
        scores=[],
        learnings=[],
        decisions=[],
        budget_left=Estimate(tokens=10000),
        rounds_done=1,
        join_failures=["graft broke ring"],
    )
    fake = FakeModel([FakeTurn(text=json.dumps(planner_output()))])
    fake.register("planner")
    await LLMPlanner("planner").plan(GOAL, planner_config(tmp_path), history)
    assert "graft broke ring" in json.dumps(fake.calls[0], default=str)


async def test_model_needs_planner_role(tmp_path: Path) -> None:
    config = planner_config(tmp_path)
    config.models.root["planner"] = config.models.root["planner"].model_copy(
        update={"roles": ["judge"]}
    )
    fake = FakeModel([])
    fake.register("planner")
    with pytest.raises(ValueError, match=r"catalog|invalid|planner"):
        await LLMPlanner("planner").plan(GOAL, config, None)
    assert fake.calls == []
    assert not await anyio.Path(config.run_root).exists()


@pytest.mark.parametrize("never", [False, True])
async def test_planner_drains_before_it_returns(tmp_path: Path, never: bool) -> None:
    release = anyio.Event()
    fake = LateFakeModel([FakeTurn(text=json.dumps(planner_output()))], delay_s=0, release=release)
    fake.register("planner")
    config = planner_config(tmp_path)
    try:
        async with open_llm_run(tmp_path, config=config) as run:
            scope = run.scope("planner")

            async def call() -> Plan:
                with llm_scope(scope):
                    return await LLMPlanner("planner").plan(GOAL, config, None)

            task = asyncio.create_task(call())
            try:
                with anyio.fail_after(10):
                    await fake.ready.wait()
                    if not never:
                        release.set()
                    await task
                assert len(scope.call_ids) == len(fake.calls) == 1
                assert set(scope.call_ids) == {c.id for c in run.calls()}
                assert (run.calls()[0].error or "").startswith("estimated:") is never
                assert run.charged(scope)._micro_usd == sum(
                    round(c.cost_usd * 1_000_000) for c in run.calls()
                )
                release.set()
                await fake.flush()
                assert len(run.calls()) == 1
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
    finally:
        await fake.aclose()


async def test_planner_preserves_public_budget_refusal(tmp_path: Path) -> None:
    from swarm_arena.llm._hooks import BudgetExceeded

    config = planner_config(tmp_path)
    config = config.model_copy(update={"limits": Limits(max_tokens=1)})
    fake = FakeModel([FakeTurn(text=json.dumps(planner_output()))])
    fake.register("planner")
    with pytest.raises(BudgetExceeded):
        await LLMPlanner("planner").plan(GOAL, config, None)
    assert not fake.calls


@pytest.mark.parametrize("field", ["inputs", "files"])
async def test_generation_refuses_changed_main_goal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    from swarm_arena.llm import _planner
    from swarm_arena.tool import FileSet

    convert = _planner.plan_from_output

    def change(*args: Any, **kwargs: Any) -> Plan:
        result = convert(*args, **kwargs)
        value = {"private": SECRET} if field == "inputs" else FileSet(manifest={"other": "a" * 64})
        return result.model_copy(update={"goal": result.goal.model_copy(update={field: value})})

    monkeypatch.setattr(_planner, "plan_from_output", change)
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    config = planner_config(tmp_path)
    with pytest.raises(ValueError, match="frozen main goal") as caught:
        await LLMPlanner("planner").plan(GOAL, config, None)
    assert_secret_absent(caught.value, SECRET)
    assert len([e for e in all_events(config) if e.type == "llm_call"]) == 1


@pytest.mark.parametrize("kind", ["terminal", "fenced", "forged_cancel", "nested_terminal"])
async def test_planner_controls_use_shared_classifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    from swarm_arena._errors import Fenced, _core_error, _stop_cleanup_error
    from swarm_arena.llm import _planner

    error = {
        "terminal": _stop_cleanup_error(),
        "fenced": _core_error(Fenced, "run ownership lost"),
        "forged_cancel": asyncio.CancelledError(SECRET),
        "nested_terminal": BaseExceptionGroup(SECRET, [_stop_cleanup_error()]),
    }[kind]

    async def fail(**kwargs: Any) -> tuple[dict[str, Any], list[str]]:
        raise error

    monkeypatch.setattr(_planner, "structured_call", fail)
    config = planner_config(tmp_path)
    async with open_llm_run(tmp_path, config=config) as run:
        with llm_scope(run.scope("planner")):
            expected = Fenced if kind == "fenced" else RuntimeError
            with pytest.raises(expected) as caught:
                await LLMPlanner("planner").plan(GOAL, config, None)
            assert_secret_absent(caught.value, SECRET)


async def test_planner_caller_cancellation_drains_held_record(tmp_path: Path) -> None:
    release = anyio.Event()
    fake = LateFakeModel([FakeTurn(text=json.dumps(planner_output()))], delay_s=0, release=release)
    fake.register("planner")
    config = planner_config(tmp_path)
    async with open_llm_run(tmp_path, config=config) as run:
        scope = run.scope("planner")
        with llm_scope(scope):
            task = asyncio.create_task(LLMPlanner("planner").plan(GOAL, config, None))
        try:
            with anyio.fail_after(10):
                await fake.ready.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert scope.tranche.pending() == 0
            assert len(run.calls()) == 1
            assert (run.calls()[0].error or "").startswith("estimated:")
            release.set()
            await fake.flush()
            assert len(run.calls()) == 1
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await fake.aclose()


def test_output_round_defaults_preserve_round_spec(tmp_path: Path) -> None:
    output = planner_output()
    output["rounds"] = [{}]
    assert _convert(tmp_path, output).rounds == [RoundSpec()]


async def test_stamps_llm_origin_revision_0(tmp_path: Path) -> None:
    await _assert_stamped(tmp_path, False)


async def test_output_origin_human_is_overwritten(tmp_path: Path) -> None:
    await _assert_stamped(tmp_path, True)


@pytest.mark.parametrize("failure", ["malformed", "limit", "overflow"])
async def test_standalone_failure_records_finished_outcome(tmp_path: Path, failure: str) -> None:
    from swarm_arena._errors import _core_category, _SignalCategory
    from swarm_arena._journal.replay import RunState
    from swarm_arena.manager import LimitReached
    from swarm_arena.tool import Usage

    config = planner_config(tmp_path)
    if failure == "limit":
        config = config.model_copy(update={"limits": Limits(max_tokens=10)})
    fake = FakeModel(
        [
            FakeTurn(
                text="not-json" if failure == "malformed" else json.dumps(planner_output()),
                usage=Usage(input_tokens=10**18 if failure == "overflow" else 1),
            )
        ]
    )
    fake.register("planner")
    with pytest.raises((ValueError, LimitReached)) as caught:
        await LLMPlanner("planner").plan(GOAL, config, None)
    if failure == "overflow":
        assert _core_category(caught.value) is _SignalCategory.TERMINAL
    events = all_events(config)
    finished = [e for e in events if e.type == "goal_finished"]
    assert len(finished) == 1
    assert finished[0].payload["reason"] == (
        "planner failed" if failure == "malformed" else "run budget"
    )
    assert RunState.from_events(events).goals["g"].finished


@pytest.mark.parametrize("kind", ["stop", "stop_now", "deadline", "takeover"])
async def test_standalone_paid_work_is_supervised(tmp_path: Path, kind: str) -> None:
    from _fakes.leases import force_take_over

    from swarm_arena._errors import Fenced
    from swarm_arena._journal import llm_record
    from swarm_arena._journal.lease import read_owner
    from swarm_arena._model.canonical import utc_now
    from swarm_arena._store.local import LocalStore

    class HeldModel(FakeModel):
        def __init__(self) -> None:
            super().__init__([FakeTurn(text=json.dumps(planner_output()))])
            self.entered, self.release = anyio.Event(), anyio.Event()

        async def get_response(self, *args: Any, **kwargs: Any) -> Any:
            self.entered.set()
            await self.release.wait()
            return await super().get_response(*args, **kwargs)

    config = planner_config(tmp_path)
    if kind == "deadline":
        config = config.model_copy(update={"limits": Limits(max_tokens=1_000_000, run_timeout_s=2)})
    fake = HeldModel()
    fake.register("planner")
    task = asyncio.create_task(LLMPlanner("planner").plan(GOAL, config, None))
    replacement = None
    try:
        with anyio.fail_after(15):
            await fake.entered.wait()
            (session,) = await anyio.to_thread.run_sync(
                lambda: list(Path(config.run_root).iterdir())
            )
            store = LocalStore(session)
            if kind in {"stop", "stop_now"}:
                (
                    session / "control" / ("stop-now-e1" if kind == "stop_now" else "stop-e1")
                ).write_text("")
            if kind == "takeover":
                replacement = force_take_over(store, owner_id="replacement", host="test", ttl_s=60)
                fenced_events = all_events(config)
                fake.release.set()
                with pytest.raises(Fenced):
                    await task
            else:
                with pytest.raises(RuntimeError, match=r"^planning session stopped$"):
                    await task
        assert session.name not in llm_record._bindings
        owner = read_owner(store)
        assert owner is not None
        events = all_events(config)
        if kind == "takeover":
            assert owner[0].owner_id == "replacement"
            assert owner[0].epoch == 2
            assert replacement is not None
            assert owner[0] == replacement.doc
            assert events == fenced_events
        else:
            assert owner[0].expires_at <= utc_now()
            calls = [e.payload["call"] for e in events if e.type == "llm_call"]
            assert len(calls) == 1
            assert calls[0]["error"].startswith("estimated:")
            finished = [e for e in events if e.type == "goal_finished"]
            assert len(finished) == 1
            assert (
                finished[0].payload["reason"]
                == {"stop": "stop", "stop_now": "stop --now", "deadline": "run_timeout_s"}[kind]
            )
        fake.release.set()
        assert all_events(config) == events
    finally:
        fake.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if replacement is not None:
            replacement.release()


@pytest.mark.parametrize("handled", ["budget", "fenced"])
async def test_public_planning_ignores_callers_handled_exception(
    tmp_path: Path,
    handled: str,
) -> None:
    from swarm_arena._errors import Fenced, _core_error
    from swarm_arena.llm import BudgetExceeded

    try:
        if handled == "fenced":
            raise _core_error(Fenced, "unrelated owner")
        small = planner_config(tmp_path / "unrelated")
        small = small.model_copy(update={"limits": Limits(max_tokens=10)})
        FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
        await LLMPlanner("planner").plan(GOAL, small, None)
        pytest.fail("expected budget refusal")
    except (BudgetExceeded, Fenced):
        config = planner_config(tmp_path / "success")
        FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
        result = await LLMPlanner("planner").plan(GOAL, config, None)
        assert result.planning_session is not None
        finished = [e for e in all_events(config) if e.type == "goal_finished"]
        assert [e.payload["reason"] for e in finished] == ["planning complete"]


@pytest.mark.parametrize("operation", ["run", "resume"])
@pytest.mark.parametrize("handled", ["budget", "fenced"])
def test_board_close_ignores_sync_callers_handled_exception(
    run_config: Any,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    handled: str,
) -> None:
    from _fakes.run import plan as human_plan

    from swarm_arena import Swarm
    from swarm_arena._board.arena import ArenaBoard
    from swarm_arena._engine import run as run_module
    from swarm_arena._errors import Fenced, _core_error
    from swarm_arena.llm import BudgetExceeded

    if operation == "resume":
        asyncio.run(Swarm(run_config).run(human_plan(rounds=1), run_id="original"))
    flags: list[bool] = []
    close_board = run_module._close_board
    original_close = ArenaBoard.close

    async def observe_close(board: Any, config: Any, *, failed: bool) -> None:
        flags.append(failed)
        await close_board(board, config, failed=failed)

    def fail_close(board: ArenaBoard) -> None:
        original_close(board)
        raise OSError("unrelated close failure")

    monkeypatch.setattr(run_module, "_close_board", observe_close)
    monkeypatch.setattr(ArenaBoard, "close", fail_close)
    try:
        if handled == "fenced":
            raise _core_error(Fenced, "unrelated owner")
        raise BudgetExceeded("budget", "unrelated budget")
    except (BudgetExceeded, Fenced):
        swarm = Swarm(run_config)
        with pytest.raises(RuntimeError, match="board close failed: OSError"):
            asyncio.run(
                swarm.run(human_plan(rounds=1))
                if operation == "run"
                else swarm.resume("original", force=True)
            )
        assert flags == [False]


@pytest.mark.parametrize("accounting_failure", [False, True])
async def test_falsey_body_failure_is_retained_with_accounting_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accounting_failure: bool
) -> None:
    from swarm_arena._errors import StorageError
    from swarm_arena.llm import _invocation, _planner

    class FalseyError(Exception):
        def __bool__(self) -> bool:
            return False

    async def prepare(*args: Any, **kwargs: Any) -> Any:
        raise FalseyError(SECRET)

    async def finish(*args: Any, **kwargs: Any) -> None:
        if accounting_failure:
            raise ValueError(SECRET)

    monkeypatch.setattr(_planner, "_prepare", prepare)
    monkeypatch.setattr(_invocation, "_finish_accounting", finish)
    config = planner_config(tmp_path)
    async with open_llm_run(tmp_path, config=config) as run:
        parent = run.scope("planner")
        expected = StorageError if accounting_failure else ValueError
        with pytest.raises(expected) as shared:
            async with _invocation._invocation(parent, name="planner", error_type=ValueError):
                raise FalseyError(SECRET)
        with llm_scope(parent), pytest.raises(expected) as planner:
            await LLMPlanner("planner")._draft_at(GOAL, config, None, base_dir=tmp_path)
        assert type(planner.value) is type(shared.value)
        assert str(planner.value) == str(shared.value)
        assert str(planner.value) == (
            "LLM accounting failed: ValueError"
            if accounting_failure
            else "LLM planner failed: FalseyError"
        )
        assert_secret_absent(planner.value, SECRET)


async def test_custom_reviewer_cleanup_hook(tmp_path: Path) -> None:
    class Reviews(_Reviews):
        cleared = False

        def _clear_edit(self) -> None:
            self.cleared = True

    reviewer = Reviews([Approve()])
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    await CollaborativePlanner(LLMPlanner("planner"), reviewer).plan(
        GOAL, planner_config(tmp_path), None
    )
    assert reviewer.cleared


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("hook_failure", ["lookup", "call"])
async def test_custom_reviewer_cleanup_failure_preserves_public_outcome(
    tmp_path: Path, cancelled: bool, hook_failure: str
) -> None:
    import traceback

    from swarm_arena._errors import _is_core_signal

    entered = anyio.Event()

    class Reviews:
        cleared = False

        async def review(self, plan: Plan) -> Approve:
            entered.set()
            if cancelled:
                await anyio.sleep_forever()
            return Approve()

        @property
        def _clear_edit(self) -> Any:
            if hook_failure == "lookup":
                raise ValueError(SECRET)
            return self.clear

        def clear(self) -> None:
            self.cleared = True
            raise ValueError(SECRET)

    reviewer = Reviews()
    config = planner_config(tmp_path)
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    task = asyncio.create_task(
        CollaborativePlanner(LLMPlanner("planner"), reviewer).plan(GOAL, config, None)
    )
    try:
        with anyio.fail_after(15):
            await entered.wait()
            if cancelled:
                task.cancel()
            expected = asyncio.CancelledError if cancelled else RuntimeError
            with pytest.raises(
                expected, match=None if cancelled else r"^review cleanup failed: ValueError$"
            ) as caught:
                await task
        assert_secret_absent(caught.value, SECRET)
        assert SECRET not in "".join(traceback.format_exception(caught.value))
        if cancelled:
            assert isinstance(caught.value, asyncio.CancelledError)
        else:
            assert isinstance(caught.value, RuntimeError)
            assert _is_core_signal(caught.value)
            assert str(caught.value) == "review cleanup failed: ValueError"
        finished = [e.payload["reason"] for e in all_events(config) if e.type == "goal_finished"]
        assert finished == ["stopped" if cancelled else "planning complete"]
        assert reviewer.cleared is (hook_failure == "call")
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("primary", ["body", "control", "handled"])
async def test_custom_cleanup_preserves_primary_and_ignores_handled_exception(
    tmp_path: Path, primary: str
) -> None:
    from swarm_arena._errors import Fenced, _core_error

    class Reviews:
        async def review(self, plan: Plan) -> Approve:
            if primary == "control":
                raise _core_error(Fenced, "review ownership lost")
            if primary == "body":
                raise ValueError(SECRET)
            return Approve()

        def _clear_edit(self) -> None:
            raise ValueError(SECRET)

    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    config = planner_config(tmp_path)
    planner = CollaborativePlanner(LLMPlanner("planner"), Reviews())
    try:
        raise _core_error(Fenced, "unrelated handled ownership loss")
    except Fenced:
        with pytest.raises(Fenced if primary == "control" else RuntimeError) as caught:
            await planner.plan(GOAL, config, None)
    assert (
        str(caught.value)
        == {
            "control": "run ownership lost",
            "body": "reviewer failed: ValueError",
            "handled": "review cleanup failed: ValueError",
        }[primary]
    )
    assert_secret_absent(caught.value, SECRET)
