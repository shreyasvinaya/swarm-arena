"""Exercise user callbacks at the conversion boundaries found by the source audit."""

import sys
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from _fakes.pr1_r2 import SECRET, failure
from _fakes.run import plan
from _fakes.tools import cand
from pydantic import BaseModel, Field, model_validator

from swarm_arena import Chain, FunctionTool, Goal, Ref, RunConfig, Score, validate_plan
from swarm_arena._errors import Fenced
from swarm_arena._model import ref as ref_module
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.catalog import CatalogEntry
from swarm_arena._model.goal import FileSet
from swarm_arena._model.ref import Configurable, EmptyConfig
from swarm_arena._planning import catalog_check, validate
from swarm_arena._planning.provenance import RefSite
from swarm_arena._tooling import function_tool, staging
from swarm_arena.executor import LocalExecutor
from swarm_arena.planning import Decision, Policy, RoundState
from swarm_arena.scoring import Judge, JudgeContext, Lexicographic


@pytest.mark.parametrize("kind", ["fenced", "cancel", "group"])
@pytest.mark.parametrize(
    "site",
    [
        "import",
        "config_validator",
        "safe_validator",
        "policy_constructor",
        "judge_constructor",
        "judge_rubric",
        "lex_rubric",
        "lex_panel",
        "config_serializer",
        "function_annotations",
        "function_schema",
        "import_root",
        "catalog_import",
        "catalog_default",
        "admission_hook",
        "admission_ref",
        "admission_override",
        "admission_snapshot",
        "admission_location",
    ],
)
def test_sync_conversion_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: str, kind: str
) -> None:
    calls: list[None] = []

    def fail(*args: Any, **kwargs: Any) -> Any:
        calls.append(None)
        raise failure(kind)

    class BadConfig(BaseModel):
        @model_validator(mode="before")
        @classmethod
        def validate_hook(cls, data: Any) -> Any:
            return fail()

    class BadSafe(FrozenModel):
        @model_validator(mode="before")
        @classmethod
        def validate_hook(cls, data: Any) -> Any:
            return fail()

    class DumpConfig(BaseModel):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            return fail()

    class BadPolicy(Policy):
        def __init__(self, config: Any) -> None:
            fail()

        def after_round(self, state: RoundState) -> Decision:
            return Decision()

    class BadJudge(Judge):
        def __init__(self, config: Any) -> None:
            fail()

        async def judge(self, goal: Goal, candidates: Any, ctx: JudgeContext) -> list[Score]:
            return []

    class SchemaHook:
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
            return fail()

    current = plan()
    ref = Ref(path="swarm_arena:ScriptTool")
    invoke: Callable[[], Any]
    fallback = False
    if site == "import":
        module = ModuleType("pr1_r2_import")
        monkeypatch.setattr(module, "__getattr__", fail, raising=False)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        invoke = partial(ref_module.load_object, "pr1_r2_import:missing")
    elif site == "config_validator":
        invoke = partial(ref_module._validate_config, BadConfig, {}, where="probe")
    elif site == "safe_validator":
        invoke = BadSafe
    elif site in {"policy_constructor", "judge_constructor"}:
        cls = BadPolicy if site == "policy_constructor" else BadJudge
        monkeypatch.setattr(Ref, "_resolve_config", lambda self, base: (cls, EmptyConfig()))
        if site == "policy_constructor":
            composite = Chain(ref)
            state = RoundState(
                goal_id="g",
                round=1,
                rounds_total=2,
                plan=current,
                candidates=[],
                scores=[],
                status={},
                ranking_identity={},
            )
            invoke = partial(composite.after_round, state)
        else:
            invoke = Lexicographic([ref]).anchor_count
    elif site == "judge_rubric":
        judge = object.__new__(BadJudge)
        judge._config = DumpConfig()
        invoke = judge.rubric_hash
    elif site in {"lex_rubric", "lex_panel"}:
        from _fakes.tools import FixedJudge

        judge = FixedJudge(FixedJudge.Config(value=1))
        monkeypatch.setattr(judge, "rubric_hash" if site == "lex_rubric" else "panel_hash", fail)
        if site == "lex_rubric":
            composite = Lexicographic([ref])
            monkeypatch.setattr(composite, "_get_members", lambda: [judge])
            invoke = composite.rubric_hash
        else:
            invoke = partial(
                Lexicographic._combine_scores, [cand(id="c")], [{"c": Score.maximize(1)}], [judge]
            )
            fallback = True
    elif site == "config_serializer":
        monkeypatch.setattr(Ref, "_resolve_config", lambda self, base: (Configurable, DumpConfig()))
        invoke = partial(staging.fixed_config, ref, None)
    elif site in {"function_annotations", "function_schema"}:
        from _fakes.pr1_r2 import score as annotated

        monkeypatch.setattr(annotated, "__annotations__", dict(annotated.__annotations__))
        if site == "function_annotations":
            annotated.__annotations__["goal"] = "fail()"
            monkeypatch.setitem(annotated.__globals__, "fail", fail)
        else:
            annotated.__annotations__["goal"] = SchemaHook
        invoke = partial(function_tool.function_input_model, annotated)
    elif site == "import_root":
        module = ModuleType("pr1_r2_root")
        monkeypatch.setattr(module, "__getattr__", fail, raising=False)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        invoke = partial(function_tool.import_root, module.__name__)
    elif site in {"catalog_import", "catalog_default"}:

        class Defaults(Configurable):
            class Config(BaseModel):
                files: FileSet = Field(default_factory=fail)

        entry = CatalogEntry(name="probe", kind="tool", ref=ref.path, description="probe")
        source = RefSite(
            pointer="/tool",
            kind="tool",
            ref=ref.model_copy(update={"config": {"files": {"globs": [], "manifest": {}}}}),
        )
        monkeypatch.setattr(
            Ref, "load", fail if site == "catalog_import" else lambda self: Defaults
        )
        invoke = partial(catalog_check._check_catalog_defaults, [(source, entry, ("files",))])
        fallback = True
    elif site == "admission_hook":
        monkeypatch.setattr(validate, "_MODEL_CHECKS", [fail])
        invoke = partial(validate_plan, current, RunConfig())
    elif site == "admission_ref":
        monkeypatch.setattr(Ref, "load", fail)
        invoke = partial(validate._rule_configs, current, RunConfig(), [], built={})
        fallback = True
    elif site == "admission_override":
        monkeypatch.setattr(validate, "_round_refs", lambda plan: [("/rounds/0", ref, {})])
        monkeypatch.setattr(validate, "effective_tool_ref", fail)
        invoke = partial(validate._validate_round_refs, current)
    elif site == "admission_snapshot":
        from swarm_arena._store.artifacts import ArtifactStore
        from swarm_arena._store.local import LocalStore

        monkeypatch.setattr(validate, "snapshot_plan", fail)
        invoke = partial(
            validate.admit_revision, current, RunConfig(), ArtifactStore(LocalStore(tmp_path))
        )
    else:
        parent = Ref(path=ref.path, config={"child": {"path": ref.path}})
        sites = [
            RefSite(pointer="/tool", kind="tool", ref=parent),
            RefSite(pointer="/tool/config/child", kind="tool", ref=ref),
        ]
        monkeypatch.setattr(Ref, "load", fail)
        invoke = partial(validate._site_pointers, sites)
        fallback = True
    caught: BaseException | None = None
    try:
        invoke()
    except BaseException as error:
        caught = error
    if caught is not None:
        assert isinstance(caught, Fenced if kind == "fenced" else Exception)
        assert SECRET not in "".join(traceback.format_exception(caught))
    else:
        assert fallback
        assert kind != "fenced"
    assert calls


@pytest.mark.parametrize("kind", ["fenced", "cancel", "group"])
async def test_offline_manager_outcomes(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    from _fakes.tools import GOAL, NoopManager

    from swarm_arena import Limits
    from swarm_arena.manager import AgentContext, RoundReport
    from swarm_arena.testing._harness import OfflineHarness

    async def fail(self: NoopManager, ctx: AgentContext) -> RoundReport:
        raise failure(kind)

    monkeypatch.setattr(NoopManager, "run_round", fail)
    harness = OfflineHarness()
    invoke = partial(
        harness.run_rounds,
        NoopManager(),
        goal=GOAL,
        tool=FunctionTool("_fakes.tools:square"),
        rounds=2,
        limits=Limits(),
        config=RunConfig(),
        judges=[],
    )
    if kind == "fenced":
        with pytest.raises(Fenced) as caught:
            await invoke()
        assert SECRET not in "".join(traceback.format_exception(caught.value))
    else:
        assert len(await invoke()) == 2
        assert len(harness.manager_errors) == 2
        assert SECRET not in repr(harness.manager_errors)


@pytest.mark.parametrize("kind", ["fenced", "cancel", "group"])
async def test_script_judge_child_outcome(
    tmp_path: Path, local_executor: LocalExecutor, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from _fakes.tools import GOAL

    from swarm_arena import ScriptJudge

    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure(kind)

    monkeypatch.setattr(ScriptJudge, "_score", fail)
    judge_ctx = JudgeContext(goal=GOAL, executor=local_executor, artifact_root=str(tmp_path))
    with pytest.raises(Fenced if kind == "fenced" else Exception) as caught:
        await ScriptJudge(command=["unused"]).judge(GOAL, [cand(id="c")], judge_ctx)
    assert SECRET not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("kind", ["fenced", "cancel", "group"])
@pytest.mark.parametrize(
    "site",
    [
        "repeat",
        "strict_copy",
        "collect_probe",
        "output_dump",
        "validation_details",
        "validation_fallback",
    ],
)
def test_remaining_sync_conversions(monkeypatch: pytest.MonkeyPatch, site: str, kind: str) -> None:
    from types import SimpleNamespace

    from pydantic import ValidationError
    from pydantic_core import ErrorDetails

    from swarm_arena._agent import repeats
    from swarm_arena._agent.runtime import AgentRuntime
    from swarm_arena._model import safe
    from swarm_arena.testing import _contracts, _tool
    from swarm_arena.tool import Tool, ToolOutput, ToolRequest

    calls: list[None] = []

    def fail(*args: Any, **kwargs: Any) -> Any:
        calls.append(None)
        raise failure(kind)

    invoke: Callable[[], Any]
    fallback = False
    if site == "repeat":
        monkeypatch.setattr(repeats, "repeat_key", fail)
        runtime = SimpleNamespace(
            spec=SimpleNamespace(replicates=1, name="a"),
            team=SimpleNamespace(name="t"),
            goal_key=Goal(id="g", namespace="n").key(),
            tool_ref=Ref(path="swarm_arena:ScriptTool"),
        )
        invoke = partial(
            repeats.RepeatIndex("r").next_replicate, cast(AgentRuntime, runtime), ToolRequest()
        )
    elif site == "strict_copy":

        class BadScore(Score):
            def model_dump(self, **kwargs: Any) -> dict[str, Any]:
                return fail()

        invoke = partial(_contracts._strict_copy, Score, BadScore.maximize(1))
        fallback = True
    elif site == "collect_probe":
        invoke = partial(
            _tool._probe_collect, cast(Tool, SimpleNamespace(collect=fail)), ToolRequest()
        )
        fallback = True
    elif site == "output_dump":

        class BadOutput(ToolOutput):
            def model_dump(self, **kwargs: Any) -> dict[str, Any]:
                return fail()

        invoke = partial(
            _tool._check_tool_output,
            FunctionTool("_fakes.tools:square"),
            ToolRequest(),
            BadOutput(status="ok"),
        )
    elif site == "validation_details":

        class BadDetails(ValidationError):
            def errors(self, **kwargs: Any) -> list[ErrorDetails]:
                return fail()

        invoke = partial(
            safe._sanitize_validation_error,
            EmptyConfig,
            BadDetails.from_exception_data("probe", []),
        )
        fallback = True
    else:
        monkeypatch.setattr(safe, "_sanitize_validation_error", fail)

        def invalid(data: Any) -> Any:
            raise ValueError(SECRET)

        invoke = partial(safe._validation_wrapper(BaseModel), {}, invalid)
    caught: BaseException | None = None
    try:
        invoke()
    except BaseException as error:
        caught = error
    if caught is not None:
        assert isinstance(caught, Fenced if kind == "fenced" else Exception)
        assert SECRET not in "".join(traceback.format_exception(caught))
    else:
        assert fallback
        assert kind != "fenced"
    assert calls


@pytest.mark.parametrize("kind", ["fenced", "cancel", "group"])
async def test_returned_score_serialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    from _fakes.tools import GOAL

    from swarm_arena import ProgramJudge
    from swarm_arena._scoring import program_judge

    class BadScore(Score):
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            raise failure(kind)

    monkeypatch.setattr(
        program_judge, "load_object", lambda path: lambda g, c: BadScore.maximize(1)
    )
    ctx = JudgeContext(goal=GOAL, executor=LocalExecutor(), artifact_root=str(tmp_path))
    judge = ProgramJudge("example:score")
    if kind == "fenced":
        with pytest.raises(Fenced):
            await judge.judge(GOAL, [cand(id="c")], ctx)
    else:
        assert await judge.judge(GOAL, [cand(id="c")], ctx) == []
        assert ctx.errors == {"c": "invalid Score returned"}


async def test_offline_harness_preserves_native_caller_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import anyio
    from _fakes.tools import GOAL, NoopManager

    from swarm_arena import Limits
    from swarm_arena.manager import AgentContext, RoundReport
    from swarm_arena.testing._harness import OfflineHarness

    ready = anyio.Event()

    async def wait(self: NoopManager, ctx: AgentContext) -> RoundReport:
        ready.set()
        try:
            await anyio.sleep_forever()
        except asyncio.CancelledError:
            raise failure("group") from None
        raise AssertionError("unreachable")

    monkeypatch.setattr(NoopManager, "run_round", wait)
    harness = OfflineHarness()
    task = asyncio.create_task(
        harness.run_rounds(
            NoopManager(),
            goal=GOAL,
            tool=FunctionTool("_fakes.tools:square"),
            rounds=2,
            limits=Limits(),
            config=RunConfig(),
            judges=[],
        )
    )
    try:
        with anyio.fail_after(5):
            await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert SECRET not in "".join(traceback.format_exception(caught.value))
        assert harness.manager_errors == []
    finally:
        if not task.done():
            task.cancel()
            from contextlib import suppress

            with suppress(BaseException):
                await task
