"""Startup and shutdown failure surfaces through public Swarm runs."""

import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from _fakes.pr1_r2 import SECRET, failure
from _fakes.run import plan

from swarm_arena import FunctionTool, Ref, RunConfig, ScriptedManager, Swarm
from swarm_arena._errors import Fenced
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize(
    "kind", ["fenced", "cancel", "group", "ordinary", "forged", "forged_group", "interrupt"]
)
@pytest.mark.parametrize(
    "site",
    [
        "constructor",
        "capacity",
        "executor_preflight",
        "tool_preflight",
        "estimate",
        "judge_models",
        "judge_providers",
        "manager_models",
        "planner_models",
        "curator_models",
        "model_check",
        "price_source",
        "manager_factory",
        "judge_factory",
        "curator_factory",
        "resume_planner_models",
        "resume_curator_models",
        "resume_model_check",
        "resume_price_source",
        "resume_curator_factory",
        "approve",
        "close",
    ],
)
async def test_public_startup_and_shutdown_outcomes(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, site: str, kind: str
) -> None:
    from swarm_arena.executor import LocalExecutor

    calls: list[None] = []

    def fail(*args: Any, **kwargs: Any) -> Any:
        calls.append(None)
        if kind == "interrupt":
            raise KeyboardInterrupt(SECRET)
        raise ValueError(SECRET) if kind == "ordinary" else failure(kind)

    async def fail_async(*args: Any, **kwargs: Any) -> Any:
        return fail()

    current = plan(rounds=1)
    if site.endswith("price_source"):
        run_config = run_config.model_copy(
            update={
                "limits": run_config.limits.model_copy(
                    update={"max_tokens": 1_000_000_000, "max_usd": 1_000_000}
                )
            }
        )
    resuming = site.startswith("resume_")
    store: LocalStore | None = None
    before: dict[str, bytes] = {}
    if resuming:
        site = site.removeprefix("resume_")
        current.teams[0].agents[0] = (
            current.teams[0].agents[0].model_copy(update={"manager": Ref.of(ScriptedManager([]))})
        )
        await Swarm(run_config).run(current, run_id="startup")
        store = LocalStore(Path(run_config.run_root) / "startup")
        before = {key: store.read(key).data for key in store.list("")}
    approve = None
    replanner = None
    if site == "constructor":
        monkeypatch.setattr(LocalExecutor, "__init__", fail)
    elif site == "capacity":
        monkeypatch.setattr(LocalExecutor, "capacity", fail_async)
    elif site == "executor_preflight":
        monkeypatch.setattr(LocalExecutor, "preflight", fail)
    elif site == "tool_preflight":
        monkeypatch.setattr(FunctionTool, "preflight", fail)
    elif site == "estimate":
        monkeypatch.setattr(FunctionTool, "estimate", fail)
    elif site.startswith("judge_"):
        from _fakes.tools import FixedJudge

        method = {
            "judge_models": "models",
            "judge_providers": "providers",
            "judge_factory": "__init__",
        }[site]
        monkeypatch.setattr(FixedJudge, method, fail)
        current = current.model_copy(
            update={"judges": [Ref(path="_fakes.tools:FixedJudge", name="y", config={"value": 1})]}
        )
    elif site.startswith("manager_"):
        monkeypatch.setattr(
            ScriptedManager, "models" if site == "manager_models" else "__init__", fail
        )
    elif site == "planner_models":
        from _fakes.planners import SequencePlanner

        replanner = SequencePlanner([current])
        monkeypatch.setattr(SequencePlanner, "models", fail)
    elif site.startswith("curator_"):
        pytest.importorskip("litellm")
        from _fakes.llm_components import AliasCurator

        monkeypatch.setattr(
            AliasCurator, "models" if site == "curator_models" else "__init__", fail
        )
        run_config = run_config.model_copy(
            update={
                "curator": Ref(path="_fakes.llm_components:AliasCurator", config={"model": "c"})
            }
        )
    elif site == "model_check":
        from swarm_arena._planning import validate as validation

        monkeypatch.setattr(validation, "_MODEL_CHECKS", [fail])
    elif site == "price_source":
        from _fakes.planners import SequencePlanner

        from swarm_arena._budget import estimate as estimation
        from swarm_arena.config import ModelCatalog, ModelSpec

        replanner = SequencePlanner([current])
        monkeypatch.setattr(SequencePlanner, "models", lambda self: ("p",))
        monkeypatch.setattr(estimation, "_price_sources", [fail])
        run_config = run_config.model_copy(
            update={"models": ModelCatalog({"p": ModelSpec(model="fake/p", roles=["planner"])})}
        )
    elif site == "approve":
        approve = fail_async
    else:
        assert site == "close"
        close = LocalExecutor.aclose

        async def close_then_fail(self: LocalExecutor) -> None:
            await close(self)
            fail()

        monkeypatch.setattr(LocalExecutor, "aclose", close_then_fail)
    swarm = Swarm(run_config, replanner=replanner)
    operation = (
        swarm.resume("startup", force=True) if resuming else swarm.run(current, approve=approve)
    )
    with pytest.raises(Fenced if kind == "fenced" else Exception) as caught:
        await operation
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert calls
    if kind != "fenced":
        assert not isinstance(caught.value, Fenced)
    if store is not None:
        assert {key: store.read(key).data for key in store.list("")} == before


@pytest.mark.parametrize("kind", ["ordinary", "cancel", "group", "forged", "forged_group"])
def test_cli_startup_outcome_is_type_only(tmp_path: Path, run_config: RunConfig, kind: str) -> None:
    import os
    import subprocess
    import sys

    from swarm_arena import Ref

    (tmp_path / "run.toml").write_text(run_config.dumps())
    current = plan(
        tool_ref=Ref(
            path="_fakes.pr1_r2_startup:StartupTool",
            config={"function": "_fakes.tools:square", "kind": kind},
        )
    )
    (tmp_path / "plan.toml").write_text(current.dumps())
    process = subprocess.run(
        [
            str(Path(sys.executable).with_name("swarm-arena")),
            "run",
            "--yes",
            str(tmp_path / "run.toml"),
            str(tmp_path / "plan.toml"),
        ],
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1])),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 1
    output = process.stdout + process.stderr
    assert SECRET not in output
    assert "Traceback" not in output
    assert "estimate failed" in output


@pytest.mark.parametrize("site", ["capacity", "approve"])
async def test_startup_caller_cancel_wins_over_replacement(
    run_config: RunConfig, monkeypatch: pytest.MonkeyPatch, site: str
) -> None:
    import asyncio

    import anyio

    from swarm_arena.executor import LocalExecutor

    ready = anyio.Event()

    async def replace(*args: Any, **kwargs: Any) -> Any:
        ready.set()
        try:
            await anyio.sleep_forever()
        except asyncio.CancelledError:
            raise failure("fenced") from None

    if site == "capacity":
        monkeypatch.setattr(LocalExecutor, "capacity", replace)
    task = asyncio.create_task(
        Swarm(run_config).run(plan(), approve=replace if site == "approve" else None)
    )
    try:
        with anyio.fail_after(5):
            await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert SECRET not in "".join(traceback.format_exception(caught.value))
    finally:
        if not task.done():
            task.cancel()
            with suppress(BaseException):
                await task
