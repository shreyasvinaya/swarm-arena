"""Startup and shutdown failure surfaces through public Swarm runs."""

import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from _fakes.pr1_r2 import SECRET, failure
from _fakes.run import plan

from swarm_arena import FunctionTool, RunConfig, Swarm
from swarm_arena._errors import Fenced


@pytest.mark.parametrize(
    "kind", ["fenced", "cancel", "group", "ordinary", "forged", "forged_group"]
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
        raise ValueError(SECRET) if kind == "ordinary" else failure(kind)

    async def fail_async(*args: Any, **kwargs: Any) -> Any:
        return fail()

    current = plan(rounds=1)
    approve = None
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

        from swarm_arena import Ref

        method = {
            "judge_models": "models",
            "judge_providers": "providers",
        }[site]
        monkeypatch.setattr(FixedJudge, method, fail)
        current = current.model_copy(
            update={"judges": [Ref(path="_fakes.tools:FixedJudge", name="y", config={"value": 1})]}
        )
    elif site == "approve":
        approve = fail_async
    else:
        close = LocalExecutor.aclose

        async def close_then_fail(self: LocalExecutor) -> None:
            await close(self)
            fail()

        monkeypatch.setattr(LocalExecutor, "aclose", close_then_fail)
    with pytest.raises(Fenced if kind == "fenced" else Exception) as caught:
        await Swarm(run_config).run(current, approve=approve)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert calls
    if kind != "fenced":
        assert not isinstance(caught.value, Fenced)


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
