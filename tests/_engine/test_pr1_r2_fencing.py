"""Only registered ownership loss may terminate a public run as Fenced."""

import asyncio
import json
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes import pr1_fm as hooks

from swarm_arena import Swarm
from swarm_arena._errors import Fenced, _core_error
from tests._engine.test_pr1_fm_outcomes import config, events, plan


def fence(*, registered: bool, depth: int) -> BaseException:
    error: BaseException = (
        _core_error(Fenced, "run ownership lost") if registered else Fenced(hooks.MARK)
    )
    error.add_note(hooks.MARK)
    error.__cause__ = ValueError(hooks.MARK)
    for level in range(depth):
        error = (
            BaseExceptionGroup(hooks.MARK, [error, asyncio.CancelledError(hooks.MARK)])
            if level % 2
            else ExceptionGroup(hooks.MARK, [error])
            if isinstance(error, Exception)
            else BaseExceptionGroup(hooks.MARK, [error])
        )
        error.add_note(hooks.MARK)
        error.__cause__ = ValueError(hooks.MARK)
    return error


@pytest.mark.parametrize("where", ["manager", "action", "check", "command", "collect", "estimate"])
@pytest.mark.parametrize("depth", [0, 4])
@pytest.mark.parametrize("registered", [False, True])
async def test_fencing_authority_public_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, depth: int, registered: bool
) -> None:
    hooks.PEER_READY = None
    monkeypatch.setattr(hooks, "failure", lambda kind: fence(registered=registered, depth=depth))
    if registered:
        with pytest.raises(Fenced) as caught:
            await Swarm(config(tmp_path)).run(plan(where, "fence"), run_id="r")
        assert hooks.MARK not in "".join(traceback.format_exception(caught.value))
        assert caught.value.__cause__ is None
    else:
        result = await Swarm(config(tmp_path)).run(plan(where, "fence"), run_id="r")
        assert result["g"]
        journal = events(tmp_path)
        assert any(e["type"] == "tool_end" and e["agent"] == "peer" for e in journal)
        assert any(e["type"] == "goal_finished" for e in journal)
    assert hooks.MARK not in json.dumps(events(tmp_path))


@pytest.mark.parametrize("where", ["manager", "action", "check"])
@pytest.mark.parametrize("registered", [False, True])
async def test_caller_cancel_wins_over_replacement_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, registered: bool
) -> None:
    ready = anyio.Event()
    hooks.PEER_READY = None

    async def replace(*args: Any, **kwargs: Any) -> Any:
        ready.set()
        try:
            await anyio.sleep_forever()
        except asyncio.CancelledError:
            raise fence(registered=registered, depth=4) from None

    if where == "manager":
        monkeypatch.setattr(hooks.Broken, "run_round", replace)
    elif where == "check":
        monkeypatch.setattr(hooks.BrokenCheck, "check", replace)
    else:
        from swarm_arena.tool import Action

        original = hooks.PaidTool.extra_actions

        def actions(self: hooks.PaidTool) -> tuple[Action[Any], Action[Any]]:
            return (
                Action(
                    name="broken",
                    description="replace cancellation",
                    input_model=hooks.Inputs,
                    handler=replace,
                ),
                original(self)[1],
            )

        monkeypatch.setattr(hooks.PaidTool, "extra_actions", actions)
    task = asyncio.create_task(Swarm(config(tmp_path)).run(plan(where, "fence"), run_id="r"))
    try:
        with anyio.fail_after(60):
            await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert hooks.MARK not in "".join(traceback.format_exception(caught.value))
    finally:
        if not task.done():
            task.cancel()
            with suppress(BaseException):
                await task
