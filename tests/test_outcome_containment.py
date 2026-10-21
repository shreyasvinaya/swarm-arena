"""One boundary reconstructs untrusted failures and preserves core controls."""

import asyncio

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import Fenced, StorageError, _core_error, _is_core_signal
from swarm_arena._outcomes import _Outcome


@pytest.mark.parametrize("kind", ["ordinary", "cancel", "group", "fenced", "core"])
async def test_containment(kind: str) -> None:
    secret = "private-boundary-marker"
    errors: dict[str, BaseException] = {
        "ordinary": OSError(secret),
        "cancel": asyncio.CancelledError(secret),
        "group": BaseExceptionGroup(secret, [asyncio.CancelledError(secret), ValueError(secret)]),
        "fenced": Fenced(secret),
        "core": _core_error(Fenced, "run ownership lost"),
    }
    with pytest.raises(Fenced if kind == "core" else StorageError) as caught:
        _Outcome().contain(errors[kind], StorageError, "storage")
    assert _is_core_signal(caught.value)
    assert_secret_absent(caught.value, secret)


async def test_pending_cancellation_wins_containment() -> None:
    task = asyncio.current_task()
    assert task is not None
    outcome = _Outcome()
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            outcome.contain(ValueError("private-boundary-marker"), StorageError, "storage")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
    finally:
        task.uncancel()
