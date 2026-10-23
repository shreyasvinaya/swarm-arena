"""Exercise failure cleanup with a task that deliberately resists cancellation."""

import asyncio
from contextlib import suppress

import pytest
from _fakes.cleanup import settle_tasks


async def test_cleanup_deadline_preserves_failure_and_runs_nested_cleanup() -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    joined: list[bool] = []

    async def held() -> None:
        entered.set()
        while not release.is_set():
            with suppress(asyncio.CancelledError):
                await release.wait()

    task = asyncio.create_task(held())
    original = TimeoutError("original watchdog")
    try:
        await asyncio.wait_for(entered.wait(), 60)
        with pytest.raises(AssertionError, match="cleanup did not finish") as caught:  # noqa: PT012
            try:
                raise original
            finally:
                try:
                    # Zero observation time is deterministic: release remains unset.
                    await settle_tasks(task, timeout_s=0)
                finally:
                    joined.append(True)
        assert caught.value.__context__ is original
        assert joined == [True]
        assert not task.done()
    finally:
        release.set()
        await settle_tasks(task)


async def test_cleanup_accepts_completed_and_empty_tasks() -> None:
    await settle_tasks()
    task = asyncio.create_task(asyncio.sleep(0))
    await asyncio.wait({task}, timeout=60)
    await settle_tasks(task)
    assert task.done()


def test_emergency_reaper_kills_held_children() -> None:
    import subprocess
    import sys

    from _fakes.process_cleanup import reap_children

    children: list[tuple[subprocess.Popen[bytes], bool]] = []
    try:
        for own_group in (False, True):
            child = subprocess.Popen(
                [sys.executable, "-c", "import threading; threading.Event().wait(300)"],
                start_new_session=own_group,
            )
            children.append((child, own_group))
        assert reap_children(children) == []
        assert all(child.returncode is not None for child, _ in children)
    finally:
        assert reap_children(children) == []


def test_pool_failure_releases_and_joins_owned_worker() -> None:
    import threading

    from _fakes.cleanup import BoundedThreadPoolExecutor

    entered, release = threading.Event(), threading.Event()
    workers: list[threading.Thread] = []

    def held() -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(300)

    with pytest.raises(ValueError, match="body failure"):  # noqa: PT012, SIM117
        with BoundedThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(held)
            try:
                assert entered.wait(60)
                raise ValueError("body failure")
            finally:
                release.set()
    assert workers
    assert all(not worker.is_alive() for worker in workers)


async def test_observation_timeout_does_not_cancel_owned_work() -> None:
    from _fakes.cleanup import observe_task

    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    try:
        with pytest.raises(AssertionError, match="did not finish"):
            await observe_task(task, timeout_s=0)
        assert not task.cancelling()
        release.set()
        assert await observe_task(task) is True
    finally:
        release.set()
        await settle_tasks(task)


@pytest.mark.parametrize("check", ["raises_outer", "cancelled_assertion"])
async def test_cancellation_watchdog_cannot_pass_for_a_resisting_task(check: str) -> None:
    import anyio
    from _fakes.cleanup import observe_task

    entered, resisted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def held() -> None:
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                resisted.set()

    task = asyncio.create_task(held())
    try:
        with anyio.fail_after(60):
            await entered.wait()
            task.cancel()
            await resisted.wait()
        if check == "raises_outer":
            with pytest.raises(TimeoutError):  # noqa: PT012, SIM117
                with pytest.raises(asyncio.CancelledError), anyio.fail_after(60) as watchdog:  # noqa: PT012
                    watchdog.deadline = float("-inf")
                    await observe_task(task)
        else:
            with pytest.raises(AssertionError):  # noqa: PT012
                with anyio.fail_after(60) as watchdog, pytest.raises(asyncio.CancelledError):  # noqa: PT012
                    watchdog.deadline = float("-inf")
                    await observe_task(task)
                assert task.cancelled()
        assert not task.done()
        assert not release.is_set()
    finally:
        release.set()
        await settle_tasks(task)
