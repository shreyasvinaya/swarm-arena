"""Exercise failure cleanup with a task that deliberately resists cancellation."""

import asyncio
import time
from contextlib import suppress
from pathlib import Path

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

    from swarm_arena._exec.procinfo import process_start_time

    children: list[tuple[subprocess.Popen[bytes], bool, str | None]] = []
    try:
        for own_group in (False, True):
            child = subprocess.Popen(
                [sys.executable, "-c", "import threading; threading.Event().wait(300)"],
                start_new_session=own_group,
            )
            children.append(
                (child, own_group, process_start_time(child.pid) if own_group else None)
            )
        assert reap_children(children) == []
        assert all(child.returncode is not None for child, _, _ in children)
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


@pytest.mark.parametrize("body_error", [False, True], ids=["clean-exit", "escaping-error"])
def test_pool_join_deadline_preserves_the_body_failure(
    monkeypatch: pytest.MonkeyPatch, body_error: bool
) -> None:
    import threading

    from _fakes import cleanup

    entered, release = threading.Event(), threading.Event()
    pool = cleanup.BoundedThreadPoolExecutor(max_workers=1)
    workers: list[threading.Thread] = []
    original = ValueError("body failure")

    def held() -> None:
        workers.append(threading.current_thread())
        entered.set()
        assert release.wait(300)

    try:
        # The first clock read starts the deadline; the join sees it exhausted.
        ticks = iter((0.0, 60.0))
        monkeypatch.setattr(cleanup, "monotonic", lambda: next(ticks))
        with pytest.raises(ValueError if body_error else AssertionError) as caught:  # noqa: PT012, SIM117
            with pool:
                future = pool.submit(held)
                assert entered.wait(60)
                if body_error:
                    raise original
        if body_error:
            assert caught.value is original
            assert original.__notes__ == ["test-owned pool cleanup did not finish"]
        else:
            assert str(caught.value) == "test-owned pool cleanup did not finish"
        assert workers
        assert all(worker.is_alive() for worker in workers)
    finally:
        release.set()
        pool.shutdown(wait=False, cancel_futures=True)
        for worker in workers:
            worker.join(60)
            assert not worker.is_alive()
    future.result(timeout=60)


@pytest.mark.parametrize("own_group", [False, True], ids=["individual", "group"])
@pytest.mark.parametrize("held", [False, True], ids=["already-reaped", "held-child"])
def test_owned_process_fixture_captures_and_reports_leaks(
    monkeypatch: pytest.MonkeyPatch, own_group: bool, held: bool
) -> None:
    import signal
    import subprocess
    import sys
    from collections.abc import Generator
    from inspect import unwrap
    from typing import cast

    from _fakes import process_cleanup

    group_kills: list[int] = []
    real_killpg = process_cleanup.os.killpg
    original_popen = subprocess.Popen

    def killpg(pgid: int, sig: int) -> None:
        if sig == signal.SIGKILL:
            group_kills.append(pgid)
        real_killpg(pgid, sig)

    with monkeypatch.context() as patch:
        patch.setattr(process_cleanup.os, "killpg", killpg)
        owner = cast(Generator[None, None, None], unwrap(process_cleanup.owned_processes)(patch))
        next(owner)
        child = None
        try:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import threading; threading.Event().wait(300)" if held else "pass",
                ],
                start_new_session=own_group,
            )
            assert type(child) is not original_popen
            if held:
                assert child.poll() is None
                with pytest.raises(AssertionError, match="left an owned child alive"):
                    next(owner)
                assert child.returncode == -signal.SIGKILL
                assert group_kills == ([child.pid] if own_group else [])
            else:
                child.wait(timeout=60)
                with pytest.raises(StopIteration):
                    next(owner)
                assert group_kills == []
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=60)
            owner.close()


@pytest.mark.parametrize(
    ("captured", "current", "live", "leader_exists", "signal_group"),
    [
        ("original", "original", True, True, True),
        ("original", "replacement", True, True, False),
        ("original", None, False, False, False),
        ("original", None, True, False, True),
        (None, "unverified", True, True, False),
        ("original", None, True, True, False),
        ("original", None, True, None, False),
    ],
    ids=[
        "same-leader",
        "reused-pid",
        "empty-group",
        "orphan-members",
        "unknown-capture",
        "lookup-failed-live-leader",
        "lookup-and-existence-unverified",
    ],
)
def test_owned_group_cleanup_checks_identity_and_liveness(
    monkeypatch: pytest.MonkeyPatch,
    captured: str | None,
    current: str | None,
    live: bool,
    leader_exists: bool | None,
    signal_group: bool,
) -> None:
    import signal
    import subprocess
    import sys
    from collections.abc import Generator
    from inspect import unwrap
    from typing import cast

    from _fakes import process_cleanup

    identity = [captured]
    signals: list[tuple[int, int]] = []

    def check_leader(pid: int, sig: int) -> None:
        assert sig == 0
        if leader_exists is None:
            raise PermissionError
        if not leader_exists:
            raise ProcessLookupError

    with monkeypatch.context() as patch:
        patch.setattr(process_cleanup, "process_start_time", lambda pid: identity[0], raising=False)
        patch.setattr(process_cleanup, "group_has_live_member", lambda pid: live, raising=False)
        owner = cast(Generator[None, None, None], unwrap(process_cleanup.owned_processes)(patch))
        next(owner)
        child = None
        try:
            child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
            child.wait(timeout=60)
            identity[0] = current
            patch.setattr(process_cleanup.os, "kill", check_leader)
            patch.setattr(process_cleanup.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
            if signal_group:
                # A live verified group means the test left a member running.
                with pytest.raises(AssertionError, match="left an owned child alive"):
                    next(owner)
            else:
                with pytest.raises(StopIteration):
                    next(owner)
            assert signals == ([(child.pid, signal.SIGKILL)] if signal_group else [])
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=60)
            owner.close()


def test_unreaped_own_group_child_is_killed_when_identity_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import signal
    import subprocess
    import sys
    from collections.abc import Generator
    from inspect import unwrap
    from typing import cast

    from _fakes import process_cleanup

    with monkeypatch.context() as patch:
        patch.setattr(process_cleanup, "process_start_time", lambda pid: None, raising=False)
        owner = cast(Generator[None, None, None], unwrap(process_cleanup.owned_processes)(patch))
        next(owner)
        child = None
        try:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True
            )
            with pytest.raises(AssertionError, match="left an owned child alive"):
                next(owner)
            assert child.poll() == -signal.SIGKILL
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=60)
            owner.close()


def test_reaped_leader_with_live_descendant_is_reported_and_killed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os
    import signal
    import subprocess
    import sys
    from collections.abc import Generator
    from inspect import unwrap
    from typing import cast

    from _fakes import process_cleanup

    pid_file = tmp_path / "descendant.pid"
    code = (
        "import os, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(300)\n"
        "    os._exit(0)\n"
        f"open({str(pid_file)!r}, 'w').write(str(pid))\n"
    )
    with monkeypatch.context() as patch:
        owner = cast(Generator[None, None, None], unwrap(process_cleanup.owned_processes)(patch))
        next(owner)
        descendant = None
        try:
            leader = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            assert leader.wait(timeout=60) == 0
            descendant = int(pid_file.read_text())
            os.kill(descendant, 0)
            with pytest.raises(AssertionError, match="left an owned child alive"):
                next(owner)
            deadline = time.monotonic() + 60
            while _alive(descendant):
                assert time.monotonic() < deadline, "emergency cleanup left the descendant"
                time.sleep(0.05)
        finally:
            if descendant is not None and _alive(descendant):
                os.kill(descendant, signal.SIGKILL)
            owner.close()


def _alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
