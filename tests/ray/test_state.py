"""Named task and worker evidence on the shared, two-CPU Ray head."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

import pytest
from _fakes.tools import WORKER_SCRIPTS, make_job

pytest.importorskip("swarm_arena.ray")

# isort: split
import ray

from swarm_arena import Ref
from swarm_arena.ray._files import MIB
from swarm_arena.ray._state import PENDING_STATES, TERMINAL_STATES, TaskInfo, find_task, worker_dead
from swarm_arena.ray._task import TaskSettings, job_payload, run_job
from swarm_arena.tool import ToolRequest

pytestmark = pytest.mark.ray
T = TypeVar("T")


def _payload(script: str, *, pid_file: Path | None = None) -> bytes:
    args = [script] if pid_file is None else [script, "--pid-file", str(pid_file)]
    code = f"import sys; sys.argv = {args!r}\n" + WORKER_SCRIPTS[script]
    job = make_job(
        kind="tool",
        command=None,
        tool=Ref(path="_fakes.tools:PyTool", config={"code": code}),
        request=ToolRequest(),
        timeout_s=90 if pid_file else 10,
    )
    return job_payload(job, TaskSettings(cap_bytes=256 * MIB))


def _wait_for(check: Callable[[float], T], *, timeout_s: float = 45) -> T:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = check(deadline)
        if result:
            return result
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    raise AssertionError("Ray state evidence did not arrive within the test deadline")


def _task(name: str, *, terminal: bool, deadline: float) -> TaskInfo | None:
    info = find_task(name, deadline=deadline)
    if info is None:
        return None
    if terminal:
        return info if info.state in TERMINAL_STATES else None
    return info if info.state not in PENDING_STATES | TERMINAL_STATES else None


def test_find_task_by_name(ray_driver: None) -> None:
    name = f"x-test-e1-{uuid4().hex}"
    ref = run_job.options(name=name).remote(_payload("report_env.py"))
    ray.get(ref, timeout=45)
    info = _wait_for(lambda deadline: _task(name, terminal=True, deadline=deadline))
    assert info is not None
    assert info.name == name
    assert info.state in TERMINAL_STATES
    assert info.attempt_number == 0
    assert info.task_id
    assert info.worker_id
    assert info.node_id


def test_find_task_unknown_name_is_none(ray_driver: None) -> None:
    assert find_task(f"x-never-e1-{uuid4().hex}") is None


def test_worker_not_dead_while_running(ray_driver: None, tmp_path: Path) -> None:
    name = f"x-test-e1-{uuid4().hex}"
    pid_file = tmp_path / "running.pid"
    ref = run_job.options(name=name).remote(_payload("ignore_sigterm.py", pid_file=pid_file))
    try:
        _wait_for(lambda deadline: pid_file.exists())
        info = _wait_for(lambda deadline: _task(name, terminal=False, deadline=deadline))
        assert info is not None
        assert worker_dead(name, expected=info) is False
    finally:
        ray.cancel(ref, force=True)


def test_worker_dead_after_finish(ray_driver: None) -> None:
    name = f"x-test-e1-{uuid4().hex}"
    ray.get(run_job.options(name=name).remote(_payload("report_env.py")), timeout=45)
    assert _wait_for(lambda deadline: worker_dead(name, deadline=deadline) is True)


def test_worker_dead_after_force_cancel(ray_driver: None, tmp_path: Path) -> None:
    name = f"x-test-e1-{uuid4().hex}"
    pid_file = tmp_path / "cancel.pid"
    ref = run_job.options(name=name).remote(_payload("ignore_sigterm.py", pid_file=pid_file))
    try:
        _wait_for(lambda deadline: pid_file.exists())
        info = _wait_for(lambda deadline: _task(name, terminal=False, deadline=deadline))
        assert info is not None
        assert info.worker_id
        assert info.node_id
    finally:
        ray.cancel(ref, force=True)
    assert _wait_for(lambda deadline: worker_dead(name, expected=info, deadline=deadline) is True)


def test_unscheduled_task_has_no_worker_death_proof(ray_driver: None) -> None:
    name = f"x-pending-e1-{uuid4().hex}"
    ref = run_job.options(name=name, num_cpus=3).remote(_payload("report_env.py"))
    try:
        info = _wait_for(lambda deadline: find_task(name, deadline=deadline))
        assert info is not None
        assert info.state in PENDING_STATES
        assert info.worker_id is None
        assert worker_dead(name, expected=info) is False
    finally:
        ray.cancel(ref, force=True)
