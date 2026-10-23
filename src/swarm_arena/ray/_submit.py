"""Build and submit the unattended driver through Ray Jobs."""

import os
import shlex
from collections.abc import Mapping, Sequence
from threading import Event
from typing import Any
from uuid import uuid4

import anyio
from ray.job_submission import JobSubmissionClient

from swarm_arena._errors import _core_error
from swarm_arena._exec.threads import run_sync_daemon

_SUBMIT_TIMEOUT_S = 300


def driver_entrypoint(files: Sequence[str]) -> str:
    """Build a shell-quoted driver command with explicit approval.

    Parameters
    ----------
    files : Sequence[str]
        Run configuration followed by one or more plan paths.

    Returns
    -------
    str
        Shell command for the Ray Job driver.

    Examples
    --------
    >>> driver_entrypoint(["run.toml", "plans dir/a.toml"])
    "swarm-arena run --yes -- run.toml 'plans dir/a.toml'"
    """
    return shlex.join(["swarm-arena", "run", "--yes", "--", *files])


def submit_runtime_env(*, working_dir: str | None, pip: Sequence[str]) -> dict[str, Any] | None:
    """Build explicit code and dependency packaging without environment variables.

    Parameters
    ----------
    working_dir : str or None
        Directory to upload when nonempty.
    pip : Sequence[str]
        Dependency specifications to install in the job environment.

    Returns
    -------
    dict[str, Any] or None
        Only nonempty working_dir and pip entries, or None if both are empty.

    Examples
    --------
    >>> submit_runtime_env(working_dir=".", pip=["deepretro-swarm"])
    {'working_dir': '.', 'pip': ['deepretro-swarm']}
    >>> submit_runtime_env(working_dir=None, pip=[]) is None
    True
    """
    env: dict[str, Any] = {}
    if working_dir:
        env["working_dir"] = working_dir
    if pip:
        env["pip"] = list(pip)
    return env or None


def submit_driver(
    address: str | None, entrypoint: str, *, runtime_env: Mapping[str, Any] | None
) -> str:
    """Submit a driver to the Ray Jobs server and return its submission ID.

    Parameters
    ----------
    address : str or None
        Jobs server address. Conflicting RAY_ADDRESS or RAY_API_SERVER_ADDRESS
        values are refused. None lets Ray resolve its environment fallback.
    entrypoint : str
        Shell command to execute as the driver.
    runtime_env : Mapping[str, Any] or None
        Explicit working_dir and pip packaging, without secrets.

    Returns
    -------
    str
        Ray's submission ID, generated locally before contacting the server.

    Raises
    ------
    RuntimeError
        On conflicting addresses, a 300-second submission timeout, or an
        interrupt. Timeout and interrupt diagnostics include the submission
        ID for checking job status before retrying. A blocked daemon is
        abandoned; an in-flight server request may still create the job.

    Examples
    --------
    >>> submit_driver("http://head:8265", driver_entrypoint(["run.toml", "plan.toml"]),
    ...               runtime_env=None)  # doctest: +SKIP
    'swarm-arena-...'
    """
    if address is not None and any(
        os.environ.get(name) not in (None, address)
        for name in ("RAY_ADDRESS", "RAY_API_SERVER_ADDRESS")
    ):
        raise _core_error(
            RuntimeError,
            "--address conflicts with RAY_ADDRESS/RAY_API_SERVER_ADDRESS; "
            "unset conflicting variables",
        )

    submission_id = f"swarm-arena-{uuid4().hex}"
    abandoned = Event()

    def submit() -> str:
        client = JobSubmissionClient(address)
        # A constructor returning after abandonment must not start a new POST.
        if abandoned.is_set():
            return submission_id
        return client.submit_job(
            entrypoint=entrypoint,
            runtime_env=dict(runtime_env) if runtime_env is not None else None,
            submission_id=submission_id,
        )

    async def wait() -> str:
        result = submission_id
        try:
            with anyio.move_on_after(_SUBMIT_TIMEOUT_S) as scope:
                result = await run_sync_daemon(submit)
        finally:
            abandoned.set()
        if scope.cancel_called:
            raise _core_error(
                RuntimeError,
                f"Ray job submission timed out: {submission_id}; "
                f"check ray job status {submission_id} before retrying",
            )
        return result

    try:
        return anyio.run(wait, backend="asyncio")
    except KeyboardInterrupt:
        raise _core_error(
            RuntimeError,
            f"Ray job submission interrupted: {submission_id}; "
            f"check ray job status {submission_id} before retrying",
        ) from None
    finally:
        abandoned.set()
