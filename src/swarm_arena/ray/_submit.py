"""Build and submit the unattended driver through Ray Jobs."""

import shlex
from collections.abc import Mapping, Sequence
from typing import Any

from ray.job_submission import JobSubmissionClient


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
    "swarm-arena run --yes run.toml 'plans dir/a.toml'"
    """
    return shlex.join(["swarm-arena", "run", "--yes", *files])


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
        Jobs server address. None lets Ray resolve RAY_ADDRESS.
    entrypoint : str
        Shell command to execute as the driver.
    runtime_env : Mapping[str, Any] or None
        Explicit working_dir and pip packaging, without secrets.

    Returns
    -------
    str
        Ray's submission ID, rather than the deprecated job_id parameter.

    Examples
    --------
    >>> submit_driver("http://head:8265", driver_entrypoint(["run.toml", "plan.toml"]),
    ...               runtime_env=None)  # doctest: +SKIP
    'raysubmit_123'
    """
    return JobSubmissionClient(address).submit_job(
        entrypoint=entrypoint,
        runtime_env=dict(runtime_env) if runtime_env is not None else None,
    )
