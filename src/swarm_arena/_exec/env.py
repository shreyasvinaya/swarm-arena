"""Build allowlisted tool environments without exposing secret values in errors."""

import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from swarm_arena._exec.executor import Job
from swarm_arena._model.tooling import Command

BASE_ENV_NAMES: Final = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TZ")
_MAX_SECRETS_BYTES: Final = 2**20


def read_secrets_file(path: str) -> dict[str, str]:
    r"""
    Read stripped KEY=VALUE entries from a UTF-8 secrets file.

    Ignore blank lines and comments whose first nonspace character is ``#``.
    Split at the first equals sign, preserve quotes and inline hashes literally,
    and retain the last value for a repeated key. Accept only an opened regular
    file (including a symlink's regular target), at most 1 MiB of UTF-8 bytes.
    Diagnostics name the file and line number, never its contents or a
    value-bearing exception chain.

    Parameters
    ----------
    path : str
        Secrets file to read synchronously.

    Returns
    -------
    dict[str, str]
        Parsed environment names and values.

    Raises
    ------
    ValueError
        If the source is missing, unreadable, non-regular, over 1 MiB, or
        contains a malformed entry.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     source = Path(directory) / "secrets.env"
    ...     _ = source.write_text("API_KEY=example\n", encoding="utf-8")
    ...     read_secrets_file(str(source))
    {'API_KEY': 'example'}
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"secrets source must be a regular file: {path}")
            content = bytearray()
            while len(content) <= _MAX_SECRETS_BYTES:
                chunk = os.read(fd, min(65536, _MAX_SECRETS_BYTES + 1 - len(content)))
                if not chunk:
                    break
                content.extend(chunk)
            if len(content) > _MAX_SECRETS_BYTES:
                raise ValueError(f"secrets file exceeds 1 MiB: {path}")
            text = content.decode("utf-8")
        finally:
            os.close(fd)
    except FileNotFoundError:
        raise ValueError(f"secrets file not found: {path}") from None
    except (OSError, UnicodeError):
        raise ValueError(f"cannot read secrets file: {path}") from None

    values: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(f"{path}: line {number}: expected KEY=VALUE")
        values[key] = value.strip()
    return values


def tool_env_vars(
    *, run_id: str, goal_id: str, candidate_id: str, team: str, agent: str, round: int
) -> dict[str, str]:
    """
    Build the tool invocation's six SWARM_ARENA identity variables.

    Parameters
    ----------
    run_id : str
        Run identifier.
    goal_id : str
        Goal identifier.
    candidate_id : str
        Candidate identifier.
    team : str
        Team name.
    agent : str
        Agent name.
    round : int
        Round number, converted to a decimal string.

    Returns
    -------
    dict[str, str]
        Identity environment supplied through ``Job.env`` by the driver.

    Examples
    --------
    >>> env = tool_env_vars(run_id="r", goal_id="g", candidate_id="c",
    ...                     team="t", agent="a", round=2)
    >>> env["SWARM_ARENA_ROUND"]
    '2'
    """
    return {
        "SWARM_ARENA_RUN_ID": run_id,
        "SWARM_ARENA_GOAL_ID": goal_id,
        "SWARM_ARENA_CANDIDATE_ID": candidate_id,
        "SWARM_ARENA_TEAM": team,
        "SWARM_ARENA_AGENT": agent,
        "SWARM_ARENA_ROUND": str(round),
    }


def build_env(
    job: Job, command: Command, *, workspace: Path, node_env: Mapping[str, str]
) -> dict[str, str]:
    """
    Build a fresh environment using the job's explicit precedence rules.

    Start with base node names, then resolve command and secret allowlists from
    the node before falling back to the secrets file, skipping ``job.drop_env``
    names before reading any source. Then apply ``job.env``, set the
    workspace result path, and finally apply driver-supplied ``secret_values``.
    Missing allowlisted names stay unset. Read the file at most once, only when
    an allowlisted name is absent from the node. Async callers must run this
    synchronous helper through ``anyio.to_thread.run_sync``.

    Parameters
    ----------
    job : Job
        Job carrying allowlists, driver environment, and secret sources.
    command : Command
        Command carrying its environment allowlist.
    workspace : Path
        Workspace containing ``swarm/result.json``.
    node_env : Mapping[str, str]
        Environment of the executor's node, left unchanged.

    Returns
    -------
    dict[str, str]
        Environment to pass to the tool process.

    Examples
    --------
    >>> env = build_env(job, command, workspace=workspace,
    ...                 node_env={"PATH": "/bin"})  # doctest: +SKIP
    """
    dropped = frozenset(job.drop_env)
    env = {
        name: node_env[name] for name in BASE_ENV_NAMES if name not in dropped and name in node_env
    }
    secrets: dict[str, str] | None = None
    for name in (*command.env, *job.secret_env):
        if name in dropped:
            continue
        if name in node_env:
            env[name] = node_env[name]
        elif job.secrets_file is not None:
            if secrets is None:
                secrets = read_secrets_file(job.secrets_file)
            if name in secrets:
                env[name] = secrets[name]
    env.update(job.env)
    env["SWARM_ARENA_RESULT"] = str(workspace / "swarm" / "result.json")
    env.update(job.secret_values)
    return env
