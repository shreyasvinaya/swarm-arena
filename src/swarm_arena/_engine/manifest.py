"""Build validated run provenance without resolving secret values."""

import os
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from swarm_arena._exec.procinfo import host_id
from swarm_arena._model.base import StoredModel
from swarm_arena._model.canonical import UtcDatetime, format_utc, sha256_hex, utc_now
from swarm_arena._model.data import FiniteJsonObject, Sha256
from swarm_arena._model.plan import Plan, Resources
from swarm_arena._model.run_config import RunConfig
from swarm_arena._planning.catalog_check import find_entry
from swarm_arena._tooling.staging import tool_identity
from swarm_arena._version import __version__


class RunManifest(StoredModel):
    """
    Validate the private versioned run manifest and its finite JSON subtrees.

    JSON nulls are preserved, including unavailable capacity or git commits.
    Readers use model_validate_json for migrations and newer-version refusal.
    This model is separate from the workspace-file Manifest alias.

    Parameters
    ----------
    schema_version : int, default=1
        Stored format version, upgraded when loading.
    run_id : str
        Run identifier.
    created_at : datetime
        UTC creation timestamp with millisecond precision.
    versions : dict[str, Any]
        Package, Python and platform versions.
    host : dict[str, Any]
        Boot-scoped host identity and CPU count.
    executor_capacity : dict[str, Any] or None
        Executor capacity, or None when unavailable.
    run_config_sha256 : str
        SHA-256 of deterministic UTF-8 run configuration TOML.
    plans : list[dict[str, Any]]
        Plan goal identifiers, revisions, content hashes and file hashes.
    tools : list[dict[str, Any]]
        Reference paths, base identities and nullable git commits.

    Examples
    --------
    >>> RunManifest.schema_kind
    'manifest'
    """

    schema_kind: ClassVar[str] = "manifest"
    run_id: str
    created_at: UtcDatetime
    versions: FiniteJsonObject
    host: FiniteJsonObject
    executor_capacity: FiniteJsonObject | None
    run_config_sha256: Sha256
    plans: list[FiniteJsonObject]
    tools: list[FiniteJsonObject]


def _git_commit(base_dir: Path) -> str | None:
    """Bound local git discovery and discard all subprocess diagnostics."""
    try:
        done = subprocess.run(
            ["git", "-C", str(base_dir), "rev-parse", "HEAD"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def build_manifest(
    *, run_id: str, config: RunConfig, plans: Sequence[Plan], capacity: Resources | None
) -> dict[str, Any]:
    """
    Record run versions, host profile, configuration hashes and base tool identity.

    Parameters
    ----------
    run_id : str
        Run identifier.
    config : RunConfig
        Human configuration, hashed as its deterministic UTF-8 TOML dump.
    plans : Sequence[Plan]
        Admitted snapshotted plans, hashed as deterministic UTF-8 TOML dumps.
        Tools include ordinary and converge agents; identities use base refs
        and catalog settable fields, independent of round overrides.
    capacity : Resources or None
        Executor-reported capacity, or None when unavailable.

    Returns
    -------
    dict[str, Any]
        Validated JSON-mode RunManifest fields. Tool rows contain the reference
        path, base identity and git commit at the plan's base directory. Resolved
        secret values are never read. Async callers must offload this function
        with anyio.to_thread.run_sync because host and git probes perform I/O.

    Examples
    --------
    >>> from swarm_arena import RunConfig
    >>> manifest = build_manifest(  # doctest: +SKIP
    ...     run_id="r", config=RunConfig(), plans=[], capacity=None)
    >>> manifest["run_id"]  # doctest: +SKIP
    'r'
    """
    tools: list[FiniteJsonObject] = []
    seen: set[tuple[str, str, str | None]] = set()
    for plan in plans:
        teams = list(plan.teams)
        if plan.converge.mode == "team" and plan.converge.team is not None:
            teams.append(plan.converge.team)
        commit = _git_commit(plan.base_dir)
        for team in teams:
            for agent in team.agents:
                ref = agent.tool
                identity = tool_identity(ref, find_entry(ref, "tool", config.planner_catalog))
                key = (ref.path, identity, commit)
                if key not in seen:
                    seen.add(key)
                    tools.append({"ref": ref.path, "identity": identity, "git_commit": commit})
    return RunManifest(
        run_id=run_id,
        created_at=format_utc(utc_now()),
        versions={
            "swarm_arena": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        host={"host_id": host_id(), "cpu_count": os.cpu_count()},
        executor_capacity=capacity.model_dump(mode="json") if capacity is not None else None,
        run_config_sha256=sha256_hex(config.dumps().encode("utf-8")),
        plans=[
            {
                "goal_id": plan.goal.id,
                "revision": plan.revision,
                "sha256": sha256_hex(plan.dumps().encode("utf-8")),
                "files_sha256": plan.goal.key().files_sha256,
            }
            for plan in plans
        ],
        tools=tools,
    ).model_dump(mode="json")
