"""Build validated run provenance without resolving secret values."""

import os
import platform
import subprocess
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self
from urllib.parse import urlsplit

import anyio
from pydantic import Field, field_validator, model_validator

from swarm_arena._exec.procinfo import host_id
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._gateway.labels import _profile_failures
from swarm_arena._model.base import FrozenModel, StoredModel
from swarm_arena._model.canonical import UtcDatetime, format_utc, sha256_hex, utc_now
from swarm_arena._model.data import FiniteJsonObject, Sha256
from swarm_arena._model.plan import Plan, Resources
from swarm_arena._model.run_config import RunConfig
from swarm_arena._planning.catalog_check import find_entry
from swarm_arena._tooling.staging import tool_identity
from swarm_arena._version import __version__


class _GatewayManifest(FrozenModel):
    """Retain validated capability evidence without credential-bearing URLs."""

    url: Annotated[str, Field(strict=True)]
    cap: Literal["hard", "soft"]
    verified: Annotated[bool, Field(strict=True)]
    failures: tuple[Annotated[str, Field(strict=True)], ...]
    terminal_accounting: Annotated[bool, Field(strict=True)]

    @field_validator("failures")
    @classmethod
    def _bounded_failures(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Keep printable bounded evidence through serialization and rereads."""
        return _profile_failures(value)

    @field_validator("url")
    @classmethod
    def _safe_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or any(c.isspace() or ord(c) < 32 for c in value)
            ):
                raise ValueError
            _ = parsed.port
        except ValueError:
            raise ValueError("invalid gateway url") from None
        return value

    @model_validator(mode="after")
    def _consistent_cap(self) -> Self:
        if self.verified != (not self.failures) or self.verified != (self.cap == "hard"):
            raise ValueError("gateway cap conflicts with profile verification")
        return self


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
    llm_gateway : _GatewayManifest or None, default=None
        Secret-free gateway URL, cap label and verified profile evidence.

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
    llm_gateway: _GatewayManifest | None = None

    @field_validator("llm_gateway", mode="before")
    @classmethod
    def _gateway_fields(cls, value: Any) -> Any:
        """Revalidate copied instances without invoking a potentially unsafe serializer."""
        return dict(vars(value)) if isinstance(value, _GatewayManifest) else value


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
    *,
    run_id: str,
    config: RunConfig,
    plans: Sequence[Plan],
    capacity: Resources | None,
    llm_gateway: _GatewayManifest | dict[str, Any] | None = None,
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
    llm_gateway : _GatewayManifest or dict or None, default=None
        Validated preflight evidence, absent when no gateway is configured.

    Returns
    -------
    dict[str, Any]
        Validated JSON-mode RunManifest fields. Tool rows contain the reference
        path, base identity and git commit at the plan's base directory. Resolved
        secret values are never read.

    Notes
    -----
    This synchronous entry point runs user config validators and serializers.
    Async callers must use build_manifest_async, which separates abandonable
    user code from bounded host/git work. Do not offload this entire function
    to an ordinary AnyIO worker: a blocked user hook can prevent cancellation
    and interpreter exit.

    Examples
    --------
    >>> from swarm_arena import RunConfig
    >>> manifest = build_manifest(  # doctest: +SKIP
    ...     run_id="r", config=RunConfig(), plans=[], capacity=None)
    >>> manifest["run_id"]  # doctest: +SKIP
    'r'
    """
    return _build_manifest(
        run_id=run_id,
        config=config,
        plans=plans,
        capacity=capacity,
        identities=_tool_identities(config, plans),
        llm_gateway=llm_gateway,
    )


async def build_manifest_async(
    *,
    run_id: str,
    config: RunConfig,
    plans: Sequence[Plan],
    capacity: Resources | None,
    llm_gateway: _GatewayManifest | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build run provenance with abandonable user configuration hooks.

    Parameters
    ----------
    run_id : str
        Run identifier.
    config : RunConfig
        Human configuration, hashed as deterministic UTF-8 TOML.
    plans : Sequence[Plan]
        Admitted snapshotted plans. Ordinary and converge tools use base refs
        and catalog settable fields, independent of round overrides.
    capacity : Resources or None
        Executor-reported capacity, or None when unavailable.
    llm_gateway : _GatewayManifest or dict or None, default=None
        Validated preflight evidence, absent when no gateway is configured.

    Returns
    -------
    dict[str, Any]
        The same validated JSON-mode RunManifest fields as build_manifest.

    Notes
    -----
    This is the supported entry point for async callers. Tool identity
    validation and serialization run through run_sync_daemon; cancellation
    abandons a blocked hook without preventing interpreter exit. Only after
    that phase succeeds does an AnyIO worker assemble the manifest and perform
    bounded host/git probes. Do not wrap build_manifest in to_thread.run_sync.

    Examples
    --------
    >>> import anyio
    >>> from functools import partial
    >>> from swarm_arena import RunConfig
    >>> anyio.run(partial(build_manifest_async, run_id="r",  # doctest: +SKIP
    ...     config=RunConfig(), plans=[], capacity=None))["run_id"]
    'r'
    """
    identities = await run_sync_daemon(_tool_identities, config, plans)
    return await anyio.to_thread.run_sync(
        partial(
            _build_manifest,
            run_id=run_id,
            config=config,
            plans=plans,
            capacity=capacity,
            identities=identities,
            llm_gateway=llm_gateway,
        )
    )


def _tool_identities(config: RunConfig, plans: Sequence[Plan]) -> list[list[tuple[str, str]]]:
    """Resolve user tool configs without performing host/git probes."""
    identities: list[list[tuple[str, str]]] = []
    for plan in plans:
        rows: list[tuple[str, str]] = []
        teams = list(plan.teams)
        if plan.converge.mode == "team" and plan.converge.team is not None:
            teams.append(plan.converge.team)
        for team in teams:
            for agent in team.agents:
                ref = agent.tool
                identity = tool_identity(ref, find_entry(ref, "tool", config.planner_catalog))
                rows.append((ref.path, identity))
        identities.append(rows)
    return identities


def _build_manifest(
    *,
    run_id: str,
    config: RunConfig,
    plans: Sequence[Plan],
    capacity: Resources | None,
    identities: Sequence[Sequence[tuple[str, str]]],
    llm_gateway: _GatewayManifest | dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble host provenance using already resolved tool identities."""
    tools: list[FiniteJsonObject] = []
    seen: set[tuple[str, str, str | None]] = set()
    for plan, rows in zip(plans, identities, strict=True):
        commit = _git_commit(plan.base_dir)
        for path, identity in rows:
            key = (path, identity, commit)
            if key not in seen:
                seen.add(key)
                tools.append({"ref": path, "identity": identity, "git_commit": commit})
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
        llm_gateway=_GatewayManifest.model_validate(
            dict(vars(llm_gateway)) if isinstance(llm_gateway, _GatewayManifest) else llm_gateway
        )
        if llm_gateway is not None
        else None,
    ).model_dump(mode="json")
