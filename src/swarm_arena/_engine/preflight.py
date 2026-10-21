"""Check component prerequisites for admitted plan configurations."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import anyio

from swarm_arena._agent.manager import Manager
from swarm_arena._board.arena import _effective_arena_root
from swarm_arena._errors import ConfigError, StorageError, _core_error, _exception_type_name
from swarm_arena._exec.executor import Executor
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.plan import Plan, RoundSpec
from swarm_arena._model.ref import Ref
from swarm_arena._model.run_config import RunConfig
from swarm_arena._outcomes import _Outcome
from swarm_arena._scoring.check import Check
from swarm_arena._scoring.judge import Judge
from swarm_arena._store.base import open_store
from swarm_arena._tooling.tool import Tool, effective_tool_ref

_Component = Tool | Manager | Check | Judge
_Site = tuple[Ref, type[_Component], Mapping[str, Any] | None]


def _sites(plans: Sequence[Plan]) -> Iterator[_Site]:
    """Enumerate only interfaces with hooks, including all execution variants."""
    for plan in plans:
        for ref in plan.checks:
            yield ref, Check, None
        for ref in plan.judges:
            yield ref, Judge, None
        for part in plan.parts:
            for ref in part.checks or []:
                yield ref, Check, None
            for ref in part.judges or []:
                yield ref, Judge, None
        rounds = plan.rounds or [RoundSpec()]
        teams = [(team, rounds) for team in plan.teams]
        if plan.converge.mode == "team" and plan.converge.team is not None:
            teams.append((plan.converge.team, [plan.converge.round or rounds[-1]]))
        for team, specs in teams:
            for agent in team.agents:
                yield agent.manager, Manager, None
                for spec in specs:
                    yield agent.tool, Tool, spec.tool_overrides


def _run_hook(
    ref: Ref,
    base: type[_Component],
    overrides: Mapping[str, Any] | None,
    seen: set[tuple[str, bytes]],
) -> None:
    """Resolve, construct and invoke one hook in an abandonable daemon."""
    effective = effective_tool_ref(ref, overrides) if overrides is not None else ref
    cls, settings = effective._resolve_config(base)
    key = (effective.path, canonical_json(settings.model_dump(mode="json")))
    if key not in seen:
        seen.add(key)
        cls(settings).preflight()


async def run_preflight(plans: Sequence[Plan], config: RunConfig, executor: Executor) -> None:
    """
    Run prerequisite hooks once per distinct normalized component configuration.

    Parameters
    ----------
    plans : Sequence[Plan]
        Admitted plans. Main and part checks/judges, managers and effective tools
        across ordinary and converge rounds are checked. Policies and planners
        have no preflight interface and are excluded. Call again on each newly
        admitted revision before dispatch; deduplication is local to this call.
    config : RunConfig
        Run configuration identifying the supplied executor.
    executor : Executor
        Already constructed executor whose hook is also checked once.

    Raises
    ------
    RuntimeError
        If construction, configuration or hooks fail. All failures are collected
        as reference paths and exception types, without exception messages.
        Synchronous user work runs in a cancellable daemon thread. Each hook,
        including construction, is bounded by ``config.settle_timeout_s``.

    Examples
    --------
    >>> import anyio
    >>> from swarm_arena import RunConfig
    >>> from swarm_arena.executor import LocalExecutor
    >>> anyio.run(run_preflight, [], RunConfig(), LocalExecutor())
    """
    await _check_storage(config)
    seen: set[tuple[str, bytes]] = set()
    failures: list[str] = []
    for ref, base, overrides in _sites(plans):
        outcome = _Outcome()
        try:
            with anyio.fail_after(config.settle_timeout_s):
                await run_sync_daemon(_run_hook, ref, base, overrides, seen)
        except BaseException as exc:
            exc = outcome.classify(exc)
            failures.append(f"- {ref.path}: {_exception_type_name(exc)}")
    outcome = _Outcome()
    try:
        with anyio.fail_after(config.settle_timeout_s):
            await run_sync_daemon(executor.preflight)
    except BaseException as exc:
        exc = outcome.classify(exc)
        failures.append(f"- {config.executor.path}: {_exception_type_name(exc)}")
    if failures:
        raise _core_error(RuntimeError, "preflight failed:\n" + "\n".join(failures)) from None


def _board_arena_root(config: RunConfig) -> str:
    """Resolve the shared root through the board construction error boundary."""
    try:
        root = _effective_arena_root(config)
        if not isinstance(root, str):
            raise TypeError("arena root must be a string")
        return str.__str__(root)
    except BaseException as error:
        _Outcome(capture_caller=False).contain(error, RuntimeError, "board construction")


async def _check_storage(config: RunConfig) -> None:
    """Probe before ownership, with caller cancellation and type-only failures."""
    outcome = _Outcome()
    try:
        await run_sync_daemon(check_storage, config)
    except BaseException as error:
        outcome.contain(error, StorageError, "storage preflight")


def check_storage(config: RunConfig) -> None:
    """Check root reachability and probe each distinct S3 root.

    Parameters
    ----------
    config : RunConfig
        Run, effective board arena and artifact roots to validate before mutation.

    Raises
    ------
    ConfigError
        If an S3 arena points at a local run root.
    ConditionalWritesUnsupported
        If a remote root cannot demonstrate conditional writes.

    Examples
    --------
    >>> check_storage(RunConfig())
    """
    arena_root = _board_arena_root(config)
    if arena_root.startswith("s3://") and not config.run_root.startswith("s3://"):
        raise _core_error(
            ConfigError,
            "run_root must be reachable by other readers: a local run_root cannot "
            "be used with an s3:// arena_root",
        )
    for root in dict.fromkeys((config.run_root, arena_root, config.artifact_root)):
        if root.startswith("s3://"):
            open_store(root).probe_conditional_writes()
