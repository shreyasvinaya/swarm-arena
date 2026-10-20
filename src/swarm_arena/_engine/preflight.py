"""Check component prerequisites for admitted plan configurations."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from swarm_arena._agent.manager import Manager
from swarm_arena._exec.executor import Executor
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.plan import Plan, RoundSpec
from swarm_arena._model.ref import Ref
from swarm_arena._model.run_config import RunConfig
from swarm_arena._scoring.check import Check
from swarm_arena._scoring.judge import Judge
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
        Synchronous user work runs in a cancellable daemon thread.

    Examples
    --------
    >>> import anyio
    >>> from swarm_arena import RunConfig
    >>> from swarm_arena.executor import LocalExecutor
    >>> anyio.run(run_preflight, [], RunConfig(), LocalExecutor())
    """
    seen: set[tuple[str, bytes]] = set()
    failures: list[str] = []
    for ref, base, overrides in _sites(plans):
        try:
            await run_sync_daemon(_run_hook, ref, base, overrides, seen)
        except Exception as exc:
            failures.append(f"- {ref.path}: {type(exc).__name__}")
    try:
        await run_sync_daemon(executor.preflight)
    except Exception as exc:
        failures.append(f"- {config.executor.path}: {type(exc).__name__}")
    if failures:
        raise RuntimeError("preflight failed:\n" + "\n".join(failures)) from None
