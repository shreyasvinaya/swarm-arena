"""Small run plans and committed-event readers for orchestration tests."""

from pathlib import Path

import anyio

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    FunctionTool,
    Goal,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    ScriptedManager,
    TeamSpec,
)
from swarm_arena._journal.event import Event
from swarm_arena._journal.reader import read_committed
from swarm_arena._store.local import LocalStore
from swarm_arena._tooling.tool import Tool

WRITE = "import json, os; json.dump({}, open(os.environ['SWARM_ARENA_RESULT'], 'w')); "
WRITE_THEN_SLEEP = WRITE.format("{'data': {'y': 1.0}}") + "import time; time.sleep(600)"


def plan(
    *,
    goal_id: str = "g",
    teams: int = 1,
    agents: int = 1,
    rounds: int = 2,
    tool: Tool | None = None,
    tool_ref: Ref | None = None,
    check: str | None = None,
    converge_team: bool = False,
    converge_agents: int = 1,
) -> Plan:
    def team(name: str, count: int, *, final: bool = False) -> TeamSpec:
        return TeamSpec(
            name=name,
            rationale="Explore",
            agents=[
                AgentSpec(
                    name=f"a{i}",
                    rationale="Explore",
                    tool=tool_ref
                    or Ref.of(
                        tool
                        or FunctionTool(
                            "_fakes.tools:square",
                            timeout_s=60 + i + (100 if final else int(name[1:]) * 10),
                        )
                    ),
                    manager=Ref.of(
                        ScriptedManager(
                            [
                                {
                                    "inputs": {"x": float(i * 10 + r)},
                                    **({} if final else {"round": r}),
                                }
                                for r in range(1, (1 if final else rounds) + 1)
                            ]
                        )
                    ),
                )
                for i in range(count)
            ],
        )

    return Plan(
        goal=Goal(id=goal_id, namespace="n"),
        teams=[team(f"t{i}", agents) for i in range(teams)],
        judges=[Ref.of(ProgramJudge("_fakes.tools:score_min_y")).model_copy(update={"name": "y"})],
        ranking="y",
        checks=[Ref(path=check)] if check else [],
        rounds=[RoundSpec() for _ in range(rounds)],
        converge=ConvergeSpec(
            mode="team" if converge_team else "select",
            team=team("conv", converge_agents, final=True) if converge_team else None,
        ),
    )


def run_dir(config: RunConfig) -> Path:
    return next(Path(config.run_root).iterdir())


def all_events(config: RunConfig) -> list[Event]:
    read = read_committed(LocalStore(run_dir(config)))
    assert read is not None
    return read[1]


def events(config: RunConfig, type: str) -> list[Event]:
    return [e for e in all_events(config) if e.type == type]


def _started_store(config: RunConfig) -> LocalStore | None:
    """Treat unpublished startup state as pending; require a committed start."""
    try:
        store = LocalStore(run_dir(config))
    except (FileNotFoundError, StopIteration):
        return None
    read = read_committed(store)
    if read is not None and any(e.type == "tool_start" for e in read[1]):
        return store
    return None


async def wait_tool_started(config: RunConfig) -> LocalStore:
    """
    Wait for a tool start in a committed journal snapshot.

    Parameters
    ----------
    config : RunConfig
        Test configuration with one run below its run root.

    Returns
    -------
    LocalStore
        The run store whose committed chain contains a tool start.

    Notes
    -----
    The root, run directory and owner document may appear in separate steps.
    Their absence is pending startup, while malformed journals still fail.

    Examples
    --------
    >>> store = await wait_tool_started(config)  # doctest: +SKIP
    """
    with anyio.fail_after(20):
        while True:
            if (store := await anyio.to_thread.run_sync(_started_store, config)) is not None:
                return store
            await anyio.sleep(0.02)
