"""Deterministic planners for offline engine and provenance tests."""

from collections.abc import Sequence

from swarm_arena import Goal, Plan, Ref, RunConfig
from swarm_arena.planning import History, Planner


class SequencePlanner(Planner):
    """Return supplied plans in order, repeating the last after exhaustion."""

    def __init__(self, plans: Sequence[Plan]) -> None:
        if not plans:
            raise ValueError("at least one plan is required")
        self._plans = tuple(plans)
        self._index = 0

    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        """Return the next supplied plan without changing its provenance."""
        result = self._plans[self._index]
        self._index = min(self._index + 1, len(self._plans) - 1)
        return result


class AddAgentPlanner(Planner):
    """Copy the first agent with a fresh name and the selected catalog tool."""

    def __init__(self, entry_name: str) -> None:
        self._entry_name = entry_name

    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        """Add a catalog tool agent to the latest revision's first team."""
        if history is None or not history.revisions:
            raise ValueError("agent addition requires plan history")
        parent = history.revisions[-1]
        if not parent.teams or not parent.teams[0].agents:
            raise ValueError("agent addition requires an existing agent")
        entry = config.planner_catalog.get(self._entry_name)
        if entry.kind != "tool":
            raise ValueError("agent addition requires a tool entry")
        team = parent.teams[0]
        names = {agent.name for agent in team.agents}
        index = 1
        while f"new-{index}" in names:
            index += 1
        agent = team.agents[0].model_copy(
            update={
                "name": f"new-{index}",
                "rationale": "Explore another catalog tool",
                "tool": Ref(path=entry.ref, config=entry.config),
            }
        )
        updated = team.model_copy(update={"agents": [*team.agents, agent]})
        return parent.model_copy(update={"teams": [updated, *parent.teams[1:]]})


class RaisingPlanner(Planner):
    """Fail deterministically to exercise planner failure containment."""

    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        """Raise a fixed planner error without external effects."""
        raise RuntimeError("planner failed")
