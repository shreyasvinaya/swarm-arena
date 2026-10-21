"""Define pruning policies and enforce group survival."""

from abc import ABC, abstractmethod

from swarm_arena._model.planning_types import Decision, RoundState
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import AgentStatus

_INACTIVE = frozenset({AgentStatus.PRUNED, AgentStatus.STOPPED, AgentStatus.WATCHDOG_STOPPED})


class Policy(Configurable, ABC):
    """
    Decide which teams or agents survive a completed round.

    Implement ``after_round`` to return team or team/agent pruning reasons.
    The engine guards every decision to retain a group's last surviving team
    and protect starved agents.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None, optional
        Settings validated by the policy's nested configuration model.

    Examples
    --------
    >>> from swarm_arena.planning import Decision, Policy
    >>> class Continue(Policy):
    ...     def after_round(self, state):
    ...         return Decision()
    >>> Continue().config.model_dump()
    {}
    """

    @abstractmethod
    def after_round(self, state: RoundState) -> Decision:
        """
        Select pruning actions from the completed round's evidence.

        Parameters
        ----------
        state : RoundState
            Accumulated results, current plan, and agent statuses.

        Returns
        -------
        Decision
            Pruning reasons and an optional no-result reason.

        Examples
        --------
        >>> from swarm_arena import Goal, KeepTopTeams, Plan
        >>> from swarm_arena.planning import RoundState
        >>> state = RoundState(goal_id='g', round=1, rounds_total=2,
        ...     plan=Plan(goal=Goal(id='g', namespace='demo'), teams=[], judges=[]),
        ...     candidates=[], scores=[], status={}, ranking_identity={})
        >>> KeepTopTeams(1).after_round(state).prune
        {}
        """


def _live_agents(state: RoundState) -> dict[str, list[str]]:
    """Return planned agent addresses that have not permanently stopped."""
    return {
        team.name: [
            f"{team.name}/{agent.name}"
            for agent in team.agents
            if state.status.get(f"{team.name}/{agent.name}") not in _INACTIVE
        ]
        for team in state.plan.teams
    }


def guard(decision: Decision, state: RoundState) -> Decision:
    """
    Drop unsafe prunes and mark an empty final ranking as no result.

    Parameters
    ----------
    decision : Decision
        Proposed pruning actions, processed in mapping order.
    state : RoundState
        Current plan, statuses, and ranking evidence.

    Returns
    -------
    Decision
        A new decision preserving starved agents and at least one live team
        per group. Neither the input decision nor the round state is modified.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan
    >>> from swarm_arena.planning import Decision, RoundState
    >>> state = RoundState(goal_id='g', round=1, rounds_total=1,
    ...     plan=Plan(goal=Goal(id='g', namespace='demo'), teams=[], judges=[]),
    ...     candidates=[], scores=[], status={}, ranking_identity={})
    >>> guard(Decision(), state).no_result
    'no passing candidate'
    """
    remaining = {team: set(agents) for team, agents in _live_agents(state).items()}
    groups = state.groups()
    team_groups = {team: teams for teams in groups.values() for team in teams}
    prune: dict[str, str] = {}
    for address, reason in decision.prune.items():
        team, separator, _ = address.partition("/")
        if team not in remaining:
            continue
        targets = {address} & remaining[team] if separator else remaining[team].copy()
        if not targets or any(state.status.get(agent) == AgentStatus.STARVED for agent in targets):
            continue
        if targets == remaining[team] and not any(
            remaining[other] for other in team_groups[team] if other != team
        ):
            continue
        remaining[team].difference_update(targets)
        prune[address] = reason
    no_result = decision.no_result
    if state.round == state.rounds_total and not any(state.ranked(group) for group in groups):
        no_result = "no passing candidate"
    return Decision(prune=prune, no_result=no_result, policy=decision.policy)
