"""Provide configurable pruning policies with deterministic per-group ranking."""

from math import ceil
from typing import cast

from pydantic import Field

from swarm_arena._model.planning_types import Decision, RoundState
from swarm_arena._model.ref import Ref, RefField, _ConfigInputModel, _validate_config
from swarm_arena._model.results import AgentStatus, Score
from swarm_arena._planning.policy import Policy, _live_agents, guard


def _best_value(score: Score | None) -> float | None:
    """Format the display value without changing the ranking key."""
    return None if score is None else score.value if score.value is not None else score.key[0]


class KeepTopTeams(Policy):
    """
    Retain the best k live teams within each ranking group.

    Teams without a score rank last; ties follow plan order.

    Parameters
    ----------
    k : int or KeepTopTeams.Config
        Positive number of teams to keep, or validated configuration.

    Examples
    --------
    >>> from swarm_arena import KeepTopTeams
    >>> KeepTopTeams(2).config.k
    2
    """

    class Config(_ConfigInputModel):
        """
        Configure the number of surviving teams per group.

        Parameters
        ----------
        k : int
            Positive number of teams to keep.

        Examples
        --------
        >>> from swarm_arena import KeepTopTeams
        >>> KeepTopTeams.Config(k=1).k
        1
        """

        k: int = Field(ge=1)

    def __init__(self, k: int | Config, /) -> None:
        """Validate the team limit without importing components."""
        self._config = _validate_config(
            self.Config, k if isinstance(k, self.Config) else {"k": k}, where="KeepTopTeams"
        )

    @property
    def config(self) -> Config:
        """
        Return the validated team limit.

        Returns
        -------
        KeepTopTeams.Config
            Settings used by this policy.

        Examples
        --------
        >>> from swarm_arena import KeepTopTeams
        >>> KeepTopTeams(1).config.k
        1
        """
        return cast(KeepTopTeams.Config, self._config)

    def after_round(self, state: RoundState) -> Decision:
        """
        Prune live teams below the group retention limit.

        Parameters
        ----------
        state : RoundState
            Accumulated evidence on the current ranking scale.

        Returns
        -------
        Decision
            Guarded team pruning reasons.

        Examples
        --------
        >>> from swarm_arena import KeepTopTeams, Goal, Plan
        >>> from swarm_arena.planning import RoundState
        >>> state = RoundState(goal_id='g', round=1, rounds_total=2,
        ...     plan=Plan(goal=Goal(id='g', namespace='demo'), teams=[], judges=[]),
        ...     candidates=[], scores=[], status={}, ranking_identity={})
        >>> KeepTopTeams(1).after_round(state).prune
        {}
        """
        live = _live_agents(state)
        prune: dict[str, str] = {}
        for teams in state.groups().values():
            ranked = [(team, state.team_best(team)) for team in teams if live[team]]
            ranked.sort(
                key=lambda item: (item[1] is not None, item[1].key if item[1] else ()), reverse=True
            )
            for rank, (team, score) in enumerate(ranked, 1):
                if rank > self.config.k:
                    prune[team] = (
                        f"KeepTopTeams: rank {rank} of {len(ranked)} (best {_best_value(score)})"
                    )
        return guard(Decision(prune=prune, policy="KeepTopTeams"), state)


class SuccessiveHalving(Policy):
    """
    Retain a fraction of each group's agents and each team's passing minimum.

    Parameters
    ----------
    keep : float or SuccessiveHalving.Config, default=0.5
        Fraction in (0, 1] to retain, or validated configuration.
    min_per_team : int, default=1
        Nonnegative minimum of agents with ranked results to retain per team.

    Examples
    --------
    >>> from swarm_arena import SuccessiveHalving
    >>> SuccessiveHalving(keep=0.5, min_per_team=1).config.keep
    0.5
    """

    class Config(_ConfigInputModel):
        """
        Configure agent retention and passing-team protection.

        Parameters
        ----------
        keep : float, default=0.5
            Fraction in (0, 1] of eligible agents to keep.
        min_per_team : int, default=1
            Nonnegative minimum of passing agents kept per team.

        Examples
        --------
        >>> from swarm_arena import SuccessiveHalving
        >>> SuccessiveHalving.Config().min_per_team
        1
        """

        keep: float = Field(default=0.5, gt=0, le=1)
        min_per_team: int = Field(default=1, ge=0)

    def __init__(self, keep: float | Config = 0.5, min_per_team: int = 1) -> None:
        """Validate retention settings without importing components."""
        self._config = _validate_config(
            self.Config,
            keep if isinstance(keep, self.Config) else {"keep": keep, "min_per_team": min_per_team},
            where="SuccessiveHalving",
        )

    @property
    def config(self) -> Config:
        """
        Return the validated agent retention settings.

        Returns
        -------
        SuccessiveHalving.Config
            Fraction and passing-team minimum.

        Examples
        --------
        >>> from swarm_arena import SuccessiveHalving
        >>> SuccessiveHalving().config.keep
        0.5
        """
        return cast(SuccessiveHalving.Config, self._config)

    def after_round(self, state: RoundState) -> Decision:
        """
        Rank live agents per group and prune those outside the retained set.

        Parameters
        ----------
        state : RoundState
            Accumulated evidence on the current ranking scale.

        Returns
        -------
        Decision
            Guarded team or agent pruning reasons, protecting starved agents.

        Examples
        --------
        >>> from swarm_arena import SuccessiveHalving, Goal, Plan
        >>> from swarm_arena.planning import RoundState
        >>> state = RoundState(goal_id='g', round=1, rounds_total=2,
        ...     plan=Plan(goal=Goal(id='g', namespace='demo'), teams=[], judges=[]),
        ...     candidates=[], scores=[], status={}, ranking_identity={})
        >>> SuccessiveHalving().after_round(state).prune
        {}
        """
        live = _live_agents(state)
        prune: dict[str, str] = {}
        for group, teams in state.groups().items():
            scores: dict[str, Score] = {}
            for candidate, score in state.ranked(group):
                scores.setdefault(f"{candidate.team}/{candidate.agent}", score)
            agents = [
                address
                for team in teams
                for address in live[team]
                if state.status.get(address) != AgentStatus.STARVED
            ]
            # Ranked insertion order retains the shared candidate tie-breakers.
            order = {address: index for index, address in enumerate(scores)}
            agents.sort(key=lambda address: order.get(address, len(order)))
            kept = set(agents[: ceil(self.config.keep * len(agents))])
            for team in teams:
                passing = [
                    address for address in agents if address in scores and address in live[team]
                ]
                kept.update(passing[: self.config.min_per_team])
            reasons = {
                address: (
                    f"SuccessiveHalving: rank {rank} of {len(agents)} "
                    f"(best {_best_value(scores.get(address))})"
                )
                for rank, address in enumerate(agents, 1)
                if address not in kept
            }
            for team in teams:
                rejected = [
                    address for address in agents if address in live[team] and address in reasons
                ]
                if rejected and len(rejected) == len(live[team]):
                    prune[team] = reasons[rejected[0]]
                else:
                    prune.update((address, reasons[address]) for address in rejected)
        return guard(Decision(prune=prune, policy="SuccessiveHalving"), state)


class Patience(Policy):
    """
    Prune teams whose best has not improved for enough rounds.

    Comparisons use the group's current ranking identity and newest fidelity.

    Parameters
    ----------
    rounds : int or Patience.Config, default=2
        Positive number of rounds without improvement, or validated configuration.

    Examples
    --------
    >>> from swarm_arena import Patience
    >>> Patience(rounds=2).config.rounds
    2
    """

    class Config(_ConfigInputModel):
        """
        Configure the number of rounds without improvement.

        Parameters
        ----------
        rounds : int, default=2
            Positive patience interval.

        Examples
        --------
        >>> from swarm_arena import Patience
        >>> Patience.Config().rounds
        2
        """

        rounds: int = Field(default=2, ge=1)

    def __init__(self, rounds: int | Config = 2) -> None:
        """Validate the patience interval without importing components."""
        self._config = _validate_config(
            self.Config,
            rounds if isinstance(rounds, self.Config) else {"rounds": rounds},
            where="Patience",
        )

    @property
    def config(self) -> Config:
        """
        Return the validated patience interval.

        Returns
        -------
        Patience.Config
            Number of rounds without improvement.

        Examples
        --------
        >>> from swarm_arena import Patience
        >>> Patience().config.rounds
        2
        """
        return cast(Patience.Config, self._config)

    def after_round(self, state: RoundState) -> Decision:
        """
        Prune stagnant teams using comparable cumulative best scores.

        Parameters
        ----------
        state : RoundState
            Accumulated evidence on the current ranking scale.

        Returns
        -------
        Decision
            Guarded team pruning reasons.

        Examples
        --------
        >>> from swarm_arena import Patience, Goal, Plan
        >>> from swarm_arena.planning import RoundState
        >>> state = RoundState(goal_id='g', round=1, rounds_total=2,
        ...     plan=Plan(goal=Goal(id='g', namespace='demo'), teams=[], judges=[]),
        ...     candidates=[], scores=[], status={}, ranking_identity={})
        >>> Patience().after_round(state).prune
        {}
        """
        prune: dict[str, str] = {}
        for team, agents in _live_agents(state).items():
            if not agents:
                continue
            history = state.team_best_by_round(team)
            if len(history) <= self.config.rounds:
                continue
            baseline, best = history[-self.config.rounds - 1], history[-1]
            if best is None or (baseline is not None and best.key <= baseline.key):
                prune[team] = (
                    f"Patience: no improvement for {self.config.rounds} rounds "
                    f"(best {_best_value(best)})"
                )
        return guard(Decision(prune=prune, policy="Patience"), state)


class Chain(Policy):
    """
    Merge policies in order, keeping the first reason for each pruning target.

    References are resolved and cached on first use. Construction never imports
    member components. The merged decision is guarded again for group survival.

    Parameters
    ----------
    *policies : Policy or Ref or Chain.Config
        One or more policies or references, or one validated configuration.

    Examples
    --------
    >>> from swarm_arena import Chain, KeepTopTeams, Patience
    >>> len(Chain(KeepTopTeams(2), Patience(2)).config.policies)
    2
    """

    class Config(_ConfigInputModel):
        """
        Store a nonempty ordered list of policy references.

        Parameters
        ----------
        policies : list[Ref]
            Member references, accepting configurable policy instances.

        Examples
        --------
        >>> from swarm_arena import Chain, Patience
        >>> Chain.Config(policies=[Patience()]).policies[0].path
        'swarm_arena:Patience'
        """

        policies: list[RefField] = Field(min_length=1)

    def __init__(self, *policies: Policy | Ref | Config) -> None:
        """Validate references without importing or constructing members."""
        data = (
            policies[0]
            if len(policies) == 1 and isinstance(policies[0], self.Config)
            else {"policies": list(policies)}
        )
        self._config = _validate_config(self.Config, data, where="Chain")
        self._members: list[Policy] | None = None

    @property
    def config(self) -> Config:
        """
        Return the ordered member references.

        Returns
        -------
        Chain.Config
            Validated configuration for the policy chain.

        Examples
        --------
        >>> from swarm_arena import Chain, Patience
        >>> Chain(Patience()).config.policies[0].path
        'swarm_arena:Patience'
        """
        return cast(Chain.Config, self._config)

    def _get_members(self) -> list[Policy]:
        """Build and cache members, sanitizing only constructor failures here."""
        if self._members is None:
            members: list[Policy] = []
            for ref in self.config.policies:
                cls, validated = ref._resolve_config(Policy)
                try:
                    members.append(cls(validated))
                except TypeError:
                    raise TypeError(f"cannot build policy {ref.path}") from None
                except Exception:
                    raise ValueError(f"cannot build policy {ref.path}") from None
            self._members = members
        return self._members

    def after_round(self, state: RoundState) -> Decision:
        """
        Merge member decisions in order and guard the combined pruning actions.

        Parameters
        ----------
        state : RoundState
            Same completed-round evidence supplied to every member.

        Returns
        -------
        Decision
            Guarded union with nonempty policy names joined by ``+``.

        Raises
        ------
        ValueError
            If a member cannot be imported, configured, or constructed.
        TypeError
            If a member reference identifies an incompatible component.
        Exception
            If a member's policy hook raises.

        Examples
        --------
        >>> from swarm_arena import Chain, Patience, Goal, Plan
        >>> from swarm_arena.planning import RoundState
        >>> state = RoundState(goal_id='g', round=1, rounds_total=2,
        ...     plan=Plan(goal=Goal(id='g', namespace='demo'), teams=[], judges=[]),
        ...     candidates=[], scores=[], status={}, ranking_identity={})
        >>> Chain(Patience()).after_round(state).prune
        {}
        """
        result = Decision()
        for member in self._get_members():
            result = result.merge(member.after_round(state))
        return guard(result, state)
