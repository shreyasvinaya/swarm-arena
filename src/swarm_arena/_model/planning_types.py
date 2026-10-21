"""Define in-memory policy decisions, round state, and replanning history."""

from swarm_arena._model.base import FrozenModel
from swarm_arena._model.plan import Plan
from swarm_arena._model.results import AgentStatus, Candidate, Learning, Score
from swarm_arena._model.tooling import Estimate
from swarm_arena._scoring import ranking


class Decision(FrozenModel):
    """
    Record policy pruning and an optional no-result reason.

    Parameters
    ----------
    prune : dict[str, str], default={}
        Team or team/agent names mapped to pruning reasons.
    no_result : str or None, default=None
        Reason to stop without a result.
    policy : str, default=""
        Name of the policy producing this decision.

    Examples
    --------
    >>> from swarm_arena.planning import Decision
    >>> Decision(prune={"search": "No improvement"}, policy="patience").policy
    'patience'
    """

    prune: dict[str, str] = {}
    no_result: str | None = None
    policy: str = ""

    def merge(self, other: "Decision") -> "Decision":
        """
        Merge pruning decisions while retaining the first reason for each name.

        Parameters
        ----------
        other : Decision
            Later policy decision to combine with this one.

        Returns
        -------
        Decision
            New decision retaining the first non-None no-result reason and joining
            nonempty policy names with ``+``. Neither input is modified.

        Examples
        --------
        >>> from swarm_arena.planning import Decision
        >>> Decision(policy="first").merge(Decision(policy="second")).policy
        'first+second'
        """
        return Decision(
            prune={**other.prune, **self.prune},
            no_result=self.no_result if self.no_result is not None else other.no_result,
            policy="+".join(name for name in (self.policy, other.policy) if name),
        )


class RoundState(FrozenModel):
    """
    Present a goal's accumulated results to a pruning policy.

    This model stays in memory because ranking groups include a None mapping key.
    Comparisons use each group's current identity and newest fidelity throughout
    the history, so different scales never count as improvements.

    Parameters
    ----------
    goal_id : str
        Goal identifier.
    round : int
        Current completed round number.
    rounds_total : int
        Total planned rounds.
    plan : Plan
        Current plan revision.
    candidates : list[Candidate]
        This run and goal's candidates from all rounds so far.
    scores : list[Score]
        This run and goal's scores in recorded order.
    status : dict[str, AgentStatus]
        Team/agent names mapped to their status this round.
    ranking_identity : dict[str or None, tuple[str, str, str]]
        Current judge, rubric, and panel identity per part or main-goal group.
        Groups without a ranking score are absent.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan
    >>> from swarm_arena.planning import RoundState
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[])
    >>> state = RoundState(goal_id="g", round=1, rounds_total=3, plan=plan,
    ...     candidates=[], scores=[], status={}, ranking_identity={})
    >>> state.ranked(None)
    []
    """

    goal_id: str
    round: int
    rounds_total: int
    plan: Plan
    candidates: list[Candidate]
    scores: list[Score]
    status: dict[str, AgentStatus]
    ranking_identity: dict[str | None, tuple[str, str, str]]

    def groups(self) -> dict[str | None, list[str]]:
        """
        Return ordinary teams grouped in main-goal and then part order.

        Returns
        -------
        dict[str or None, list[str]]
            Group names mapped to teams, excluding the converge team.

        Examples
        --------
        >>> from swarm_arena import Goal, Plan
        >>> from swarm_arena.planning import RoundState
        >>> state = RoundState(goal_id="g", round=0, rounds_total=1,
        ...     plan=Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[]),
        ...     candidates=[], scores=[], status={}, ranking_identity={})
        >>> state.groups()
        {None: []}
        """
        return ranking.groups(self.plan)

    def ranked(self, group: str | None) -> list[tuple[Candidate, Score]]:
        """
        Rank a group's complete passing candidates on its current scale.

        Parameters
        ----------
        group : str or None
            Part name, or None for the main goal.

        Returns
        -------
        list[tuple[Candidate, Score]]
            Ranking under the group's current identity and newest fidelity.
            Empty when the group has no identity.

        Examples
        --------
        >>> from swarm_arena import Goal, Plan
        >>> from swarm_arena.planning import RoundState
        >>> state = RoundState(goal_id="g", round=0, rounds_total=1,
        ...     plan=Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[]),
        ...     candidates=[], scores=[], status={}, ranking_identity={})
        >>> state.ranked("unknown")
        []
        """
        identity = self.ranking_identity.get(group)
        if identity is None:
            return []
        return ranking.ranked(
            (candidate for candidate in self.candidates if candidate.part == group),
            self.scores,
            identity=identity,
            fidelity=None,
            allow_partial=False,
        )

    def team_best(self, team: str) -> Score | None:
        """
        Find a team's best score on its group's current scale.

        Parameters
        ----------
        team : str
            Ordinary team name in the current plan.

        Returns
        -------
        Score or None
            Best eligible score, or None when the team has no ranked candidate.

        Raises
        ------
        ValueError
            If the team is absent from the plan's ordinary teams.

        Examples
        --------
        >>> from swarm_arena import Goal, Plan, TeamSpec
        >>> from swarm_arena.planning import RoundState
        >>> plan = Plan(goal=Goal(id="g", namespace="demo"), judges=[],
        ...     teams=[TeamSpec(name="t", rationale="Explore", agents=[])])
        >>> state = RoundState(goal_id="g", round=1, rounds_total=3, plan=plan,
        ...     candidates=[], scores=[], status={}, ranking_identity={})
        >>> state.team_best("t") is None
        True
        """
        group = ranking.group_of(self.plan, team)
        return next(
            (score for candidate, score in self.ranked(group) if candidate.team == team), None
        )

    def team_best_by_round(self, team: str) -> list[Score | None]:
        """
        Return cumulative team bests on one current identity and fidelity.

        Parameters
        ----------
        team : str
            Ordinary team name in the current plan.

        Returns
        -------
        list[Score or None]
            Entry i uses candidates through round i + 1, up to the current round.
            Earlier fidelities and identities are excluded throughout, yielding
            None before the team has an eligible result on the current scale.

        Raises
        ------
        ValueError
            If the team is absent from the plan's ordinary teams.

        Examples
        --------
        >>> from swarm_arena import Goal, Plan, TeamSpec
        >>> from swarm_arena.planning import RoundState
        >>> plan = Plan(goal=Goal(id="g", namespace="demo"), judges=[],
        ...     teams=[TeamSpec(name="t", rationale="Explore", agents=[])])
        >>> state = RoundState(goal_id="g", round=2, rounds_total=3, plan=plan,
        ...     candidates=[], scores=[], status={}, ranking_identity={})
        >>> state.team_best_by_round("t")
        [None, None]
        """
        group = ranking.group_of(self.plan, team)
        pairs = [
            (candidate, score) for candidate, score in self.ranked(group) if candidate.team == team
        ]
        return [
            next((score for candidate, score in pairs if candidate.round <= round), None)
            for round in range(1, self.round + 1)
        ]


class History(FrozenModel):
    """
    Collect the evidence and remaining budget supplied to a replanner.

    Parameters
    ----------
    goal_id : str
        Goal identifier.
    revisions : list[Plan]
        Plan revisions in recorded order.
    scores : list[Score]
        Recorded judging results.
    learnings : list[Learning]
        Available learnings.
    decisions : list[tuple[int, Decision]]
        Round number and policy decision pairs forming the pruning log.
    budget_left : Estimate
        Remaining token and monetary budget.
    rounds_done : int
        Number of completed ordinary rounds.
    join_failures : list[str], default=[]
        Reasons previous join attempts failed.

    Examples
    --------
    >>> from swarm_arena.planning import History
    >>> from swarm_arena.tool import Estimate
    >>> history = History(goal_id="g", revisions=[], scores=[], learnings=[],
    ...     decisions=[], budget_left=Estimate(tokens=100), rounds_done=0)
    >>> history.join_failures
    []
    """

    goal_id: str
    revisions: list[Plan]
    scores: list[Score]
    learnings: list[Learning]
    decisions: list[tuple[int, Decision]]
    budget_left: Estimate
    rounds_done: int
    join_failures: list[str] = []


class RoundReport(FrozenModel):
    """
    Record a manager's summary of its completed round.

    Parameters
    ----------
    summary : str, default=""
        Human-readable account of the round's work.

    Examples
    --------
    >>> from swarm_arena.manager import RoundReport
    >>> RoundReport(summary="Completed the search").summary
    'Completed the search'
    """

    summary: str = ""
