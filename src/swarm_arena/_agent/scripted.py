"""Run deterministic request sequences through the manager action interface."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from swarm_arena._agent.context import AgentContext
from swarm_arena._agent.manager import Manager
from swarm_arena._errors import SoftError
from swarm_arena._model.planning_types import RoundReport
from swarm_arena._model.ref import _ConfigInputModel, _validate_config
from swarm_arena._model.results import Score


class ScriptedManager(Manager):
    """
    Run configured tool requests in order without a model.

    Requests with no round selector run every round. Soft action errors skip
    the request and appear in the summary; ``LimitReached`` ends the round by
    propagating to its runner. Optional learnings cite the best scored own
    candidate visible at the start of the round.

    Parameters
    ----------
    requests : Sequence[Mapping[str, Any] or ScriptedManager.Request] or ScriptedManager.Config
        Ordered requests, or configuration including the learning option.

    Examples
    --------
    >>> from swarm_arena import ScriptedManager
    >>> manager = ScriptedManager(requests=[{"inputs": {"x": 1.0}}])
    >>> manager.config.requests[0].inputs
    {'x': 1.0}
    """

    class Request(_ConfigInputModel):
        """
        Configure one tool request and its optional round selector.

        Parameters
        ----------
        inputs : dict[str, Any], optional
            Tool parameters validated by the tool when the request runs.
        parents : list[str] or {'best'}, optional
            Explicit parent identifiers, or the top visible leaderboard row.
        edits : dict[str, str], optional
            Workspace paths mapped to new UTF-8 text.
        round : int or None, default=None
            Selected round, or every round when omitted.

        Examples
        --------
        >>> from swarm_arena import ScriptedManager
        >>> ScriptedManager.Request(inputs={"x": 1.0}, parents="best").round is None
        True
        """

        inputs: dict[str, Any] = {}
        parents: list[str] | Literal["best"] = []
        edits: dict[str, str] = {}
        round: int | None = None

    class Config(_ConfigInputModel):
        """
        Store requests and the optional best-candidate learning setting.

        Parameters
        ----------
        requests : list[ScriptedManager.Request]
            Ordered tool requests.
        post_learnings : bool, default=False
            Publish one learning when an own scored candidate is visible.

        Examples
        --------
        >>> from swarm_arena import ScriptedManager
        >>> ScriptedManager.Config(requests=[]).post_learnings
        False
        """

        requests: list["ScriptedManager.Request"]
        post_learnings: bool = False

    def __init__(self, requests: Sequence[Mapping[str, Any] | Request] | Config) -> None:
        """Validate request configuration through the shared safe boundary."""
        self._config = _validate_config(
            self.Config,
            requests if isinstance(requests, self.Config) else {"requests": list(requests)},
            where="ScriptedManager",
        )

    @property
    def config(self) -> Config:
        """
        Return the validated request configuration.

        Returns
        -------
        ScriptedManager.Config
            Requests and learning publication setting.

        Examples
        --------
        >>> from swarm_arena import ScriptedManager
        >>> ScriptedManager([]).config.requests
        []
        """
        return cast(ScriptedManager.Config, self._config)

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        """
        Run this round's selected requests and report successes and soft errors.

        Parameters
        ----------
        ctx : AgentContext
            Core-created context with snapshot reads and checked actions.

        Returns
        -------
        RoundReport
            Counts of completed and selected requests, plus soft errors.

        Raises
        ------
        LimitReached
            If an action reaches a configured limit, including budget,
            terminal usage overflow, or a stop request.

        Examples
        --------
        >>> import anyio
        >>> from unittest.mock import Mock
        >>> from swarm_arena import ScriptedManager
        >>> report = anyio.run(ScriptedManager([]).run_round, Mock(round=1))
        >>> report.summary
        'ran 0 of 0 requests'
        """
        requests = [r for r in self.config.requests if r.round is None or r.round == ctx.round]
        ran = errors = 0
        for request in requests:
            try:
                parents = request.parents
                if parents == "best":
                    rows = ctx.best(1)
                    parents = [rows[0].candidate_id] if rows else []
                await ctx.run_tool(request.inputs, parents=parents, edits=request.edits)
            except SoftError:
                errors += 1
            else:
                ran += 1
        if self.config.post_learnings:
            await self._post_learning(ctx)
        summary = f"ran {ran} of {len(requests)} requests"
        if errors:
            summary += f" ({errors} soft {'error' if errors == 1 else 'errors'})"
        return RoundReport(summary=summary)

    async def _post_learning(self, ctx: AgentContext) -> None:
        """Find the greatest visible own score key and cite its display value."""
        best: Score | None = None
        for candidate in ctx.my_candidates():
            for score in ctx.scores_of(candidate.id):
                if best is None or score.key > best.key:
                    best = score
        if best is not None:
            value = best.value if best.value is not None else best.key[0]
            await ctx.post_learning(
                title=f"Best so far: {best.candidate_id}",
                text=f"score {value:.6g}",
                evidence=[best.candidate_id],
            )


ScriptedManager.Config.model_rebuild()
