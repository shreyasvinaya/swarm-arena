"""Public entry point for approved runs and epoch-scoped stop requests."""

from collections.abc import Awaitable, Callable, Sequence

from swarm_arena._engine.control import request_stop
from swarm_arena._engine.run import RunRunner
from swarm_arena._errors import _core_error
from swarm_arena._model.plan import Plan
from swarm_arena._model.results import Candidate
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.tooling import Estimate
from swarm_arena._planning.planner import Planner


class Swarm:
    """
    Run goal plans with shared budgets, approval and durable results.

    Parameters
    ----------
    config : RunConfig
        Human-controlled storage, executor and run limits.
    replanner : Planner or None, default=None
        Optional live planner for autonomous revisions.

    Attributes
    ----------
    run_id : str or None
        Identifier published when an approved run starts.

    Examples
    --------
    >>> from swarm_arena import RunConfig, Swarm
    >>> swarm = Swarm(RunConfig())
    >>> swarm.run_id is None
    True
    >>> results = await swarm.run(plan)  # doctest: +SKIP
    """

    def __init__(self, config: RunConfig, *, replanner: Planner | None = None) -> None:
        self.config, self.replanner = config, replanner
        self._runner: RunRunner | None = None
        self._run_id: str | None = None

    @property
    def run_id(self) -> str | None:
        """
        Return the current run identifier, including during execution.

        Returns
        -------
        str or None
            Identifier once assigned, otherwise None.

        Examples
        --------
        >>> Swarm(RunConfig()).run_id is None
        True
        """
        return self._runner.run_id if self._runner is not None else self._run_id

    @run_id.setter
    def run_id(self, value: str | None) -> None:
        self._run_id = value
        if self._runner is not None:
            self._runner.run_id = value

    async def run(
        self,
        plans: Plan | Sequence[Plan],
        *,
        run_id: str | None = None,
        approve: Callable[[Estimate, Sequence[str]], Awaitable[bool]] | None = None,
    ) -> dict[str, list[Candidate]]:
        """
        Validate, estimate, approve and execute one or more goal plans.

        Parameters
        ----------
        plans : Plan or Sequence[Plan]
            Initial plans. Each retains its origin and base directory.
        run_id : str or None, default=None
            Requested identifier, otherwise generated after approval.
        approve : Callable or None, default=None
            Async callback receiving a numeric estimate and warnings. False
            refuses the run without writing to run or artifact roots. Temporary
            snapshots are discarded. None accepts an estimate within the limits.

        Returns
        -------
        dict[str, list[Candidate]]
            Goal ids mapped to finalists in final-judge order; empty for no result.

        Raises
        ------
        ValueError
            If a plan is invalid.
        RuntimeError
            If approval, estimates, preflight or the asyncio backend check fails,
            or ownership is lost. Lost ownership raises a bare Fenced exception.

        Notes
        -----
        Executor shutdown is shielded for at most ``config.settle_timeout_s``.
        On expiry the original result or exception is preserved; shutdown failures
        never replace an exception already raised by the run.

        Examples
        --------
        >>> swarm = Swarm(RunConfig())
        >>> results = await swarm.run(plan)  # doctest: +SKIP
        """
        self._runner = RunRunner(
            self.config,
            [plans] if isinstance(plans, Plan) else plans,
            run_id=run_id,
            approve=approve,
            replanner=self.replanner,
        )
        return await self._runner.run()

    async def resume(self, run_id: str, *, force: bool = False) -> dict[str, list[Candidate]]:
        """
        Continue a run from its committed journal under a new ownership epoch.

        Parameters
        ----------
        run_id : str
            Existing run to recover.
        force : bool, default=False
            Permit takeover while the previous owner's lease is still live.

        Returns
        -------
        dict[str, list[Candidate]]
            Latest finalists in plan order, or empty lists for no-result goals.

        Raises
        ------
        ValueError
            A goal's peak exceeds max_agents_total, a budget cap decreases,
            or committed state is invalid.
        RuntimeError
            The old owner is live without force, a competing resume wins
            ownership, or a runtime service fails.

        Notes
        -----
        Executor shutdown is shielded for at most ``config.settle_timeout_s``.
        On expiry the original result or exception is preserved; shutdown failures
        never replace an exception already raised by the run.

        A goal stopped by ``stop`` or ``stop --now`` with rounds left continues.
        A goal stopped by a limit continues only when this resume changes the
        limits. Every goal end returns its unused pool; continued goals reopen
        with their goal_weights share of the remaining unallocated work budget
        before limits_changed.

        A resumed stopped goal gets only its share of the remaining unallocated
        work budget, which can be 0 when other goals were still running and
        already got its unused budget; unless the resume raises the limits, such
        a goal stops again with `goal budget` at its first run.

        A continued goal's earlier lifecycle stays in the run journal: its
        `final_order` and its `goal_finished` stop reason are kept there, even
        after the later lifecycle replaces `solution.json` and the replay clears
        the goal's `end_reason`. `status` and `solution.json` show only the latest
        lifecycle.

        Examples
        --------
        >>> results = await swarm.resume("run-id", force=True)  # doctest: +SKIP
        """
        from swarm_arena._engine.resume import resume_run

        self._runner = await resume_run(self.config, run_id, force=force, replanner=self.replanner)
        return await self._runner.run()

    async def stop(self, *, now: bool = False) -> None:
        """
        Request a stop from the current run's live owner.

        Parameters
        ----------
        now : bool, default=False
            Cancel in-flight work when true; otherwise drain it gracefully.

        Raises
        ------
        RuntimeError
            If no run is selected, no owner is live, or its epoch changes.

        Examples
        --------
        >>> await swarm.stop()  # doctest: +SKIP
        """
        if self.run_id is None:
            raise _core_error(RuntimeError, "no run has started")
        await request_stop(self.config.run_root, self.run_id, now=now)
