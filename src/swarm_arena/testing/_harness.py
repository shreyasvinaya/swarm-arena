"""Drive real manager actions and round-barrier judging without a swarm run."""

import atexit
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path

import anyio

from swarm_arena._agent.context import AgentContext
from swarm_arena._agent.manager import LimitReached, Manager
from swarm_arena._agent.repeats import RepeatIndex
from swarm_arena._agent.runtime import AgentCounters, AgentRuntime
from swarm_arena._agent.tool_run import Inflight, command_journal, track_inflight
from swarm_arena._agent.watchdog import Watchdog
from swarm_arena._board.digest import build_digest
from swarm_arena._board.run_board import RunBoard
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.ledger import Ledger
from swarm_arena._budget.tranche import Tranche
from swarm_arena._errors import Fenced, _core_error, _exception_type_name
from swarm_arena._journal.a2a import agent_round_task, agent_task_id, task_to_json
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.lease import Lease
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.goal import Goal
from swarm_arena._model.ids import IdGen
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import AgentSpec, Plan, RoundSpec, TeamSpec
from swarm_arena._model.planning_types import RoundReport
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import AgentStatus, Candidate, Learning
from swarm_arena._model.run_config import RunConfig
from swarm_arena._outcomes import _Outcome
from swarm_arena._planning.validate import judge_private_paths
from swarm_arena._scoring.judge import Judge, JudgeContext
from swarm_arena._scoring.runner import judge_round
from swarm_arena._store.local import LocalStore
from swarm_arena._tooling.staging import tool_identity
from swarm_arena._tooling.tool import Tool, effective_tool_ref
from swarm_arena.testing._components import _build
from swarm_arena.testing._contracts import _fail, _strict_copy
from swarm_arena.testing._execution import _executor, _failed_runs, _failed_runs_on
from swarm_arena.testing._state import _artifact_root, _artifacts


class OfflineHarness:
    """
    Retain the candidates, learnings, and readable store from an offline run.

    The private ranking convention uses the first supplied judge's component
    name. Without judges, an unscored private ranking makes empty ``best`` reads
    return no rows. Artifacts use the conformance kit's process-local root.
    The temporary run folder remains readable until process exit.

    Attributes
    ----------
    candidates : list[Candidate]
        Tool candidates made in the most recent run.
    learnings : list[Learning]
        Learnings posted in the most recent run.
    manager_errors : list[Exception]
        Original manager failures in round order, with failed-run log notes.
    store : LocalStore
        Readable run journal after ``run_rounds`` has returned.

    Examples
    --------
    >>> harness = OfflineHarness()
    >>> harness.candidates, harness.learnings
    ([], [])
    """

    def __init__(self) -> None:
        """Initialize empty observations without allocating a run folder."""
        self.candidates: list[Candidate] = []
        self.learnings: list[Learning] = []
        self.manager_errors: list[Exception] = []
        self.store: LocalStore

    async def run_rounds(
        self,
        manager: Manager,
        *,
        goal: Goal,
        tool: Tool,
        rounds: int,
        limits: Limits,
        config: RunConfig,
        judges: Sequence[Ref] = (),
    ) -> list[RoundReport]:
        """
        Run a manager through real action, journal, budget, and judging services.

        Each round captures a committed frontier and fixed snapshot. Its new
        gated-in candidates are judged and scores indexed before the following
        frontier. ``LimitReached`` ends only that manager round, returning an
        empty report for it. Ordinary manager exceptions are retained in
        ``manager_errors`` with failed-run log notes and journaled by type only;
        their rounds still receive barrier judging and closure, return empty
        reports, and do not prevent later rounds. Cancellation and fencing
        propagate. Returned reports pass strict validation and retain their identity.
        Tranche outcomes may be ignored here: durable ``settle.usage_overflow``
        and ledger exhaustion remain authoritative for subsequent admission.

        Parameters
        ----------
        manager : Manager
            Manager instance to exercise.
        goal : Goal
            Goal whose files are snapshotted relative to the current directory.
        tool : Tool
            Tool instance rebuilt from its snapshotted reference.
        rounds : int
            Positive number of rounds to execute.
        limits : Limits
            Action limits and optional budget caps; unset budget units are unlimited.
        config : RunConfig
            Run settings; storage is isolated in the temporary run and kit roots.
        judges : Sequence[Ref], default=()
            Private harness judges; the first defines the ranking for ``best``.

        Returns
        -------
        list[RoundReport]
            Original valid reports in round order.

        Examples
        --------
        >>> from swarm_arena import FunctionTool, Goal, Limits, RunConfig, ScriptedManager
        >>> import anyio
        >>> from functools import partial
        >>> harness = OfflineHarness()
        >>> reports = anyio.run(partial(harness.run_rounds, ScriptedManager([]),  # doctest: +SKIP
        ...     goal=Goal(id="g", namespace="demo"), tool=FunctionTool("my_package:solve"),
        ...     rounds=1, limits=Limits(), config=RunConfig()))
        """
        if rounds < 1:
            raise ValueError("rounds must be positive")
        self.candidates, self.learnings = [], []
        self.manager_errors = []
        root = await anyio.to_thread.run_sync(self._run_folder)
        self.store = LocalStore(root / "local")
        config = config.model_copy(
            update={
                "limits": limits,
                "run_root": str(root),
                "artifact_root": _artifact_root(),
            }
        )
        base_ref, _, goal = await anyio.to_thread.run_sync(_build, tool, Tool, goal)
        execution_ref = effective_tool_ref(base_ref, {})
        built_tool = await anyio.to_thread.run_sync(execution_ref.build, Tool)
        built_judges: list[tuple[Ref, Judge]] = []
        for ref in judges:
            snapshot, judge, _ = await anyio.to_thread.run_sync(_build, ref, Judge, goal)
            built_judges.append((snapshot, judge))
        ranking = built_judges[0][0].component_name() if built_judges else ":unscored"
        ids = IdGen("local", 1)
        lease = await anyio.to_thread.run_sync(
            partial(
                Lease.acquire_new,
                self.store,
                owner_id="testing",
                host="local",
                ttl_s=config.lease_ttl_s,
            )
        )
        async with (
            _executor(None) as executor,
            _failed_runs_on(executor),
            JournalWriter(
                self.store,
                lease,
                run_id="local",
                next_seq=1,
                local_path=root / "journal.jsonl",
                sync_interval_s=config.sync_interval_s,
            ) as journal,
        ):
            ledger = Ledger(
                limits,
                reserve_fraction=config.reserve_fraction,
                goal_weights=None,
                journal=journal,
                ids=ids,
            )
            await journal.append_durable(
                EventDraft(
                    type=EventType.RUN_CONFIG,
                    payload={"config": config.model_dump(mode="json")},
                )
            )
            await ledger.hold_final_slices({goal.id: Amount()})
            await journal.append_durable(
                EventDraft(
                    type=EventType.RUN_STARTED,
                    payload={"final_slices": {goal.id: Amount().model_dump()}},
                )
            )
            ledger.open_goal(goal.id)
            await journal.append_durable(EventDraft(type=EventType.GOAL_STARTED, goal_id=goal.id))
            board = RunBoard(run_id="local", run_root=str(root), head=lambda: journal.head)
            board.register_goal(goal.key(), goal.id, None)
            spec = AgentSpec(
                name="agent",
                rationale="Offline manager check",
                tool=base_ref,
                manager=Ref.of(manager),
            )
            team = TeamSpec(name="team", rationale="Offline manager check", agents=[spec])
            address = f"{goal.id}/team/agent"
            round_spec = RoundSpec()
            view = await anyio.to_thread.run_sync(
                board.snapshot,
                goal.key(),
                await anyio.to_thread.run_sync(board.frontier),
            )
            rt = AgentRuntime(
                address=address,
                holder=address,
                goal_id=goal.id,
                goal=goal,
                goal_key=goal.key(),
                part=None,
                team=team,
                spec=spec,
                round=1,
                round_spec=round_spec,
                fidelity=round_spec.effective_fidelity(),
                tool=built_tool,
                tool_ref=base_ref,
                execution_ref=execution_ref,
                tool_identity=tool_identity(base_ref, None),
                catalog_entry=None,
                checks=[],
                view=view,
                own=[],
                digest="",
                notes="",
                ledger=ledger,
                tranche=self._tranche(ledger, address, goal.id, limits),
                limits=limits,
                counters=AgentCounters(),
                repeats=RepeatIndex("local"),
                watchdog=Watchdog(limits),
                journal=journal,
                board=board,
                executor=executor,
                artifacts=_artifacts(),
                ids=ids,
                config=config,
                resumed=(),
                rankings={"": ranking},
                private_paths=judge_private_paths(
                    Plan(
                        goal=goal,
                        teams=[team],
                        judges=[ref for ref, _ in built_judges],
                    )
                ),
            )
            with track_inflight(Inflight()):
                reports = await self._rounds(manager, rt, rounds, ranking, built_judges)
        notes = await anyio.to_thread.run_sync(_failed_runs, executor.outputs)
        if self.manager_errors:
            for error in self.manager_errors:
                for note in notes:
                    error.add_note(note)
        else:
            for note in notes:
                warnings.warn(note, UserWarning, stacklevel=2)
        return reports

    @staticmethod
    def _run_folder() -> Path:
        """Keep the temporary journal readable, using the kit's exit cleanup policy."""
        root = Path(tempfile.mkdtemp(prefix="swarm-arena-manager-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        return root

    @staticmethod
    def _tranche(ledger: Ledger, address: str, goal_id: str, limits: Limits) -> Tranche:
        """Fund a fresh round tranche under the normal holder ceiling."""
        return Tranche(
            ledger,
            address,
            goal_id,
            size=Amount(
                tokens=limits.llm_tranche_tokens,
                usd=limits.llm_tranche_usd,
            ),
        )

    async def _rounds(
        self,
        manager: Manager,
        rt: AgentRuntime,
        rounds: int,
        ranking: str,
        judges: Sequence[tuple[Ref, Judge]],
    ) -> list[RoundReport]:
        """Advance snapshots and judge each completed round before the next."""
        reports: list[RoundReport] = []
        for number in range(1, rounds + 1):
            frontier = await anyio.to_thread.run_sync(rt.board.frontier)
            await rt.journal.append_durable(
                EventDraft(
                    type=EventType.FRONTIER,
                    goal_id=rt.goal_id,
                    round=number,
                    payload={"round": number, "frontier": frontier.model_dump(mode="json")},
                )
            )
            view = await anyio.to_thread.run_sync(rt.board.snapshot, rt.goal_key, frontier)
            grants = rt.ledger.open_round(
                rt.goal_id, number, [rt.holder], rounds_left=rounds - number + 1
            )
            await rt.journal.append_durable(
                EventDraft(
                    type=EventType.ROUND_STARTED,
                    goal_id=rt.goal_id,
                    round=number,
                    payload={
                        "round": number,
                        "shares": {h: a.model_dump() for h, a in grants.items()},
                    },
                )
            )
            rt = replace(
                rt,
                round=number,
                view=view,
                own=[],
                counters=AgentCounters(),
                started_runs=0,
                tranche=self._tranche(rt.ledger, rt.holder, rt.goal_id, rt.limits),
                digest=build_digest(
                    view,
                    goal=rt.goal_key,
                    round=number,
                    ranking=ranking,
                    cap_tokens=rt.limits.digest_tokens,
                ),
            )
            manager_error: Exception | None = None
            outcome = _Outcome()
            try:
                report = await manager.run_round(AgentContext(rt, gate=lambda: None))
            except BaseException as error:
                error = outcome.classify(error)
                if rt.journal.fenced.is_set():
                    raise _core_error(Fenced, "run ownership lost") from None
                if not isinstance(error, LimitReached):
                    self.manager_errors.append(error)
                    manager_error = error
                report = RoundReport()
            finally:
                self.candidates.extend(c for c in rt.own if isinstance(c, Candidate))
                self.learnings.extend(item for item in rt.own if isinstance(item, Learning))
                with anyio.CancelScope(shield=True):
                    await rt.tranche.close()
            if manager_error is not None:
                reason = f"manager error: {_exception_type_name(manager_error)}"
                rt.journal.append(
                    EventDraft(
                        type=EventType.AGENT_STATUS,
                        goal_id=rt.goal_id,
                        team=rt.team.name,
                        agent=rt.spec.name,
                        round=number,
                        task_id=agent_task_id(rt.address, number),
                        payload={
                            "status": AgentStatus.FINISHED.value,
                            "reason": reason,
                            "a2a": task_to_json(
                                agent_round_task(
                                    context_id="local",
                                    address=rt.address,
                                    round=number,
                                    learnings=[
                                        item for item in rt.own if isinstance(item, Learning)
                                    ],
                                    status=AgentStatus.FINISHED,
                                    summary=reason,
                                    at=utc_now(),
                                )
                            ),
                        },
                    )
                )
            if not isinstance(report, RoundReport):
                raise _fail("Manager.run_round() must return a RoundReport")
            if _strict_copy(RoundReport, report) is None:
                raise _fail("Manager.run_round() must return a valid RoundReport")
            reports.append(report)
            candidates = [c for c in rt.own if isinstance(c, Candidate)]
            leaders = [
                view.get_candidate(row.candidate_id)
                for row in view.leaderboard(rt.goal_key, judge=ranking)
            ]
            await self._judge(rt, judges, candidates, leaders)
            await rt.journal.append_durable(
                EventDraft(
                    type=EventType.ROUND_CLOSED,
                    goal_id=rt.goal_id,
                    round=number,
                    payload={"round": number},
                )
            )
        return reports

    @staticmethod
    async def _judge(
        rt: AgentRuntime,
        judges: Sequence[tuple[Ref, Judge]],
        candidates: Sequence[Candidate],
        leaders: Sequence[Candidate],
    ) -> None:
        """Journal and index barrier outcomes with the harness id source."""
        names = {id(judge): ref.component_name() for ref, judge in judges}

        def ctx_for(judge: Judge) -> JudgeContext:
            executor, on_started = command_journal(
                rt.journal,
                rt.executor,
                candidate_id="",
                component=names[id(judge)],
            )
            return JudgeContext(
                goal=rt.goal,
                executor=executor,
                artifact_root=rt.config.artifact_root,
                run_id="local",
                round=rt.round,
                fidelity=rt.fidelity,
                ids=rt.ids,
                on_started=on_started,
                tool_timeout_s=rt.limits.tool_timeout_s,
                secrets_file=rt.config.secrets.file,
            )

        outcome = await judge_round(rt.goal_id, judges, candidates, leaders, ctx_for)
        for score in outcome.scores:
            event = await rt.journal.append_durable(
                EventDraft(
                    type=EventType.SCORE,
                    goal_id=rt.goal_id,
                    round=rt.round,
                    payload={"score": score.model_dump(mode="json")},
                )
            )
            rt.board.put_score(score, epoch=event.epoch, seq=event.seq)
        for error in outcome.errors:
            rt.journal.append(
                EventDraft(
                    type=EventType.JUDGE_ERROR,
                    goal_id=rt.goal_id,
                    round=rt.round,
                    payload=error.model_dump(mode="json"),
                )
            )
