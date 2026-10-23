"""Hold the loop-owned state and dependencies of one agent round."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from swarm_arena._agent.repeats import RepeatIndex
    from swarm_arena._agent.watchdog import Watchdog
    from swarm_arena._board.base import Board, BoardView
    from swarm_arena._budget.ledger import Ledger
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._engine.stops import ProviderStops
    from swarm_arena._exec.executor import Executor
    from swarm_arena._journal.writer import JournalWriter
    from swarm_arena._model.catalog import CatalogEntry
    from swarm_arena._model.goal import Goal, GoalKey
    from swarm_arena._model.ids import IdGen
    from swarm_arena._model.limits import Limits
    from swarm_arena._model.plan import AgentSpec, RoundSpec, TeamSpec
    from swarm_arena._model.ref import Ref
    from swarm_arena._model.results import Candidate, Learning
    from swarm_arena._model.run_config import RunConfig
    from swarm_arena._scoring.check import Check
    from swarm_arena._store.artifacts import ArtifactStore
    from swarm_arena._tooling.tool import Tool


@dataclass
class AgentCounters:
    """
    Track actions within an agent round without intervening awaits.

    Parameters
    ----------
    turns : int, default=0
        Manager actions attempted.
    tool_runs : int, default=0
        New tool runs admitted.
    board_reads : int, default=0
        Charged board reads.
    learnings : int, default=0
        Learnings posted.
    refused_repeats : int, default=0
        Repeated requests refused.
    consecutive_failures : int, default=0
        Consecutive counted tool failures.

    Examples
    --------
    >>> counters = AgentCounters()
    >>> counters.tool_runs += 1
    >>> counters.tool_runs, AgentCounters().tool_runs
    (1, 0)
    """

    turns: int = 0
    tool_runs: int = 0
    board_reads: int = 0
    learnings: int = 0
    refused_repeats: int = 0
    consecutive_failures: int = 0


@dataclass(kw_only=True)
class AgentRuntime:
    """
    Bundle an agent round's state and shared engine services by keyword.

    Parameters
    ----------
    stops : ProviderStops
        Run-wide provider latch, shared with every goal and component scope.
    address : str
        Full goal/team/agent address.
    holder : str
        Ledger holder for tool runs and the tranche, including the goal's
        final slice for a converge-team agent.
    goal_id : str
        Main goal identifier for events, candidates, learnings and ledger ownership.
    goal : Goal
        Main goal or this part's sub-goal.
    goal_key : GoalKey
        Snapshotted identity used in repeat keys.
    part : str or None
        Part name, if any.
    team : TeamSpec
        Agent's team.
    spec : AgentSpec
        Agent's specification, including its replicate bound.
    round : int
        Current round number.
    round_spec : RoundSpec
        Current effort settings.
    fidelity : str
        Current effort identity.
    tool : Tool
        Effective built tool, paired with execution_ref.
    tool_ref : Ref
        Snapshotted base reference used for repeat keys and identity.
    execution_ref : Ref
        Required effective reference used privately for executor jobs.
    tool_identity : str
        Base tool identity used for overlays.
    catalog_entry : CatalogEntry or None
        Matching catalog entry, if any.
    checks : Sequence[tuple[str, Check]]
        Named checks for each tool result.
    view : BoardView
        Round-start snapshot.
    own : list[Candidate or Learning]
        This agent's current-round results and learnings.
    digest : str
        Manager-visible board digest.
    notes : str
        Agent notes carried between rounds.
    ledger : Ledger
        Shared durable budget ledger.
    tranche : Tranche
        LLM tranche funded by holder.
    limits : Limits
        Effective run limits.
    counters : AgentCounters
        Mutable action counters.
    repeats : RepeatIndex
        Earlier-round and own-result repeat index.
    watchdog : Watchdog
        Agent's watchdog, retained between rounds.
    journal : JournalWriter
        Loop-owned event writer.
    board : Board
        Run's live board.
    executor : Executor
        Tool executor.
    artifacts : ArtifactStore
        Content-addressed artifact store.
    ids : IdGen
        Run's identifier generator.
    config : RunConfig
        Run configuration.
    resumed : tuple[Candidate, ...]
        Own committed candidates of a restarted round.
    rankings : Mapping[str, str], optional
        Group to ranking judge name for all declared current groups. The main group key is empty.
    sibling_views : Mapping[str, BoardView], optional
        Fixed declared sibling capabilities; the main group key is empty.
    started_runs : int, default=0
        Starts whose durable tool_start or tool_dispatch returned in this round.
        On resume the runner restores the current goal/team/agent/round count
        from committed tool_start or tool_dispatch events, including completed
        and lost runs.
    seed_parents : tuple[str, ...], default=()
        Ordered assigned parent ids; exposed read-only to the manager and
        prepended to explicit parents with stable deduplication. The runner
        supplies a view permitting these assigned candidates.
    private_paths : frozenset[str], default=frozenset()
        Every check and judge file path of the current revision.

    Examples
    --------
    >>> from dataclasses import fields, MISSING
    >>> from unittest.mock import Mock
    >>> services = {f.name: Mock(name=f.name) for f in fields(AgentRuntime)
    ...             if f.init and f.default is MISSING and f.default_factory is MISSING}
    >>> services.update(address="g/t/a", holder="g/:final")
    >>> rt = AgentRuntime(**services)
    >>> rt.holder, rt.private_paths
    ('g/:final', frozenset())
    """

    stops: ProviderStops = field(init=False)
    address: str
    holder: str
    goal_id: str
    goal: Goal
    goal_key: GoalKey
    part: str | None
    team: TeamSpec
    spec: AgentSpec
    round: int
    round_spec: RoundSpec
    fidelity: str
    tool: Tool
    tool_ref: Ref
    execution_ref: Ref
    tool_identity: str
    catalog_entry: CatalogEntry | None
    checks: Sequence[tuple[str, Check]]
    view: BoardView
    own: list[Candidate | Learning]
    digest: str
    notes: str
    ledger: Ledger
    tranche: Tranche
    limits: Limits
    counters: AgentCounters
    repeats: RepeatIndex
    watchdog: Watchdog
    journal: JournalWriter
    board: Board
    executor: Executor
    artifacts: ArtifactStore
    ids: IdGen
    config: RunConfig
    resumed: tuple[Candidate, ...]
    started_runs: int = 0
    seed_parents: tuple[str, ...] = ()
    private_paths: frozenset[str] = frozenset()
    rankings: Mapping[str, str] = field(default_factory=dict)
    sibling_views: Mapping[str, BoardView] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Share the journal owner's provider service with the manager path."""
        from swarm_arena._engine.stops import provider_stops

        self.stops = provider_stops(self.journal)
