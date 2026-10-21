"""Expose loop-owned, limit-checked actions to manager connectors."""

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from threading import get_ident
from typing import Any, Literal, TypeVar

import anyio
from pydantic import BaseModel, ValidationError

from swarm_arena._agent.manager import LimitReached
from swarm_arena._agent.runtime import AgentRuntime
from swarm_arena._agent.tool_run import _prepare_sync, execute_tool_run
from swarm_arena._budget.llm_scope import current_requesting_call
from swarm_arena._errors import (
    ConfigError,
    SoftError,
    _core_signal,
    _exception_type_name,
    _is_core_signal,
)
from swarm_arena._journal.a2a import agent_task_id
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._model.base import JsonValue
from swarm_arena._model.board_types import Row
from swarm_arena._model.diagnostics import validation_summary
from swarm_arena._model.goal import Goal
from swarm_arena._model.plan import AgentSpec, RoundSpec
from swarm_arena._model.ref import _validate_config
from swarm_arena._model.results import Candidate, Learning, Score
from swarm_arena._model.tooling import Estimate
from swarm_arena._outcomes import _classify_outcome, _Outcome
from swarm_arena._scoring.check import Check
from swarm_arena._scoring.runner import for_manager
from swarm_arena._tooling.staging import effective_files
from swarm_arena._tooling.tool import Action, Tool

_T = TypeVar("_T")


@dataclass(eq=False)
class _ActionLifetime:
    """Revoke accounting suppression in every inherited task context at once."""

    owner: object
    live: bool = True


_nested: ContextVar[tuple[_ActionLifetime, ...]] = ContextVar("agent_nested_actions", default=())


@dataclass(frozen=True)
class _ToolMetadata:
    """Carry prepared hook results without running connector code on the loop."""

    description: str
    input_model: type[BaseModel]
    editable: tuple[str, ...]
    actions: tuple[Action[Any], ...]
    feedback: Mapping[str, str]


def _prepare_metadata(tool: Tool, checks: Sequence[tuple[str, Check]]) -> _ToolMetadata:
    """Read and validate metadata in the runner's abandonable daemon boundary."""
    try:
        description = tool.get_description()
        input_model = tool.get_input_model()
        editable = tuple(tool.editable())
        actions = tuple(tool.extra_actions())
        feedback = {name: check.feedback for name, check in checks}
        if not isinstance(description, str):
            raise TypeError("invalid description")
        if not isinstance(input_model, type) or not issubclass(input_model, BaseModel):
            raise TypeError("invalid input model")
        if not all(isinstance(path, str) for path in editable):
            raise TypeError("invalid editable paths")
        if not all(isinstance(action, Action) for action in actions):
            raise TypeError("invalid actions")
        if not all(mode in {"pass_fail", "reasons"} for mode in feedback.values()):
            raise TypeError("invalid feedback policy")
        return _ToolMetadata(description, input_model, editable, actions, feedback)
    except BaseException as error:
        error = _classify_outcome(error)
        raise _core_signal(
            SoftError(f"invalid tool metadata: {_exception_type_name(error)}")
        ) from None


class AgentContext:
    """
    Provide one agent's round view and limit-checked manager actions.

    Constructed by the core on the owning event-loop thread. Synchronous reads
    from AnyIO action workers marshal to that loop; arbitrary unmanaged threads
    are unsupported. Extra actions charge once, including their nested reads.
    Child tasks released after the enclosing action exits are charged normally.
    The runner prepares connector metadata in supervised daemon workers before
    constructing this loop-owned context. Feedback uses the producing policy;
    unknown policies hide reasons and details.

    Parameters
    ----------
    rt : AgentRuntime
        Agent services, snapshot and mutable round state.
    gate : Callable[[], LimitReached or None], optional
        Return a stopping exception when new actions are blocked.

    Attributes
    ----------
    goal : Goal
        Agent's goal or part sub-goal.
    spec : AgentSpec
        Agent specification.
    round : int
        Current round number.
    round_spec : RoundSpec
        Current effort settings.
    digest : str
        Round-start board digest.
    seed_parents : tuple[str, ...]
        Read-only ordered assigned parent ids, prepended to every tool request.
    resumed : bool
        Whether committed results exist for the restarted round.
    tool_description : str
        Effective tool description.
    input_model : type[BaseModel]
        Effective tool input model.
    editable : tuple[str, ...]
        Editable file patterns.
    actions : tuple[Action, ...]
        Extra tool actions.

    Examples
    --------
    Metadata and actions can run in memory with mocked service boundaries.
    New tool executions require the core's active execution services.

    >>> from unittest.mock import Mock
    >>> from swarm_arena import AgentSpec, Goal, Ref, RoundSpec, ScriptTool, TeamSpec
    >>> tool = ScriptTool(command=["echo"])
    >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
    ...                  manager=Ref(path="example:Manager"))
    >>> rt = Mock(goal=Goal(id="g", namespace="demo"),
    ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
    ...           round=1, round_spec=RoundSpec(), digest="", notes="try a smaller step",
    ...           resumed=(), checks=[], tool=tool, own=[],
    ...           repeats=Mock(candidates_of=lambda author: []))
    >>> rt.spec = spec
    >>> ctx = AgentContext(rt)
    >>> ctx.notes
    'try a smaller step'
    >>> ctx.my_candidates()
    []
    """

    def __init__(
        self, rt: AgentRuntime, *, gate: Callable[[], LimitReached | None] = lambda: None
    ) -> None:
        """Prepare metadata for standalone synchronous callers and bind state.

        The async engine uses the private binding seam with a supervised
        metadata snapshot, so no connector hooks run on its event loop.
        """
        self._bind(rt, gate=gate, metadata=_prepare_metadata(rt.tool, rt.checks))

    def _bind(
        self,
        rt: AgentRuntime,
        *,
        gate: Callable[[], LimitReached | None],
        metadata: _ToolMetadata,
        feedback_for: Callable[[Candidate], Mapping[str, str]] | None = None,
    ) -> None:
        """Bind runtime state and already prepared metadata on the owner loop."""
        self._rt, self._gate, self._thread = rt, gate, get_ident()
        self._read_ids: dict[str, None] = {}
        self._feedback = dict(metadata.feedback)
        self._feedback_for = feedback_for
        self.goal: Goal = rt.goal.model_copy(deep=True)
        self.spec: AgentSpec = rt.spec.model_copy(deep=True)
        self.round: int = rt.round
        self.round_spec: RoundSpec = rt.round_spec.model_copy(deep=True)
        self.digest: str = rt.digest
        self.resumed: bool = bool(rt.resumed)
        self.tool_description: str = metadata.description
        self.input_model: type[BaseModel] = metadata.input_model
        self.editable: tuple[str, ...] = metadata.editable
        self.actions: tuple[Action[Any], ...] = metadata.actions

    def _on_loop(self, fn: Callable[[], _T]) -> _T:
        """Marshal an AnyIO worker's complete lookup and accounting operation."""
        return fn() if get_ident() == self._thread else anyio.from_thread.run_sync(fn)

    def _event(self, type: EventType, payload: dict[str, Any]) -> EventDraft:
        """Stamp the owning agent task on an ordinary event."""
        rt = self._rt
        return EventDraft(
            type=type,
            task_id=agent_task_id(rt.address, rt.round),
            goal_id=rt.goal_id,
            team=rt.team.name,
            agent=rt.spec.name,
            round=rt.round,
            payload=payload,
        )

    def _charge(self, *, board_read: bool = False) -> None:
        """Check gates and update counters atomically on the loop."""
        rt = self._rt
        blocked = self._gate()
        if blocked is not None:
            raise _core_signal(LimitReached(blocked.limit, str(blocked)))
        if rt.watchdog.limit is not None:
            raise _core_signal(
                LimitReached(rt.watchdog.limit, rt.watchdog.reason or "agent stopped")
            )
        lifetime = next((item for item in reversed(_nested.get()) if item.owner is self), None)
        if lifetime is not None and lifetime.live:
            return
        if rt.counters.turns >= rt.limits.max_turns_per_round:
            raise _core_signal(LimitReached("max_turns_per_round", "turns used up for this round"))
        rt.counters.turns += 1
        exhausted = board_read and rt.counters.board_reads >= rt.limits.max_board_reads_per_round
        if board_read and not exhausted:
            rt.counters.board_reads += 1
        rt.journal.append(
            self._event(
                EventType.AGENT_COUNTERS,
                {"turns": rt.counters.turns, "board_reads": rt.counters.board_reads},
            )
        )
        if exhausted:
            raise _core_signal(SoftError("max_board_reads_per_round reached"))

    def _manager(self, candidate: Candidate) -> Candidate:
        """Filter feedback without exposing mutable runtime-owned containers."""
        if self._feedback_for is not None:
            feedback = self._feedback_for(candidate)
        elif (
            candidate.run_id == self._rt.repeats.run_id
            and candidate.goal_id == self._rt.goal_id
            and candidate.part == self._rt.part
            and candidate.team == self._rt.team.name
            and candidate.agent == self._rt.spec.name
            and candidate.round == self.round
        ):
            feedback = self._feedback
        else:
            feedback = {}
        return for_manager(candidate.model_copy(deep=True), feedback)

    def _candidate(self, id: str) -> Candidate:
        """Resolve an own item or snapshot result without charging a nested read."""
        candidate = next((c for c in self._rt.own if isinstance(c, Candidate) and c.id == id), None)
        if candidate is None:
            try:
                candidate = self._rt.view.get_candidate(id)
            except (KeyError, ValueError):
                raise _core_signal(SoftError(f"unknown candidate: {id}")) from None
        return candidate

    def _learning(self, id: str) -> Learning:
        """Resolve a visible learning without a separate action charge."""
        learning = next(
            (
                item
                for item in self._rt.own
                if isinstance(item, Learning) and item.id == id and not item.hidden
            ),
            None,
        )
        if learning is None:
            try:
                learning = self._rt.view.get_learning(id)
            except (KeyError, ValueError):
                raise _core_signal(SoftError(f"unknown learning: {id}")) from None
        return learning.model_copy(deep=True)

    @property
    def seed_parents(self) -> tuple[str, ...]:
        """
        Return the ordered parent identifiers assigned by the runner.

        Returns
        -------
        tuple[str, ...]
            Read-only seeds prepended to explicit parents, with first occurrence
            winning when the pipeline removes duplicates.

        Examples
        --------
        >>> AgentContext.seed_parents.fset is None
        True
        """
        return self._rt.seed_parents

    @property
    def notes(self) -> str:
        """
        Return carried notes without charging a turn.

        Returns
        -------
        str
            Most recently saved notes.

        Examples
        --------
        The :class:`AgentContext` example constructs a context in memory and
        reads its carried notes without an action or a running journal.
        """
        return self._rt.notes

    @property
    def budget(self) -> Estimate:
        """
        Return the holder's remaining capped budget without charging a turn.

        Returns
        -------
        Estimate
            Remaining units; uncapped units are None.

        Examples
        --------
        Use in-memory service doubles to exercise the action without storage.

        >>> from unittest.mock import Mock
        >>> from swarm_arena import AgentSpec, Goal, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
        >>> tool = ScriptTool(command=["echo"])
        >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
        ...                  manager=Ref(path="example:Manager"))
        >>> rt = Mock(goal=Goal(id="g", namespace="demo"), goal_id="g", address="g/t/a",
        ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        ...           round=1, round_spec=RoundSpec(), digest="", resumed=(), checks=[],
        ...           tool=tool, own=[], limits=Limits(), watchdog=Mock(limit=None),
        ...           counters=Mock(turns=0, board_reads=0, learnings=0))
        >>> rt.spec = spec
        >>> ctx = AgentContext(rt)
        >>> from swarm_arena.tool import Estimate
        >>> rt.holder = "g/t/a"
        >>> rt.ledger.limits = Limits(max_usd=5)
        >>> rt.ledger.remaining.return_value.to_estimate.return_value = Estimate(usd=3)
        >>> assert ctx.budget == Estimate(tokens=None, usd=3)
        >>> rt.ledger.remaining.assert_called_once_with("g/t/a")
        >>> rt.ledger.remaining.return_value.to_estimate.assert_called_once_with(
        ...     cap_tokens=False, cap_usd=True)
        >>> assert rt.counters.turns == 0
        """

        def read() -> Estimate:
            ledger = self._rt.ledger
            return ledger.remaining(self._rt.holder).to_estimate(
                cap_tokens=ledger.limits.max_tokens is not None,
                cap_usd=ledger.limits.max_usd is not None,
            )

        return self._on_loop(read)

    async def run_tool(
        self,
        inputs: Mapping[str, Any],
        *,
        parents: Sequence[str] = (),
        replicate: int | None = None,
        edits: Mapping[str, str] | None = None,
    ) -> Candidate:
        """
        Run a tool after gate, watchdog and turn admission.

        Parameters
        ----------
        inputs : Mapping[str, Any]
            Tool inputs.
        parents : Sequence[str], default=()
            Visible parent candidate identifiers. Assigned seed_parents are
            prepended, with stable deduplication (first occurrence wins).
        replicate : int or None, default=None
            Explicit replicate, or automatic allocation.
        edits : Mapping[str, str] or None, default=None
            Editable file contents.

        Returns
        -------
        Candidate
            Result with manager-visible check feedback. New results record the
            ordered read identifiers at action entry. Linked or recovered results
            retain their original read provenance.

        Examples
        --------
        >>> candidate = await ctx.run_tool({"x": 1.0})  # doctest: +SKIP
        """
        self._charge()
        read_ids = tuple(self._read_ids)
        candidate = await execute_tool_run(
            self._rt, inputs, parents=parents, replicate=replicate, edits=edits, read_ids=read_ids
        )
        return self._manager(candidate)

    def my_candidates(self) -> list[Candidate]:
        """
        Return this agent's prior and current results without charging a turn.

        Returns
        -------
        list[Candidate]
            Independent filtered copies, deduplicated by identifier.

        Examples
        --------
        The :class:`AgentContext` example calls this method with an empty
        in-memory result index and returns an empty list.
        """

        def read() -> list[Candidate]:
            rt = self._rt
            candidates = rt.repeats.candidates_of(f"{rt.team.name}/{rt.spec.name}")
            candidates += [c for c in rt.own if isinstance(c, Candidate)]
            unique = {c.id: c for c in candidates}
            return [self._manager(c) for c in unique.values()]

        return self._on_loop(read)

    def scores_of(self, candidate_id: str) -> list[Score]:
        """
        Read a visible candidate's scores with one turn and board-read charge.

        Parameters
        ----------
        candidate_id : str
            Visible candidate identifier.

        Returns
        -------
        list[Score]
            Independent score records from the snapshot; new unjudged candidates have none.

        Examples
        --------
        Use in-memory service doubles to exercise the action without storage.

        >>> from unittest.mock import Mock
        >>> from swarm_arena import AgentSpec, Goal, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
        >>> tool = ScriptTool(command=["echo"])
        >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
        ...                  manager=Ref(path="example:Manager"))
        >>> rt = Mock(goal=Goal(id="g", namespace="demo"), goal_id="g", address="g/t/a",
        ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        ...           round=1, round_spec=RoundSpec(), digest="", resumed=(), checks=[],
        ...           tool=tool, own=[], limits=Limits(), watchdog=Mock(limit=None),
        ...           counters=Mock(turns=0, board_reads=0, learnings=0))
        >>> rt.spec = spec
        >>> ctx = AgentContext(rt)
        >>> from swarm_arena.board import Candidate
        >>> from swarm_arena.tool import ToolOutput, ToolRequest
        >>> candidate = Candidate(id="c-1", run_id="r", goal_id="g", team="t", agent="a",
        ...     round=1, request=ToolRequest(), output=ToolOutput(status="ok"))
        >>> rt.view.get_candidate.return_value = candidate
        >>> from swarm_arena import Score
        >>> score = Score(id="s-1", candidate_id="c-1", judge="rank", key=(2.0,))
        >>> rt.view.scores.return_value = [score]
        >>> scores = ctx.scores_of("c-1")
        >>> assert scores == [score] and scores[0] is not score
        >>> assert (rt.counters.turns, rt.counters.board_reads) == (1, 1)
        """

        def read() -> list[Score]:
            self._charge(board_read=True)
            self._candidate(candidate_id)
            self._read_ids[candidate_id] = None
            try:
                scores = self._rt.view.scores(candidate_id)
            except (KeyError, ValueError):
                scores = []
            return [s.model_copy(deep=True) for s in scores]

        return self._on_loop(read)

    def search_learnings(
        self,
        query: str,
        *,
        tags: Sequence[str] = (),
        scope: Literal["goal", "namespace"] = "goal",
        limit: int = 10,
    ) -> list[Learning]:
        """
        Search snapshot and own learnings with one board-read charge.

        Parameters
        ----------
        query : str
            Case-insensitive title or text substring.
        tags : Sequence[str], default=()
            Required tags.
        scope : {'goal', 'namespace'}, default='goal'
            Search scope.
        limit : int, default=10
            Maximum number of results.

        Returns
        -------
        list[Learning]
            Independent visible matches, newest own items before snapshot items.

        Examples
        --------
        Use in-memory service doubles to exercise the action without storage.

        >>> from unittest.mock import Mock
        >>> from swarm_arena import AgentSpec, Goal, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
        >>> tool = ScriptTool(command=["echo"])
        >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
        ...                  manager=Ref(path="example:Manager"))
        >>> rt = Mock(goal=Goal(id="g", namespace="demo"), goal_id="g", address="g/t/a",
        ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        ...           round=1, round_spec=RoundSpec(), digest="", resumed=(), checks=[],
        ...           tool=tool, own=[], limits=Limits(), watchdog=Mock(limit=None),
        ...           counters=Mock(turns=0, board_reads=0, learnings=0))
        >>> rt.spec = spec
        >>> ctx = AgentContext(rt)
        >>> from swarm_arena.board import Learning
        >>> hint = Learning(id="L-1", namespace="demo", run_id="r", goal_id="g", round=1,
        ...     author="t/a", title="Baseline", text="It converged.", evidence=["c-1"])
        >>> rt.view.search_learnings.return_value = [hint]
        >>> hints = ctx.search_learnings("baseline")
        >>> assert hints == [hint] and hints[0] is not hint
        >>> assert (rt.counters.turns, rt.counters.board_reads) == (1, 1)
        """

        def read() -> list[Learning]:
            self._charge(board_read=True)
            if scope not in ("goal", "namespace"):
                raise _core_signal(SoftError("invalid learning scope"))
            rt = self._rt
            found = rt.view.search_learnings(
                rt.goal.namespace,
                query,
                tags=tags,
                goal=rt.goal_key if scope == "goal" else None,
                limit=limit,
            )
            own = [
                item
                for item in reversed(rt.own)
                if isinstance(item, Learning)
                and not item.hidden
                and item.namespace == rt.goal.namespace
                and (scope == "namespace" or item.goal_id == rt.goal_id)
                and set(tags).issubset(item.tags)
                and (
                    query.casefold() in item.title.casefold()
                    or query.casefold() in item.text.casefold()
                )
            ]
            unique = {item.id: item for item in [*own, *found]}
            result = list(unique.values())[: max(0, limit)]
            self._read_ids.update((item.id, None) for item in result)
            return [item.model_copy(deep=True) for item in result]

        return self._on_loop(read)

    def read_learning(self, id: str) -> Learning:
        """
        Read a visible learning and record its provenance.

        Parameters
        ----------
        id : str
            Learning identifier.

        Returns
        -------
        Learning
            Independent learning copy.

        Examples
        --------
        Use in-memory service doubles to exercise the action without storage.

        >>> from unittest.mock import Mock
        >>> from swarm_arena import AgentSpec, Goal, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
        >>> tool = ScriptTool(command=["echo"])
        >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
        ...                  manager=Ref(path="example:Manager"))
        >>> rt = Mock(goal=Goal(id="g", namespace="demo"), goal_id="g", address="g/t/a",
        ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        ...           round=1, round_spec=RoundSpec(), digest="", resumed=(), checks=[],
        ...           tool=tool, own=[], limits=Limits(), watchdog=Mock(limit=None),
        ...           counters=Mock(turns=0, board_reads=0, learnings=0))
        >>> rt.spec = spec
        >>> ctx = AgentContext(rt)
        >>> from swarm_arena.board import Learning
        >>> hint = Learning(id="L-1", namespace="demo", run_id="r", goal_id="g", round=1,
        ...     author="t/a", title="Baseline", text="It converged.", evidence=["c-1"])
        >>> rt.view.get_learning.return_value = hint
        >>> found = ctx.read_learning("L-1")
        >>> assert found == hint and found is not hint
        >>> assert (rt.counters.turns, rt.counters.board_reads) == (1, 1)
        """

        def read() -> Learning:
            self._charge(board_read=True)
            learning = self._learning(id)
            self._read_ids[id] = None
            return learning

        return self._on_loop(read)

    def get_candidate(self, id: str) -> Candidate:
        """
        Read a visible candidate with private feedback removed.

        Parameters
        ----------
        id : str
            Candidate identifier.

        Returns
        -------
        Candidate
            Independent manager-visible copy.

        Examples
        --------
        Use in-memory service doubles to exercise the action without storage.

        >>> from unittest.mock import Mock
        >>> from swarm_arena import AgentSpec, Goal, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
        >>> tool = ScriptTool(command=["echo"])
        >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
        ...                  manager=Ref(path="example:Manager"))
        >>> rt = Mock(goal=Goal(id="g", namespace="demo"), goal_id="g", address="g/t/a",
        ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        ...           round=1, round_spec=RoundSpec(), digest="", resumed=(), checks=[],
        ...           tool=tool, own=[], limits=Limits(), watchdog=Mock(limit=None),
        ...           counters=Mock(turns=0, board_reads=0, learnings=0))
        >>> rt.spec = spec
        >>> ctx = AgentContext(rt)
        >>> from swarm_arena.board import Candidate
        >>> from swarm_arena.tool import ToolOutput, ToolRequest
        >>> candidate = Candidate(id="c-1", run_id="r", goal_id="g", team="t", agent="a",
        ...     round=1, request=ToolRequest(), output=ToolOutput(status="ok"))
        >>> rt.view.get_candidate.return_value = candidate
        >>> found = ctx.get_candidate("c-1")
        >>> assert found == candidate and found is not candidate
        >>> assert (rt.counters.turns, rt.counters.board_reads) == (1, 1)
        """

        def read() -> Candidate:
            self._charge(board_read=True)
            candidate = self._candidate(id)
            self._read_ids[id] = None
            return self._manager(candidate)

        return self._on_loop(read)

    def _read_artifact(self, sha256: str) -> bytes:
        """Authorize snapshot and own file digests without a prior charged lookup."""

        def authorize() -> None:
            rt = self._rt
            allowed = set(rt.goal.files.manifest.values())
            candidates = [rt.view.get_candidate(id) for id in rt.view._visible_candidate_ids()]
            for candidate in [*candidates, *rt.own]:
                if isinstance(candidate, Candidate):
                    allowed.update(effective_files(rt.goal, candidate).values())
            if sha256 not in allowed:
                raise _core_signal(SoftError("file not in your view"))

        self._on_loop(authorize)
        return self._rt.artifacts.get_bytes(sha256)

    def best(self, k: int, *, part: str | None = None) -> list[Row]:
        """
        Return leaderboard rows from this agent's own group snapshot.

        Parameters
        ----------
        k : int
            Maximum number of rows.
        part : str or None, default=None
            This agent's part, or None for its default group. Cross-part reads
            require the future multi-goal board view.

        Returns
        -------
        list[Row]
            Ranked rows under this group's configured ranking judge.

        Examples
        --------
        Use in-memory service doubles to exercise the action without storage.

        >>> from unittest.mock import Mock
        >>> from swarm_arena import AgentSpec, Goal, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
        >>> tool = ScriptTool(command=["echo"])
        >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
        ...                  manager=Ref(path="example:Manager"))
        >>> rt = Mock(goal=Goal(id="g", namespace="demo"), goal_id="g", address="g/t/a",
        ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        ...           round=1, round_spec=RoundSpec(), digest="", resumed=(), checks=[],
        ...           tool=tool, own=[], limits=Limits(), watchdog=Mock(limit=None),
        ...           counters=Mock(turns=0, board_reads=0, learnings=0))
        >>> rt.spec = spec
        >>> ctx = AgentContext(rt)
        >>> from swarm_arena.board import Row
        >>> rt.part, rt.rankings, rt.goal_key = None, {"": "rank"}, rt.goal.key()
        >>> row = Row(rank=1, candidate_id="c-1", run_id="r", goal_id="g",
        ...     team="t", agent="a", part=None, fidelity="full", key=(2.0,), value=2.0,
        ...     judge="rank", rubric_hash="r", panel_hash="p", built_on=[], partial=False)
        >>> rt.view.leaderboard.return_value = [row]
        >>> assert ctx.best(1) == [row]
        >>> rt.view.leaderboard.assert_called_once_with(rt.goal_key, judge="rank")
        >>> assert (rt.counters.turns, rt.counters.board_reads) == (1, 1)
        """

        def read() -> list[Row]:
            self._charge(board_read=True)
            rt = self._rt
            if part is not None and part != rt.part:
                raise _core_signal(SoftError("best(part=...) can read only this agent's own part"))
            judge = rt.rankings.get(rt.part or "")
            if judge is None:
                raise _core_signal(SoftError("missing ranking for agent group"))
            rows = rt.view.leaderboard(rt.goal_key, judge=judge)[: max(0, k)]
            self._read_ids.update((row.candidate_id, None) for row in rows)
            return rows

        return self._on_loop(read)

    async def post_learning(
        self,
        title: str,
        text: str,
        evidence: Sequence[str],
        *,
        tags: Sequence[str] = (),
        refines: Sequence[str] = (),
        duplicate_reason: str | None = None,
    ) -> Learning:
        """
        Validate and publish one evidence-backed learning atomically.

        Parameters
        ----------
        title : str
            Nonempty single-line title, at most 120 characters.
        text : str
            Body, at most 1,500 characters.
        evidence : Sequence[str]
            Visible candidate identifiers supporting the learning.
        tags : Sequence[str], default=()
            Search tags.
        refines : Sequence[str], default=()
            Visible learnings refined by this one.
        duplicate_reason : str or None, default=None
            Accepted for compatibility; duplicate detection arrives in M2.

        Returns
        -------
        Learning
            Published learning with core-generated identity and call provenance.

        Raises
        ------
        SoftError
            If validation fails or max_learnings_per_agent_round is reached.
            This learning rate limit leaves the round open for other actions.

        Examples
        --------
        Use in-memory service doubles to exercise the action without storage.

        >>> from unittest.mock import Mock
        >>> from swarm_arena import AgentSpec, Goal, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
        >>> tool = ScriptTool(command=["echo"])
        >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
        ...                  manager=Ref(path="example:Manager"))
        >>> rt = Mock(goal=Goal(id="g", namespace="demo"), goal_id="g", address="g/t/a",
        ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        ...           round=1, round_spec=RoundSpec(), digest="", resumed=(), checks=[],
        ...           tool=tool, own=[], limits=Limits(), watchdog=Mock(limit=None),
        ...           counters=Mock(turns=0, board_reads=0, learnings=0))
        >>> rt.spec = spec
        >>> ctx = AgentContext(rt)
        >>> from swarm_arena.board import Candidate
        >>> from swarm_arena.tool import ToolOutput, ToolRequest
        >>> candidate = Candidate(id="c-1", run_id="r", goal_id="g", team="t", agent="a",
        ...     round=1, request=ToolRequest(), output=ToolOutput(status="ok"))
        >>> rt.view.get_candidate.return_value = candidate
        >>> import anyio
        >>> rt.repeats.run_id = "r"
        >>> rt.ids.next.return_value = "L-1"
        >>> rt.journal.append.return_value = Mock(epoch=1, seq=2)
        >>> hint = anyio.run(ctx.post_learning, "Baseline", "It converged.", ["c-1"])
        >>> assert (hint.id, hint.author, hint.evidence) == ("L-1", "t/a", ["c-1"])
        >>> rt.board.put_learning.assert_called_once_with(hint, epoch=1, seq=2)
        >>> assert rt.own == [hint] and rt.counters.learnings == 1
        >>> assert (rt.counters.turns, rt.counters.board_reads) == (1, 0)
        """
        self._charge()
        rt = self._rt
        try:
            learning = Learning(
                id="pending",
                namespace=rt.goal.namespace,
                run_id=rt.repeats.run_id,
                goal_id=rt.goal_id,
                round=rt.round,
                author=f"{rt.team.name}/{rt.spec.name}",
                title=title,
                text=text,
                evidence=list(evidence),
                tags=list(tags),
                refines=list(refines),
                posted_by_call=current_requesting_call(),
            )
        except ValidationError as error:
            raise _core_signal(
                SoftError(f"invalid learning: {validation_summary(error)}")
            ) from None
        for id in learning.evidence:
            self._candidate(id)
        for id in learning.refines:
            self._learning(id)
        if rt.counters.learnings >= rt.limits.max_learnings_per_agent_round:
            raise _core_signal(SoftError("max_learnings_per_agent_round reached"))
        rt.counters.learnings += 1
        learning = learning.model_copy(update={"id": rt.ids.next("L")})
        event = rt.journal.append(
            self._event(EventType.LEARNING, {"learning": learning.model_dump(mode="json")})
        )
        rt.board.put_learning(learning, epoch=event.epoch, seq=event.seq)
        rt.own.append(learning)
        return learning.model_copy(deep=True)

    async def save_notes(self, text: str) -> None:
        """
        Save bounded UTF-8 notes for the next round.

        Parameters
        ----------
        text : str
            Notes within the configured byte cap.

        Examples
        --------
        Use in-memory service doubles to exercise the action without storage.

        >>> from unittest.mock import Mock
        >>> from swarm_arena import AgentSpec, Goal, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
        >>> tool = ScriptTool(command=["echo"])
        >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
        ...                  manager=Ref(path="example:Manager"))
        >>> rt = Mock(goal=Goal(id="g", namespace="demo"), goal_id="g", address="g/t/a",
        ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        ...           round=1, round_spec=RoundSpec(), digest="", resumed=(), checks=[],
        ...           tool=tool, own=[], limits=Limits(), watchdog=Mock(limit=None),
        ...           counters=Mock(turns=0, board_reads=0, learnings=0))
        >>> rt.spec = spec
        >>> ctx = AgentContext(rt)
        >>> import anyio
        >>> anyio.run(ctx.save_notes, "Try a smaller step.")
        >>> assert ctx.notes == "Try a smaller step."
        >>> assert rt.journal.append.call_args.args[0].payload == {"text": ctx.notes}
        >>> assert (rt.counters.turns, rt.counters.board_reads) == (1, 0)
        """
        self._charge()
        try:
            size = len(text.encode("utf-8"))
        except (AttributeError, UnicodeError):
            raise _core_signal(SoftError("notes must be UTF-8 text")) from None
        if size > self._rt.limits.max_notes_bytes:
            raise _core_signal(SoftError("max_notes_bytes reached"))
        self._rt.journal.append(self._event(EventType.NOTES_SAVED, {"text": text}))
        self._rt.notes = text

    async def call_action(self, name: str, args: Mapping[str, Any]) -> JsonValue:
        """
        Validate and invoke an extra action with exactly one action charge.

        Parameters
        ----------
        name : str
            Declared extra-action name.
        args : Mapping[str, Any]
            Arguments validated against the action's input model.

        Returns
        -------
        JsonValue
            Handler result.

        Examples
        --------
        Use in-memory service doubles to exercise the action without storage.

        >>> from unittest.mock import Mock
        >>> from swarm_arena import AgentSpec, Goal, Limits, Ref, RoundSpec, ScriptTool, TeamSpec
        >>> tool = ScriptTool(command=["echo"])
        >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref.of(tool),
        ...                  manager=Ref(path="example:Manager"))
        >>> rt = Mock(goal=Goal(id="g", namespace="demo"), goal_id="g", address="g/t/a",
        ...           team=TeamSpec(name="t", rationale="Explore", agents=[spec]),
        ...           round=1, round_spec=RoundSpec(), digest="", resumed=(), checks=[],
        ...           tool=tool, own=[], limits=Limits(), watchdog=Mock(limit=None),
        ...           counters=Mock(turns=0, board_reads=0, learnings=0))
        >>> rt.spec = spec
        >>> ctx = AgentContext(rt)
        >>> import anyio
        >>> from swarm_arena.tool import FileSet
        >>> rt.goal = Goal(id="g", namespace="demo",
        ...                files=FileSet(manifest={"problem/a.phyk": "a" * 64}))
        >>> ctx = AgentContext(rt)
        >>> assert anyio.run(ctx.call_action, "list_files", {}) == {"files": ["problem/a.phyk"]}
        >>> assert (rt.counters.turns, rt.counters.board_reads) == (1, 1)
        >>> rt.artifacts.get_bytes.assert_not_called()
        """
        action = next((a for a in self.actions if a.name == name), None)
        self._charge(board_read=action is not None and action.counts_as == "board_read")
        if action is None:
            raise _core_signal(SoftError(f"unknown action: {name}"))
        try:

            def validate() -> BaseModel:
                return _validate_config(action.input_model, dict(args), where="action arguments")

            validated = await _prepare_sync(validate)
        except ConfigError as error:
            raise _core_signal(SoftError(str(error))) from None
        lifetime = _ActionLifetime(self)
        token = _nested.set((*_nested.get(), lifetime))
        outcome = _Outcome()
        try:
            return await action.handler(self, validated)
        except BaseException as error:
            error = outcome.classify(error)
            if _is_core_signal(error) and isinstance(error, (SoftError, LimitReached)):
                raise error from None
            raise _core_signal(SoftError(f"action failed: {_exception_type_name(error)}")) from None
        finally:
            lifetime.live = False
            _nested.reset(token)
