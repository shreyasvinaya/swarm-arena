"""Index a single run's committed records and freeze position-bounded views."""

from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field

from swarm_arena._board.base import Board, BoardView
from swarm_arena._board.frontier import build_frontier
from swarm_arena._journal.lease import Head
from swarm_arena._journal.replay import RunState
from swarm_arena._model.board_types import Frontier, LineageGraph, Row
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.planning_types import Decision
from swarm_arena._model.results import Candidate, Learning, Score
from swarm_arena._scoring.ranking import ranked


@dataclass
class _Records:
    """Own independently copied indexed records and their journal positions."""

    goals: dict[GoalKey, tuple[str, str | None]] = field(default_factory=dict)
    candidates: dict[str, Candidate] = field(default_factory=dict)
    scores: dict[str, Score] = field(default_factory=dict)
    learnings: dict[str, Learning] = field(default_factory=dict)
    positions: dict[str, tuple[int, int]] = field(default_factory=dict)


class _MemoryReads:
    """Implement the common live-board and fixed-snapshot read operations."""

    _records: _Records
    _scope: GoalKey | None = None

    def _group(self, goal: GoalKey, part: str | None = None) -> tuple[str, str | None]:
        """Resolve a registered goal after enforcing the optional view scope."""
        if self._scope is not None and self._scope != goal:
            raise ValueError("goal outside board snapshot")
        if goal not in self._records.goals:
            raise ValueError("unregistered board goal")
        goal_id, registered_part = self._records.goals[goal]
        return goal_id, registered_part if part is None else part

    def _namespace(self, namespace: str, goal: GoalKey | None) -> None:
        """Enforce namespace and explicit goal arguments at the view boundary."""
        if self._scope is not None and namespace != self._scope.namespace:
            raise ValueError("namespace outside board snapshot")
        if goal is not None:
            self._group(goal)
            if namespace != goal.namespace:
                raise ValueError("goal belongs to another namespace")

    def _pairs(
        self,
        goal: GoalKey,
        *,
        judge: str,
        run_id: str | None = None,
        part: str | None = None,
        fidelity: str | None = None,
    ) -> list[tuple[Candidate, Score]]:
        """Rank one group using the last put score's complete judge identity."""
        goal_id, part = self._group(goal, part)
        candidates = [
            c
            for c in self._records.candidates.values()
            if c.goal_id == goal_id and c.part == part and (run_id is None or c.run_id == run_id)
        ]
        ids = {candidate.id for candidate in candidates}
        scores = [
            s for s in self._records.scores.values() if s.candidate_id in ids and s.judge == judge
        ]
        if not scores:
            return []
        return ranked(
            candidates, scores, identity=scores[-1].identity, fidelity=fidelity, allow_partial=False
        )

    def best(
        self,
        goal: GoalKey,
        k: int,
        *,
        judge: str,
        run_id: str | None = None,
        part: str | None = None,
        fidelity: str | None = None,
    ) -> list[Candidate]:
        """
        Return independent copies of the highest-ranked candidates.

        Parameters
        ----------
        goal : GoalKey
            Registered goal identity.
        k : int
            Maximum result count.
        judge : str
            Ranking judge name.
        run_id : str or None, default=None
            Optional producing run filter.
        part : str or None, default=None
            Part override, defaulting to the registered part.
        fidelity : str or None, default=None
            Explicit fidelity, or the group's newest fidelity.

        Returns
        -------
        list[Candidate]
            Complete, gated-in candidates under one judge identity.

        Examples
        --------
        >>> board.best(key, 5, judge="program")  # doctest: +SKIP
        []
        """
        pairs = self._pairs(goal, judge=judge, run_id=run_id, part=part, fidelity=fidelity)
        return [deepcopy(candidate) for candidate, _ in pairs[: max(0, k)]]

    def learnings(
        self,
        namespace: str,
        *,
        tags: Sequence[str] = (),
        goal: GoalKey | None = None,
        limit: int = 50,
    ) -> list[Learning]:
        """
        Return namespace learnings in pinned, goal, verified, tag, newest order.

        Parameters
        ----------
        namespace : str
            Problem family.
        tags : Sequence[str], default=()
            Preferred tags.
        goal : GoalKey or None, default=None
            Goal preferred after pinned records.
        limit : int, default=50
            Maximum result count.

        Returns
        -------
        list[Learning]
            Independent visible learning copies in relevance order.

        Examples
        --------
        >>> RunBoard(run_id="r", run_root="/runs", head=lambda: None).learnings("n")
        []
        """
        self._namespace(namespace, goal)
        goal_id = self._group(goal)[0] if goal is not None else None
        wanted = set(tags)
        items = [
            item
            for item in self._records.learnings.values()
            if item.namespace == namespace and not item.hidden
        ]
        items.sort(
            key=lambda item: (
                item.pinned,
                item.goal_id == goal_id,
                item.verified,
                bool(wanted.intersection(item.tags)),
                self._records.positions[item.id],
            ),
            reverse=True,
        )
        return deepcopy(items[: max(0, limit)])

    def search_learnings(
        self,
        namespace: str,
        query: str,
        *,
        tags: Sequence[str] = (),
        goal: GoalKey | None = None,
        limit: int = 10,
    ) -> list[Learning]:
        """
        Find visible substring matches in descending journal order.

        Parameters
        ----------
        namespace : str
            Problem family.
        query : str
            Case-insensitive title or body substring.
        tags : Sequence[str], default=()
            Required tags.
        goal : GoalKey or None, default=None
            Optional validated goal scope.
        limit : int, default=10
            Maximum result count.

        Returns
        -------
        list[Learning]
            Independent matches with all requested tags.

        Examples
        --------
        >>> board.search_learnings("n", "baseline", tags=["fit"])  # doctest: +SKIP
        []
        """
        self._namespace(namespace, goal)
        query, wanted = query.casefold(), set(tags)
        items = [
            item
            for item in self._records.learnings.values()
            if item.namespace == namespace
            and not item.hidden
            and wanted.issubset(item.tags)
            and (query in item.title.casefold() or query in item.text.casefold())
        ]
        items.sort(key=lambda item: self._records.positions[item.id], reverse=True)
        return deepcopy(items[: max(0, limit)])

    def get_learning(self, id: str) -> Learning:
        """
        Read a visible learning without sharing mutable containers.

        Parameters
        ----------
        id : str
            Learning identifier.

        Returns
        -------
        Learning
            Independent learning copy.

        Raises
        ------
        KeyError
            If the record is absent or hidden.
        ValueError
            If the namespace is outside the view.

        Examples
        --------
        >>> view.get_learning("L-1").id  # doctest: +SKIP
        'L-1'
        """
        item = self._records.learnings[id]
        self._namespace(item.namespace, None)
        if item.hidden:
            raise KeyError(id)
        return deepcopy(item)

    def get_candidate(self, id: str) -> Candidate:
        """
        Read a visible candidate from the view's goal and part.

        Parameters
        ----------
        id : str
            Candidate identifier.

        Returns
        -------
        Candidate
            Independent candidate copy.

        Raises
        ------
        KeyError
            If the record is outside the frontier.
        ValueError
            If it belongs to another goal or part.

        Examples
        --------
        >>> view.get_candidate("c-1").id  # doctest: +SKIP
        'c-1'
        """
        candidate = self._records.candidates[id]
        if self._scope is not None:
            goal_id, part = self._group(self._scope)
            if (candidate.goal_id, candidate.part) != (goal_id, part):
                raise ValueError("candidate outside board snapshot goal")
        return deepcopy(candidate)

    def scores(self, candidate_id: str) -> list[Score]:
        """
        Return all scores for a visible candidate in recorded order.

        Parameters
        ----------
        candidate_id : str
            Visible candidate identifier.

        Returns
        -------
        list[Score]
            Independent score copies, including old judge identities.

        Examples
        --------
        >>> view.scores("c-1")  # doctest: +SKIP
        []
        """
        self.get_candidate(candidate_id)
        return deepcopy(
            [s for s in self._records.scores.values() if s.candidate_id == candidate_id]
        )

    def lineage(self, candidate_id: str) -> LineageGraph:
        """
        Traverse visible same-group ancestry with cycle detection.

        Parameters
        ----------
        candidate_id : str
            Root candidate identifier.

        Returns
        -------
        LineageGraph
            Independent visible nodes and ordered child-to-parent edges.

        Examples
        --------
        >>> view.lineage("c-1").root  # doctest: +SKIP
        'c-1'
        """
        root = self.get_candidate(candidate_id)
        nodes, edges = {}, []
        queue = deque([root.id])
        while queue:
            id = queue.popleft()
            if id in nodes:
                continue
            candidate = self._records.candidates[id]
            nodes[id] = deepcopy(candidate)
            for parent_id in candidate.parent_ids:
                parent = self._records.candidates.get(parent_id)
                if parent is not None and (parent.goal_id, parent.part) == (
                    root.goal_id,
                    root.part,
                ):
                    edges.append((id, parent_id))
                    queue.append(parent_id)
        return LineageGraph(root=root.id, nodes=nodes, edges=edges)

    def leaderboard(self, goal: GoalKey, *, judge: str) -> list[Row]:
        """
        Build one-based leaderboard rows with complete ranking provenance.

        Parameters
        ----------
        goal : GoalKey
            Registered goal identity.
        judge : str
            Ranking judge name.

        Returns
        -------
        list[Row]
            Complete ranking under the latest identity and newest fidelity.

        Examples
        --------
        >>> view.leaderboard(view.goal, judge="program")  # doctest: +SKIP
        []
        """
        return [
            Row(
                rank=rank,
                candidate_id=c.id,
                run_id=c.run_id,
                goal_id=c.goal_id,
                part=c.part,
                team=c.team,
                agent=c.agent,
                fidelity=c.fidelity,
                key=s.key,
                value=s.value,
                judge=s.judge,
                rubric_hash=s.rubric_hash,
                panel_hash=s.panel_hash,
                built_on=list(c.parent_ids),
                partial=c.output.partial,
            )
            for rank, (c, s) in enumerate(self._pairs(goal, judge=judge), 1)
        ]


class _RunView(_MemoryReads, BoardView):
    """Own records frozen at a frontier, enforcing one goal's read boundary."""

    def __init__(self, goal: GoalKey, frontier: Frontier, records: _Records) -> None:
        self.goal = goal
        self.frontier = deepcopy(frontier)
        self._scope = goal
        self._records = records


class RunBoard(_MemoryReads, Board):
    """
    Index one run's journaled records in memory without writing storage.

    Parameters
    ----------
    run_id : str
        Producing run identifier.
    run_root : str
        Storage root containing the run.
    head : Callable[[], Head or None]
        Current committed journal-head supplier.

    Examples
    --------
    >>> board = RunBoard(run_id="r", run_root="/runs", head=lambda: None)
    >>> board.config.model_dump(), board.frontier().marks
    ({}, [])
    """

    def __init__(self, *, run_id: str, run_root: str, head: Callable[[], Head | None]) -> None:
        super().__init__()
        self._run_id = run_id
        self._run_root = run_root
        self._head = head
        self._records = _Records()
        self._decisions: dict[str, list[tuple[int, Decision]]] = {}

    def register_goal(self, key: GoalKey, goal_id: str, part: str | None) -> None:
        """
        Associate a snapshotted goal identity with its stored ranking group.

        Parameters
        ----------
        key : GoalKey
            Already-snapshotted main or part goal identity.
        goal_id : str
            Main goal identifier stored on candidates and learnings.
        part : str or None
            Part name, or None for the main group.

        Examples
        --------
        >>> from swarm_arena import Goal
        >>> board = RunBoard(run_id="r", run_root="/runs", head=lambda: None)
        >>> board.register_goal(Goal(id="g", namespace="n").key(), "g", None)
        """
        self._records.goals[key] = (goal_id, part)

    def put_candidate(self, candidate: Candidate, *, epoch: int, seq: int) -> None:
        """
        Index an independent copy of a journaled candidate.

        Parameters
        ----------
        candidate : Candidate
            Committed candidate.
        epoch : int
            Recorded ownership epoch.
        seq : int
            Recorded run-wide sequence number.

        Examples
        --------
        >>> board.put_candidate(candidate, epoch=1, seq=1)  # doctest: +SKIP
        """
        self._records.candidates[candidate.id] = deepcopy(candidate)
        self._records.positions[candidate.id] = (epoch, seq)

    def put_score(self, score: Score, *, epoch: int, seq: int) -> None:
        """
        Index a stamped score; the last put defines the current identity.

        Parameters
        ----------
        score : Score
            Committed stamped score.
        epoch : int
            Recorded ownership epoch.
        seq : int
            Recorded run-wide sequence number.

        Examples
        --------
        >>> board.put_score(score, epoch=1, seq=2)  # doctest: +SKIP
        """
        self._records.scores.pop(score.id, None)
        self._records.scores[score.id] = deepcopy(score)
        self._records.positions[score.id] = (epoch, seq)

    def put_learning(self, learning: Learning, *, epoch: int, seq: int) -> None:
        """
        Index an independent copy of a journaled namespace learning.

        Parameters
        ----------
        learning : Learning
            Committed learning, including visibility flags.
        epoch : int
            Recorded ownership epoch.
        seq : int
            Recorded run-wide sequence number.

        Examples
        --------
        >>> board.put_learning(learning, epoch=1, seq=3)  # doctest: +SKIP
        """
        self._records.learnings[learning.id] = deepcopy(learning)
        self._records.positions[learning.id] = (epoch, seq)

    def put_decision(
        self,
        goal_id: str,
        round: int,
        decision: Decision,
        *,
        epoch: int,
        seq: int,
    ) -> None:
        """
        Retain a journaled policy decision without exposing it in board reads.

        Parameters
        ----------
        goal_id : str
            Main goal identifier.
        round : int
            Completed round number.
        decision : Decision
            Policy result.
        epoch : int
            Recorded ownership epoch; unused by board reads.
        seq : int
            Recorded sequence number; unused by board reads.

        Examples
        --------
        >>> board = RunBoard(run_id="r", run_root="/runs", head=lambda: None)
        >>> board.put_decision("g", 1, Decision(), epoch=1, seq=4)
        """
        self._decisions.setdefault(goal_id, []).append((round, deepcopy(decision)))

    def frontier(self) -> Frontier:
        """
        Capture this run's current committed head, or an empty frontier.

        Returns
        -------
        Frontier
            A single exact run mark when a committed head exists.

        Examples
        --------
        >>> RunBoard(run_id="r", run_root="/runs", head=lambda: None).frontier().marks
        []
        """
        head = self._head()
        return build_frontier(
            [] if head is None else [(self._run_id, self._run_root, head)], [], []
        )

    def snapshot(self, goal: GoalKey, frontier: Frontier) -> BoardView:
        """
        Copy records at or before this run's exact frontier mark.

        Parameters
        ----------
        goal : GoalKey
            Registered main or part identity.
        frontier : Frontier
            Recorded committed marks; a missing run mark admits no records.

        Returns
        -------
        BoardView
            Independent fixed snapshot with namespace-wide learning visibility.

        Raises
        ------
        ValueError
            If the goal is not registered.

        Examples
        --------
        >>> from swarm_arena import Goal
        >>> board = RunBoard(run_id="r", run_root="/runs", head=lambda: None)
        >>> key = Goal(id="g", namespace="n").key()
        >>> board.register_goal(key, "g", None)
        >>> board.snapshot(key, board.frontier()).best(key, 1, judge="program")
        []
        """
        self._group(goal)
        mark = next(
            (
                mark
                for mark in frontier.marks
                if (mark.run_id, mark.run_root) == (self._run_id, self._run_root)
            ),
            None,
        )
        records = _Records(goals=self._records.goals.copy())
        if mark is not None:
            visible = {
                id
                for id, position in self._records.positions.items()
                if position <= (mark.epoch, mark.seq)
            }
            records.candidates = {
                id: c for id, c in self._records.candidates.items() if id in visible
            }
            records.scores = {id: s for id, s in self._records.scores.items() if id in visible}
            records.learnings = {
                id: item for id, item in self._records.learnings.items() if id in visible
            }
            records.positions = {id: self._records.positions[id] for id in visible}
        return _RunView(goal, frontier, deepcopy(records))

    @classmethod
    def from_state(
        cls,
        state: RunState,
        *,
        run_root: str,
        head: Callable[[], Head | None],
    ) -> "RunBoard":
        """
        Rebuild registered goals and positioned records from committed replay.

        Parameters
        ----------
        state : RunState
            Pure replay state with already-snapshotted latest plan revisions.
        run_root : str
            Storage root containing the run.
        head : Callable[[], Head or None]
            Current committed head supplier.

        Returns
        -------
        RunBoard
            Board equivalent to indexing the replayed events in journal order.
            Decisions need no board position because no read exposes them.

        Examples
        --------
        >>> RunBoard.from_state(RunState(), run_root="/runs", head=lambda: None).frontier().marks
        []
        """
        board = cls(run_id=state.run_id, run_root=run_root, head=head)
        for revisions in state.plans.values():
            if not revisions:
                continue
            plan = revisions[-1][0]
            board.register_goal(plan.goal.key(), plan.goal.id, None)
            for part in plan.parts:
                board.register_goal(part.goal.key(), plan.goal.id, part.name)
        for candidate in sorted(
            state.candidates.values(), key=lambda item: state.positions[item.id]
        ):
            epoch, seq = state.positions[candidate.id]
            board.put_candidate(candidate, epoch=epoch, seq=seq)
        for score in sorted(state.scores.values(), key=lambda item: state.positions[item.id]):
            epoch, seq = state.positions[score.id]
            board.put_score(score, epoch=epoch, seq=seq)
        for learning in sorted(state.learnings.values(), key=lambda item: state.positions[item.id]):
            epoch, seq = state.positions[learning.id]
            board.put_learning(learning, epoch=epoch, seq=seq)
        board._decisions = {
            goal_id: deepcopy(goal.decisions) for goal_id, goal in state.goals.items()
        }
        return board
