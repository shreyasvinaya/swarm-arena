"""Read shared committed journals through synchronized, fixed arena views."""

import logging
import os
from collections.abc import Callable, Sequence
from contextlib import ExitStack, suppress
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, cast

from swarm_arena._board.base import Board, BoardView
from swarm_arena._board.cache import ArenaCache, LearningFilter, Position, goal_key_digest
from swarm_arena._board.frontier import build_frontier
from swarm_arena._board.human_objects import (
    HumanObject,
    _validate_human_key,
    human_object_order,
    list_human_objects,
    read_human_object,
)
from swarm_arena._board.ingest import _validate_marks, ensure_covered, ingest_run, run_store
from swarm_arena._board.leaderboard import flatten, group_judges, group_rows
from swarm_arena._board.pointers import (
    RunPointer,
    _validate_namespace,
    _validate_run_id,
    list_pointers,
    pointer_key,
    write_pointer,
)
from swarm_arena._errors import StorageError, _core_error
from swarm_arena._journal.lease import Head, read_owner
from swarm_arena._model.board_types import Frontier, LineageGraph, Row
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.plan import Plan
from swarm_arena._model.planning_types import Decision
from swarm_arena._model.ref import _ConfigInputModel, config_or_fields
from swarm_arena._model.results import Candidate, Learning, Score
from swarm_arena._store.base import open_store

_LOG = logging.getLogger("swarm_arena.board")


@dataclass(frozen=True)
class _RunPlans:
    """Retain fixed plan resolution without aliasing mutable replay progress."""

    revisions: dict[str, tuple[Plan, ...]]
    current: dict[str, Plan]


def _resolved_root(root: str) -> str:
    """Keep remote roots intact and normalize local paths before publication."""
    return root if root.startswith("s3://") else str(Path(root).resolve())


def _namespace_frontier(frontier: Frontier, namespace: str) -> Frontier:
    """Keep all marked runs and only this cache's named human objects."""
    return frontier.model_copy(
        update={
            "reviews": [name for name in frontier.reviews if name.startswith(namespace + "/")],
            "curation": [name for name in frontier.curation if name.startswith(namespace + "/")],
        },
        deep=True,
    )


def _find_learnings(
    cache: ArenaCache,
    frontier: Frontier,
    *,
    query: str | None,
    filters: LearningFilter,
    relevance_goal: tuple[str, str] | None,
    limit: int,
) -> list[Learning]:
    """Hydrate only returned IDs while preserving the cache's relevance order."""
    ids = cache.find(
        frontier, query=query, filters=filters, relevance_goal=relevance_goal, limit=max(0, limit)
    )
    by_id = {item.id: item for item in cache.learnings_by_id(frontier, ids)}
    return [by_id[id] for id in ids]


def _lineage(
    read: Callable[[Sequence[str]], dict[str, Candidate]], candidate_id: str
) -> LineageGraph:
    """Batch each breadth-first level and look up every parent ID at most once."""
    known = read([candidate_id])
    if candidate_id not in known:
        raise KeyError(candidate_id)
    attempted = {candidate_id}
    nodes: dict[str, Candidate] = {}
    edges: list[tuple[str, str]] = []
    level = {candidate_id: known[candidate_id]}
    while level:
        nodes.update(level)
        ids = list(
            dict.fromkeys(
                parent
                for candidate in level.values()
                for parent in candidate.parent_ids
                if parent not in attempted
            )
        )
        if ids:
            attempted.update(ids)
            known.update(read(ids))
        following: dict[str, Candidate] = {}
        for candidate in level.values():
            for parent_id in candidate.parent_ids:
                if parent_id not in known:
                    continue
                edges.append((candidate.id, parent_id))
                if parent_id not in nodes:
                    following[parent_id] = known[parent_id]
        level = following
    return LineageGraph(root=candidate_id, nodes=nodes, edges=edges)


class ArenaBoard(Board):
    """
    Index shared run journals and capture exact, reproducible arena snapshots.

    Construction performs no I/O. Caches open lazily and share one board lock
    with lifecycle operations, latest-frontier publication, and hint maps.
    Journal heads and named human objects alone determine snapshot membership.

    Parameters
    ----------
    config : ArenaBoard.Config or None, optional
        Validated configuration, mutually exclusive with keyword fields.
    **fields : Any
        Configuration fields when ``config`` is omitted.

    Examples
    --------
    >>> import tempfile
    >>> from swarm_arena.board import ArenaBoard
    >>> root = tempfile.mkdtemp()
    >>> board = ArenaBoard(arena_root=root + "/arena", cache_dir=root + "/cache")
    >>> board.open("", "", ["physika"])
    >>> board.frontier().marks
    []
    >>> board.close()
    >>> import shutil
    >>> shutil.rmtree(root)
    """

    class Config(_ConfigInputModel):
        """
        Configure shared arena storage and the local disposable cache.

        Parameters
        ----------
        arena_root : str
            Local directory or S3 root holding pointers and human objects.
        cache_dir : str or None, default=None
            Local cache directory. None uses ``XDG_CACHE_HOME`` (or
            ``~/.cache``), then ``swarm-arena/<resolved-root-hash>``.

        Examples
        --------
        >>> ArenaBoard.Config(arena_root="/arena").cache_dir is None
        True
        """

        arena_root: str
        cache_dir: str | None = None

    def __init__(self, config: "ArenaBoard.Config | None" = None, /, **fields: Any) -> None:
        """Validate configuration without opening storage or resolving paths."""
        super().__init__(config_or_fields(config, fields))
        self._lock = RLock()
        self._caches: dict[str, ArenaCache] = {}
        self._namespaces: tuple[str, ...] = ()
        self._run_id = ""
        self._run_root = ""
        self._latest: Frontier | None = None
        self._candidates: dict[str, Candidate] = {}
        self._scores: dict[str, Score] = {}
        self._learnings: dict[str, Learning] = {}
        self._decisions: dict[tuple[str, int], Decision] = {}
        self._run_plans: dict[tuple[str, str], tuple[Frontier.Mark, _RunPlans]] = {}

    @property
    def config(self) -> Config:
        """
        Return the immutable arena configuration.

        Returns
        -------
        ArenaBoard.Config
            Validated storage and cache locations.

        Examples
        --------
        >>> ArenaBoard(arena_root="/arena").config.arena_root
        '/arena'
        """
        return cast(ArenaBoard.Config, self._config)

    def _cache_dir(self) -> Path:
        """Resolve the configured cache location without creating it."""
        if self.config.cache_dir is not None:
            return Path(self.config.cache_dir).expanduser().resolve()
        root = _resolved_root(self.config.arena_root)
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        return base / "swarm-arena" / sha256_hex(root.encode())[:12]

    def _cache(self, namespace: str) -> ArenaCache:
        """Validate a namespace before a synchronized first cache open."""
        _validate_namespace(namespace)
        with self._lock:
            if namespace not in self._caches:
                self._caches[namespace] = ArenaCache(self._cache_dir() / f"{namespace}.sqlite")
            return self._caches[namespace]

    def _plans_at(self, cache: ArenaCache, mark: Frontier.Mark) -> _RunPlans:
        """Memoize one committed plan projection per run under the board lock.

        Reuse the ingest checkpoint and its suffix catch-up seam under the
        cache lock. Historical reads never rewind a newer ingest checkpoint.
        Retain independent plans, not the state that later ingest mutates.
        """
        key = (cache.namespace, mark.run_id)
        previous = self._run_plans.get(key)
        if previous is not None and previous[0] == mark:
            return previous[1]
        with cache._lock:
            checkpoint = cache._ingest_states.get(mark.run_id)
            if checkpoint is not None and checkpoint[1] > mark.seq:
                state = cache._replay(mark.run_id, through_seq=mark.seq)
            else:
                state = cache._ingest_state(mark.run_id, mark.seq)
                # Coverage was validated before resolution. This fold contains
                # only committed cached events, so it can seed later ingestion.
                cache._ingest_states[mark.run_id] = (mark.segment, mark.seq, state)
            revisions = {
                goal_id: tuple(plan.model_copy(deep=True) for plan, _ in plans)
                for goal_id, plans in state.plans.items()
            }
            current = {}
            for goal_id in revisions:
                progress = state.goals[goal_id]
                round = max(1, progress.round, progress.rounds_closed + 1)
                current[goal_id] = state.plan_for_round(goal_id, round).model_copy(deep=True)
            result = _RunPlans(revisions, current)
        self._run_plans[key] = (mark.model_copy(deep=True), result)
        return result

    def open(self, run_id: str, run_root: str, namespaces: Sequence[str]) -> None:
        """
        Attach namespaces and publish write-once pointers for an owning run.

        Parameters
        ----------
        run_id : str
            Own run identifier, or an empty string for a read-only attachment.
        run_root : str
            Own run root; ignored for a reader. Local roots become absolute.
        namespaces : Sequence[str]
            Namespace slugs whose journals and objects will be read.

        Raises
        ------
        StorageError
            If a pointer cannot be written or has a conflicting identity.
        ValueError
            If a namespace or run identifier is unsafe.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root)
        ...     board.open("", "", ["toy"])
        ...     board.close()
        """
        names = tuple(sorted({_validate_namespace(ns) for ns in namespaces}))
        if run_id:
            _validate_run_id(run_id)
        resolved = _resolved_root(run_root) if run_id else ""
        with self._lock:
            if run_id:
                arena = open_store(self.config.arena_root)
                pointer = RunPointer(run_id=run_id, run_root=resolved)
                warned = False
                for namespace in names:
                    try:
                        write_pointer(arena, namespace, pointer)
                    except StorageError:
                        existing = None
                        with suppress(ValueError, StorageError, OSError):
                            existing = RunPointer.model_validate_json(
                                arena.read(pointer_key(namespace, run_id)).data
                            )
                        if (
                            existing is None
                            or existing.run_id != run_id
                            or existing.run_root == resolved
                        ):
                            raise
                        if not warned:
                            _LOG.warning(
                                "run %s resumed at %s; keeping original pointer %s",
                                run_id,
                                resolved,
                                existing.run_root,
                            )
                            warned = True
            self._run_id, self._run_root, self._namespaces = run_id, resolved, names
            self._latest = None

    def close(self) -> None:
        """
        Close every cache and release its lifetime file lock.

        Examples
        --------
        >>> board = ArenaBoard(arena_root="/arena")
        >>> board.close()
        >>> board.close()
        """
        with self._lock:
            for cache in self._caches.values():
                cache.close()
            self._caches.clear()
            self._run_plans.clear()
            self._latest = None

    def put_candidate(self, candidate: Candidate, *, epoch: int, seq: int) -> None:
        """
        Retain a candidate hint without changing committed snapshot membership.

        Parameters
        ----------
        candidate : Candidate
            Core-journaled candidate to copy.
        epoch : int
            Journal ownership epoch.
        seq : int
            Journal event sequence number.

        Examples
        --------
        >>> from swarm_arena.testing import make_candidate
        >>> from swarm_arena.tool import ToolOutput
        >>> board = ArenaBoard(arena_root="/arena")
        >>> board.put_candidate(make_candidate(ToolOutput(status="ok")), epoch=1, seq=1)
        """
        with self._lock:
            self._candidates[candidate.id] = candidate.model_copy(deep=True)

    def put_score(self, score: Score, *, epoch: int, seq: int) -> None:
        """
        Retain a score hint for board-level fallback reads.

        Parameters
        ----------
        score : Score
            Core-journaled score to copy.
        epoch : int
            Journal ownership epoch.
        seq : int
            Journal event sequence number.

        Examples
        --------
        >>> from swarm_arena import Score
        >>> board = ArenaBoard(arena_root="/arena")
        >>> board.put_score(Score.maximize(1.0), epoch=1, seq=2)
        """
        with self._lock:
            self._scores[score.id] = score.model_copy(deep=True)

    def put_learning(self, learning: Learning, *, epoch: int, seq: int) -> None:
        """
        Retain a learning hint without publishing it to fixed views.

        Parameters
        ----------
        learning : Learning
            Core-journaled learning to copy.
        epoch : int
            Journal ownership epoch.
        seq : int
            Journal event sequence number.

        Examples
        --------
        >>> from swarm_arena.board import Learning
        >>> board = ArenaBoard(arena_root="/arena")
        >>> hint = Learning(id="L-1", namespace="toy", run_id="r", goal_id="g",
        ...     round=1, author="human:a", title="Baseline", text="Start here")
        >>> board.put_learning(hint, epoch=1, seq=3)
        """
        with self._lock:
            self._learnings[learning.id] = learning.model_copy(deep=True)

    def put_decision(
        self, goal_id: str, round: int, decision: Decision, *, epoch: int, seq: int
    ) -> None:
        """
        Retain a decision hint without writing a journal or changing a frontier.

        Parameters
        ----------
        goal_id : str
            Owning main goal identifier.
        round : int
            Decision's round number.
        decision : Decision
            Core-journaled policy decision to copy.
        epoch : int
            Journal ownership epoch.
        seq : int
            Journal event sequence number.

        Examples
        --------
        >>> from swarm_arena.planning import Decision
        >>> board = ArenaBoard(arena_root="/arena")
        >>> board.put_decision("g", 1, Decision(), epoch=1, seq=4)
        """
        with self._lock:
            self._decisions[(goal_id, round)] = decision.model_copy(deep=True)

    def _load_objects(self, cache: ArenaCache, frontier: Frontier) -> None:
        """Read complete local-namespace objects and apply one atomic batch."""
        arena = open_store(self.config.arena_root)
        names: list[str] = []
        operations: list[HumanObject] = []
        empty: list[str] = []
        for name in sorted(set(frontier.reviews + frontier.curation), key=human_object_order):
            _validate_human_key(name)
            if not name.startswith(cache.namespace + "/") or cache.has_object(name):
                continue
            objects = read_human_object(arena, name)
            if objects:
                names.extend([name] * len(objects))
                operations.extend(objects)
            else:
                empty.append(name)
        if names or empty:
            cache.apply_human_objects(names, operations, covered=empty)

    def frontier(self) -> Frontier:
        """
        Capture each run's committed head once and cache its committed chain.

        Returns
        -------
        Frontier
            Canonically ordered marks and exact review and curation names.

        Raises
        ------
        StorageError
            If the owning run is unreachable or has no owner document.

        Notes
        -----
        Broken neighboring runs are warned about and omitted. The own run uses
        its open root; a not-yet-committed own run contributes no mark.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root, cache_dir=root + "/cache")
        ...     board.open("", "", ["toy"])
        ...     assert board.frontier().marks == []
        ...     board.close()
        """
        with self._lock:
            arena = open_store(self.config.arena_root)
            roots: dict[str, str] = {}
            for namespace in self._namespaces:
                pointers, warnings = list_pointers(arena, namespace)
                for warning in warnings:
                    if self._run_id and warning.startswith(
                        pointer_key(namespace, self._run_id) + ":"
                    ):
                        raise _core_error(StorageError, "invalid own run pointer")
                    _LOG.warning("%s", warning)
                for pointer in pointers:
                    roots.setdefault(pointer.run_id, pointer.run_root)
            if self._run_id:
                roots[self._run_id] = self._run_root
            for namespace in self._namespaces:
                self._cache(namespace)
            caches = list(self._caches.values())
            reads: list[tuple[str, str, Head]] = []
            for run_id, root in sorted(roots.items()):
                try:
                    store = run_store(root, run_id)
                    owner = read_owner(store)
                    if owner is None:
                        raise _core_error(StorageError, "run has no owner document")
                    head = owner[0].head
                    if head is None:
                        if run_id != self._run_id:
                            _LOG.warning("run %s has no committed head", run_id)
                        continue
                    for cache in caches:
                        ingest_run(cache, store, run_id, root, head)
                    reads.append((run_id, root, head))
                except (StorageError, OSError, ValueError):
                    if run_id == self._run_id:
                        raise
                    _LOG.warning("cannot read arena run %s", run_id)
            reviews, curation = list_human_objects(arena, self._namespaces)
            frontier = build_frontier(reads, reviews, curation)
            for cache in caches:
                self._load_objects(cache, frontier)
            self._latest = frontier.model_copy(deep=True)
            return frontier

    def snapshot(self, goal: GoalKey, frontier: Frontier) -> "ArenaView":
        """
        Read an exact recorded frontier and freeze goal and sibling membership.

        Parameters
        ----------
        goal : GoalKey
            Main or explicit historical part identity.
        frontier : Frontier
            Complete committed marks and exact named human objects.

        Returns
        -------
        BoardView
            Fixed view independent of hints and later cache ingestion.

        Raises
        ------
        StorageError
            If a required recorded chain or named object cannot be read.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> from swarm_arena import Goal
        >>> from swarm_arena.board import Frontier
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root, cache_dir=root + "/cache")
        ...     view = board.snapshot(Goal(id="g", namespace="toy").key(), Frontier())
        ...     assert view._visible_candidate_ids() == frozenset()
        ...     board.close()
        """
        frontier = frontier.model_copy(deep=True)
        marks = _validate_marks(frontier.marks)
        with self._lock:
            cache = self._cache(goal.namespace)
            for mark in marks:
                _validate_run_id(mark.run_id)
                ensure_covered(cache, mark)
            self._load_objects(cache, frontier)
            return ArenaView(self, goal, frontier)

    def _latest_frontier(self) -> Frontier:
        """Capture once on first board read; callers hold the board lock."""
        return self.frontier() if self._latest is None else self._latest.model_copy(deep=True)

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
        Rank a current group across the namespace at the latest frontier.

        Parameters
        ----------
        goal : GoalKey
            Main or part identity to read.
        k : int
            Maximum returned candidates; nonpositive values return none.
        judge : str
            Recorded judge name.
        run_id : str or None, default=None
            Optional producing-run filter, independent of judge identity.
        part : str or None, default=None
            Main-view part name, or the part view's own name.
        fidelity : str or None, default=None
            Effort filter, defaulting to the newest eligible round's fidelity.

        Returns
        -------
        list[Candidate]
            Ranked, complete, gated-in candidates.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> from swarm_arena import Goal
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root, cache_dir=root + "/cache")
        ...     assert board.best(Goal(id="g", namespace="toy").key(), 5, judge="j") == []
        ...     board.close()
        """
        with self._lock:
            return self.snapshot(goal, self._latest_frontier()).best(
                goal, k, judge=judge, run_id=run_id, part=part, fidelity=fidelity
            )

    def learnings(
        self,
        namespace: str,
        *,
        tags: Sequence[str] = (),
        goal: GoalKey | None = None,
        limit: int = 50,
    ) -> list[Learning]:
        """
        List visible namespace learnings at the latest frontier.

        Parameters
        ----------
        namespace : str
            Namespace slug.
        tags : Sequence[str], default=()
            Any-of tag filter and relevance preferences.
        goal : GoalKey or None, default=None
            Optional main or part goal scope.
        limit : int, default=50
            Maximum returned records; nonpositive values return none.

        Returns
        -------
        list[Learning]
            Visible learnings in relevance order.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root, cache_dir=root + "/cache")
        ...     assert board.learnings("toy") == []
        ...     board.close()
        """
        with self._lock:
            return self._learning_read(namespace, None, tags=tags, goal=goal, limit=limit)

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
        Search visible learnings with literal text and fixed-frontier relevance.

        Parameters
        ----------
        namespace : str
            Namespace slug.
        query : str
            Literal search text; FTS syntax is never executed directly.
        tags : Sequence[str], default=()
            Any-of tag filter and relevance preferences.
        goal : GoalKey or None, default=None
            Optional main or part goal scope.
        limit : int, default=10
            Maximum returned records; nonpositive values return none.

        Returns
        -------
        list[Learning]
            Text-ranked, visible learning records.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root, cache_dir=root + "/cache")
        ...     assert board.search_learnings("toy", "log(nu)") == []
        ...     board.close()
        """
        with self._lock:
            return self._learning_read(namespace, query, tags=tags, goal=goal, limit=limit)

    def _learning_read(
        self,
        namespace: str,
        query: str | None,
        *,
        tags: Sequence[str],
        goal: GoalKey | None,
        limit: int,
    ) -> list[Learning]:
        """Delegate scoped reads or use namespace-wide relevance without a goal."""
        _validate_namespace(namespace)
        frontier = self._latest_frontier()
        if goal is not None:
            view = self.snapshot(goal, frontier)
            return view._learning_read(namespace, query, tags=tags, goal=goal, limit=limit)
        cache = self._cache(namespace)
        for mark in _validate_marks(frontier.marks):
            ensure_covered(cache, mark)
        self._load_objects(cache, frontier)
        return _find_learnings(
            cache,
            _namespace_frontier(frontier, namespace),
            query=query,
            filters=LearningFilter(tags=list(tags)),
            relevance_goal=None,
            limit=limit,
        )

    def get_candidate(self, id: str) -> Candidate:
        """
        Get a committed candidate, falling back to an own-process hint.

        Parameters
        ----------
        id : str
            Candidate identifier.

        Returns
        -------
        Candidate
            Independent candidate model.

        Raises
        ------
        KeyError
            If no committed record or hint has that identifier.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root)
        ...     try:
        ...         board.get_candidate("missing")
        ...     except KeyError:
        ...         pass
        ...     board.close()
        """
        with self._lock:
            frontier = self._latest_frontier()
            for namespace in self._namespaces:
                rows = self._cache(namespace).candidates(frontier, ids=[id])
                if rows:
                    return rows[0][0]
            if id in self._candidates:
                return self._candidates[id].model_copy(deep=True)
            raise KeyError(id)

    def get_learning(self, id: str) -> Learning:
        """
        Get a visible committed learning or an uncommitted learning hint.

        Parameters
        ----------
        id : str
            Learning identifier.

        Returns
        -------
        Learning
            Independent model with fixed-frontier curation flags.

        Raises
        ------
        KeyError
            If absent or hidden, including a hidden committed hinted record.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root)
        ...     try:
        ...         board.get_learning("missing")
        ...     except KeyError:
        ...         pass
        ...     board.close()
        """
        with self._lock:
            frontier = self._latest_frontier()
            for namespace in self._namespaces:
                rows = self._cache(namespace).learnings_by_id(frontier, [id])
                if rows:
                    if rows[0].hidden:
                        raise KeyError(id)
                    return rows[0]
            hint = self._learnings.get(id)
            if hint is None or hint.hidden:
                raise KeyError(id)
            cache = self._caches.get(hint.namespace)
            pinned, hidden = (
                cache.flags(frontier).get(id, (hint.pinned, hint.hidden))
                if cache is not None
                else (hint.pinned, hint.hidden)
            )
            if hidden:
                raise KeyError(id)
            return hint.model_copy(update={"pinned": pinned, "hidden": hidden}, deep=True)

    def scores(self, candidate_id: str) -> list[Score]:
        """
        Get committed scores, falling back to hints when no scores are marked.

        Parameters
        ----------
        candidate_id : str
            Candidate identifier whose existence is checked first.

        Returns
        -------
        list[Score]
            Independent recorded scores or process-local score hints.

        Raises
        ------
        KeyError
            If the candidate is unknown.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> from swarm_arena.testing import make_candidate
        >>> from swarm_arena.tool import ToolOutput
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root)
        ...     candidate = make_candidate(ToolOutput(status="ok"))
        ...     board.put_candidate(candidate, epoch=1, seq=1)
        ...     assert board.scores(candidate.id) == []
        ...     board.close()
        """
        with self._lock:
            self.get_candidate(candidate_id)
            frontier = self._latest_frontier()
            found: dict[str, Score] = {}
            for namespace in self._namespaces:
                for score, _ in self._cache(namespace).scores(
                    frontier, candidate_ids=[candidate_id]
                ):
                    found[score.id] = score
            if found:
                return list(found.values())
            return [
                s.model_copy(deep=True)
                for s in self._scores.values()
                if s.candidate_id == candidate_id
            ]

    def lineage(self, candidate_id: str) -> LineageGraph:
        """
        Follow committed ancestry at the latest frontier without using hints.

        Parameters
        ----------
        candidate_id : str
            Root candidate identifier.

        Returns
        -------
        LineageGraph
            Breadth-first resolvable nodes and child-to-parent edges.

        Raises
        ------
        KeyError
            If the root is absent from committed records.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root)
        ...     try:
        ...         board.lineage("missing")
        ...     except KeyError:
        ...         pass
        ...     board.close()
        """
        with self._lock, ExitStack() as reads:
            frontier = self._latest_frontier()
            caches = []
            for namespace in self._namespaces:
                cache = self._cache(namespace)
                selected = _namespace_frontier(frontier, namespace)
                reads.enter_context(cache._lock)
                reads.enter_context(cache._frontier(selected))
                caches.append((cache, selected))

            def read(ids: Sequence[str]) -> dict[str, Candidate]:
                found: dict[str, Candidate] = {}
                missing = list(ids)
                for cache, selected in caches:
                    found.update((c.id, c) for c, _ in cache.candidates(selected, ids=missing))
                    missing = [id for id in missing if id not in found]
                    if not missing:
                        break
                return found

            return _lineage(read, candidate_id)

    def leaderboard(self, goal: GoalKey, *, judge: str) -> list[Row]:
        """
        Rank all current groups at the latest frontier without a run filter.

        Parameters
        ----------
        goal : GoalKey
            Main or explicit part identity.
        judge : str
            Recorded ranking judge name.

        Returns
        -------
        list[Row]
            Flattened current groups with independent ranks and provenance.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> from swarm_arena import Goal
        >>> with TemporaryDirectory() as root:
        ...     board = ArenaBoard(arena_root=root, cache_dir=root + "/cache")
        ...     assert board.leaderboard(Goal(id="g", namespace="toy").key(), judge="j") == []
        ...     board.close()
        """
        with self._lock:
            return self.snapshot(goal, self._latest_frontier()).leaderboard(goal, judge=judge)


class ArenaView(BoardView):
    """Read a fixed namespace frontier with current or explicit historical keys."""

    def __init__(self, board: ArenaBoard, goal: GoalKey, frontier: Frontier) -> None:
        self._board = board
        self.goal = goal.model_copy(deep=True)
        self.frontier = frontier.model_copy(deep=True)
        self._frontier = _namespace_frontier(frontier, goal.namespace)
        self._cache = board._cache(goal.namespace)
        self._identity_run = board._run_id or None
        self.part: str | None = None
        self.learning_goal = (goal_key_digest(goal), goal.id)
        self._plan: Plan | None = None
        self._ranking_plan: Plan | None = None
        self._active_goals: dict[str | None, GoalKey] = {}
        self._candidate_groups: dict[str, str | None] = {}
        self._resolve()
        self._candidate_ids = frozenset(c.id for c, _ in self._selected())

    def _resolve(self) -> None:
        """Resolve fixed identities using the producer's core replay selector."""
        digest = goal_key_digest(self.goal)
        mains: list[tuple[str, str, int, Plan, _RunPlans]] = []
        parts: list[tuple[str, str, int, Plan, _RunPlans, str]] = []
        for mark in _validate_marks(self._frontier.marks):
            state = self._board._plans_at(self._cache, mark)
            for goal_id, revisions in state.revisions.items():
                for plan in revisions:
                    if plan.goal.namespace != self.goal.namespace:
                        continue
                    row = (mark.run_id, goal_id, plan.revision, plan, state)
                    if goal_key_digest(plan.goal.key()) == digest:
                        mains.append(row)
                    for part in plan.parts:
                        if goal_key_digest(part.goal.key()) == digest:
                            parts.append((*row, part.name))
        matches = mains if mains else [row[:5] for row in parts]
        if not matches:
            return
        ordered = sorted(matches, key=lambda row: row[:3])
        if not mains:
            resolved = min(parts, key=lambda row: row[:3])
            self.part = resolved[5]
            self.learning_goal = (goal_key_digest(resolved[3].goal.key()), resolved[1])
        anchor = next((row for row in ordered if row[0] == self._identity_run), ordered[0])
        _, goal_id, _, _, state = anchor
        self._plan = state.current[goal_id]
        self._ranking_plan = self._plan
        if not mains:
            current = next(
                (part for part in self._plan.parts if goal_key_digest(part.goal.key()) == digest),
                None,
            )
            if current is not None:
                self.part = current.name
            else:
                self._ranking_plan = anchor[3]
                self.part = next(
                    part.name
                    for part in anchor[3].parts
                    if goal_key_digest(part.goal.key()) == digest
                )
        self._active_goals = {None: self._plan.goal.key()}
        self._active_goals.update((part.name, part.goal.key()) for part in self._plan.parts)

    def _selected(self) -> list[tuple[Candidate, Position]]:
        """Filter immutable producing identities before ranking or visibility."""
        self._candidate_groups = {}
        if self.part is not None:
            pairs = self._cache.candidates(self._frontier, part_key=goal_key_digest(self.goal))
            self._candidate_groups = {c.id: self.part for c, _ in pairs}
            return pairs
        if self._plan is None:
            return []
        selected: dict[str, tuple[Candidate, Position]] = {}
        for part, key in self._active_goals.items():
            for candidate, position in self._cache.candidates(
                self._frontier, goal_key=goal_key_digest(self.goal), part_key=goal_key_digest(key)
            ):
                selected[candidate.id] = (candidate, position)
                self._candidate_groups[candidate.id] = part
        return sorted(selected.values(), key=lambda pair: pair[1])

    def _visible_candidate_ids(self) -> frozenset[str]:
        """Return fixed membership without I/O, hints, or mutable engine state."""
        return self._candidate_ids

    def _sibling_view(self, part: str | None) -> "ArenaView":
        """Select only a declared group from this frontier's executed revision."""
        if part not in self._active_goals:
            raise _core_error(ValueError, "part outside board snapshot")
        with self._board._lock:
            result = copy(self)
            result.goal = self._active_goals[part].model_copy(deep=True)
            result._frontier = self._frontier.model_copy(deep=True)
            main_goal = self._active_goals[None]
            result.learning_goal = (goal_key_digest(main_goal), main_goal.id)
            # The capability belongs to this owner/revision, even when a sibling
            # key also occurs under another run's different main goal.
            result._plan = self._plan
            result._ranking_plan = self._plan
            result._active_goals = dict(self._active_goals)
            result._identity_run = self._identity_run
            result.frontier = self.frontier.model_copy(deep=True)
            result.part = part
            result._candidate_ids = frozenset(c.id for c, _ in result._selected())
            return result

    def _check_goal(self, goal: GoalKey) -> None:
        if goal != self.goal:
            raise _core_error(ValueError, "goal outside board snapshot")

    def _filters(
        self, namespace: str, goal: GoalKey | None, tags: Sequence[str] = ()
    ) -> LearningFilter:
        if namespace != self.goal.namespace:
            raise _core_error(ValueError, "namespace outside board snapshot")
        if goal is not None:
            self._check_goal(goal)
        return LearningFilter(
            goal_key=self.learning_goal[0] if goal is not None else None,
            goal_id=self.learning_goal[1] if goal is not None else None,
            tags=list(tags),
        )

    def _pairs(self) -> list[tuple[Candidate, Position]]:
        return self._cache.candidates(self._frontier, ids=tuple(self._candidate_ids))

    def _ranking_judges(self) -> dict[str | None, str]:
        """Resolve fixed group names from recorded data without loading plugins."""
        with self._board._lock:
            if self._ranking_plan is None:
                return {}
            pairs = self._ranking_pairs(self._pairs())
            scores = self._cache.scores(self._frontier, candidate_ids=tuple(self._candidate_ids))
            names = group_judges(
                self._ranking_plan, candidates=[c for c, _ in pairs], scores=[s for s, _ in scores]
            )
            return (
                names
                if self.part is None
                else {p: name for p, name in names.items() if p == self.part}
            )

    def _ranking_pairs(
        self, pairs: list[tuple[Candidate, Position]]
    ) -> list[tuple[Candidate, Position]]:
        """Map selected producing keys to current groups without altering records."""
        return [
            (c.model_copy(update={"part": self._candidate_groups[c.id]}), position)
            for c, position in pairs
        ]

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
        """Rank one selected producing-key group under the captured own identity."""
        self._check_goal(goal)
        if self.part is not None and part not in (None, self.part):
            raise _core_error(ValueError, "part outside board snapshot")
        group = self.part if part is None else part
        with self._board._lock:
            pairs = self._pairs()
            scores = self._cache.scores(self._frontier, candidate_ids=tuple(self._candidate_ids))
            rows = group_rows(
                self._ranking_pairs(pairs),
                scores,
                judge=judge,
                identity_run=self._identity_run,
                run_id=run_id,
                fidelity=fidelity,
            ).get(group, [])
            by_id = {c.id: c for c, _ in pairs}
            return [by_id[row.candidate_id] for row in rows[: max(0, k)]]

    def leaderboard(self, goal: GoalKey, *, judge: str) -> list[Row]:
        """Flatten selected groups without filtering a run or applying a margin."""
        self._check_goal(goal)
        with self._board._lock:
            pairs = self._pairs()
            original_parts = {c.id: c.part for c, _ in pairs}
            rows = flatten(
                group_rows(
                    self._ranking_pairs(pairs),
                    self._cache.scores(self._frontier, candidate_ids=tuple(self._candidate_ids)),
                    judge=judge,
                    identity_run=self._identity_run,
                )
            )
            return [
                row.model_copy(update={"part": original_parts[row.candidate_id]}) for row in rows
            ]

    def _learning_read(
        self,
        namespace: str,
        query: str | None,
        *,
        tags: Sequence[str],
        goal: GoalKey | None,
        limit: int,
    ) -> list[Learning]:
        filters = self._filters(namespace, goal, tags)
        with self._board._lock:
            return _find_learnings(
                self._cache,
                self._frontier,
                query=query,
                filters=filters,
                relevance_goal=self.learning_goal,
                limit=limit,
            )

    def learnings(
        self,
        namespace: str,
        *,
        tags: Sequence[str] = (),
        goal: GoalKey | None = None,
        limit: int = 50,
    ) -> list[Learning]:
        """List visible learnings, always preferring the resolving main goal."""
        return self._learning_read(namespace, None, tags=tags, goal=goal, limit=limit)

    def search_learnings(
        self,
        namespace: str,
        query: str,
        *,
        tags: Sequence[str] = (),
        goal: GoalKey | None = None,
        limit: int = 10,
    ) -> list[Learning]:
        """Search through the shared fixed-frontier find path and goal relevance."""
        return self._learning_read(namespace, query, tags=tags, goal=goal, limit=limit)

    def count_learnings(self, namespace: str, *, goal: GoalKey | None = None) -> int:
        """Count visible learning membership with the same scope as list reads."""
        filters = self._filters(namespace, goal)
        with self._board._lock:
            return self._cache.count(self._frontier, filters=filters)

    def get_candidate(self, id: str) -> Candidate:
        """Return a candidate only when it belongs to the fixed selected keys."""
        if id not in self._candidate_ids:
            raise KeyError(id)
        with self._board._lock:
            rows = self._cache.candidates(self._frontier, ids=[id])
            if not rows:
                raise KeyError(id)
            return rows[0][0]

    def get_learning(self, id: str) -> Learning:
        """Return a namespace learning only when visible at this frontier."""
        with self._board._lock:
            rows = self._cache.learnings_by_id(self._frontier, [id])
            if not rows or rows[0].hidden:
                raise KeyError(id)
            return rows[0]

    def scores(self, candidate_id: str) -> list[Score]:
        """Read marked scores only for a candidate in this view's membership."""
        with self._board._lock:
            self.get_candidate(candidate_id)
            return [s for s, _ in self._cache.scores(self._frontier, candidate_ids=[candidate_id])]

    def lineage(self, candidate_id: str) -> LineageGraph:
        """Traverse only resolvable fixed-view nodes, with cycle detection."""
        with self._board._lock, self._cache._lock, self._cache._frontier(self._frontier):

            def read(ids: Sequence[str]) -> dict[str, Candidate]:
                selected = [id for id in ids if id in self._candidate_ids]
                return {c.id: c for c, _ in self._cache.candidates(self._frontier, ids=selected)}

            return _lineage(read, candidate_id)
