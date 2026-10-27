"""Read shared committed journals through synchronized, fixed arena views."""

import logging
import os
from collections.abc import Callable, Generator, Sequence
from contextlib import ExitStack, contextmanager, suppress
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Condition, RLock, get_ident
from typing import TYPE_CHECKING, Any, cast

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
from swarm_arena._journal.replay import _GatewayKey, _GatewaySpend
from swarm_arena._model.board_types import Frontier, LineageGraph, Row
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.plan import Plan
from swarm_arena._model.planning_types import Decision
from swarm_arena._model.ref import _ConfigInputModel, config_or_fields
from swarm_arena._model.results import Candidate, Learning, LLMCall, Score
from swarm_arena._model.run_config import RunConfig
from swarm_arena._outcomes import _Outcome
from swarm_arena._store.base import open_store

if TYPE_CHECKING:
    from swarm_arena._board.trace import _SessionCalls

_LOG = logging.getLogger("swarm_arena.board")


class _ReadLifetime:
    """Pin the shared reader pool until every captured read has returned."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._depths: dict[int, int] = {}
        self.closed = False

    @contextmanager
    def pin(self) -> Generator[None]:
        """Reject new reads after close while allowing pinned nested operations."""
        thread = get_ident()
        with self._condition:
            if self.closed and thread not in self._depths:
                raise _core_error(RuntimeError, "arena board is closed")
            self._depths[thread] = self._depths.get(thread, 0) + 1
        try:
            yield
        finally:
            with self._condition:
                self._depths[thread] -= 1
                if not self._depths[thread]:
                    del self._depths[thread]
                self._condition.notify_all()

    def stop(self) -> None:
        """Stop admission without waiting under the publication lock."""
        with self._condition:
            self.closed = True

    def wait(self) -> None:
        """Drain admitted reads before the detached pool is physically closed."""
        with self._condition:
            self._condition.wait_for(lambda: not self._depths)


@dataclass(frozen=True)
class _RunPlans:
    """Retain fixed plan resolution without aliasing mutable replay progress."""

    revisions: dict[str, tuple[Plan, ...]]
    current: dict[str, Plan]
    calls: dict[str, LLMCall]
    gateway_keys: dict[tuple[str, str], _GatewayKey]
    gateway_reconcile: dict[tuple[str, str], _GatewaySpend]


@dataclass(frozen=True)
class _Published:
    """Keep one complete reader generation independent of refresh preparation."""

    frontier: Frontier
    readers: dict[str, ArenaCache]
    plans: dict[str, _RunPlans]
    namespaces: tuple[str, ...]
    run_id: str | None
    lifetime: _ReadLifetime
    sessions: dict[tuple[str, str], "_SessionCalls | None"] = field(default_factory=dict)
    source_namespaces: dict[tuple[str, str], frozenset[str]] = field(default_factory=dict)


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

    Construction performs no I/O. Storage refresh uses a worker-only lock and
    writer connections; board and view reads use separate WAL reader connections.
    A short in-memory lock publishes prepared generations and protects hint maps.
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
        self._refresh_lock = RLock()
        self._caches: dict[str, ArenaCache] = {}
        self._read_caches: dict[str, ArenaCache] = {}
        self._namespaces: tuple[str, ...] = ()
        self._attached = False
        self._run_id = ""
        self._run_root = ""
        self._lifetime = _ReadLifetime()
        self._published = _Published(Frontier(), {}, {}, (), None, self._lifetime)
        self._candidates: dict[str, Candidate] = {}
        self._scores: dict[str, Score] = {}
        self._learnings: dict[str, Learning] = {}
        self._run_plans: dict[tuple[str, str], tuple[Frontier.Mark, _RunPlans]] = {}
        self._session_calls: dict[tuple[str, str], tuple[Head, _SessionCalls]] = {}
        self._session_read_warnings: set[tuple[tuple[str, str], Head | None]] = set()

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

    def _cache_settings(self) -> tuple[str, Path]:
        """Return the resolved storage and local-cache locations for maintenance."""
        return self.config.arena_root, self._cache_dir()

    def _diagnostics(self) -> tuple[dict[str, int], tuple[str, ...]]:
        """Collect committed cache counts and value-free warnings on a worker."""
        with self._refresh_lock:
            counts = {}
            warnings = []
            for namespace, cache in self._caches.items():
                count, messages = cache._diagnostics()
                counts[namespace] = count
                warnings.extend(messages)
            return counts, tuple(warnings)

    @contextmanager
    def _read_namespace(self) -> Generator["_ArenaReader"]:
        """Pin one committed read capability for CLI, viewer and trace consumers."""
        with self._pin_state() as state:
            yield _ArenaReader(self, state)

    def _cache(self, namespace: str) -> ArenaCache:
        """Open a writer without holding any fixed-view read lock."""
        _validate_namespace(namespace)
        with self._refresh_lock:
            self._require_open()
            if namespace not in self._caches:
                cache = ArenaCache(self._cache_dir() / f"{namespace}.sqlite")
                with self._lock:
                    self._caches[namespace] = cache
            return self._caches[namespace]

    def _read_cache(self, namespace: str) -> ArenaCache:
        """Prepare a WAL reader whose lock is never taken by ingest or storage."""
        _validate_namespace(namespace)
        with self._refresh_lock:
            self._require_open()
            if namespace not in self._read_caches:
                cache = ArenaCache(self._cache_dir() / f"{namespace}.sqlite")
                with self._lock:
                    self._read_caches[namespace] = cache
            return self._read_caches[namespace]

    def _plans_at(self, cache: ArenaCache, mark: Frontier.Mark) -> _RunPlans:
        """Memoize one committed plan projection per run under the refresh lock.

        Reuse the ingest checkpoint and its suffix catch-up seam under the
        cache lock. Historical reads never rewind a newer ingest checkpoint.
        Retain independent plans, not the state that later ingest mutates.
        """
        key = (_resolved_root(mark.run_root), mark.run_id)
        previous = self._run_plans.get(key)
        if previous is not None and previous[0] == mark:
            return previous[1]
        state = cache._committed_state(mark)
        revisions = {
            goal_id: tuple(plan for plan, _ in plans) for goal_id, plans in state.plans.items()
        }
        current = {}
        for goal_id in revisions:
            progress = state.goals[goal_id]
            round = max(1, progress.round, progress.rounds_closed + 1)
            current[goal_id] = state.plan_for_round(goal_id, round)
        result = _RunPlans(
            revisions, current, state.llm_calls, state.gateway_keys, state.gateway_reconcile
        )
        self._run_plans[key] = (mark.model_copy(deep=True), result)
        return result

    def _sessions_at(
        self, frontier: Frontier, plans: dict[str, _RunPlans]
    ) -> dict[tuple[str, str], "_SessionCalls | None"]:
        """Capture sessions from producing roots; warn once per unreadable head."""
        from swarm_arena._board.trace import _calls_at, _SessionCalls

        previous_state = self._read_state()
        previous_marks = {
            (_resolved_root(mark.run_root), mark.run_id): mark
            for mark in previous_state.frontier.marks
        }
        requested: dict[tuple[str, str], bool] = {}
        for mark in frontier.marks:
            root = _resolved_root(mark.run_root)
            changed = previous_marks.get((root, mark.run_id)) != mark
            for revisions in plans[mark.run_id].revisions.values():
                for plan in revisions:
                    if plan.planning_session:
                        key = (root, plan.planning_session)
                        requested[key] = requested.get(key, False) or changed
        sessions: dict[tuple[str, str], _SessionCalls | None] = {}
        for key, changed in requested.items():
            if not changed and key in previous_state.sessions:
                sessions[key] = previous_state.sessions[key]
                continue
            root, session_id = key
            # Publication is synchronous worker preparation. Lease-owning
            # callers retain cancellation on their existing outer boundary.
            outcome = _Outcome(capture_caller=False)
            head: Head | None = None
            try:
                store = run_store(root, session_id)
                owner = read_owner(store)
                if owner is None or owner[0].head is None:
                    sessions[key] = None
                    continue
                head = owner[0].head
                previous = self._session_calls.get(key)
                if previous is not None and previous[0] == head:
                    session = previous[1]
                else:
                    calls, reason = _calls_at(store, head)
                    session = _SessionCalls(calls, reason)
                    self._session_calls[key] = (head.model_copy(deep=True), session)
                sessions[key] = session
            except (StorageError, OSError, ValueError) as error:
                outcome.propagate_control(error)
                warning = (key, head)
                log = _LOG.debug if warning in self._session_read_warnings else _LOG.warning
                log("cannot read planning session %s", session_id)
                self._session_read_warnings.add(warning)
                # An unreadable session is not a capture; retry next refresh.
        return sessions

    def open(self, run_id: str, run_root: str, namespaces: Sequence[str]) -> None:
        """
        Attach namespaces and publish write-once pointers for an owning run.

        Reattaching the same live owner extends its namespace set while retaining
        the published read generation until the next frontier refresh.

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
        with self._refresh_lock:
            with self._lock:
                extending = (
                    self._attached
                    and not self._lifetime.closed
                    and (run_id, resolved)
                    == (
                        self._run_id,
                        self._run_root,
                    )
                )
                if extending:
                    names = tuple(sorted(set(names) | set(self._namespaces)))
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
                        if existing is None or existing.run_id != run_id:
                            raise
                        if existing == pointer:
                            continue
                        if not warned:
                            _LOG.warning(
                                "run %s resumed at a different root; keeping original pointer",
                                run_id,
                            )
                            warned = True
            with self._lock:
                if self._lifetime.closed:
                    self._lifetime = _ReadLifetime()
                self._attached = True
                self._run_id, self._run_root, self._namespaces = run_id, resolved, names
                if not extending:
                    self._published = _Published(
                        Frontier(), {}, {}, names, run_id or None, self._lifetime
                    )

    def close(self) -> None:
        """
        Drain active reads, close every cache, and release lifetime file locks.

        Notes
        -----
        New reads raise RuntimeError once shutdown begins. Active reads finish
        against their captured state before connections close. Waiting holds
        neither the refresh nor publication lock. Reopening starts a new reader
        lifetime; existing views remain closed.

        Examples
        --------
        >>> board = ArenaBoard(arena_root="/arena")
        >>> board.close()
        >>> board.close()
        """
        with self._refresh_lock, self._lock:
            lifetime = self._lifetime
            lifetime.stop()
            caches = [*self._read_caches.values(), *self._caches.values()]
            self._read_caches.clear()
            self._caches.clear()
            self._run_plans.clear()
            self._session_calls.clear()
            self._session_read_warnings.clear()
            self._published = _Published(Frontier(), {}, {}, (), None, lifetime)
        # Neither publication nor refresh locks may be held while readers drain.
        lifetime.wait()
        for cache in caches:
            cache.close()

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
        Accept a journaled decision notification without retaining a duplicate.

        Parameters
        ----------
        goal_id : str
            Owning main goal identifier.
        round : int
            Decision's round number.
        decision : Decision
            Core-journaled policy decision; committed replay owns this state.
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
        return None

    def _load_objects(
        self, cache: ArenaCache, frontier: Frontier, *, tolerate_unreadable: bool = False
    ) -> set[str]:
        """Read complete local-namespace objects and apply one atomic batch."""
        arena = open_store(self.config.arena_root)
        names: list[str] = []
        operations: list[HumanObject] = []
        empty: list[str] = []
        omitted: set[str] = set()
        for name in sorted(set(frontier.reviews + frontier.curation), key=human_object_order):
            _validate_human_key(name)
            if not name.startswith(cache.namespace + "/") or cache.has_object(name):
                continue
            try:
                objects = read_human_object(arena, name)
            except (StorageError, OSError, ValueError):
                if not tolerate_unreadable:
                    raise
                _LOG.warning("skipping unreadable arena human object")
                omitted.add(name)
                continue
            if objects:
                names.extend([name] * len(objects))
                operations.extend(objects)
            else:
                empty.append(name)
        if names or empty:
            cache.apply_human_objects(names, operations, covered=empty)
        return omitted

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
        with self._refresh_lock:
            self._require_open()
            arena = open_store(self.config.arena_root)
            roots: dict[str, str] = {}
            source_namespaces: dict[tuple[str, str], set[str]] = {}
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
                    source_namespaces.setdefault(
                        (_resolved_root(pointer.run_root), pointer.run_id), set()
                    ).add(namespace)
            if self._run_id:
                roots[self._run_id] = self._run_root
                source_namespaces.setdefault((self._run_root, self._run_id), set()).update(
                    self._namespaces
                )
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
            omitted = set()
            for cache in caches:
                omitted.update(self._load_objects(cache, frontier, tolerate_unreadable=True))
            frontier = frontier.model_copy(
                update={
                    "reviews": [name for name in reviews if name not in omitted],
                    "curation": [name for name in curation if name not in omitted],
                }
            )
            published = replace(
                self._prepare_published(frontier),
                source_namespaces={
                    (_resolved_root(mark.run_root), mark.run_id): frozenset(
                        source_namespaces.get((_resolved_root(mark.run_root), mark.run_id), ())
                    )
                    for mark in frontier.marks
                },
            )
            with self._lock:
                self._published = published
            return frontier

    def _prepare_published(self, frontier: Frontier) -> _Published:
        """Prepare readers and plan projections before the short publication swap."""
        plans = {}
        if self._caches:
            # Every namespace cache contains the same committed plan events.
            cache = next(iter(self._caches.values()))
            plans = {mark.run_id: self._plans_at(cache, mark) for mark in frontier.marks}
        namespaces = set(self._caches)
        for state in plans.values():
            for revisions in state.revisions.values():
                for plan in revisions:
                    namespaces.update(
                        goal.namespace for goal in [plan.goal, *(p.goal for p in plan.parts)]
                    )
        readers = {}
        for namespace in sorted(namespaces):
            cache = self._cache(namespace)
            for mark in frontier.marks:
                ensure_covered(cache, mark)
            self._load_objects(cache, frontier)
            readers[namespace] = self._read_cache(namespace)
        return _Published(
            frontier.model_copy(deep=True),
            readers,
            plans,
            self._namespaces,
            self._run_id or None,
            self._lifetime,
            self._sessions_at(frontier, plans),
        )

    def _read_state(self) -> _Published:
        """Capture a complete generation without entering the refresh lock."""
        with self._lock:
            return self._published

    def _require_open(self) -> None:
        """Prevent closed boards from reopening cache connections implicitly."""
        with self._lock:
            if self._lifetime.closed:
                raise _core_error(RuntimeError, "arena board is closed")

    @contextmanager
    def _pin_state(self) -> Generator[_Published]:
        """Atomically capture and pin before close can detach the reader pool."""
        with ExitStack() as pins:
            with self._lock:
                state = self._published
                pins.enter_context(state.lifetime.pin())
            yield state

    def _published_view(self, goal: GoalKey, state: _Published) -> "ArenaView | None":
        """Resolve a prepared generation without opening caches or reading storage."""
        _validate_namespace(goal.namespace)
        if goal.namespace not in state.readers:
            return None
        return ArenaView(self, goal, state.frontier, published=state)

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
        ...     assert view.leaderboard(view.goal, judge="program") == []
        ...     board.close()
        """
        frontier = frontier.model_copy(deep=True)
        marks = _validate_marks(frontier.marks)
        with self._refresh_lock:
            cache = self._cache(goal.namespace)
            for mark in marks:
                _validate_run_id(mark.run_id)
                ensure_covered(cache, mark)
            self._load_objects(cache, frontier)
            return ArenaView(self, goal, frontier)

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
        with self._pin_state() as state:
            view = self._published_view(goal, state)
            return (
                view.best(goal, k, judge=judge, run_id=run_id, part=part, fidelity=fidelity)
                if view is not None
                else []
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
        with self._pin_state() as state:
            _validate_namespace(namespace)
            if goal is not None:
                if namespace != goal.namespace:
                    raise _core_error(ValueError, "namespace outside board snapshot")
                view = self._published_view(goal, state)
                return (
                    view._learning_read(namespace, query, tags=tags, goal=goal, limit=limit)
                    if view is not None
                    else []
                )
            cache = state.readers.get(namespace)
            if cache is None:
                return []
            return _find_learnings(
                cache,
                _namespace_frontier(state.frontier, namespace),
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
        with self._pin_state() as state:
            return self._candidate_at(id, state)

    def _candidate_at(self, id: str, state: _Published) -> Candidate:
        """Read a candidate using the same captured generation as its caller."""
        for namespace in state.namespaces:
            cache = state.readers.get(namespace)
            if cache is not None:
                rows = cache.candidates(_namespace_frontier(state.frontier, namespace), ids=[id])
                if rows:
                    return rows[0][0]
        with self._lock:
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
        with self._pin_state() as state:
            for namespace in state.namespaces:
                cache = state.readers.get(namespace)
                if cache is None:
                    continue
                rows = cache.learnings_by_id(_namespace_frontier(state.frontier, namespace), [id])
                if rows:
                    if rows[0].hidden:
                        raise KeyError(id)
                    return rows[0]
            with self._lock:
                hint = self._learnings.get(id)
            if hint is None or hint.hidden:
                raise KeyError(id)
            cache = state.readers.get(hint.namespace)
            pinned, hidden = (
                cache.flags(_namespace_frontier(state.frontier, hint.namespace)).get(
                    id, (hint.pinned, hint.hidden)
                )
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
        with self._pin_state() as state:
            retained = self._candidate_at(candidate_id, state)
            found: dict[str, Score] = {}
            for namespace in state.namespaces:
                cache = state.readers.get(namespace)
                if cache is None:
                    continue
                for score, _ in cache.scores(
                    _namespace_frontier(state.frontier, namespace), candidate_ids=[candidate_id]
                ):
                    found[score.id] = score
            if found:
                return list(found.values())
            with self._lock:
                hinted = self._candidates.get(candidate_id)
                if hinted is not None and hinted.run_id != retained.run_id:
                    return []
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
        with self._pin_state() as state, ExitStack() as reads:
            caches = []
            for namespace in state.namespaces:
                cache = state.readers.get(namespace)
                if cache is None:
                    continue
                selected = _namespace_frontier(state.frontier, namespace)
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
        with self._pin_state() as state:
            view = self._published_view(goal, state)
            return view.leaderboard(goal, judge=judge) if view is not None else []


class ArenaView(BoardView):
    """Read a fixed namespace frontier with current or explicit historical keys."""

    def __init__(
        self,
        board: ArenaBoard,
        goal: GoalKey,
        frontier: Frontier,
        *,
        published: _Published | None = None,
    ) -> None:
        self._board = board
        self._lifetime = board._lifetime if published is None else published.lifetime
        self._lock = RLock()
        self.goal = goal.model_copy(deep=True)
        self.frontier = frontier.model_copy(deep=True)
        self._frontier = _namespace_frontier(frontier, goal.namespace)
        self._cache = (
            board._read_cache(goal.namespace)
            if published is None
            else published.readers[goal.namespace]
        )
        self._identity_run = (board._run_id or None) if published is None else published.run_id
        self.part: str | None = None
        self.learning_goal = (goal_key_digest(goal), goal.id)
        self._plan: Plan | None = None
        self._ranking_plan: Plan | None = None
        self._candidate_main_key: str | None = None
        self._active_goals: dict[str | None, GoalKey] = {}
        self._candidate_groups: dict[str, str | None] = {}
        self._resolve(published)
        namespaces = {goal.namespace, *(key.namespace for key in self._active_goals.values())}
        self._group_caches = {}
        for namespace in sorted(namespaces):
            if published is None:
                cache = board._cache(namespace)
                for mark in _validate_marks(frontier.marks):
                    ensure_covered(cache, mark)
                board._load_objects(cache, frontier)
                self._group_caches[namespace] = board._read_cache(namespace)
            else:
                self._group_caches[namespace] = published.readers[namespace]
        self._candidate_ids = frozenset(c.id for c, _ in self._selected())

    @contextmanager
    def _read(self) -> Generator[None]:
        """Keep the pool alive for the entire compound fixed-view operation."""
        with self._lifetime.pin(), self._lock:
            yield

    def _resolve(self, published: _Published | None = None) -> None:
        """Resolve fixed identities using the producer's core replay selector."""
        digest = goal_key_digest(self.goal)
        mains: list[tuple[str, str, int, Plan, _RunPlans]] = []
        parts: list[tuple[str, str, int, Plan, _RunPlans, str]] = []
        for mark in _validate_marks(self._frontier.marks):
            state = (
                self._board._plans_at(self._board._cache(self.goal.namespace), mark)
                if published is None
                else published.plans[mark.run_id]
            )
            for goal_id, revisions in state.revisions.items():
                for plan in revisions:
                    row = (mark.run_id, goal_id, plan.revision, plan, state)
                    if goal_key_digest(plan.goal.key()) == digest:
                        mains.append(row)
                    for part in plan.parts:
                        if goal_key_digest(part.goal.key()) == digest:
                            parts.append((*row, part.name))
        own_mains = [row for row in mains if row[0] == self._identity_run]
        own_parts = [row for row in parts if row[0] == self._identity_run]
        if own_mains:
            mains = own_mains
        elif own_parts:
            mains, parts = [], own_parts
        matches = mains if mains else [row[:5] for row in parts]
        if not matches:
            return
        ordered = sorted(matches, key=lambda row: row[:3])
        if not mains:
            resolved = min(parts, key=lambda row: row[:3])
            self.part = resolved[5]
            self.learning_goal = (goal_key_digest(resolved[3].goal.key()), resolved[1])
        anchor = next((row for row in ordered if row[0] == self._identity_run), ordered[0])
        if anchor[0] == self._identity_run:
            self._candidate_main_key = goal_key_digest(anchor[3].goal.key())
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
            pairs = self._cache.candidates(
                self._frontier,
                goal_key=self._candidate_main_key,
                part_key=goal_key_digest(self.goal),
            )
            self._candidate_groups = {c.id: self.part for c, _ in pairs}
            return pairs
        if self._plan is None:
            return []
        selected: dict[str, tuple[Candidate, Position]] = {}
        for part, key in self._active_goals.items():
            for candidate, position in self._group_caches[key.namespace].candidates(
                _namespace_frontier(self.frontier, key.namespace),
                goal_key=goal_key_digest(self.goal),
                part_key=goal_key_digest(key),
            ):
                selected[candidate.id] = (candidate, position)
                self._candidate_groups[candidate.id] = part
        return sorted(selected.values(), key=lambda pair: pair[1])

    def _visible_candidate_ids(self) -> frozenset[str]:
        """Return fixed membership without I/O, hints, or mutable engine state."""
        return self._candidate_ids

    def _sibling_view(self, part: str | None) -> "ArenaView":
        """Select only a declared group from this frontier's executed revision."""
        with self._read():
            if part not in self._active_goals:
                raise _core_error(ValueError, "part outside board snapshot")
            result = copy(self)
            result.goal = self._active_goals[part].model_copy(deep=True)
            result._frontier = _namespace_frontier(self.frontier, result.goal.namespace)
            result._cache = self._group_caches[result.goal.namespace]
            main_goal = self._active_goals[None]
            result.learning_goal = (goal_key_digest(main_goal), main_goal.id)
            if self._candidate_main_key is not None:
                result._candidate_main_key = goal_key_digest(main_goal)
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

    def _pairs(self, ids: Sequence[str] | None = None) -> list[tuple[Candidate, Position]]:
        selected = tuple(self._candidate_ids) if ids is None else ids
        pairs = [
            pair
            for namespace, cache in self._group_caches.items()
            for pair in cache.candidates(
                _namespace_frontier(self.frontier, namespace), ids=selected
            )
        ]
        return sorted(pairs, key=lambda pair: pair[1])

    def _scores(self, ids: Sequence[str]) -> list[tuple[Score, Position]]:
        scores = [
            pair
            for namespace, cache in self._group_caches.items()
            for pair in cache.scores(
                _namespace_frontier(self.frontier, namespace), candidate_ids=ids
            )
        ]
        return sorted(scores, key=lambda pair: pair[1])

    def _ranking_judges(self) -> dict[str | None, str]:
        """Resolve fixed group names from recorded data without loading plugins."""
        with self._read():
            if self._ranking_plan is None:
                return {}
            pairs = self._ranking_pairs(self._pairs())
            scores = self._scores(tuple(self._candidate_ids))
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
        with self._read():
            self._check_goal(goal)
            if self.part is not None and part not in (None, self.part):
                raise _core_error(ValueError, "part outside board snapshot")
            group = self.part if part is None else part
            pairs = self._pairs()
            scores = self._scores(tuple(self._candidate_ids))
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
        with self._read():
            self._check_goal(goal)
            pairs = self._pairs()
            original_parts = {c.id: c.part for c, _ in pairs}
            rows = flatten(
                group_rows(
                    self._ranking_pairs(pairs),
                    self._scores(tuple(self._candidate_ids)),
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
        with self._read():
            filters = self._filters(namespace, goal, tags)
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
        with self._read():
            filters = self._filters(namespace, goal)
            return self._cache.count(self._frontier, filters=filters)

    def get_candidate(self, id: str) -> Candidate:
        """Return a candidate only when it belongs to the fixed selected keys."""
        with self._read():
            if id not in self._candidate_ids:
                raise KeyError(id)
            rows = self._pairs([id])
            if not rows:
                raise KeyError(id)
            return rows[0][0]

    def get_learning(self, id: str) -> Learning:
        """Return a namespace learning only when visible at this frontier."""
        with self._read():
            rows = self._cache.learnings_by_id(self._frontier, [id])
            if not rows or rows[0].hidden:
                raise KeyError(id)
            return rows[0]

    def scores(self, candidate_id: str) -> list[Score]:
        """Read marked scores only for a candidate in this view's membership."""
        with self._read():
            self.get_candidate(candidate_id)
            return [s for s, _ in self._scores([candidate_id])]

    def lineage(self, candidate_id: str) -> LineageGraph:
        """Traverse only resolvable fixed-view nodes, with cycle detection."""
        with self._read(), ExitStack() as reads:
            for namespace, cache in self._group_caches.items():
                reads.enter_context(cache._lock)
                reads.enter_context(cache._frontier(_namespace_frontier(self.frontier, namespace)))

            def read(ids: Sequence[str]) -> dict[str, Candidate]:
                selected = [id for id in ids if id in self._candidate_ids]
                return {c.id: c for c, _ in self._pairs(selected)}

            return _lineage(read, candidate_id)


def past_run_repeat_entries(view: BoardView, run_id: str) -> list[tuple[str, Candidate]]:
    """
    Read other-run repeat hints from an arena snapshot's resolved membership.

    Parameters
    ----------
    view : BoardView
        Fixed snapshot; other board implementations have no arena repeat hints.
    run_id : str
        Owning run excluded from the returned entries.

    Returns
    -------
    list[tuple[str, Candidate]]
        Stored repeat keys and candidates in committed position order.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> past_run_repeat_entries(Mock(spec=BoardView), "r")
    []
    """
    if not isinstance(view, ArenaView):
        return []
    with view._read():
        entries = view._cache.repeat_entries(
            view._frontier, goal_key=goal_key_digest(view.goal), exclude_run=run_id
        )
        visible = view._visible_candidate_ids()
        return [(key, candidate) for key, candidate in entries if candidate.id in visible]


def _effective_arena_root(config: RunConfig) -> str:
    """Resolve the arena location from the configured board when it owns one."""
    if config.board is not None:
        board = config.board.build(Board)
        if isinstance(board, ArenaBoard):
            return board.config.arena_root
    return config.arena_root


class _ArenaReader:
    """Expose named committed reads while keeping cache/view details in the board."""

    def __init__(self, board: ArenaBoard, state: _Published) -> None:
        self._board = board
        self._state = state

    def namespaces(self) -> tuple[str, ...]:
        """Return the namespaces explicitly attached by the caller."""
        return self._state.namespaces

    def plan_namespaces(self) -> list[str]:
        """Include namespaces declared by every current main and part group."""
        return sorted(
            {
                goal.namespace
                for run in self._state.plans.values()
                for plan in run.current.values()
                for goal in [plan.goal, *(part.goal for part in plan.parts)]
            }
        )

    def candidates(self, namespaces: Sequence[str], ids: Sequence[str]) -> list[Candidate]:
        """Read selected committed candidates, including explicit historical IDs."""
        return [
            candidate
            for ns in namespaces
            if (cache := self._state.readers.get(ns)) is not None
            for candidate, _ in cache.candidates(
                _namespace_frontier(self._state.frontier, ns), ids=ids
            )
        ]

    def _candidate_source(
        self, namespaces: Sequence[str], identifier: str
    ) -> tuple[ArenaCache, Frontier, Candidate, int]:
        """Resolve a committed candidate and its fixed source prefix."""
        for ns in namespaces:
            cache = self._state.readers.get(ns)
            if cache is not None:
                frontier = _namespace_frontier(self._state.frontier, ns)
                pairs = cache.candidates(frontier, ids=[identifier])
                if pairs:
                    candidate, position = pairs[0]
                    return cache, frontier, candidate, position[2]
        raise KeyError(identifier)

    def candidate(self, namespaces: Sequence[str], identifier: str) -> Candidate:
        """Read an explicit committed candidate without consulting local hints."""
        return self._candidate_source(namespaces, identifier)[2]

    def learning(self, namespaces: Sequence[str], identifier: str) -> Learning:
        """Read a committed learning, including hidden mutation targets."""
        for ns in namespaces:
            cache = self._state.readers.get(ns)
            if cache is not None:
                items = cache.learnings_by_id(
                    _namespace_frontier(self._state.frontier, ns), [identifier]
                )
                if items:
                    return items[0]
        raise KeyError(identifier)

    def scores(self, namespaces: Sequence[str], identifier: str) -> list[Score]:
        """Read scores bound to a retained candidate in this generation."""
        cache, frontier, _, _ = self._candidate_source(namespaces, identifier)
        return [score for score, _ in cache.scores(frontier, candidate_ids=[identifier])]

    def trace(self, namespaces: Sequence[str], identifier: str) -> list[str]:
        """Render committed provenance under one namespace frontier scope."""
        from swarm_arena._board.trace import _lines

        for namespace in namespaces:
            cache = self._state.readers.get(namespace)
            if cache is None:
                continue
            frontier = _namespace_frontier(self._state.frontier, namespace)
            with cache._read_scope(frontier):
                pairs = cache.candidates(frontier, ids=[identifier])
                if pairs:
                    candidate, position = pairs[0]
                    revisions = self._state.plans[candidate.run_id].revisions[candidate.goal_id]
                    root = next(
                        _resolved_root(mark.run_root)
                        for mark in frontier.marks
                        if mark.run_id == candidate.run_id
                    )
                    sessions = {
                        identifier: session
                        for (source, identifier), session in self._state.sessions.items()
                        if source == root
                    }
                    return _lines(
                        cache,
                        frontier,
                        candidate,
                        position[2],
                        revisions,
                        {run: projection.calls for run, projection in self._state.plans.items()},
                        sessions,
                        self._state.plans[candidate.run_id].gateway_keys,
                        self._state.plans[candidate.run_id].gateway_reconcile,
                    )
        raise KeyError(identifier)

    def llm_calls(self, run_id: str, *, namespaces: Sequence[str] | None = None) -> list[LLMCall]:
        """Return detached calls only for a source exposed by this fixed selection."""
        selected = set(self._state.namespaces if namespaces is None else namespaces)
        visible = any(
            identifier == run_id and selected.intersection(names)
            for (_, identifier), names in self._state.source_namespaces.items()
        )
        if not visible or run_id not in self._state.plans:
            raise KeyError(run_id)
        return [call.model_copy(deep=True) for call in self._state.plans[run_id].calls.values()]

    def find(
        self,
        namespaces: Sequence[str],
        *,
        query: str | None,
        filters: LearningFilter,
        limit: int,
        offset: int = 0,
    ) -> list[Learning]:
        """Filter and paginate across namespaces, hydrating returned rows only."""
        selected: list[Learning] = []
        skip = max(0, offset)
        for ns in namespaces:
            cache = self._state.readers.get(ns)
            if cache is None or len(selected) >= limit:
                continue
            frontier = _namespace_frontier(self._state.frontier, ns)
            if query is None:
                count = cache.count(frontier, filters=filters)
                if skip >= count:
                    skip -= count
                    continue
                ids = cache.find(
                    frontier,
                    query=None,
                    filters=filters,
                    relevance_goal=None,
                    offset=skip,
                    limit=limit - len(selected),
                )
            else:
                ids = cache.find(
                    frontier,
                    query=query,
                    filters=filters,
                    relevance_goal=None,
                    limit=skip + limit - len(selected),
                )
                if skip >= len(ids):
                    skip -= len(ids)
                    continue
                ids = ids[skip:]
            skip = 0
            items = {item.id: item for item in cache.learnings_by_id(frontier, ids)}
            selected.extend(items[identifier] for identifier in ids)
        return selected

    def search_markers(
        self, namespaces: Sequence[str], *, query: str | None, filters: LearningFilter, limit: int
    ) -> list[tuple[Learning, bool]]:
        """Return ordered search results and batched superseded markers."""
        items = self.find(namespaces, query=query, filters=filters, limit=limit)
        superseded: set[str] = set()
        for ns in namespaces:
            cache = self._state.readers.get(ns)
            if cache is not None:
                superseded.update(
                    cache._superseded(
                        _namespace_frontier(self._state.frontier, ns),
                        [item.id for item in items if item.namespace == ns],
                    )
                )
        return [(item, item.id in superseded) for item in items]

    def leaderboards(
        self, *, run_id: str | None, namespace: str | None, goal: str | None, judge: str | None
    ) -> tuple[list[tuple[str, str, str | None, list[Row]]], set[str]]:
        """Rank current producing groups through the board's shared row policy."""
        newest: dict[tuple[str, str], tuple[str, Plan]] = {}
        for identifier, run in sorted(self._state.plans.items()):
            if run_id is not None and identifier != run_id:
                continue
            for goal_id, plan in run.current.items():
                if namespace is not None and namespace not in {
                    plan.goal.namespace,
                    *(p.goal.namespace for p in plan.parts),
                }:
                    continue
                if goal is None or goal_id == goal:
                    newest[(plan.goal.namespace, goal_id)] = identifier, plan
        result = []
        judges: set[str] = set()
        for identifier, plan in newest.values():
            view = ArenaView(
                self._board,
                plan.goal.key(),
                self._state.frontier,
                published=replace(self._state, run_id=identifier),
            )
            ranking = view._ranking_judges()
            names = dict.fromkeys(view._active_goals, judge) if judge else ranking
            if namespace is not None:
                names = {
                    part: name
                    for part, name in names.items()
                    if view._active_goals[part].namespace == namespace
                }
            judges.update(names.values())
            pairs = view._ranking_pairs(view._pairs())
            scores = view._scores(tuple(view._visible_candidate_ids()))
            for part, name in names.items():
                groups = group_rows(
                    [pair for pair in pairs if pair[0].part == part],
                    scores,
                    judge={part: name},
                    run_id=run_id,
                    identity_run=run_id,
                    min_improvement=(
                        plan.converge.min_improvement
                        if run_id and name == ranking.get(part)
                        else None
                    ),
                )
                rows = groups.get(part, [])
                if rows:
                    result.append((view._active_goals[part].namespace, plan.goal.id, part, rows))
        return result, judges
