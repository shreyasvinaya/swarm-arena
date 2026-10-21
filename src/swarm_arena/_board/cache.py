"""Cache committed arena records, retaining replay's producing identities."""

import fcntl
import json
import logging
import os
import re
import sqlite3
import time
from collections import OrderedDict
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar, Literal

from pydantic import JsonValue

from swarm_arena._board.human_objects import (
    HumanObject,
    _check_folder_payload,
    _validate_human_key,
    human_object_order,
)
from swarm_arena._errors import ChainError, _core_error
from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.lease import Head
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.segment import parse_segment_name
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.plan import Plan
from swarm_arena._model.results import Candidate, Learning, Score

_LOG = logging.getLogger("swarm_arena.board")
Position = tuple[str, str, int]
_SCHEMA = (
    "runs(run_id TEXT PRIMARY KEY, run_root TEXT NOT NULL)",
    (
        "segments(run_id TEXT, name TEXT, epoch INTEGER, first_seq INTEGER, last_seq INTEGER, "
        "PRIMARY KEY(run_id, name))"
    ),
    "goals(run_id TEXT, goal_id TEXT, namespace TEXT, goal_key TEXT, PRIMARY KEY(run_id, goal_id))",
    (
        "plans(run_id TEXT, goal_id TEXT, revision INTEGER, origin TEXT, seq INTEGER, ts TEXT, "
        "plan_json TEXT, first_round INTEGER NOT NULL, PRIMARY KEY(run_id, goal_id, revision))"
    ),
    (
        "part_goals(run_id TEXT, goal_id TEXT, revision INTEGER, part TEXT, part_key TEXT, "
        "PRIMARY KEY(run_id, goal_id, revision, part))"
    ),
    (
        "candidates(id TEXT, producer TEXT NOT NULL, run_id TEXT, epoch INTEGER, seq INTEGER, "
        "ts TEXT, "
        "goal_id TEXT, goal_key TEXT, part_key TEXT, part TEXT, team TEXT, agent TEXT, round "
        "INTEGER, fidelity TEXT, partial INTEGER, gated_out INTEGER, repeat_key TEXT, json "
        "TEXT, PRIMARY KEY(id, producer))"
    ),
    (
        "scores(id TEXT, producer TEXT NOT NULL, source TEXT, run_id TEXT, epoch INTEGER, "
        "seq INTEGER, ts TEXT, obj_name TEXT, line INTEGER, candidate_id TEXT, "
        "judge TEXT, rubric_hash TEXT, "
        "panel_hash TEXT, json TEXT, PRIMARY KEY(id, producer))"
    ),
    (
        "learnings(rowid INTEGER PRIMARY KEY, id TEXT, producer TEXT NOT NULL, source TEXT, "
        "run_id TEXT, epoch INTEGER, seq INTEGER, ts TEXT, obj_name TEXT, line INTEGER, "
        "goal_id TEXT, goal_key TEXT, author TEXT, verified INTEGER, from_human INTEGER, "
        "json TEXT, UNIQUE(id, producer))"
    ),
    "learning_tags(learning_id TEXT, producer TEXT, tag TEXT)",
    "learning_links(learning_id TEXT, producer TEXT, kind TEXT, target_id TEXT)",
    "learning_evidence(learning_id TEXT, producer TEXT, candidate_id TEXT)",
    "curation_ops(obj_name TEXT, line INTEGER, kind TEXT, learning_id TEXT)",
    "human_objects(name TEXT PRIMARY KEY)",
    # Lossless events let every reader use core activation and lifecycle semantics.
    "_events(run_id TEXT, seq INTEGER, json TEXT NOT NULL, PRIMARY KEY(run_id, seq))",
)


def goal_key_digest(key: GoalKey) -> str:
    """
    Hash the complete serialized goal identity for SQLite equality queries.

    Parameters
    ----------
    key : GoalKey
        Snapshotted main or part identity.

    Returns
    -------
    str
        Canonical SHA-256 digest.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> len(goal_key_digest(Goal(id='g', namespace='toy').key()))
    64
    """
    return sha256_hex(canonical_json(key.model_dump(mode="json")))


def fts_query(text: str) -> str | None:
    """
    Convert literal user text into a disjunction of quoted FTS tokens.

    Parameters
    ----------
    text : str
        Text whose Unicode alphanumeric tokens are lowercased; underscores split.

    Returns
    -------
    str or None
        Quoted OR query, or None when no tokens remain.

    Examples
    --------
    >>> fts_query('log(nu) "x"')
    '"log" OR "nu" OR "x"'
    >>> fts_query('___') is None
    True
    """
    tokens = re.findall(r"[^\W_]+", text.lower())
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens) or None


class LearningFilter(FrozenModel):
    """
    Select learning metadata with intersected optional predicates.

    Parameters
    ----------
    goal_key : str or None, default=None
        Exact goal digest. With goal_id, also accepts NULL digests of that ID.
    goal_id : str or None, default=None
        Goal ID, used alone or as the human-learning fallback with goal_key.
    run : str or None, default=None
        Producing run ID.
    team : str or None, default=None
        Team component of the author address.
    agent : str or None, default=None
        Agent component of the author address.
    round : int or None, default=None
        Posting round.
    part : str or None, default=None
        Part of at least one evidence candidate inside the frontier.
    tags : list[str], default=[]
        Any-of tags; an empty list imposes no tag condition.
    verified : bool or None, default=None
        Evidence verification flag.
    human : bool or None, default=None
        Whether the author has the human prefix.
    pinned : bool or None, default=None
        Frontier-local pin status.
    hidden : bool or None, default=False
        False excludes hidden, True selects hidden, None includes both.

    Examples
    --------
    >>> LearningFilter(tags=['solver']).hidden
    False
    """

    goal_key: str | None = None
    goal_id: str | None = None
    run: str | None = None
    team: str | None = None
    agent: str | None = None
    round: int | None = None
    part: str | None = None
    tags: list[str] = []
    verified: bool | None = None
    human: bool | None = None
    pinned: bool | None = None
    hidden: bool | None = False


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Retry WAL's lock upgrade, which can bypass SQLite's busy handler."""
    deadline = time.monotonic() + 30
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
            if time.monotonic() >= deadline:
                raise _core_error(
                    RuntimeError, "arena cache WAL initialization timed out"
                ) from None
            time.sleep(0.01)


def _has_fts5(conn: sqlite3.Connection) -> bool:
    """Probe the actual SQLite runtime without adding a persistent FTS table."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp._fts5_probe USING fts5(text)")
        conn.execute("DROP TABLE temp._fts5_probe")
    except sqlite3.OperationalError:
        return False
    return True


def _validate_head(head: Head) -> None:
    """Check even cached heads before their sequence can widen a snapshot."""
    try:
        numbers = parse_segment_name(head.segment)
    except ValueError:
        raise _core_error(ChainError, "invalid head segment name") from None
    if numbers != (head.epoch, head.seq):
        raise _core_error(ChainError, "head epoch or seq does not match its segment name")


class ArenaCache:
    """
    Index one namespace's committed journals in a shared SQLite WAL database.

    Every public operation holds one reentrant lock through its entire query or
    transaction. A shared flock on the persistent ``<path>.lock`` file protects
    the database lifetime against reindex. Downloads belong outside this lock.
    Ingest retains a core replay checkpoint per run, advances only missing events,
    and discards mutated state after rollback. A per-run sequence index supports
    segment junction and newest-head lookups without scanning prior segments.
    Named indexed TEMP sets hold frontier membership and explicit ID selections.
    Frontier sets are filled once per read, including nested lookups, and all
    scratch selection rows are released on successful or failed outermost exit.
    Search filtering, relevance, pagination and counts execute in SQLite.
    Colliding IDs retain the first row from each producer. At a frontier, the
    holder is the visible row with the lexicographically smallest producer key:
    ``run:<run_id>`` for journals or ``object:<object_name>`` for human objects.
    Visibility includes the committed sequence or exact named object, and holder
    selection precedes query filters. Later arrivals cannot change old frontiers.
    Private ``_replay`` and ``_plan_for_round`` lookups accept a committed prefix
    sequence and reuse core replay, including future activation boundaries.

    Parameters
    ----------
    path : Path
        Database path; its stem identifies the namespace.

    Examples
    --------
    >>> cache = ArenaCache(Path('/tmp/arena/toy.sqlite'))  # doctest: +SKIP
    >>> cache.close()  # doctest: +SKIP
    """

    SCHEMA_VERSION: ClassVar[int] = 3

    def __init__(self, path: Path) -> None:
        self.namespace = path.stem
        self.warnings: list[str] = []
        self._lock = RLock()
        self._frontier_depth = 0
        self._fts_tables: OrderedDict[str, None] = OrderedDict()
        self._closed = False
        self._ingest_states: dict[str, tuple[str, int, RunState]] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        conn: sqlite3.Connection | None = None
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_SH)
            conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            self._conn = conn
            conn.execute("PRAGMA busy_timeout=30000")
            _enable_wal(conn)
            if not _has_fts5(conn):
                raise _core_error(
                    RuntimeError,
                    "this Python's sqlite3 has no FTS5; ArenaBoard needs SQLite built with FTS5",
                )
            with self._lock:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    version = conn.execute("PRAGMA user_version").fetchone()[0]
                    if version not in (0, self.SCHEMA_VERSION):
                        raise _core_error(
                            RuntimeError, "arena cache schema mismatch; run `swarm-arena reindex`"
                        )
                    if version == 0:
                        for table in _SCHEMA:
                            conn.execute("CREATE TABLE IF NOT EXISTS " + table)
                        conn.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS segments_run_last_seq "
                        "ON segments(run_id, last_seq)"
                    )
                    for index, definition in (
                        ("learning_tags_lookup", "learning_tags(learning_id, producer, tag)"),
                        (
                            "learning_evidence_lookup",
                            "learning_evidence(learning_id, producer, candidate_id)",
                        ),
                        (
                            "learning_superseders_lookup",
                            "learning_links(target_id, kind, learning_id, producer)",
                        ),
                        ("learning_goal_id_lookup", "learnings(goal_id)"),
                        ("learning_goal_key_lookup", "learnings(goal_key)"),
                    ):
                        conn.execute(f"CREATE INDEX IF NOT EXISTS {index} ON {definition}")
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
        except BaseException:
            if conn is not None:
                conn.close()
            os.close(self._lock_fd)
            raise

    def close(self) -> None:
        """
        Close the connection and release the lifetime lock, idempotently.

        Examples
        --------
        >>> cache.close()  # doctest: +SKIP
        """
        with self._lock:
            if not self._closed:
                try:
                    self._conn.close()
                finally:
                    os.close(self._lock_fd)
                    self._closed = True
                    self._ingest_states.clear()

    def newest_segment(self, run_id: str) -> str | None:
        """
        Return the newest locally committed segment for a run.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        str or None
            Segment key, or None for an uncached run.

        Examples
        --------
        >>> cache.newest_segment('r1')  # doctest: +SKIP
        'journal/e1-2.jsonl'
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM segments WHERE run_id=? ORDER BY last_seq DESC LIMIT 1", (run_id,)
            ).fetchone()
            return row[0] if row else None

    def has_segment(self, run_id: str, segment: str) -> bool:
        """
        Test membership in the cached committed chain.

        Parameters
        ----------
        run_id : str
            Run identifier.
        segment : str
            Exact segment key.

        Returns
        -------
        bool
            Whether the segment was committed to this cache.

        Examples
        --------
        >>> cache.has_segment('r1', 'journal/e1-2.jsonl')  # doctest: +SKIP
        True
        """
        with self._lock:
            return (
                self._conn.execute(
                    "SELECT 1 FROM segments WHERE run_id=? AND name=?", (run_id, segment)
                ).fetchone()
                is not None
            )

    def run_root(self, run_id: str) -> str | None:
        """
        Return the last ingested storage root for a run.

        Parameters
        ----------
        run_id : str
            Run identifier.

        Returns
        -------
        str or None
            Recorded root, or None for an uncached run.

        Examples
        --------
        >>> cache.run_root('r1')  # doctest: +SKIP
        '/runs'
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT run_root FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return row[0] if row else None

    def apply_human_objects(
        self,
        names: Sequence[str],
        objects: Sequence[HumanObject],
        *,
        covered: Sequence[str] = (),
    ) -> None:
        """
        Atomically index complete human objects, retaining their original lines.

        Parameters
        ----------
        names : Sequence[str]
            One arena-relative name per operation, repeated for each object's
            lines. Supply every line of each named object in its original order.
        objects : Sequence[HumanObject]
            Corresponding validated operations. Foreign namespaces are ignored.
        covered : Sequence[str], default=()
            Names of completely read objects with no operations. These receive
            the same durable completion markers as nonempty objects. Repeated
            names are harmless; a name also in ``names`` retains its operations.

        Raises
        ------
        ValueError
            If lengths, reserved locations, folders, or namespaces disagree.

        Notes
        -----
        Each name is immutable. Reapplying it has no effect. The line number is
        zero-based within that name, even when names are interleaved. Completion
        markers, including those of empty objects, and indexed rows commit in
        one transaction under the cache lock and survive reopening the cache.

        Examples
        --------
        >>> cache.apply_human_objects([name], [operation])  # doctest: +SKIP
        """
        if len(names) != len(objects):
            raise _core_error(ValueError, "human object names and operations must align")
        grouped: dict[str, list[HumanObject]] = {}
        for name in covered:
            _validate_human_key(name)
            grouped.setdefault(name, [])
        for name, obj in zip(names, objects, strict=True):
            folder = _validate_human_key(name)
            _check_folder_payload(folder, [obj])
            if obj.learning is not None and obj.learning.namespace != name.split("/", 1)[0]:
                raise _core_error(ValueError, "human learning namespace does not match object")
            grouped.setdefault(name, []).append(obj)
        with self._lock:
            pending: list[str] = []
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for name in sorted(grouped, key=human_object_order):
                    if name.split("/", 1)[0] != self.namespace or self.has_object(name):
                        continue
                    ts = datetime.strptime(human_object_order(name)[0], "%Y%m%dT%H%M%S.%fZ")
                    stamp = ts.replace(tzinfo=UTC).isoformat(timespec="microseconds")
                    for line, obj in enumerate(grouped[name]):
                        self._index_human(name, line, stamp, obj, pending)
                    self._conn.execute("INSERT INTO human_objects VALUES (?)", (name,))
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self.warnings.extend(pending)
            for warning in pending:
                _LOG.warning("%s", warning)

    def _insert_learning_children(self, learning: Learning, producer: str) -> None:
        """Project child rows after parent insertion within the caller's transaction."""
        self._conn.executemany(
            "INSERT INTO learning_tags VALUES (?, ?, ?)",
            [(learning.id, producer, tag) for tag in learning.tags],
        )
        self._conn.executemany(
            "INSERT INTO learning_evidence VALUES (?, ?, ?)",
            [(learning.id, producer, cid) for cid in learning.evidence],
        )
        self._conn.executemany(
            "INSERT INTO learning_links VALUES (?, ?, ?, ?)",
            [
                (learning.id, producer, kind, target)
                for kind in ("refines", "supersedes")
                for target in getattr(learning, kind)
            ],
        )

    def _index_human(
        self, name: str, line: int, ts: str, obj: HumanObject, pending: list[str]
    ) -> None:
        """Project one validated operation inside its object's transaction."""
        if obj.kind in {"pin", "hide"}:
            self._conn.execute(
                "INSERT INTO curation_ops VALUES (?, ?, ?, ?)",
                (name, line, obj.kind, obj.learning_id),
            )
        elif obj.kind == "review":
            assert obj.score is not None
            score = obj.score
            self._insert(
                "scores",
                dict(
                    id=score.id,
                    source="review",
                    run_id=None,
                    epoch=None,
                    seq=None,
                    ts=ts,
                    obj_name=name,
                    line=line,
                    candidate_id=score.candidate_id,
                    judge=score.judge,
                    rubric_hash=score.rubric_hash,
                    panel_hash=score.panel_hash,
                    json=score.model_dump_json(),
                ),
                "score",
                pending,
            )
        else:
            assert obj.learning is not None
            learning = obj.learning
            if producer := self._insert(
                "learnings",
                dict(
                    id=learning.id,
                    source="human",
                    run_id=learning.run_id,
                    epoch=None,
                    seq=None,
                    ts=ts,
                    obj_name=name,
                    line=line,
                    goal_id=learning.goal_id,
                    goal_key=None,
                    author=learning.author,
                    verified=learning.verified,
                    from_human=learning.from_human,
                    json=learning.model_dump_json(),
                ),
                "learning",
                pending,
            ):
                self._insert_learning_children(learning, producer)

    def has_object(self, name: str) -> bool:
        """
        Test whether a complete human object has been indexed.

        Parameters
        ----------
        name : str
            Exact arena-relative object name.

        Returns
        -------
        bool
            Whether the object's transaction committed in this namespace.

        Examples
        --------
        >>> cache.has_object(name)  # doctest: +SKIP
        True
        """
        with self._lock:
            return (
                self._conn.execute("SELECT 1 FROM human_objects WHERE name=?", (name,)).fetchone()
                is not None
            )

    def _replace_sets(self, sets: Sequence[tuple[str, Iterable[tuple[str, int | None]]]]) -> None:
        """Atomically replace named indexed sets with bounded row bindings."""
        if not sets:
            return
        self._conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _read_sets "
            "(set_name TEXT, id TEXT, seq INTEGER, PRIMARY KEY(set_name, id)) WITHOUT ROWID"
        )
        self._conn.execute("SAVEPOINT read_sets")
        try:
            for name, rows in sets:
                self._conn.execute("DELETE FROM _read_sets WHERE set_name=?", (name,))
                self._conn.executemany(
                    "INSERT OR IGNORE INTO _read_sets VALUES (?, ?, ?)",
                    ((name, item, seq) for item, seq in rows),
                )
            self._conn.execute("RELEASE read_sets")
        except BaseException:
            self._conn.execute("ROLLBACK TO read_sets")
            self._conn.execute("RELEASE read_sets")
            raise

    @contextmanager
    def _frontier(self, frontier: Frontier) -> Generator[None, None, None]:
        """Fill named frontier sets once, sharing storage with explicit ID sets."""
        if self._frontier_depth:
            yield
            return
        # Ingest depends on this cache; defer its shared validation import.
        from swarm_arena._board.ingest import _validate_marks

        marks = _validate_marks(frontier.marks)
        for mark in marks:
            if not self._covered(
                mark.run_id, Head(epoch=mark.epoch, seq=mark.seq, segment=mark.segment)
            ):
                raise _core_error(ChainError, "frontier mark is not covered by cache")
        self._replace_sets(
            (
                ("marks", ((mark.run_id, mark.seq) for mark in marks)),
                ("reviews", ((name, None) for name in frontier.reviews)),
                ("curation", ((name, None) for name in frontier.curation)),
            )
        )
        self._frontier_depth += 1
        try:
            yield
        finally:
            self._frontier_depth -= 1
            if self._frontier_depth == 0:
                self._conn.execute("DELETE FROM temp._read_sets")

    @staticmethod
    def _member_join(table: Literal["candidates", "scores", "learnings"], alias: str) -> str:
        """Join requested marks and source-appropriate human names by their keys."""
        marked = (
            f" JOIN _read_sets m ON m.set_name='marks' AND m.id={alias}.run_id "
            f"AND {alias}.seq<=m.seq "
        )
        if table == "candidates":
            return marked
        kind = "reviews" if table == "scores" else "curation"
        source = "review" if table == "scores" else "human"
        return (
            " LEFT" + marked + f"AND {alias}.source='run' "
            f"LEFT JOIN _read_sets o ON o.set_name='{kind}' AND o.id={alias}.obj_name "
            f"AND {alias}.source='{source}' "
        )

    @classmethod
    def _member_condition(
        cls, table: Literal["candidates", "scores", "learnings"], alias: str = "i"
    ) -> str:
        """Select the smallest visible producer per ID before any query filters."""
        visible = (
            "m.id IS NOT NULL"
            if table == "candidates"
            else ("(m.id IS NOT NULL OR o.id IS NOT NULL)")
        )
        return (
            visible
            + f" AND NOT EXISTS (SELECT 1 FROM {table} holder "
            + cls._member_join(table, "holder")
            + f" WHERE holder.id={alias}.id AND holder.producer<{alias}.producer AND "
            + visible
            + ")"
        )

    def _item_rows(
        self,
        frontier: Frontier,
        table: Literal["candidates", "scores", "learnings"],
        *,
        ids: Sequence[str] | None = None,
        filters: dict[str, str] | None = None,
        candidate_ids: Sequence[str] | None = None,
    ) -> list[tuple[str, Position]]:
        """Read immutable rows using only their marked run or named source object."""
        params: list[Any] = []
        if table == "candidates":
            origin, ordinal = "i.run_id", "i.seq"
        else:
            origin, ordinal = "coalesce(i.obj_name, i.run_id)", "coalesce(i.line, i.seq)"
        clauses = [self._member_condition(table)]
        selected = [
            (column, values)
            for column, values in (("id", ids), ("candidate_id", candidate_ids))
            if values is not None
        ]
        self._replace_sets(
            [(column, ((value, None) for value in values)) for column, values in selected]
        )
        selection_joins = "".join(
            f" JOIN _read_sets selected_{column} ON selected_{column}.set_name='{column}' "
            f"AND selected_{column}.id=i.{column} "
            for column, _ in selected
        )
        for column, value in (filters or {}).items():
            clauses.append(f"i.{column}=?")
            params.append(value)
        rows = self._conn.execute(
            f"SELECT i.json, i.ts, {origin}, {ordinal} FROM {table} i "
            + self._member_join(table, "i")
            + selection_joins
            + " WHERE "
            + " AND ".join(clauses)
            + f" ORDER BY i.ts, {origin}, {ordinal}, i.id",
            params,
        ).fetchall()
        return [(data, (ts, source, seq)) for data, ts, source, seq in rows]

    def candidates(
        self,
        frontier: Frontier,
        *,
        ids: Sequence[str] | None = None,
        goal_key: str | None = None,
        part_key: str | None = None,
        run_id: str | None = None,
    ) -> list[tuple[Candidate, Position]]:
        """
        Read candidates within exact committed marks and producing-key filters.

        Parameters
        ----------
        frontier : Frontier
            Covered committed marks; later cached events remain invisible.
        ids : Sequence[str] or None, default=None
            Candidate IDs to include; an empty sequence selects none. Named
            indexed TEMP sets support selections beyond SQLite binding limits.
        goal_key : str or None, default=None
            Main goal identity digest.
        part_key : str or None, default=None
            Immutable producing identity digest, never a later revision's key.
        run_id : str or None, default=None
            Producing run identifier.

        Returns
        -------
        list[tuple[Candidate, Position]]
            Independent models and UTC timestamp/run/sequence positions, ordered
            by position. All supplied filters are intersected.

        Raises
        ------
        ChainError
            If marks conflict or are not exactly covered. Reads never fetch.

        Examples
        --------
        >>> rows = cache.candidates(frontier, ids=['c-1'])  # doctest: +SKIP
        """
        with self._lock, self._frontier(frontier):
            filters = {
                key: value
                for key, value in (
                    ("goal_key", goal_key),
                    ("part_key", part_key),
                    ("run_id", run_id),
                )
                if value is not None
            }
            return [
                (Candidate.model_validate_json(data), position)
                for data, position in self._item_rows(
                    frontier, "candidates", ids=ids, filters=filters
                )
            ]

    def repeat_entries(
        self, frontier: Frontier, *, goal_key: str, exclude_run: str
    ) -> list[tuple[str, Candidate]]:
        """
        Read other-run repeat identities in committed position order.

        Parameters
        ----------
        frontier : Frontier
            Exactly covered snapshot.
        goal_key : str
            Immutable producing goal-key digest.
        exclude_run : str
            Owning run to exclude.

        Returns
        -------
        list[tuple[str, Candidate]]
            Stored repeat keys and independent candidates, including all statuses.

        Examples
        --------
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as root:
        ...     cache = ArenaCache(Path(root) / "toy.sqlite")
        ...     cache.repeat_entries(Frontier(), goal_key="key", exclude_run="r")
        ...     cache.close()
        []
        """
        with self._lock, self._frontier(frontier):
            rows = self._conn.execute(
                "SELECT i.repeat_key, i.json FROM candidates i "
                + self._member_join("candidates", "i")
                + " WHERE "
                + self._member_condition("candidates")
                + " AND i.part_key=? AND i.run_id!=? AND i.repeat_key IS NOT NULL"
                + " ORDER BY i.ts, i.run_id, i.seq, i.id",
                (goal_key, exclude_run),
            ).fetchall()
            return [(key, Candidate.model_validate_json(data)) for key, data in rows]

    def scores(
        self,
        frontier: Frontier,
        *,
        candidate_ids: Sequence[str] | None = None,
        judge: str | None = None,
    ) -> list[tuple[Score, Position]]:
        """
        Read marked journal scores and exactly named human review scores.

        Parameters
        ----------
        frontier : Frontier
            Covered marks and exact review names. Review receipts add no scores.
        candidate_ids : Sequence[str] or None, default=None
            Candidate IDs to include; an empty sequence selects none. Named
            indexed TEMP sets support selections beyond SQLite binding limits.
        judge : str or None, default=None
            Exact recorded judge name.

        Returns
        -------
        list[tuple[Score, Position]]
            Scores ordered by UTC timestamp, run/object name, and sequence/line.

        Raises
        ------
        ChainError
            If marks conflict or are not exactly covered.

        Examples
        --------
        >>> rows = cache.scores(frontier, judge='human:alice')  # doctest: +SKIP
        """
        with self._lock, self._frontier(frontier):
            filters = {} if judge is None else {"judge": judge}
            return [
                (Score.model_validate_json(data), position)
                for data, position in self._item_rows(
                    frontier, "scores", candidate_ids=candidate_ids, filters=filters
                )
            ]

    def flags(self, frontier: Frontier) -> dict[str, tuple[bool, bool]]:
        """
        Fold pins and hides from only the frontier's named curation objects.

        Parameters
        ----------
        frontier : Frontier
            Covered marks and exact curation names.

        Returns
        -------
        dict[str, tuple[bool, bool]]
            Learning IDs mapped to pinned/hidden flags. The latest operation by
            object timestamp, full name, and line wins, even for an absent ID.

        Raises
        ------
        ChainError
            If marks conflict or are not exactly covered.

        Examples
        --------
        >>> flags = cache.flags(frontier)  # doctest: +SKIP
        """
        with self._lock, self._frontier(frontier):
            rows = self._conn.execute(self._flags_sql() + "SELECT learning_id, kind FROM flags")
            return {lid: (kind == "pin", kind == "hide") for lid, kind in rows}

    def _learnings(self, frontier: Frontier, ids: Sequence[str] | None = None) -> list[Learning]:
        """Materialize frontier members, including hidden and superseded records."""
        rows = self._item_rows(frontier, "learnings", ids=ids)
        flags = self.flags(frontier)
        result: list[Learning] = []
        for data, _ in rows:
            learning = Learning.model_validate_json(data)
            pinned, hidden = flags.get(learning.id, (False, False))
            result.append(learning.model_copy(update={"pinned": pinned, "hidden": hidden}))
        return result

    def learnings_by_id(self, frontier: Frontier, ids: Sequence[str]) -> list[Learning]:
        """
        Read member learnings with frontier-local flags, including hidden ones.

        Parameters
        ----------
        frontier : Frontier
            Covered marks and exact human-object names.
        ids : Sequence[str]
            Requested learning IDs. Unknown and nonmember IDs are omitted.
            Named indexed TEMP sets avoid a binding limit on selection size.

        Returns
        -------
        list[Learning]
            Independent models in commit/object position order. Hidden and
            superseded records remain available for explicit-ID inspection.

        Raises
        ------
        ChainError
            If marks conflict or are not exactly covered.

        Examples
        --------
        >>> rows = cache.learnings_by_id(frontier, ['L-1'])  # doctest: +SKIP
        """
        with self._lock, self._frontier(frontier):
            return self._learnings(frontier, ids)

    def plans(self, frontier: Frontier, run_id: str, goal_id: str) -> list[tuple[Plan, str, int]]:
        """
        Read committed plan revisions for one marked run and goal.

        Parameters
        ----------
        frontier : Frontier
            Covered committed marks, without any latest-head fallback.
        run_id : str
            Owning run identifier.
        goal_id : str
            Owning main goal identifier.

        Returns
        -------
        list[tuple[Plan, str, int]]
            Plan, recorded origin, and event sequence, ordered by revision.
            Activation remains available through the private replay lookups;
            publication of a revision does not imply immediate activation.

        Raises
        ------
        ChainError
            If marks conflict or are not exactly covered.

        Examples
        --------
        >>> rows = cache.plans(frontier, 'r1', 'g1')  # doctest: +SKIP
        """
        with self._lock, self._frontier(frontier):
            rows = self._conn.execute(
                "SELECT plan_json, origin, p.seq FROM plans p "
                "JOIN _read_sets m ON m.set_name='marks' AND m.id=p.run_id AND p.seq<=m.seq "
                "WHERE p.run_id=? AND goal_id=? ORDER BY revision",
                (run_id, goal_id),
            ).fetchall()
            return [(Plan.model_validate_json(data), origin, seq) for data, origin, seq in rows]

    def _plan_projection(
        self,
        frontier: Frontier,
        run_id: str,
        goal_id: str,
        *,
        through_seq: int,
        revisions: Sequence[Plan],
    ) -> RunState:
        """Query historical plan metadata for the core round-applicable selector.

        Use models retained by the published generation, restricted to revisions
        inside both the covered mark and the candidate source prefix. Only
        activation rounds and origins are queried; no plan bodies are parsed
        and no journal events are folded. The core selector retains its
        initial-revision fallback.
        """
        with self._lock, self._frontier(frontier):
            rows = self._conn.execute(
                "SELECT p.revision, p.origin, p.first_round FROM plans p "
                "JOIN _read_sets m ON m.set_name='marks' AND m.id=p.run_id AND p.seq<=m.seq "
                "WHERE p.run_id=? AND p.goal_id=? AND p.seq<=? ORDER BY p.seq",
                (run_id, goal_id, through_seq),
            ).fetchall()
            if not rows:
                raise _core_error(ChainError, "no committed plan for goal")
            retained = {plan.revision: plan for plan in revisions}
            if any(revision not in retained for revision, _, _ in rows):
                raise _core_error(ChainError, "published plan revision unavailable")
            return RunState(
                plans={goal_id: [(retained[revision], origin) for revision, origin, _ in rows]},
                plan_first_rounds={goal_id: [first_round for _, _, first_round in rows]},
            )

    def _citing_learning_ids(self, frontier: Frontier, candidate_id: str) -> list[str]:
        """Select every visible citation through persisted evidence before hydration.

        Reuse exact run/object membership, human-object coverage and the same
        frontier-local hidden rules as ordinary learning reads. The evidence
        subquery returns learning IDs without parsing unrelated model bodies.
        """
        with self._lock, self._frontier(frontier):
            self._search_covered()
            source, params = self._learning_sql(LearningFilter())
            rows = self._conn.execute(
                self._flags_sql() + "SELECT l.id FROM " + source + " AND (l.id, l.producer) IN "
                "(SELECT learning_id, producer FROM learning_evidence WHERE candidate_id=?)"
                + " ORDER BY l.id",
                [*params, candidate_id],
            )
            return [row[0] for row in rows]

    def membership(self, frontier: Frontier) -> bytes:
        """
        Encode sorted candidate, score, and flagged learning membership.

        Parameters
        ----------
        frontier : Frontier
            Exact covered journal marks and human-object names.

        Returns
        -------
        bytes
            Canonical JSON with sorted candidates, scores, and learning triples
            containing ID, pinned, and hidden. Hidden learnings are retained.

        Raises
        ------
        ChainError
            If marks conflict or are not exactly covered.

        Examples
        --------
        >>> encoded = cache.membership(frontier)  # doctest: +SKIP
        """
        with self._lock, self._frontier(frontier):
            members: dict[str, JsonValue] = {
                "candidates": [cid for cid in sorted(c.id for c, _ in self.candidates(frontier))],
                "scores": [sid for sid in sorted(s.id for s, _ in self.scores(frontier))],
                "learnings": [
                    [learning.id, learning.pinned, learning.hidden]
                    for learning in sorted(self._learnings(frontier), key=lambda item: item.id)
                ],
            }
            return canonical_json(members)

    @staticmethod
    def _flags_sql() -> str:
        """Select the last pin/hide using validated curation-name ordering."""
        # All stored curation names share the namespace/folder prefix, so their
        # lexical order is timestamp then full name, exactly as fold_curation.
        return (
            "WITH flags AS (SELECT learning_id, kind FROM ("
            "SELECT learning_id, c.kind, row_number() OVER (PARTITION BY learning_id "
            "ORDER BY obj_name DESC, line DESC) AS ordinal FROM curation_ops c "
            "JOIN _read_sets o ON o.set_name='curation' AND o.id=c.obj_name"
            ") WHERE ordinal=1) "
        )

    def _search_covered(self) -> None:
        """Refuse partial human membership before any query or FTS construction."""
        if self._conn.execute(
            "SELECT 1 FROM _read_sets o LEFT JOIN human_objects h ON h.name=o.id "
            "WHERE o.set_name IN ('reviews','curation') AND h.name IS NULL LIMIT 1"
        ).fetchone():
            raise _core_error(ChainError, "frontier human object is not covered by cache")

    def _fts_index(self, frontier: Frontier) -> str:
        """Build a complete TEMP corpus in SQL, retaining eight by access order."""
        name = "fts_" + frontier.digest()[:16]
        if name in self._fts_tables:
            self._fts_tables.move_to_end(name)
            return name
        self._conn.execute(f"CREATE VIRTUAL TABLE temp.{name} USING fts5(title, text, tags)")
        try:
            self._conn.execute(
                f"INSERT INTO temp.{name}(rowid, title, text, tags) "
                "SELECT l.rowid, json_extract(l.json, '$.title'), json_extract(l.json, '$.text'), "
                "coalesce((SELECT group_concat(value, ' ') FROM json_each(l.json, '$.tags')), '') "
                "FROM learnings l "
                + self._member_join("learnings", "l")
                + " WHERE "
                + self._member_condition("learnings", "l")
                + " ORDER BY l.id"
            )
        except BaseException:
            self._conn.execute(f"DROP TABLE temp.{name}")
            raise
        self._fts_tables[name] = None
        if len(self._fts_tables) > 8:
            oldest, _ = self._fts_tables.popitem(last=False)
            self._conn.execute(f"DROP TABLE temp.{oldest}")
        return name

    def _learning_sql(
        self, filters: LearningFilter, table: str | None = None
    ) -> tuple[str, list[Any]]:
        """Compose shared SQL membership and intersected learning predicates."""
        source = (
            "learnings l "
            if table is None
            else (f"temp.{table} JOIN learnings l ON l.rowid={table}.rowid ")
        )
        source += self._member_join("learnings", "l")
        source += " LEFT JOIN flags f ON f.learning_id=l.id WHERE "
        clauses = [self._member_condition("learnings", "l")]
        params: list[Any] = []
        if filters.goal_key is not None:
            if filters.goal_id is None:
                clauses.append("l.goal_key=?")
                params.append(filters.goal_key)
            else:
                clauses.append("(l.goal_key=? OR (l.goal_key IS NULL AND l.goal_id=?))")
                params.extend((filters.goal_key, filters.goal_id))
        elif filters.goal_id is not None:
            clauses.append("l.goal_id=?")
            params.append(filters.goal_id)
        for expression, value in (
            ("l.run_id", filters.run),
            (
                "CASE WHEN instr(l.author,'/')=0 THEN l.author ELSE "
                "substr(l.author,1,instr(l.author,'/')-1) END",
                filters.team,
            ),
            (
                "CASE WHEN instr(l.author,'/')=0 THEN '' ELSE "
                "substr(l.author,instr(l.author,'/')+1) END",
                filters.agent,
            ),
            ("json_extract(l.json,'$.round')", filters.round),
            ("l.verified", filters.verified),
            ("l.from_human", filters.human),
            ("coalesce(f.kind='pin',0)", filters.pinned),
            ("coalesce(f.kind='hide',0)", filters.hidden),
        ):
            if value is not None:
                clauses.append(expression + "=?")
                params.append(value)
        if filters.tags:
            clauses.append(
                "EXISTS (SELECT 1 FROM learning_tags t JOIN json_each(?) wanted "
                "ON wanted.value=t.tag WHERE t.learning_id=l.id AND t.producer=l.producer)"
            )
            params.append(json.dumps(filters.tags))
        if filters.part is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM learning_evidence e "
                "JOIN candidates c ON c.id=e.candidate_id "
                + self._member_join("candidates", "c")
                + "WHERE e.learning_id=l.id AND e.producer=l.producer AND c.part=? AND "
                + self._member_condition("candidates", "c")
                + ")"
            )
            params.append(filters.part)
        return source + " AND ".join(clauses), params

    def _relevance_sql(
        self, filters: LearningFilter, relevance_goal: tuple[str, str] | None
    ) -> tuple[str, list[Any]]:
        """Order by visible superseders and metadata without loading JSON models."""
        superseded = (
            "EXISTS (SELECT 1 FROM learning_links link JOIN learnings s "
            "ON s.id=link.learning_id AND s.producer=link.producer "
            + self._member_join("learnings", "s")
            + "LEFT JOIN flags sf ON sf.learning_id=s.id "
            "WHERE link.target_id=l.id AND link.kind='supersedes' AND "
            + self._member_condition("learnings", "s")
            + " AND coalesce(sf.kind,'')!='hide')"
        )
        params: list[Any] = []
        goal_match = "0.0"
        if relevance_goal is not None:
            goal_match = "coalesce(l.goal_key=? OR (l.goal_key IS NULL AND l.goal_id=?),0)"
            params.extend(relevance_goal)
        matching_tags = "0.0"
        if filters.tags:
            matching_tags = (
                "(SELECT count(DISTINCT t.tag) FROM learning_tags t JOIN json_each(?) wanted "
                "ON wanted.value=t.tag WHERE t.learning_id=l.id AND t.producer=l.producer)"
            )
            params.append(json.dumps(filters.tags))
        return (
            "coalesce(f.kind='pin',0) DESC, "
            + superseded
            + " ASC, "
            + goal_match
            + " DESC, l.verified DESC, "
            + matching_tags
            + " DESC, l.ts DESC, l.id ASC",
            params,
        )

    def find(
        self,
        frontier: Frontier,
        *,
        query: str | None,
        filters: LearningFilter,
        relevance_goal: tuple[str, str] | None,
        offset: int = 0,
        limit: int = 10,
    ) -> list[str]:
        """
        Find learning IDs by frontier-local text rank and relevance.

        Parameters
        ----------
        frontier : Frontier
            Fully covered fixed journal marks and human-object names.
        query : str or None
            Literal user text, or None to use relevance alone. Tokenless text
            matches nothing; punctuation is never interpreted as FTS syntax.
        filters : LearningFilter
            Intersected metadata predicates; tags use any-of matching.
        relevance_goal : tuple[str, str] or None
            Preferred goal digest and ID, independent of the goal filter.
        offset : int, default=0
            Number of ranked matches to skip; negative values act as zero.
        limit : int, default=10
            Maximum returned IDs; a negative value returns all remaining IDs.

        Returns
        -------
        list[str]
            IDs ordered by bm25 when querying, then pinned, not superseded,
            goal match, verified, matching tag count, newest timestamp, and ID.
            Only visible frontier members can supersede another learning.

        Raises
        ------
        ChainError
            If a journal mark or named human object is not fully covered.

        Notes
        -----
        SQL applies filters, relevance, and pagination before IDs are returned.
        Each connection caches at most eight TEMP FTS corpora by least recent
        use. Corpus statistics include hidden members, but exclude all rows
        outside the fixed frontier. Later ingestion cannot alter that corpus.

        Examples
        --------
        >>> ids = cache.find(frontier, query='log(nu)', filters=LearningFilter(),
        ...     relevance_goal=None)  # doctest: +SKIP
        """
        with self._lock, self._frontier(frontier):
            self._search_covered()
            table = None
            match = None
            if query is not None:
                match = fts_query(query)
                if match is None:
                    return []
                table = self._fts_index(frontier)
            source, params = self._learning_sql(filters, table)
            if table is not None:
                source += f" AND {table} MATCH ?"
                params.append(match)
            order, order_params = self._relevance_sql(filters, relevance_goal)
            if table is not None:
                order = f"bm25({table}) ASC, " + order
            rows = self._conn.execute(
                self._flags_sql()
                + "SELECT l.id FROM "
                + source
                + " ORDER BY "
                + order
                + " LIMIT ? OFFSET ?",
                [*params, *order_params, limit, max(offset, 0)],
            )
            return [row[0] for row in rows]

    def count(self, frontier: Frontier, *, filters: LearningFilter) -> int:
        """
        Count the frontier members satisfying learning filters.

        Parameters
        ----------
        frontier : Frontier
            Fully covered fixed journal marks and human-object names.
        filters : LearningFilter
            Same predicates and hidden default as ``find``.

        Returns
        -------
        int
            Total matches before pagination, without a text query.

        Raises
        ------
        ChainError
            If the fixed frontier is not fully covered.

        Examples
        --------
        >>> total = cache.count(frontier, filters=LearningFilter())  # doctest: +SKIP
        """
        with self._lock, self._frontier(frontier):
            self._search_covered()
            source, params = self._learning_sql(filters)
            return self._conn.execute(
                self._flags_sql() + "SELECT count(*) FROM " + source, params
            ).fetchone()[0]

    def search(
        self,
        namespace: str,
        query: str,
        *,
        frontier: Frontier,
        goal: GoalKey | None,
        tags: Sequence[str],
        limit: int,
    ) -> list[str]:
        """
        Search this namespace with an optional goal and any-of tags.

        Parameters
        ----------
        namespace : str
            Must equal this cache's namespace.
        query : str
            User text to tokenize and match literally.
        frontier : Frontier
            Fully covered fixed journal marks and human-object names.
        goal : GoalKey or None
            Goal filter and relevance preference, including human goal-ID fallback.
        tags : Sequence[str]
            Any-of tags, also used for relevance tie ordering.
        limit : int
            Maximum number of learning IDs.

        Returns
        -------
        list[str]
            Visible learning IDs in text rank and relevance order.

        Raises
        ------
        ValueError
            If namespace differs from the cache namespace.
        ChainError
            If the fixed frontier is not fully covered.

        Examples
        --------
        >>> ids = cache.search('toy', 'log(nu)', frontier=frontier,
        ...     goal=None, tags=(), limit=10)  # doctest: +SKIP
        """
        if namespace != self.namespace:
            raise _core_error(ValueError, "search namespace does not match cache")
        digest = goal_key_digest(goal) if goal is not None else None
        return self.find(
            frontier,
            query=query,
            filters=LearningFilter(
                goal_key=digest, goal_id=goal.id if goal else None, tags=list(tags)
            ),
            relevance_goal=(digest, goal.id) if digest is not None and goal is not None else None,
            limit=limit,
        )

    def _covered(self, run_id: str, head: Head) -> bool:
        """Validate full metadata and ancestry, rejecting divergent older marks."""
        with self._lock:
            _validate_head(head)
            row = self._conn.execute(
                "SELECT epoch, last_seq FROM segments WHERE run_id=? AND name=?",
                (run_id, head.segment),
            ).fetchone()
            if row is not None:
                if row != (head.epoch, head.seq):
                    raise _core_error(ChainError, "cached segment metadata does not match head")
                return True
            newest = self._conn.execute(
                "SELECT max(last_seq) FROM segments WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            if newest is not None and newest >= head.seq:
                raise _core_error(ChainError, "head is not on the cached chain")
            return False

    def _replay(self, run_id: str, *, through_seq: int | None = None) -> RunState:
        """Reconstruct an independent core state from a cached committed prefix."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT json FROM _events WHERE run_id=? AND (? IS NULL OR seq<=?) ORDER BY seq",
                (run_id, through_seq, through_seq),
            ).fetchall()
            try:
                return RunState.from_events(Event.model_validate_json(row[0]) for row in rows)
            except (ValueError, TypeError, KeyError):
                raise _core_error(ChainError, "invalid journal replay payload") from None

    def _ingest_state(self, run_id: str, through_seq: int) -> RunState:
        """Consume a valid checkpoint and catch up under the write transaction.

        Taking the checkpoint out before mutation ensures a failed fold or
        transaction cannot leave uncommitted state available to the next call.
        Only COMMIT publishes it again. External appenders require only a suffix
        fold; a missing or rewound checkpoint requires independent reconstruction.
        """
        checkpoint = self._ingest_states.pop(run_id, None)
        if checkpoint is None:
            return self._replay(run_id, through_seq=through_seq)
        name, seq, state = checkpoint
        row = self._conn.execute(
            "SELECT last_seq FROM segments WHERE run_id=? AND name=?", (run_id, name)
        ).fetchone()
        if seq > through_seq or row != (seq,):
            return self._replay(run_id, through_seq=through_seq)
        if seq < through_seq:
            rows = self._conn.execute(
                "SELECT json FROM _events WHERE run_id=? AND seq>? AND seq<=? ORDER BY seq",
                (run_id, seq, through_seq),
            ).fetchall()
            try:
                for row in rows:
                    state._apply(Event.model_validate_json(row[0]))
            except (ValueError, TypeError, KeyError):
                raise _core_error(ChainError, "invalid journal replay payload") from None
        return state

    def _plan_for_round(self, run_id: str, goal_id: str, round: int, *, through_seq: int) -> Plan:
        """Select an executed revision using core replay within the supplied prefix."""
        with self._lock:
            try:
                return self._replay(run_id, through_seq=through_seq).plan_for_round(goal_id, round)
            except (KeyError, ValueError):
                raise _core_error(ChainError, "no committed plan for goal") from None

    def ingest(
        self, run_id: str, run_root: str, segments: Sequence[tuple[str, list[Event]]]
    ) -> None:
        """
        Atomically index each validated segment, skipping identical cached ones.

        Parameters
        ----------
        run_id : str
            Expected run identifier on every event.
        run_root : str
            Storage root recorded for future reads.
        segments : Sequence[tuple[str, list[Event]]]
            Oldest-first committed chain returned by ``read_chain``.

        Raises
        ------
        ChainError
            If sequence, routing, plan, or identity data are inconsistent.

        Examples
        --------
        >>> cache.ingest('r1', '/runs', segments)  # doctest: +SKIP
        """
        with self._lock:
            self._ingest(run_id, run_root, segments)

    def _ingest(
        self, run_id: str, run_root: str, segments: Sequence[tuple[str, list[Event]]]
    ) -> int:
        """Return the actual event count inserted by this connection's transactions."""
        with self._lock:
            count = 0
            for name, events in segments:
                pending: list[str] = []
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    if self._segment(run_id, name, events):
                        self._conn.execute("COMMIT")
                        continue
                    state = self._ingest_state(run_id, events[0].seq - 1)
                    for event in events:
                        try:
                            state._apply(deepcopy(event))
                        except (ValueError, TypeError, KeyError):
                            raise _core_error(
                                ChainError, "invalid journal replay payload"
                            ) from None
                        self._index(event, state, pending)
                        self._conn.execute(
                            "INSERT INTO _events VALUES (?, ?, ?)",
                            (run_id, event.seq, event.model_dump_json()),
                        )
                    self._conn.execute(
                        "INSERT INTO runs VALUES (?, ?) ON CONFLICT(run_id) "
                        "DO UPDATE SET run_root=excluded.run_root",
                        (run_id, run_root),
                    )
                    self._conn.execute(
                        "INSERT INTO segments VALUES (?, ?, ?, ?, ?)",
                        (run_id, name, events[0].epoch, events[0].seq, events[-1].seq),
                    )
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
                self._ingest_states[run_id] = (name, events[-1].seq, state)
                count += len(events)
                self.warnings.extend(pending)
                for warning in pending:
                    _LOG.warning("%s", warning)
            return count

    def _segment(self, run_id: str, name: str, events: list[Event]) -> bool:
        """Check a segment and its exact junction while holding the write lock."""
        try:
            epoch, last = parse_segment_name(name)
        except ValueError:
            raise _core_error(ChainError, "invalid segment name") from None
        if not events or events[-1].seq != last:
            raise _core_error(ChainError, "segment terminal seq mismatch")
        if any(
            event.run_id != run_id or event.epoch != epoch or event.seq != events[0].seq + i
            for i, event in enumerate(events)
        ):
            raise _core_error(ChainError, "segment run, epoch or consecutive seq mismatch")
        cached = self._conn.execute(
            "SELECT first_seq, last_seq FROM segments WHERE run_id=? AND name=?", (run_id, name)
        ).fetchone()
        if cached is not None:
            if cached != (events[0].seq, last):
                raise _core_error(ChainError, "cached segment seq mismatch")
            return True
        newest = self._conn.execute(
            "SELECT epoch, last_seq FROM segments WHERE run_id=? ORDER BY last_seq DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if events[0].seq != (newest[1] + 1 if newest else 1):
            raise _core_error(ChainError, "seq gap at cached chain junction")
        if newest and epoch < newest[0]:
            raise _core_error(ChainError, "segment epoch moves backwards")
        return False

    def _insert(
        self, table: str, values: dict[str, Any], kind: str, pending: list[str]
    ) -> str | None:
        """Retain each producer and defer identifier-only warnings until commit."""
        producer = (
            f"object:{values['obj_name']}"
            if values.get("obj_name") is not None
            else f"run:{values['run_id']}"
        )
        if self._conn.execute(
            f"SELECT 1 FROM {table} WHERE id=? AND producer=?", (values["id"], producer)
        ).fetchone():
            return None
        if self._conn.execute(f"SELECT 1 FROM {table} WHERE id=?", (values["id"],)).fetchone():
            pending.append(
                f"{kind} {values['id']} has multiple producers; holder selected at each frontier"
            )
        values["producer"] = producer
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        self._conn.execute(
            f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        return producer

    def _index(self, event: Event, state: RunState, pending: list[str]) -> None:
        """Project indexable models; all lifecycle interpretation stays in replay."""
        try:
            self._index_event(event, state, pending)
        except (ValueError, TypeError, KeyError, sqlite3.IntegrityError):
            raise _core_error(ChainError, "invalid indexed journal payload") from None

    def _index_event(self, event: Event, state: RunState, pending: list[str]) -> None:
        ts = event.ts.astimezone(UTC).isoformat(timespec="microseconds")
        run_id, goal_id = event.run_id, event.goal_id
        if event.type == EventType.PLAN_REVISION:
            plan = state.plans[goal_id][-1][0]
            if plan.goal.id != goal_id:
                raise _core_error(ChainError, "plan goal does not match event routing")
            if plan.revision == 0:
                self._conn.execute(
                    "INSERT OR IGNORE INTO goals VALUES (?, ?, ?, ?)",
                    (run_id, goal_id, plan.goal.namespace, goal_key_digest(plan.goal.key())),
                )
            elif (
                self._conn.execute(
                    "SELECT 1 FROM goals WHERE run_id=? AND goal_id=?", (run_id, goal_id)
                ).fetchone()
                is None
            ):
                raise _core_error(ChainError, "plan revision has no initial plan")
            first_round = state.plan_first_rounds[goal_id][-1]
            self._conn.execute(
                "INSERT INTO plans VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    goal_id,
                    plan.revision,
                    event.payload["origin"],
                    event.seq,
                    ts,
                    plan.model_dump_json(),
                    first_round,
                ),
            )
            for part in plan.parts:
                self._conn.execute(
                    "INSERT INTO part_goals VALUES (?, ?, ?, ?, ?)",
                    (run_id, goal_id, plan.revision, part.name, goal_key_digest(part.goal.key())),
                )
            return
        if event.type not in (EventType.TOOL_END, EventType.SCORE, EventType.LEARNING):
            return
        goal = self._conn.execute(
            "SELECT namespace, goal_key FROM goals WHERE run_id=? AND goal_id=?", (run_id, goal_id)
        ).fetchone()
        if goal is None:
            raise _core_error(ChainError, "indexed event has no committed plan")
        position = dict(run_id=run_id, epoch=event.epoch, seq=event.seq, ts=ts)
        if event.type == EventType.TOOL_END:
            candidate = Candidate.model_validate(event.payload["candidate"])
            if (candidate.run_id, candidate.goal_id) != (run_id, goal_id):
                raise _core_error(ChainError, "candidate does not match event routing")
            producing_key = state.candidate_goal_keys[candidate.id]
            if producing_key.namespace != self.namespace:
                return
            repeat_key = event.payload.get("repeat_key")
            if repeat_key is not None and not isinstance(repeat_key, str):
                raise _core_error(ChainError, "invalid repeat_key")
            self._insert(
                "candidates",
                dict(
                    id=candidate.id,
                    **position,
                    goal_id=goal_id,
                    goal_key=goal[1],
                    part_key=goal[1]
                    if candidate.part is None
                    else goal_key_digest(state.candidate_goal_keys[candidate.id]),
                    part=candidate.part,
                    team=candidate.team,
                    agent=candidate.agent,
                    round=candidate.round,
                    fidelity=candidate.fidelity,
                    partial=candidate.output.partial,
                    gated_out=candidate.gated_out,
                    repeat_key=repeat_key,
                    json=candidate.model_dump_json(),
                ),
                "candidate",
                pending,
            )
        elif event.type == EventType.SCORE:
            score = Score.model_validate(event.payload["score"])
            producing_key = state.candidate_goal_keys.get(score.candidate_id)
            namespace = goal[0] if producing_key is None else producing_key.namespace
            if namespace != self.namespace:
                return
            self._insert(
                "scores",
                dict(
                    id=score.id,
                    source="run",
                    **position,
                    obj_name=None,
                    line=None,
                    candidate_id=score.candidate_id,
                    judge=score.judge,
                    rubric_hash=score.rubric_hash,
                    panel_hash=score.panel_hash,
                    json=score.model_dump_json(),
                ),
                "score",
                pending,
            )
        else:
            learning = Learning.model_validate(event.payload["learning"])
            plan = state.plan_for_round(goal_id, learning.round)
            namespaces = {plan.goal.namespace, *(part.goal.namespace for part in plan.parts)}
            if (learning.run_id, learning.goal_id) != (run_id, goal_id) or (
                learning.namespace not in namespaces
            ):
                raise _core_error(ChainError, "learning does not match event routing")
            if learning.namespace != self.namespace:
                return
            if producer := self._insert(
                "learnings",
                dict(
                    id=learning.id,
                    source="run",
                    **position,
                    obj_name=None,
                    line=None,
                    goal_id=goal_id,
                    goal_key=goal[1],
                    author=learning.author,
                    verified=learning.verified,
                    from_human=learning.from_human,
                    json=learning.model_dump_json(),
                ),
                "learning",
                pending,
            ):
                self._insert_learning_children(learning, producer)
