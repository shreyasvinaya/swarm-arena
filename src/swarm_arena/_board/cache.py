"""Cache committed arena records, retaining replay's producing identities."""

import fcntl
import logging
import os
import sqlite3
import time
from collections.abc import Sequence
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
    fold_curation,
    human_object_order,
)
from swarm_arena._errors import ChainError, _core_error
from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.lease import Head
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.segment import parse_segment_name
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
        "candidates(id TEXT PRIMARY KEY, run_id TEXT, epoch INTEGER, seq INTEGER, ts TEXT, "
        "goal_id TEXT, goal_key TEXT, part_key TEXT, part TEXT, team TEXT, agent TEXT, round "
        "INTEGER, fidelity TEXT, partial INTEGER, gated_out INTEGER, repeat_key TEXT, json "
        "TEXT)"
    ),
    (
        "scores(id TEXT PRIMARY KEY, source TEXT, run_id TEXT, epoch INTEGER, seq INTEGER, ts "
        "TEXT, obj_name TEXT, line INTEGER, candidate_id TEXT, judge TEXT, rubric_hash TEXT, "
        "panel_hash TEXT, json TEXT)"
    ),
    (
        "learnings(rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, source TEXT, run_id TEXT, epoch "
        "INTEGER, seq INTEGER, ts TEXT, obj_name TEXT, line INTEGER, goal_id TEXT, goal_key "
        "TEXT, author TEXT, verified INTEGER, from_human INTEGER, json TEXT)"
    ),
    "learning_tags(learning_id TEXT, tag TEXT)",
    "learning_links(learning_id TEXT, kind TEXT, target_id TEXT)",
    "learning_evidence(learning_id TEXT, candidate_id TEXT)",
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

    SCHEMA_VERSION: ClassVar[int] = 1

    def __init__(self, path: Path) -> None:
        self.namespace = path.stem
        self.warnings: list[str] = []
        self._lock = RLock()
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

    def apply_human_objects(self, names: Sequence[str], objects: Sequence[HumanObject]) -> None:
        """
        Atomically index complete human objects, retaining their original lines.

        Parameters
        ----------
        names : Sequence[str]
            One arena-relative name per operation, repeated for each object's
            lines. Supply every line of each named object in its original order.
        objects : Sequence[HumanObject]
            Corresponding validated operations. Foreign namespaces are ignored.

        Raises
        ------
        ValueError
            If lengths, reserved locations, folders, or namespaces disagree.

        Notes
        -----
        Each name is immutable. Reapplying it has no effect. The line number is
        zero-based within that name, even when names are interleaved. Completion
        markers and indexed rows commit together under the cache lock.

        Examples
        --------
        >>> cache.apply_human_objects([name], [operation])  # doctest: +SKIP
        """
        if len(names) != len(objects):
            raise _core_error(ValueError, "human object names and operations must align")
        grouped: dict[str, list[HumanObject]] = {}
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
            if self._insert(
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
                self._conn.executemany(
                    "INSERT INTO learning_tags VALUES (?, ?)",
                    [(learning.id, tag) for tag in learning.tags],
                )
                self._conn.executemany(
                    "INSERT INTO learning_evidence VALUES (?, ?)",
                    [(learning.id, cid) for cid in learning.evidence],
                )
                self._conn.executemany(
                    "INSERT INTO learning_links VALUES (?, ?, ?)",
                    [
                        (learning.id, kind, target)
                        for kind in ("refines", "supersedes")
                        for target in getattr(learning, kind)
                    ],
                )

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

    def _frontier_sql(self, frontier: Frontier, names: Sequence[str]) -> tuple[str, list[Any]]:
        """Reuse exact committed-mark validation before constructing membership SQL."""
        # Ingest depends on this cache; defer its shared validation import.
        from swarm_arena._board.ingest import _validate_marks

        marks = _validate_marks(frontier.marks)
        for mark in marks:
            if not self._covered(
                mark.run_id, Head(epoch=mark.epoch, seq=mark.seq, segment=mark.segment)
            ):
                raise _core_error(ChainError, "frontier mark is not covered by cache")
        marked = [(mark.run_id, mark.seq) for mark in marks] or [("", -1)]
        named = sorted(set(names)) or [""]
        sql = (
            "WITH marks(run_id, seq) AS (VALUES "
            + ",".join("(?,?)" for _ in marked)
            + "), objs(name) AS (VALUES "
            + ",".join("(?)" for _ in named)
            + ") "
        )
        return sql, [value for pair in marked for value in pair] + list(named)

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
        names = frontier.reviews if table == "scores" else frontier.curation
        cte, params = self._frontier_sql(frontier, names)
        marked = "EXISTS (SELECT 1 FROM marks m WHERE m.run_id=i.run_id AND i.seq<=m.seq)"
        if table == "candidates":
            included = marked
            origin, ordinal = "i.run_id", "i.seq"
        else:
            source = "review" if table == "scores" else "human"
            included = (
                f"((i.source='run' AND {marked}) OR (i.source='{source}' "
                "AND EXISTS (SELECT 1 FROM objs WHERE name=i.obj_name)))"
            )
            origin, ordinal = "coalesce(i.obj_name, i.run_id)", "coalesce(i.line, i.seq)"
        clauses = [included]
        for column, values in (("id", ids), ("candidate_id", candidate_ids)):
            if values is not None:
                clauses.append(
                    f"i.{column} IN (" + ",".join("?" for _ in values) + ")" if values else "0"
                )
                params.extend(values)
        for column, value in (filters or {}).items():
            clauses.append(f"i.{column}=?")
            params.append(value)
        rows = self._conn.execute(
            cte
            + f"SELECT i.json, i.ts, {origin}, {ordinal} FROM {table} i WHERE "
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
            Candidate IDs to include; an empty sequence selects none.
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
        with self._lock:
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
            Candidate IDs to include; an empty sequence selects none.
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
        with self._lock:
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
        with self._lock:
            cte, params = self._frontier_sql(frontier, frontier.curation)
            rows = self._conn.execute(
                cte + "SELECT obj_name, line, kind, learning_id FROM curation_ops "
                "WHERE EXISTS (SELECT 1 FROM objs WHERE name=obj_name)",
                params,
            ).fetchall()
            return fold_curation(
                (
                    name,
                    line,
                    HumanObject(
                        kind=kind,
                        learning_id=lid,
                        author="human:cache",
                    ),
                )
                for name, line, kind, lid in rows
            )

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
        with self._lock:
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
        with self._lock:
            cte, params = self._frontier_sql(frontier, [])
            rows = self._conn.execute(
                cte + "SELECT plan_json, origin, p.seq FROM plans p WHERE p.run_id=? AND goal_id=? "
                "AND EXISTS (SELECT 1 FROM marks m WHERE m.run_id=p.run_id AND p.seq<=m.seq) "
                "ORDER BY revision",
                [*params, run_id, goal_id],
            ).fetchall()
            return [(Plan.model_validate_json(data), origin, seq) for data, origin, seq in rows]

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
        with self._lock:
            members: dict[str, JsonValue] = {
                "candidates": [cid for cid in sorted(c.id for c, _ in self.candidates(frontier))],
                "scores": [sid for sid in sorted(s.id for s, _ in self.scores(frontier))],
                "learnings": [
                    [learning.id, learning.pinned, learning.hidden]
                    for learning in sorted(self._learnings(frontier), key=lambda item: item.id)
                ],
            }
            return canonical_json(members)

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

    def _insert(self, table: str, values: dict[str, Any], kind: str, pending: list[str]) -> bool:
        """Insert an ID once and defer collision warnings until transaction commit."""
        row = self._conn.execute(
            f"SELECT run_id FROM {table} WHERE id=?", (values["id"],)
        ).fetchone()
        if row is not None:
            if row[0] != values["run_id"]:
                pending.append(
                    f"{kind} {values['id']} of run {values['run_id']} skipped: "
                    f"the cache holds it from run {row[0]}"
                )
            return False
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        self._conn.execute(
            f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
        return True

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
        if goal[0] != self.namespace:
            return
        position = dict(run_id=run_id, epoch=event.epoch, seq=event.seq, ts=ts)
        if event.type == EventType.TOOL_END:
            candidate = Candidate.model_validate(event.payload["candidate"])
            if (candidate.run_id, candidate.goal_id) != (run_id, goal_id):
                raise _core_error(ChainError, "candidate does not match event routing")
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
            if (learning.run_id, learning.goal_id, learning.namespace) != (
                run_id,
                goal_id,
                goal[0],
            ):
                raise _core_error(ChainError, "learning does not match event routing")
            if self._insert(
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
                self._conn.executemany(
                    "INSERT INTO learning_tags VALUES (?, ?)",
                    [(learning.id, tag) for tag in learning.tags],
                )
                self._conn.executemany(
                    "INSERT INTO learning_evidence VALUES (?, ?)",
                    [(learning.id, cid) for cid in learning.evidence],
                )
                self._conn.executemany(
                    "INSERT INTO learning_links VALUES (?, ?, ?)",
                    [
                        (learning.id, kind, target)
                        for kind in ("refines", "supersedes")
                        for target in getattr(learning, kind)
                    ],
                )
