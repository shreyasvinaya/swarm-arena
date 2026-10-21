"""Export committed candidates and publish recoverable human review receipts."""

import csv
import html
import io
import math
import os
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Final, TypeVar
from urllib.parse import quote, urlsplit

import anyio

from swarm_arena._board.human_objects import (
    HumanObject,
    _validate_reviewer,
    human_object_key,
    human_object_name,
    write_human_objects,
)
from swarm_arena._board.ingest import run_store
from swarm_arena._board.leaderboard import flatten, format_value, group_judges, group_rows
from swarm_arena._board.pages import state_positions
from swarm_arena._board.pointers import _validate_run_id
from swarm_arena._engine.preflight import _board_arena_root, check_storage
from swarm_arena._errors import (
    AlreadyExists,
    Fenced,
    LeaseHeld,
    StorageError,
    _core_error,
)
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._journal.lease import Lease, OwnerDoc, read_owner
from swarm_arena._journal.reader import read_chain
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import canonical_json, utc_now
from swarm_arena._model.ids import IdGen
from swarm_arena._model.results import Score
from swarm_arena._model.run_config import RunConfig
from swarm_arena._outcomes import _Outcome
from swarm_arena._store.base import Store, _check_key, _read_blob, open_store

CSV_HEADER: Final = ("id", "goal", "team", "agent", "scores", "verdict", "score", "notes")


def render_review(state: RunState, *, top: int = 20, goal: str | None = None) -> dict[str, bytes]:
    """
    Render deterministic CSV, candidate JSON and an escaped static HTML table.

    Parameters
    ----------
    state : RunState
        Committed single-run history, including producing goal identities.
    top : int, default=20
        Maximum rows per main goal, after flattening independently ranked parts.
    goal : str or None, default=None
        Restrict the export to one main goal identifier.

    Returns
    -------
    dict[str, bytes]
        Relative export filenames and their UTF-8 contents.

    Raises
    ------
    ValueError
        If the goal is unknown or the row limit is negative.

    Examples
    --------
    >>> sorted(render_review(RunState()))
    ['index.html', 'review.csv']
    """
    if type(top) is not int or top < 0:
        raise _core_error(ValueError, "top must be a nonnegative integer")
    if goal is not None and goal not in state.plans:
        raise _core_error(ValueError, f"unknown goal {goal}")
    files: dict[str, bytes] = {}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(CSV_HEADER)
    table = [
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Human review</title></head><body><table>",
        "<thead><tr>"
        + "".join(f"<th>{html.escape(c)}</th>" for c in (*CSV_HEADER, "judge notes"))
        + "</tr></thead><tbody>",
    ]
    for goal_id in sorted(state.plans):
        if goal is not None and goal_id != goal:
            continue
        plan = state.plans[goal_id][-1][0]
        part_keys = {p.name: p.goal.key() for p in plan.parts}
        candidates = [
            c
            for c in state.candidates.values()
            if c.goal_id == goal_id
            and state.candidate_goal_keys.get(c.id)
            == (plan.goal.key() if c.part is None else part_keys.get(c.part))
        ]
        ids = {c.id for c in candidates}
        scores = [s for s in state.scores.values() if s.candidate_id in ids]
        rows = flatten(
            group_rows(
                state_positions(state, candidates),
                state_positions(state, scores),
                judge=group_judges(plan, candidates=candidates, scores=scores),
                run_id=state.run_id,
                identity_run=state.run_id,
                min_improvement=plan.converge.min_improvement,
            )
        )[:top]
        for row in rows:
            cid = _validate_run_id(row.candidate_id)
            candidate = state.candidates[cid]
            attached = sorted(
                (s for s in scores if s.candidate_id == cid),
                key=lambda s: (state.positions[s.id][1], s.id),
            )
            cells = [
                cid,
                goal_id,
                candidate.team,
                candidate.agent,
                "; ".join(f"{s.judge}={format_value(s.value)}" for s in attached),
                "",
                "",
                "",
            ]
            writer.writerow(
                "'" + cell if cell.startswith(("=", "+", "-", "@", "\t", "\r")) else cell
                for cell in cells
            )
            key = f"candidates/{cid}.json"
            files[key] = canonical_json(
                {
                    "candidate": candidate.model_dump(mode="json"),
                    "scores": [s.model_dump(mode="json") for s in attached],
                }
            )
            escaped = [
                html.escape(cell)
                for cell in [*cells, "; ".join(s.notes for s in attached if s.notes)]
            ]
            link = html.escape("candidates/" + quote(cid, safe="") + ".json")
            escaped[0] = f'<a href="{link}">{escaped[0]}</a>'
            table.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in escaped) + "</tr>")
    table.append("</tbody></table></body></html>\n")
    files["review.csv"] = stream.getvalue().encode()
    files["index.html"] = "\n".join(table).encode()
    return files


def parse_review_csv(data: bytes, state: RunState, *, judge: str) -> list[Score]:
    r"""
    Validate all CSV rows without allocating durable IDs or touching storage.

    Parameters
    ----------
    data : bytes
        UTF-8 CSV, optionally with a BOM and spreadsheet CRLF line endings.
    state : RunState
        Committed candidates against which IDs are checked.
    judge : str
        Human judge name assigned to provisional scores.

    Returns
    -------
    list[Score]
        Nonblank reviews in CSV order, with empty IDs and batch IDs.

    Raises
    ------
    ValueError
        All invalid row numbers and value-free reasons in one exception.

    Examples
    --------
    >>> parse_review_csv((",".join(CSV_HEADER) + "\n").encode(), RunState(), judge="human:a")
    []
    """
    if not isinstance(judge, str) or not judge.startswith("human:"):
        raise _core_error(ValueError, "judge must identify a human reviewer")
    _validate_reviewer(judge.removeprefix("human:"))
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError:
        raise _core_error(ValueError, "row 1: invalid UTF-8") from None
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    errors: list[str] = []
    scores: list[Score] = []
    seen: set[str] = set()
    number = 1
    try:
        header = reader.fieldnames or []
        for column in CSV_HEADER:
            if column not in header:
                errors.append(f"row 1: missing column {column}")
        if len(set(header)) != len(header):
            errors.append("row 1: duplicate column")
        for number, row in enumerate(reader, 2):
            start = len(errors)
            if any(row.get(c) is None for c in CSV_HEADER):
                errors.append(f"row {number}: missing column")
            if None in row:
                errors.append(f"row {number}: extra column")
            cid = row.get("id") or ""
            if cid not in state.candidates:
                errors.append(f"row {number}: unknown candidate id")
            if cid in seen:
                errors.append(f"row {number}: duplicate id")
            seen.add(cid)
            verdict = (row.get("verdict") or "").strip().lower()
            raw = (row.get("score") or "").strip()
            if verdict not in {"accept", "reject", ""}:
                errors.append(f"row {number}: bad verdict")
            value = None
            if raw:
                try:
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError
                except ValueError:
                    errors.append(f"row {number}: bad score")
            if len(errors) != start or not (verdict or raw):
                continue
            scores.append(
                Score(
                    candidate_id=cid,
                    judge=judge,
                    key=({"accept": 1.0, "": 0.0, "reject": -1.0}[verdict], value or 0.0),
                    value=value,
                    verdict=verdict or None,
                    notes=row.get("notes") or "",
                )
            )
    except csv.Error:
        errors.append(f"row {number + 1}: malformed CSV")
    if errors:
        raise _core_error(ValueError, "\n".join(errors)) from None
    return scores


def _read_state(store: Store, run_id: str, doc: OwnerDoc | None = None) -> RunState:
    """Read one authority head, with no listing or mutation."""
    if doc is None:
        owner = read_owner(store)
        if owner is None:
            raise FileNotFoundError(f"run {run_id} has no control/owner.json")
        doc = owner[0]
    if doc.head is None:
        raise _core_error(ValueError, f"run {run_id} has no committed events")
    return RunState.from_events(e for _, batch in read_chain(store, doc.head) for e in batch)


def _local_path(path: Path) -> Path:
    """Reject nonlocal or malformed paths before side effects."""
    if not isinstance(path, Path) or "\x00" in str(path):
        raise _core_error(ValueError, "invalid local review path")
    return path


def _read_csv(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise _core_error(ValueError, f"cannot read review CSV: {type(error).__name__}") from None


def _write_export(out_dir: Path, files: dict[str, bytes]) -> None:
    try:
        for name, data in files.items():
            path = out_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    except OSError as error:
        raise _core_error(
            StorageError, f"cannot write review export: {type(error).__name__}"
        ) from None


def _review_keys(
    state: RunState, reviewer: str, arena_root: str, now: datetime, scores: Sequence[Score]
) -> dict[str, str]:
    """Validate all derived locations without probing or mutating storage."""
    namespaces = {state.candidate_goal_keys[score.candidate_id].namespace for score in scores}
    keys = {
        ns: human_object_key(ns, "reviews", human_object_name(now), reviewer=reviewer)
        for ns in sorted(namespaces)
    }
    try:
        for key in keys.values():
            _check_key(key)
        if arena_root.startswith("s3://"):
            prefix = urlsplit(arena_root).path.lstrip("/").rstrip("/")
            if any(
                len((prefix + "/" + key if prefix else key).encode()) > 1024
                for key in keys.values()
            ):
                raise ValueError
        else:
            root = Path(arena_root).expanduser().resolve()
            ancestor = root
            while not ancestor.exists():
                ancestor = ancestor.parent
            name_max = os.pathconf(ancestor, "PC_NAME_MAX")
            path_max = os.pathconf(ancestor, "PC_PATH_MAX")
            for key in keys.values():
                # LocalStore creates a sibling with a .tmp-<12 hex> suffix.
                temporary = Path(str(root / key) + ".tmp-" + "0" * 12)
                if (path_max > 0 and len(os.fsencode(temporary)) >= path_max) or (
                    name_max > 0 and any(len(os.fsencode(p)) > name_max for p in temporary.parts)
                ):
                    raise ValueError
    except (OSError, ValueError):
        raise _core_error(ValueError, "review object location exceeds storage limits") from None
    return keys


_T = TypeVar("_T")


async def _review_io(operation: Callable[..., _T], *args: Any) -> _T:
    """Await abandonable review I/O with reconstructed storage diagnostics."""
    outcome = _Outcome()
    try:
        return await run_sync_daemon(operation, *args)
    except BaseException as error:
        outcome.contain(error, StorageError, "review storage")


async def _publish_review_object(
    arena: Store,
    store: Store,
    lease: Lease,
    key: str,
    objects: list[HumanObject],
    *,
    reviewer: str,
    now: datetime,
) -> None:
    """Check current authority immediately before one exact create-or-compare."""
    owner = await _review_io(read_owner, store)
    if (
        owner is None
        or owner[0].epoch != lease.doc.epoch
        or owner[0].owner_id != lease.doc.owner_id
        or owner[0].expires_at <= utc_now()
    ):
        raise _core_error(Fenced, "review publication lost ownership")
    parts = key.split("/")
    try:
        await anyio.lowlevel.checkpoint_if_cancelled()
        await _review_io(
            partial(
                write_human_objects,
                arena,
                parts[0],
                "reviews",
                objects,
                now=now,
                reviewer=reviewer,
                name=parts[-1],
            )
        )
    except AlreadyExists:
        expected = b"".join(canonical_json(obj.model_dump(mode="json")) + b"\n" for obj in objects)
        existing = await _review_io(_read_blob, arena, key)
        if existing.data != expected:
            raise _core_error(
                StorageError, "review object bytes differ from committed receipt"
            ) from None


def _receipt_objects(
    payload: dict[str, Any], *, state: RunState | None = None
) -> tuple[str, dict[str, list[HumanObject]]]:
    """Validate a whole receipt before admission or the first repair write."""
    try:
        reviewer = _validate_reviewer(payload["reviewer"])
        raw_scores, keys = payload["scores"], payload["objects"]
        if not isinstance(raw_scores, list) or not isinstance(keys, list):
            raise ValueError
        if any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
            raise ValueError
        objects: dict[str, list[HumanObject]] = {}
        for key in keys:
            parts = key.split("/")
            if len(parts) != 4 or key != human_object_key(
                parts[0], "reviews", parts[-1], reviewer=reviewer
            ):
                raise ValueError
            objects[key] = []
        scores = [Score.model_validate(s) for s in raw_scores]
        if len({s.id for s in scores}) != len(scores) or len(
            {s.candidate_id for s in scores}
        ) != len(scores):
            raise ValueError
        for score in scores:
            if (
                not score.id
                or not score.candidate_id
                or score.batch_id not in objects
                or score.judge != f"human:{reviewer}"
                or score.rubric_hash
                or score.panel_hash
                or score.llm_call_ids
                or score.verdict not in {"accept", "reject", None}
                or score.key
                != ({"accept": 1.0, None: 0.0, "reject": -1.0}[score.verdict], score.value or 0.0)
            ):
                raise ValueError
            if state is not None and (
                score.candidate_id not in state.candidates
                or state.candidate_goal_keys[score.candidate_id].namespace
                != score.batch_id.split("/")[0]
            ):
                raise ValueError
            objects[score.batch_id].append(
                HumanObject(kind="review", score=score, author=f"human:{reviewer}")
            )
        # Exercise the exact canonical representation before journal admission.
        for batch in objects.values():
            for obj in batch:
                canonical_json(obj.model_dump(mode="json"))
    except (ValueError, TypeError, KeyError):
        raise _core_error(StorageError, "invalid review receipt") from None
    return reviewer, objects


async def _repair_review_objects(
    state: RunState, arena: Store, store: Store, lease: Lease, writer: JournalWriter
) -> None:
    """Recreate exact committed receipts while the existing writer retains authority."""
    for event in state._events:
        if event.type != EventType.REVIEW_IMPORTED:
            continue
        reviewer, batches = _receipt_objects(event.payload)
        for key, objects in batches.items():
            if writer.fenced.is_set():
                raise _core_error(Fenced)
            await _publish_review_object(
                arena, store, lease, key, objects, reviewer=reviewer, now=event.ts
            )
            if writer.fenced.is_set():
                raise _core_error(Fenced)


class HumanReview:
    """
    Export committed candidates and import scores under one reviewer's identity.

    Parameters
    ----------
    reviewer : str
        1-64 character lowercase slug matching ``[a-z0-9][a-z0-9._-]*``.

    Attributes
    ----------
    judge_name : str
        Ranking judge identifier ``human:<reviewer>``.

    Examples
    --------
    >>> HumanReview("alice").judge_name
    'human:alice'
    """

    def __init__(self, reviewer: str) -> None:
        self._reviewer = _validate_reviewer(reviewer)

    @property
    def judge_name(self) -> str:
        """
        Return the read-only judge identity derived from the reviewer.

        Returns
        -------
        str
            Human judge identifier for this reviewer.

        Examples
        --------
        >>> HumanReview("alice").judge_name
        'human:alice'
        """
        return f"human:{self._reviewer}"

    @staticmethod
    async def export(
        run_id: str,
        *,
        config: RunConfig,
        out_dir: Path,
        top: int = 20,
        goal: str | None = None,
    ) -> Path:
        """
        Write a local review bundle without requiring a reviewer identity.

        Parameters
        ----------
        run_id : str
            Run identifier under the configured run root.
        config : RunConfig
            Storage roots for the run.
        out_dir : Path
            Local directory receiving the CSV, HTML and candidate JSON files.
        top : int, default=20
            Maximum rows per main goal.
        goal : str or None, default=None
            Optional main goal identifier filter.

        Returns
        -------
        Path
            The supplied output directory.

        Examples
        --------
        >>> out = await HumanReview.export(  # doctest: +SKIP
        ...     "r", config=config, out_dir=Path("review"))
        """
        _validate_run_id(run_id)
        _local_path(out_dir)
        store = await anyio.to_thread.run_sync(run_store, config.run_root, run_id)
        state = await anyio.to_thread.run_sync(_read_state, store, run_id)
        files = render_review(state, top=top, goal=goal)
        await anyio.to_thread.run_sync(_write_export, out_dir, files)
        return out_dir

    async def import_csv(self, run_id: str, csv_path: Path, *, config: RunConfig) -> list[Score]:
        """
        Validate the CSV, take the ended run's lease and publish recoverable scores.

        Parameters
        ----------
        run_id : str
            Existing run identifier.
        csv_path : Path
            Local UTF-8 review CSV; BOM and CRLF are accepted.
        config : RunConfig
            Storage roots and lease/writer settings. S3 conditional writes are
            probed after pure validation and before takeover.

        Returns
        -------
        list[Score]
            Imported scores with final epoch IDs and arena object batch IDs.

        Raises
        ------
        ValueError
            If CSV rows are invalid or the run has no committed events.
        RuntimeError
            If the run still has a live owner.
        Fenced
            If another owner wins takeover or publication loses authority.

        Examples
        --------
        >>> review = HumanReview("alice")
        >>> scores = await review.import_csv(  # doctest: +SKIP
        ...     "r", Path("review.csv"), config=config)
        """
        from swarm_arena._engine.resume import _take_over
        from swarm_arena._engine.run import _supervise_writer

        config = config.model_copy(
            update={"arena_root": await _review_io(_board_arena_root, config)}
        )
        _validate_run_id(run_id)
        data = await anyio.to_thread.run_sync(_read_csv, _local_path(csv_path))
        store = await anyio.to_thread.run_sync(run_store, config.run_root, run_id)
        state = await anyio.to_thread.run_sync(_read_state, store, run_id)
        initial_scores = parse_review_csv(data, state, judge=self.judge_name)
        await run_sync_daemon(
            _review_keys, state, self._reviewer, config.arena_root, utc_now(), initial_scores
        )
        await anyio.to_thread.run_sync(check_storage, config)

        def validate_locations(doc: OwnerDoc) -> None:
            current = _read_state(store, run_id, doc)
            _review_keys(current, self._reviewer, config.arena_root, utc_now(), initial_scores)

        try:
            lease = await _take_over(
                config, run_id, store, force=False, review=True, validate=validate_locations
            )
        except LeaseHeld:
            raise _core_error(
                RuntimeError, f"run {run_id} has a live owner; run review import after the run ends"
            ) from None
        entered = False
        try:
            head = lease.doc.head
            assert head is not None
            path = (
                Path(gettempdir())
                / f"swarm-arena-{os.getuid()}"
                / "journals"
                / f"{run_id}-e{lease.doc.epoch}.jsonl"
            )
            async with JournalWriter(
                store,
                lease,
                run_id=run_id,
                next_seq=head.seq + 1,
                local_path=path,
                sync_interval_s=config.sync_interval_s,
            ) as writer:
                entered = True
                # The journal owns release even if supervision entry is cancelled.
                async with _supervise_writer(writer):
                    state = await run_sync_daemon(_read_state, store, run_id, lease.doc)
                    scores = parse_review_csv(data, state, judge=self.judge_name)
                    arena = await run_sync_daemon(open_store, config.arena_root)
                    await _repair_review_objects(state, arena, store, lease, writer)
                    now = utc_now()
                    keys = await run_sync_daemon(
                        _review_keys, state, self._reviewer, config.arena_root, now, scores
                    )
                    ids = IdGen(run_id, lease.doc.epoch, counters=state.id_counters)
                    scores = [
                        s.model_copy(
                            update={
                                "id": ids.next("s"),
                                "batch_id": keys[
                                    state.candidate_goal_keys[s.candidate_id].namespace
                                ],
                            }
                        )
                        for s in scores
                    ]
                    payload = {
                        "reviewer": self._reviewer,
                        "scores": [s.model_dump(mode="json") for s in scores],
                        "objects": list(keys.values()),
                    }
                    _receipt_objects(payload, state=state)
                    proposed = EventDraft(type=EventType.REVIEW_IMPORTED, payload=payload)
                    writer.append(EventDraft(type=EventType.TAKEOVER, payload={"force": False}))
                    receipt = writer.append(proposed)
                    await writer.commit()
                    await _repair_review_objects(
                        RunState.from_events([receipt]), arena, store, lease, writer
                    )
            return scores
        finally:
            if not entered:
                with anyio.move_on_after(config.settle_timeout_s, shield=True):
                    with suppress(Fenced):
                        await run_sync_daemon(lease.release)
