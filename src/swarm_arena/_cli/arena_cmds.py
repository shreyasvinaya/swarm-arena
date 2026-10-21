"""Read committed arena generations and perform offline human operations."""

import argparse
import getpass
import re
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import anyio

from swarm_arena._board.arena import ArenaBoard, ArenaView, _namespace_frontier
from swarm_arena._board.base import Board
from swarm_arena._board.cache import LearningFilter
from swarm_arena._board.human_ops import _apply_human_operation
from swarm_arena._board.ingest import run_store
from swarm_arena._board.leaderboard import format_value, group_rows
from swarm_arena._board.pointers import _validate_namespace, _validate_run_id
from swarm_arena._board.reindex import _reindex_with_diagnostics
from swarm_arena._board.trace import trace
from swarm_arena._errors import _core_error
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.reader import read_committed
from swarm_arena._journal.replay import RunState
from swarm_arena._model.plan import Plan
from swarm_arena._model.run_config import RunConfig
from swarm_arena._scoring.human_review import HumanReview

_RUN_ID = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z_[0-9a-f]{6}$")


def _non_negative_integer(option: str) -> Callable[[str], int]:
    """Convert numeric options without argparse's value-bearing int diagnostics."""

    def convert(value: str) -> int:
        try:
            number = int(value)
            if number >= 0:
                return number
        except ValueError:
            pass
        raise argparse.ArgumentTypeError(f"{option} must be a non-negative integer") from None

    return convert


def register(sub: argparse._SubParsersAction) -> None:
    """
    Register offline arena commands with a root parser.

    Parameters
    ----------
    sub : argparse._SubParsersAction
        Subparser collection to extend with synchronous command handlers.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> register(parser.add_subparsers())
    >>> parser.parse_args(['leaderboard', 'toy']).target
    'toy'
    """
    review = sub.add_parser("review", help="export or import human reviews")
    actions = review.add_subparsers(dest="action", required=True)
    export = actions.add_parser("export")
    export.add_argument("run")
    export.add_argument("--goal")
    export.add_argument("--top", type=_non_negative_integer("--top"), default=20)
    export.add_argument("--out", type=Path)
    imported = actions.add_parser("import")
    imported.add_argument("run")
    imported.add_argument("csv", type=Path)
    imported.add_argument("--reviewer", required=True)
    leaderboard = sub.add_parser("leaderboard", help="rank a run or namespace")
    leaderboard.add_argument("target")
    leaderboard.add_argument("--goal")
    leaderboard.add_argument("--by")
    board = sub.add_parser("board", help="search learnings or inspect evidence")
    actions = board.add_subparsers(dest="action", required=True)
    search = actions.add_parser("search")
    search.add_argument("query")
    search.add_argument("--goal")
    search.add_argument("--tag", action="append", default=[])
    for flag in ("verified", "human", "pinned", "hidden"):
        search.add_argument(f"--{flag}", action="store_true")
    search.add_argument("--limit", type=_non_negative_integer("--limit"), default=10)
    show = actions.add_parser("show")
    show.add_argument("id")
    learnings = sub.add_parser("learnings", help="add, pin or hide a learning")
    actions = learnings.add_subparsers(dest="action", required=True)
    pin = actions.add_parser("pin")
    hide = actions.add_parser("hide")
    for command in (pin, hide):
        command.add_argument("id")
    add = actions.add_parser("add")
    add.add_argument("--title", required=True)
    add.add_argument("--text", required=True)
    add.add_argument("--goal")
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--evidence", nargs="+", action="extend", default=[])
    for command in (pin, hide, add):
        command.add_argument("--namespace", required=True)
        command.add_argument("--author", default=None)
    traced = sub.add_parser("trace", help="show a candidate's provenance")
    traced.add_argument("id")
    for command in (search, show, traced):
        scope = command.add_mutually_exclusive_group(required=True)
        scope.add_argument("--namespace")
        scope.add_argument("--run")
    reindex = sub.add_parser("reindex", help="rebuild disposable arena caches")
    reindex.add_argument("--namespace", action="append", default=[])
    for command in (export, imported, leaderboard, search, show, pin, hide, add, traced, reindex):
        command.add_argument("--config", type=Path, default=Path("run.toml"))
        command.set_defaults(handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    """Enter the asyncio boundary without changing root interrupt handling."""
    anyio.run(_execute, args, backend="asyncio")
    return 0


def _make_board(config: RunConfig) -> ArenaBoard:
    """Build only the configured board, preserving optional import isolation."""
    board = (
        config.board.build(Board)
        if config.board is not None
        else ArenaBoard(arena_root=config.arena_root)
    )
    if not isinstance(board, ArenaBoard):
        raise _core_error(ValueError, "arena commands need ArenaBoard")
    return board


def _run_plans(config: RunConfig, run_id: str) -> list[Plan]:
    """Derive namespaces from a single authority-selected chain, never listings."""
    _validate_run_id(run_id)
    committed = read_committed(run_store(config.run_root, run_id))
    if committed is None or not committed[1]:
        raise _core_error(ValueError, f"run {run_id} has no committed events")
    state = RunState.from_events(committed[1])
    return [plan for revisions in state.plans.values() for plan, _ in revisions]


def _plan_namespaces(plans: list[Plan]) -> list[str]:
    """Include cross-namespace parts of every admitted revision."""
    return sorted(
        {
            key.namespace
            for plan in plans
            for key in [plan.goal, *(part.goal for part in plan.parts)]
        }
    )


async def _execute(args: argparse.Namespace) -> None:
    """Keep every board lifetime bounded and publish before the first read."""
    config = await run_sync_daemon(RunConfig.load, args.config)
    board = await run_sync_daemon(_make_board, config)
    try:
        if args.command == "review":
            review = HumanReview(args.reviewer if args.action == "import" else getpass.getuser())
            if args.action == "export":
                out = await review.export(
                    args.run,
                    config=config,
                    out_dir=args.out or Path(f"review-{args.run}"),
                    top=args.top,
                    goal=args.goal,
                )
                print(out)
            else:
                scores = await review.import_csv(args.run, args.csv, config=config)
                print(f"imported {len(scores)} scores as {review.judge_name}")
            return
        if args.command == "reindex":
            rebuilt = await run_sync_daemon(_reindex_with_diagnostics, config, args.namespace)
            for ns, count in rebuilt.run_counts.items():
                print(f"{ns}: {count} runs")
            for warning in rebuilt.warnings:
                print(warning, file=sys.stderr)
            return
        run_id = getattr(args, "run", None)
        namespace = getattr(args, "namespace", None)
        if args.command == "leaderboard":
            run_id = args.target if _RUN_ID.fullmatch(args.target) else None
            namespace = None if run_id else args.target
        names = (
            _plan_namespaces(await run_sync_daemon(_run_plans, config, run_id))
            if run_id
            else [_validate_namespace(namespace or "")]
        )
        await run_sync_daemon(board.open, "", "", names)
        await run_sync_daemon(board.frontier)
        # A main view also needs its declared parts' namespace readers.
        with board._pin_state() as state:
            expanded = sorted(
                set(names)
                | set(
                    _plan_namespaces(
                        [plan for run in state.plans.values() for plan in run.current.values()]
                    )
                )
            )
        if expanded != names:
            await run_sync_daemon(board.open, "", "", expanded)
            await run_sync_daemon(board.frontier)
        if args.command == "leaderboard":
            _leaderboard(board, args, run_id, namespace)
        elif args.command == "learnings":
            operation = await _apply_human_operation(
                board,
                args.namespace,
                args.action,
                author=args.author or getpass.getuser(),
                learning_id=getattr(args, "id", None),
                title=getattr(args, "title", ""),
                text=getattr(args, "text", ""),
                goal_id=getattr(args, "goal", None) or "",
                tags=getattr(args, "tag", ()),
                evidence=getattr(args, "evidence", ()),
            )
            if operation.learning is not None:
                print(operation.learning.id)
        elif args.command == "trace":
            print("\n".join(trace(board, args.id)))
        elif args.action == "search":
            _search(board, args, names)
        else:
            _show(board, args.id)
        for warning in await run_sync_daemon(_warnings, board):
            print(warning, file=sys.stderr)
    finally:
        with anyio.move_on_after(config.settle_timeout_s, shield=True):
            await run_sync_daemon(board.close)


def _warnings(board: ArenaBoard) -> list[str]:
    """Return writer-cache diagnostics without rebuilding a generation."""
    return [warning for cache in board._caches.values() for warning in cache.warnings]


def _leaderboard(
    board: ArenaBoard, args: argparse.Namespace, run_id: str | None, namespace: str | None
) -> None:
    """Rank current producing identities using each owning group's recorded judge."""
    with board._pin_state() as state:
        newest: dict[tuple[str, str], tuple[str, Plan]] = {}
        for identifier, run in sorted(state.plans.items()):
            if run_id is not None and identifier != run_id:
                continue
            for goal_id, plan in run.current.items():
                if namespace is not None and namespace not in _plan_namespaces([plan]):
                    continue
                if args.goal is None or goal_id == args.goal:
                    newest[(plan.goal.namespace, goal_id)] = (identifier, plan)
        any_rows = False
        judges: set[str] = set()
        for identifier, plan in newest.values():
            view = ArenaView(
                board, plan.goal.key(), state.frontier, published=replace(state, run_id=identifier)
            )
            names = view._ranking_judges()
            if args.by:
                names = dict.fromkeys(view._active_goals, args.by)
            if namespace is not None:
                names = {
                    part: judge
                    for part, judge in names.items()
                    if view._active_goals[part].namespace == namespace
                }
            judges.update(names.values())
            pairs = view._ranking_pairs(view._pairs())
            groups = group_rows(
                pairs,
                view._scores(tuple(view._visible_candidate_ids())),
                judge=names,
                run_id=run_id,
                identity_run=run_id,
                min_improvement=plan.converge.min_improvement if run_id else None,
            )
            for part, rows in groups.items():
                if not rows:
                    continue
                any_rows = True
                row = rows[0]
                suffix = f"/{part}" if part is not None else ""
                group_ns = view._active_goals[part].namespace
                print(
                    f"== {group_ns}/{plan.goal.id}{suffix} ({row.judge}, "
                    f"rubric {row.rubric_hash[:8]}, panel {row.panel_hash[:8] or '-'}) =="
                )
                for row in rows:
                    print(
                        f"{row.rank}\t{row.candidate_id}\t{row.run_id}\t{row.team}/{row.agent}"
                        f"\t{format_value(row.value)}\t{','.join(row.built_on) or '-'}"
                    )
        if not any_rows:
            for judge in sorted(judges or {args.by or "ranking judge"}):
                print(f"no scores for {judge}")


def _search(board: ArenaBoard, args: argparse.Namespace, namespaces: list[str]) -> None:
    """Apply search predicates and limit in the cache before hydrating rows."""
    remaining = args.limit
    if remaining < 0:
        raise _core_error(ValueError, "limit must be nonnegative")
    with board._pin_state() as state:
        for namespace in namespaces:
            cache = state.readers.get(namespace)
            if cache is None or remaining == 0:
                continue
            frontier = _namespace_frontier(state.frontier, namespace)
            ids = cache.find(
                frontier,
                query=args.query or None,
                filters=LearningFilter(
                    goal_id=args.goal,
                    tags=args.tag,
                    verified=args.verified or None,
                    human=args.human or None,
                    pinned=args.pinned or None,
                    hidden=args.hidden,
                ),
                relevance_goal=None,
                limit=remaining,
            )
            found = {learning.id: learning for learning in cache.learnings_by_id(frontier, ids)}
            for identifier in ids:
                item = found[identifier]
                markers = [
                    name
                    for name, enabled in (
                        ("verified", item.verified),
                        ("human", item.from_human),
                        ("pinned", item.pinned),
                        ("hidden", item.hidden),
                    )
                    if enabled
                ]
                with cache._lock, cache._frontier(frontier):
                    source, params = cache._learning_sql(LearningFilter(hidden=False))
                    superseded = cache._conn.execute(
                        cache._flags_sql()
                        + "SELECT 1 FROM "
                        + source
                        + " AND EXISTS (SELECT 1 FROM learning_links link "
                        "WHERE link.learning_id=l.id AND link.kind='supersedes' "
                        "AND link.target_id=?) LIMIT 1",
                        [*params, identifier],
                    ).fetchone()
                if superseded:
                    markers.append("superseded")
                print(f"{item.id}\t{item.title}\t{item.author}\t{','.join(markers)}")
            remaining -= len(ids)


def _show(board: ArenaBoard, identifier: str) -> None:
    """Render candidate metadata or complete learning evidence with score errors."""
    if identifier.startswith("c-"):
        candidate = board.get_candidate(identifier)
        print(f"request: {candidate.request.model_dump_json()}")
        print(f"status: {candidate.output.status.value}")
        print(f"checks: {[check.model_dump(mode='json') for check in candidate.checks]}")
        print(f"scores: {[score.model_dump(mode='json') for score in board.scores(identifier)]}")
        print(f"parents: {','.join(candidate.parent_ids) or '-'}")
        paths = sorted(
            set(candidate.request.edits) | {item.path for item in candidate.output.files}
        )
        print(f"files: {','.join(paths) or '-'}")
    else:
        item = board.get_learning(identifier)
        print(f"{item.id}: {item.title}\n{item.text}\nevidence:")
        for candidate_id in item.evidence:
            try:
                scores = board.scores(candidate_id)
                values = ", ".join(f"{score.judge}={format_value(score.value)}" for score in scores)
                print(f"{candidate_id}: {values or '-'}")
            except (KeyError, ValueError):
                print(f"{candidate_id}: error: unknown candidate")
