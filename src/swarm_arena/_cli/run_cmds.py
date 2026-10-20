"""Run and control swarms from synchronous command-line handlers.

For example, ``swarm-arena run --yes run.toml plans/`` loads sorted TOML
plans. Without ``--yes``, approval requires a terminal and accepts only y/yes.
Declining approval leaves the run and artifact roots unwritten. Resume uses
``RUN_DIR/run.toml`` unless ``--config`` supplies an override, and always
selects the run root from RUN_DIR's parent.
"""

import argparse
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import anyio

from swarm_arena._engine.control import format_status, read_status, request_stop
from swarm_arena._engine.solution import SolutionDocument
from swarm_arena._engine.swarm import Swarm
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.canonical import utc_now
from swarm_arena._model.ids import new_run_id
from swarm_arena._model.plan import Plan
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.tooling import Estimate
from swarm_arena._store.base import open_store


def register(sub: argparse._SubParsersAction) -> None:
    """
    Register run, status, stop, and resume with a root argument parser.

    Parameters
    ----------
    sub : argparse._SubParsersAction
        Subparser collection to extend. Each command stores a sync handler.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> register(parser.add_subparsers())
    >>> parser.parse_args(["stop", "--now", "runs/r"]).now
    True
    """
    run = sub.add_parser("run", help="estimate, approve, and run goal plans")
    run.add_argument("--yes", action="store_true", help="approve without a terminal prompt")
    run.add_argument("run_toml", metavar="RUN_TOML", type=Path)
    run.add_argument("plans", metavar="PLAN", nargs="+", type=Path)
    run.set_defaults(handler=_run)

    status = sub.add_parser("status", help="read committed run status")
    status.add_argument("run_dir", metavar="RUN_DIR", type=Path)
    status.set_defaults(handler=_status)

    stop = sub.add_parser("stop", help="request a stop from the live owner")
    stop.add_argument("--now", action="store_true", help="cancel in-flight work")
    stop.add_argument("run_dir", metavar="RUN_DIR", type=Path)
    stop.set_defaults(handler=_stop)

    resume = sub.add_parser("resume", help="resume a saved run")
    resume.add_argument("run_dir", metavar="RUN_DIR", type=Path)
    resume.add_argument("--config", metavar="RUN_TOML", type=Path)
    resume.add_argument("--force", action="store_true", help="take over a live owner")
    resume.set_defaults(handler=_resume)


def _prompt() -> bool:
    """Accept only an explicit y/yes, with EOF treated as a refusal."""
    try:
        return input("Start the run? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _summaries(run_root: str, run_id: str, goal_ids: Sequence[str]) -> list[str]:
    """Read versioned solution documents without interpreting published log links."""
    store = open_store(run_root).child(run_id)
    lines: list[str] = []
    for goal_id in goal_ids:
        document = SolutionDocument.model_validate_json(
            store.read(f"goals/{goal_id}/solution.json").data
        )
        if document.status == "no_result":
            lines.append(f"no result: {goal_id} {document.reason}")
        elif document.status == "unjoined":
            lines.append(f"unjoined: {goal_id} {len(document.parts)} parts")
        else:
            lines.append(f"done: {goal_id} {len(document.finalists)} finalists")
    return lines


async def _print_results(run_root: str, run_id: str, goal_ids: Sequence[str]) -> None:
    """Read committed result views off-loop and print in the supplied order."""
    for line in await anyio.to_thread.run_sync(_summaries, run_root, run_id, goal_ids):
        print(line)


def _load_run(run_toml: Path, paths: Sequence[Path]) -> tuple[RunConfig, list[Plan]]:
    """Load each plan argument, expanding directories in sorted filename order."""
    config = RunConfig.load(run_toml)
    plans = [
        Plan.load(path)
        for source in paths
        for path in (sorted(source.glob("*.toml")) if source.is_dir() else [source])
    ]
    return config, plans


async def _run_async(args: argparse.Namespace) -> None:
    """Load and execute plans with an approval callback even for --yes."""
    config, plans = await anyio.to_thread.run_sync(_load_run, args.run_toml, args.plans)
    run_id = new_run_id(utc_now())

    async def approve(estimate: Estimate, warnings: Sequence[str]) -> bool:
        tokens = "unlimited" if config.limits.max_tokens is None else str(estimate.tokens)
        usd = "unlimited" if config.limits.max_usd is None else f"${estimate.usd:.2f}"
        print(f"worst-case estimate: {tokens} tokens, {usd}")
        for warning in warnings:
            print(f"warning: {warning}")
        accepted = args.yes or await run_sync_daemon(_prompt)
        if accepted:
            print(f"run {run_id} at {Path(config.run_root) / run_id}", flush=True)
        return accepted

    await Swarm(config).run(plans, run_id=run_id, approve=approve)
    await _print_results(config.run_root, run_id, [plan.goal.id for plan in plans])


def _run(args: argparse.Namespace) -> int:
    """Refuse unattended implicit approval before constructing a swarm or writing."""
    if not args.yes and not sys.stdin.isatty():
        print("run requires a terminal or --yes", file=sys.stderr)
        return 2
    anyio.run(_run_async, args, backend="asyncio")
    return 0


async def _status_async(run_dir: Path) -> None:
    """Read status from the selected directory's committed authority."""
    print(format_status(await read_status(str(run_dir.parent), run_dir.name)))


def _status(args: argparse.Namespace) -> int:
    """Dispatch status using the asyncio backend."""
    anyio.run(_status_async, args.run_dir, backend="asyncio")
    return 0


def _stop(args: argparse.Namespace) -> int:
    """Request the selected run's current epoch stop key."""
    anyio.run(
        partial(request_stop, str(args.run_dir.parent), args.run_dir.name, now=args.now),
        backend="asyncio",
    )
    return 0


async def _resume_async(args: argparse.Namespace) -> None:
    """Load saved or override settings and resume the run at the given location."""
    config = await anyio.to_thread.run_sync(
        RunConfig.load, args.config if args.config is not None else args.run_dir / "run.toml"
    )
    config = config.model_copy(update={"run_root": str(args.run_dir.parent)})
    results = await Swarm(config).resume(args.run_dir.name, force=args.force)
    await _print_results(config.run_root, args.run_dir.name, list(results))


def _resume(args: argparse.Namespace) -> int:
    """Dispatch resume using the asyncio backend."""
    anyio.run(_resume_async, args, backend="asyncio")
    return 0
