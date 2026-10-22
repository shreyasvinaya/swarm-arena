"""Validate manual plans or save admitted, journal-linked LLM proposals."""

import argparse
import os
import sys
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import anyio
from pydantic import ValidationError

from swarm_arena._budget.amount import Amount
from swarm_arena._cli.run_cmds import _interruptible
from swarm_arena._engine.run import _estimates, mark_planning_session, planning_session
from swarm_arena._errors import ConfigError, _core_error
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.diagnostics import validation_summary
from swarm_arena._model.goal import Goal
from swarm_arena._model.plan import Plan
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.toml_io import read_toml
from swarm_arena._outcomes import _Outcome
from swarm_arena._planning.validate import admit_revision
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store


def register(sub: argparse._SubParsersAction) -> None:
    """Register manual, LLM and collaborative planning modes.

    Parameters
    ----------
    sub : argparse._SubParsersAction
        Root parser's command collection.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> register(parser.add_subparsers())
    >>> parser.parse_args(["plan", "run.toml", "plan.toml"]).llm
    False
    """
    parser = sub.add_parser("plan", help="validate a plan or generate a journal-linked proposal")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--llm", action="store_true", help="generate a plan using the catalog")
    modes.add_argument(
        "--together", action="store_true", help="generate and review in the terminal"
    )
    parser.add_argument(
        "--model", metavar="ALIAS", help="planner model alias (required for generation)"
    )
    parser.add_argument("--instructions", metavar="TEXT", default=None)
    parser.add_argument("-o", "--output", metavar="OUT", type=Path)
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    parser.add_argument("run_toml", metavar="RUN_TOML", type=Path)
    parser.add_argument("source", metavar="PLAN_OR_GOAL_TOML", type=Path)
    parser.set_defaults(handler=_plan)


def _load_goal(path: Path) -> Goal:
    """Use the core TOML and validation diagnostics for one human goal table."""
    data = read_toml(path)
    if set(data) != {"goal"} or not isinstance(data["goal"], dict):
        raise _core_error(ConfigError, "goal file must contain only a [goal] table")
    try:
        return Goal.model_validate(data["goal"])
    except ValidationError as error:
        raise _core_error(ConfigError, validation_summary(error, root="goal")) from None


def _manual(config: RunConfig, source: Path) -> tuple[Amount, list[str]]:
    """Admit before constructing estimate variants, using the run's estimator."""
    plan = Plan.load(source)
    if str(plan.base_dir) not in sys.path:
        sys.path.append(str(plan.base_dir))
    frozen, warnings = admit_revision(plan, config, ArtifactStore(open_store(config.artifact_root)))
    estimate, _ = _estimates([frozen], config)
    return estimate, warnings


def _write_plan(path: Path, text: str, force: bool) -> None:
    """Publish complete bytes, preserving an existing file unless explicitly replaced."""
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".swarm-plan-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        if force:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    except FileExistsError:
        raise _core_error(ValueError, "plan output already exists; use --force") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def _generate(args: argparse.Namespace, output: Path) -> None:
    """Consume the existing planner admission and session owner before publication."""
    try:
        from swarm_arena.llm import CollaborativePlanner, LLMPlanner
        from swarm_arena.llm._terminal_reviewer import TerminalReviewer
    except ImportError:
        raise _core_error(
            ValueError, "plan generation requires the llm extra: uv add 'swarm-arena[llm]'"
        ) from None
    base_dir = args.source.parent.resolve()
    if str(base_dir) not in sys.path:
        sys.path.append(str(base_dir))
    boundary = _Outcome()
    try:
        config = await run_sync_daemon(RunConfig.load, args.run_toml)
        goal = await run_sync_daemon(_load_goal, args.source)
    except BaseException as error:
        boundary.contain(error, ValueError, "plan input")
    draft = LLMPlanner(args.model, instructions=args.instructions or "")
    planner = CollaborativePlanner(draft, TerminalReviewer()) if args.together else draft
    async with planning_session(config, goal, planner) as scope:
        if isinstance(planner, CollaborativePlanner):
            result = await planner._plan_at(goal, config, None, base_dir=base_dir)
        else:
            result = await planner._draft_at(goal, config, None, base_dir=base_dir)
        result = mark_planning_session(result, scope)
    # No lease remains: output publication cannot precede failed session cleanup.
    boundary = _Outcome()
    try:
        text = await run_sync_daemon(result.dumps)
        await anyio.to_thread.run_sync(_write_plan, output, text, args.force)
    except BaseException as error:
        boundary.contain(error, ValueError, "plan output")
    print(f"plan written: {output}")
    print(f"planning session: {scope.run_id}")


async def _plan_async(args: argparse.Namespace, output: Path | None) -> None:
    """Keep input, manual hooks, and generation under the shared interrupt owner."""
    if output is not None:
        await _generate(args, output)
        return
    boundary = _Outcome()
    try:
        config = await run_sync_daemon(RunConfig.load, args.run_toml)
        estimate, warnings = await run_sync_daemon(_manual, config, args.source)
    except BaseException as error:
        boundary.contain(error, ValueError, "plan")
    tokens = "unlimited" if config.limits.max_tokens is None else str(estimate.tokens)
    usd = "unlimited" if config.limits.max_usd is None else f"${estimate.usd:.2f}"
    print(f"worst-case estimate: {tokens} tokens, {usd}")
    for warning in warnings:
        print(f"warning: {warning}")


def _plan(args: argparse.Namespace) -> int:
    """Check mode/output usage before entering any paid planning lifetime."""
    output = None
    if args.llm or args.together:
        if args.model is None:
            print("plan generation requires --model", file=sys.stderr)
            return 2
        base_dir = args.source.parent.resolve()
        output = (
            args.output or base_dir / f"plan.{'together' if args.together else 'llm'}.toml"
        ).absolute()
        if output.resolve().parent != base_dir or output.parent.resolve() != base_dir:
            print("plan output must be in the goal file's directory", file=sys.stderr)
            return 2
        if output.exists() and not args.force:
            raise _core_error(ValueError, "plan output already exists; use --force")
    elif (
        args.model is not None
        or args.instructions is not None
        or args.output is not None
        or args.force
    ):
        print(
            "--model, --instructions, --output and --force require --llm or --together",
            file=sys.stderr,
        )
        return 2
    anyio.run(_interruptible, partial(_plan_async, args, output), backend="asyncio")
    return 0
