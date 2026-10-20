"""Entry point of the ``swarm-arena`` console script."""

import argparse
import sys
from collections.abc import Sequence

from swarm_arena._cli.run_cmds import register
from swarm_arena._errors import ConfigError, Fenced, LeaseHeld, PlanError, StorageError
from swarm_arena._version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm-arena",
        description="Run a swarm of agents on hard goals.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    register(parser.add_subparsers(dest="command"))
    return parser


def _error_message(error: Exception) -> str:
    """Retain sanitized core diagnostics, hiding unforeseen exception values."""
    trace = error.__traceback__
    while trace is not None and trace.tb_next is not None:
        trace = trace.tb_next
    origin = trace.tb_frame.f_globals.get("__name__", "") if trace is not None else ""
    if (
        isinstance(error, (PlanError, ConfigError, StorageError, Fenced, LeaseHeld))
        and isinstance(origin, str)
        and origin.startswith("swarm_arena._")
    ):
        return str(error) or type(error).__name__
    # These core producers sanitize their plain ValueError/RuntimeError messages.
    safe_producers = {
        "swarm_arena._engine.control",
        "swarm_arena._engine.run",
        "swarm_arena._engine.resume",
        "swarm_arena._engine.preflight",
        "swarm_arena._budget.ledger",
        "swarm_arena._journal.lease",
    }
    if type(error) in (ValueError, RuntimeError) and origin in safe_producers:
        return str(error) or type(error).__name__
    return type(error).__name__


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the ``swarm-arena`` command line.

    Run goal plans, read status, request a stop, or resume a saved run. No
    arguments and ``--version`` print the version. Parser usage errors exit 2.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        The arguments without the program name. None reads ``sys.argv[1:]``.

    Returns
    -------
    int
        Zero for a returned run or successful control command, one for a runtime
        error, or two when run needs a terminal or explicit ``--yes``.

    Examples
    --------
    >>> from swarm_arena._cli.main import main
    >>> main(["--version"])
    swarm-arena ...
    0
    """
    args = _parser().parse_args(argv)
    if args.version or args.command is None:
        print(f"swarm-arena {__version__}")
        return 0
    try:
        return args.handler(args)
    except Exception as error:
        print(_error_message(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
