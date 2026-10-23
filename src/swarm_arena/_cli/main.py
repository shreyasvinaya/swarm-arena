"""Entry point of the ``swarm-arena`` console script."""

import argparse
import sys
from collections.abc import Sequence

from swarm_arena._cli import arena_cmds, cluster_cmds, plan_cmd
from swarm_arena._cli.run_cmds import register
from swarm_arena._errors import _exception_type_name, _rebuild_core_signal
from swarm_arena._version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm-arena",
        description="Run a swarm of agents on hard goals.",
        epilog="Exit codes: 0 success, 1 error, 2 usage, 130 interrupted (Ctrl-C).",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")
    register(sub)
    arena_cmds.register(sub)
    plan_cmd.register(sub)
    cluster_cmds.register(sub)
    return parser


def _error_message(error: Exception) -> str:
    """Retain sanitized core diagnostics, hiding unforeseen exception values."""
    rebuilt = _rebuild_core_signal(error)
    return str(rebuilt) if rebuilt is not None else _exception_type_name(error)


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
        error, two when run needs a terminal or explicit ``--yes``, or 130 after
        Ctrl-C cancellation cleanup.

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
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(_error_message(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
