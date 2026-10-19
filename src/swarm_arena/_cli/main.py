"""Entry point of the ``swarm-arena`` console script."""

import argparse
from collections.abc import Sequence

from swarm_arena._version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm-arena",
        description="Run a swarm of agents on hard goals.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the ``swarm-arena`` command line.

    In this version the command only prints its version, for ``--version`` and for no
    arguments. A usage error (for example an unknown option) exits with code 2.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        The arguments without the program name. None reads ``sys.argv[1:]``.

    Returns
    -------
    int
        The exit code: 0 on success.

    Examples
    --------
    >>> from swarm_arena._cli.main import main
    >>> main(["--version"])
    swarm-arena ...
    0
    """
    _parser().parse_args(argv)
    print(f"swarm-arena {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
