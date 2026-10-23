"""Submit unattended swarm drivers through the optional Ray Jobs API."""

import argparse
import re
import sys
from pathlib import Path

from swarm_arena._errors import _core_error, _exception_type_name, _rebuild_core_signal

_RAY_EXTRA_HINT = (
    "submit needs the 'ray' extra: uv add "
    '"swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena"'
)


def register(sub: argparse._SubParsersAction) -> None:
    """Register the cluster submission command without importing Ray.

    Parameters
    ----------
    sub : argparse._SubParsersAction
        Root parser's command collection.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> register(parser.add_subparsers())
    >>> parser.parse_args(["submit", "--yes", "run.toml", "plan.toml"]).yes
    True
    """
    parser = sub.add_parser("submit", help="run the driver unattended as a Ray Job")
    parser.add_argument(
        "--address",
        metavar="URL",
        help="Ray Jobs server; conflicting RAY_ADDRESS/RAY_API_SERVER_ADDRESS values are refused",
    )
    parser.add_argument("--working-dir", metavar="DIR", help="directory to upload for the job")
    parser.add_argument(
        "--pip", metavar="SPEC", action="append", default=[], help="job dependency (repeatable)"
    )
    parser.add_argument("--yes", action="store_true", help="approve unattended execution")
    parser.add_argument("run_toml", metavar="RUN_TOML")
    parser.add_argument("plans", metavar="PLAN", nargs="+")
    parser.set_defaults(handler=_submit)


def _driver_files(args: argparse.Namespace) -> list[str]:
    """Validate local packaging and rebase paths before loading Ray."""
    if any(re.search(r"://[^/?#\s]*@", spec) for spec in args.pip):
        raise _core_error(ValueError, "--pip specifications must not contain URL userinfo")
    files = [args.run_toml, *args.plans]
    if args.working_dir is None:
        return files
    try:
        directory = Path(args.working_dir).resolve(strict=True)
        if not directory.is_dir():
            raise ValueError
    except (OSError, ValueError, RuntimeError):
        raise _core_error(ValueError, "--working-dir must be an existing directory") from None
    try:
        return [str(Path(path).resolve(strict=True).relative_to(directory)) for path in files]
    except (OSError, ValueError, RuntimeError):
        raise _core_error(
            ValueError, "run and plan paths must exist inside --working-dir"
        ) from None


def _submit(args: argparse.Namespace) -> int:
    """Require approval before loading the optional runtime or contacting Ray."""
    if not args.yes:
        print(
            _core_error(
                ValueError,
                'submit starts the driver unattended: check the estimate with "swarm-arena plan", '
                "then pass --yes",
            ),
            file=sys.stderr,
        )
        return 2

    files = _driver_files(args)
    try:
        from swarm_arena.ray import _submit
    except ModuleNotFoundError as error:
        if error.name == "ray":
            raise _core_error(ValueError, _RAY_EXTRA_HINT) from None
        raise

    entrypoint = _submit.driver_entrypoint(files)
    runtime_env = _submit.submit_runtime_env(working_dir=args.working_dir, pip=args.pip)
    try:
        submission_id = _submit.submit_driver(args.address, entrypoint, runtime_env=runtime_env)
    except Exception as error:
        safe = _rebuild_core_signal(error)
        if safe is not None:
            raise safe from None
        raise _core_error(
            RuntimeError, f"Ray job submission failed: {_exception_type_name(error)}"
        ) from None
    print(submission_id)
    return 0
