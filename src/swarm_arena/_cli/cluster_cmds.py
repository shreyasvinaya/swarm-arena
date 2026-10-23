"""Submit unattended swarm drivers through the optional Ray Jobs API."""

import argparse
import sys

from swarm_arena._errors import _core_error, _exception_type_name


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
    parser.add_argument("--address", metavar="URL", help="Ray Jobs server (default: RAY_ADDRESS)")
    parser.add_argument("--working-dir", metavar="DIR", help="directory to upload for the job")
    parser.add_argument(
        "--pip", metavar="SPEC", action="append", default=[], help="job dependency (repeatable)"
    )
    parser.add_argument("--yes", action="store_true", help="approve unattended execution")
    parser.add_argument("run_toml", metavar="RUN_TOML")
    parser.add_argument("plans", metavar="PLAN", nargs="+")
    parser.set_defaults(handler=_submit)


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

    from swarm_arena.ray import _submit

    entrypoint = _submit.driver_entrypoint([args.run_toml, *args.plans])
    runtime_env = _submit.submit_runtime_env(working_dir=args.working_dir, pip=args.pip)
    try:
        submission_id = _submit.submit_driver(args.address, entrypoint, runtime_env=runtime_env)
    except Exception as error:
        raise _core_error(
            RuntimeError, f"Ray job submission failed: {_exception_type_name(error)}"
        ) from None
    print(submission_id)
    return 0
