"""Submit cluster drivers and reconcile gateway keys after a run."""

import argparse
import os
import re
import sys
from functools import partial
from pathlib import Path

import anyio

from swarm_arena._errors import ConfigError, _core_error, _exception_type_name, _rebuild_core_signal

_RAY_EXTRA_HINT = (
    "submit needs the 'ray' extra: uv add "
    '"swarm-arena[ray]"; see https://github.com/shreyasvinaya/swarm-arena#install'
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
    reconcile = sub.add_parser("reconcile", help="reconcile gateway keys after a run")
    reconcile.add_argument("run_dir", metavar="RUN_DIR", type=Path)
    reconcile.add_argument("--config", metavar="RUN_TOML", type=Path)
    reconcile.set_defaults(handler=_reconcile)


async def _reconcile_async(args: argparse.Namespace) -> None:
    """Load the addressed run's configuration and own the configured executor."""
    from swarm_arena._engine.run import _close_executor, _start_executor
    from swarm_arena._gateway.litellm_proxy import adapter_for
    from swarm_arena._gateway.reconcile import reconcile_run
    from swarm_arena._model.run_config import RunConfig

    config = await anyio.to_thread.run_sync(
        RunConfig.load, args.config if args.config is not None else args.run_dir / "run.toml"
    )
    config = config.model_copy(update={"run_root": str(args.run_dir.parent)})
    if config.llm_gateway is None:
        raise _core_error(ConfigError, "llm_gateway")
    adapter = adapter_for(config.llm_gateway, os.environ)
    executor = await _start_executor(config)
    failed, hard = True, False
    try:
        try:
            report = await reconcile_run(config, args.run_dir.name, adapter, executor)
        except FileNotFoundError:
            raise _core_error(
                ValueError, f"run {args.run_dir.name} has no control/owner.json"
            ) from None
        failed = False
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        hard = True
        raise
    except BaseExceptionGroup as error:
        hard = error.subgroup((KeyboardInterrupt, SystemExit, GeneratorExit)) is not None
        raise
    finally:
        if not hard:
            await _close_executor(executor, config, failed=failed)
    print("candidate\tkey alias\treserved USD\tspend USD\tlabel")
    for key in report.keys:
        spend = key.reserved_usd if key.observed_usd is None else key.observed_usd
        print(f"{key.candidate_id}\t{key.alias}\t{key.reserved_usd:.6f}\t{spend:.6f}\t{key.label}")


def _reconcile(args: argparse.Namespace) -> int:
    """Use the same local RUN_DIR addressing and interrupt owner as resume."""
    from swarm_arena._cli.run_cmds import _interruptible

    args.run_dir = args.run_dir.resolve()
    anyio.run(_interruptible, partial(_reconcile_async, args), backend="asyncio")
    return 0


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
