"""Run a referenced Python function using the staged tool request and result protocol."""

import argparse
import contextlib
import importlib
import json
import os
import sys
import tempfile
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from pydantic import ValidationError

from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.data import finite_json
from swarm_arena._model.migrate import CURRENT
from swarm_arena._tooling.function_tool import _load_function, function_input_model


class _Parser(argparse.ArgumentParser):
    """Avoid echoing arbitrary argument values when parsing fails."""

    def error(self, message: str) -> NoReturn:
        """Reject invalid arguments without echoing their contents."""
        raise ValueError("invalid function runner arguments")


def _write_result(payload: bytes, result_path: Path) -> None:
    """Publish a complete envelope atomically and clean up unpublished files."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=result_path.parent, prefix=".result-", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(payload)
        os.replace(temporary, result_path)
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Validate a staged request, call its function, and write an atomic result.

    Run with ``python -P -m swarm_arena.run_function``. The explicit import root
    is inserted before loading user code. ``-P`` removes only the implicit
    working-directory entry. LocalExecutor does not forward ``PYTHONPATH``;
    dependencies must be installed in the interpreter's environment or available
    under the explicit import root.

    Parameters
    ----------
    argv : Sequence[str] or None, default=None
        Optional ``--import-root DIR`` followed by ``module:qualname``.
        None reads the command line.

    Returns
    -------
    int
        Zero after publishing the result, or one on failure. Runner diagnostics
        omit values; failures importing or calling user code include their
        traceback in stderr.

    Examples
    --------
    >>> from swarm_arena.run_function import main
    >>> main(["--help"])
    usage: ...
    0
    """
    reason = "invalid function runner arguments"
    try:
        swarm_dir = Path.cwd() / "swarm"
        result_path = swarm_dir / "result.json"
        parser = _Parser(prog="python -P -m swarm_arena.run_function")
        parser.add_argument("--import-root")
        parser.add_argument("function")
        try:
            args = parser.parse_args(argv)
        except SystemExit as exc:
            return 0 if exc.code == 0 else 1
        if args.import_root is not None:
            sys.path.insert(0, args.import_root)
        reason = f"cannot import {args.function}"
        try:
            # Import here before the driver loader sanitizes exception values.
            importlib.import_module(args.function.split(":", 1)[0])
        except (Exception, SystemExit) as exc:
            print(f"{reason} ({type(exc).__name__})", file=sys.stderr)
            traceback.print_exception(exc, file=sys.stderr)
            return 1
        fn = _load_function(args.function)
        reason = f"FunctionTool {args.function}: unsupported function signature"
        model = function_input_model(fn)
        reason = "cannot read swarm/request.json"
        request = json.loads((swarm_dir / "request.json").read_bytes())
        reason = f"FunctionTool {args.function}: invalid request params"
        try:
            inputs = model.model_validate(request["params"])
        except ValidationError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        reason = f"FunctionTool {args.function}: function failed"
        try:
            result = fn(**{name: getattr(inputs, name) for name in type(inputs).model_fields})
        except (Exception, SystemExit) as exc:
            print(f"{reason} ({type(exc).__name__})", file=sys.stderr)
            traceback.print_exception(exc, file=sys.stderr)
            return 1
        if not isinstance(result, dict):
            print(f"FunctionTool {args.function}: must return a JSON object", file=sys.stderr)
            return 1
        reason = f"FunctionTool {args.function}: values must be finite JSON"
        payload = canonical_json(
            {"schema_version": CURRENT["tool_result"], "data": finite_json(result)}
        )
        reason = "cannot write swarm/result.json"
        _write_result(payload, result_path)
        return 0
    except (Exception, SystemExit) as exc:
        print(f"{reason} ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
