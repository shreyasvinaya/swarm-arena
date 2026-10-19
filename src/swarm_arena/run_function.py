"""Run a referenced Python function using the staged tool request and result protocol."""

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from pydantic import ValidationError

from swarm_arena._model.canonical import canonical_json
from swarm_arena._tooling.function_tool import _load_function, function_input_model


class _Parser(argparse.ArgumentParser):
    """Avoid echoing arbitrary argument values when parsing fails."""

    def error(self, message: str) -> NoReturn:
        """Reject invalid arguments without echoing their contents."""
        raise ValueError("invalid function runner arguments")


def _write_result(payload: bytes) -> None:
    """Publish a complete envelope atomically and clean up unpublished files."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir="swarm", prefix=".result-", delete=False) as stream:
            temporary = stream.name
            stream.write(payload)
        os.replace(temporary, "swarm/result.json")
    finally:
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Validate a staged request, call its function, and write an atomic result.

    Run with ``python -P -m swarm_arena.run_function``. The explicit import root
    is inserted before loading user code. ``-P`` removes only the implicit
    working-directory entry; explicit ``PYTHONPATH`` entries remain effective.

    Parameters
    ----------
    argv : Sequence[str] or None, default=None
        Optional ``--import-root DIR`` followed by ``module:qualname``.
        None reads the command line.

    Returns
    -------
    int
        Zero after publishing the result, or one after a sanitized diagnostic.

    Examples
    --------
    >>> main(["--import-root", "/project", "my_package.tools:solve"])  # doctest: +SKIP
    0
    """
    reason = "invalid function runner arguments"
    try:
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
        fn = _load_function(args.function)
        reason = f"FunctionTool {args.function}: unsupported function signature"
        model = function_input_model(fn)
        reason = "cannot read swarm/request.json"
        request = json.loads(Path("swarm/request.json").read_bytes())
        reason = f"FunctionTool {args.function}: invalid request params"
        try:
            inputs = model.model_validate(request["params"])
        except ValidationError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        reason = f"FunctionTool {args.function}: function failed"
        result = fn(**{name: getattr(inputs, name) for name in type(inputs).model_fields})
        if not isinstance(result, dict):
            print(f"FunctionTool {args.function}: must return a JSON object", file=sys.stderr)
            return 1
        reason = f"FunctionTool {args.function}: values must be finite JSON"
        payload = canonical_json({"schema_version": 1, "data": result})
        reason = "cannot write swarm/result.json"
        _write_result(payload)
        return 0
    except (Exception, SystemExit) as exc:
        print(f"{reason} ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
