"""Evaluate a line on public points with only the Python standard library."""

import json
import math
import os
import tomllib
from pathlib import Path


def _coefficient(path: str, name: str) -> float:
    """Read one finite numeric TOML assignment without executing model text."""
    model = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    value = model[name]
    if set(model) != {name} or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid coefficient")
    return float(value)


def _main() -> None:
    """Read staged inputs and write the tool result to its assigned path."""
    try:
        request = json.loads(Path("swarm/request.json").read_text(encoding="utf-8"))
        scale = request["params"]["scale"]
        if type(scale) not in (int, float) or not 0.1 <= scale <= 10.0:
            raise ValueError("invalid scale")
        slope = scale * _coefficient("problem/model_a.txt", "slope")
        intercept = scale * _coefficient("problem/model_b.txt", "intercept")
        # Public format: one whitespace-separated x y pair per line, no header.
        points = [
            tuple(map(float, line.split()))
            for line in Path("problem/points.txt").read_text(encoding="utf-8").splitlines()
        ]
        fit_error = sum((slope * x + intercept - y) ** 2 for x, y in points) / len(points)
        result = {"data": {"slope": slope, "intercept": intercept, "fit_error": fit_error}}
        encoded = json.dumps(result, allow_nan=False)
    except (KeyError, TypeError, ValueError, ArithmeticError, OSError):
        raise SystemExit("invalid line-fit inputs") from None
    Path(os.environ["SWARM_ARENA_RESULT"]).write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    _main()
