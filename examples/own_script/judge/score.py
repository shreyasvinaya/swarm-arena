"""Score line coefficients against private points with mean squared error."""

import csv
import json
import math
import os
from pathlib import Path


def _main() -> None:
    """Read nested candidate data and write an independently computed score."""
    try:
        data = json.loads(Path("candidate/output.json").read_text(encoding="utf-8"))["data"]
        slope, intercept = data["slope"], data["intercept"]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (slope, intercept)):
            raise ValueError("invalid coefficients")
        with Path("judge/hidden/points.csv").open(encoding="utf-8", newline="") as source:
            points = [(float(row["x"]), float(row["y"])) for row in csv.DictReader(source)]
        mse = sum((slope * x + intercept - y) ** 2 for x, y in points) / len(points)
        encoded = json.dumps({"value": mse}, allow_nan=False)
    except (KeyError, TypeError, ValueError, ArithmeticError, OSError):
        raise SystemExit("invalid line-fit scoring inputs") from None
    Path(os.environ["SWARM_ARENA_RESULT"]).write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    _main()
