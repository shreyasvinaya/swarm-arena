"""Check that a line-fit candidate supplies finite numeric result data."""

import json
import math
import os
from pathlib import Path


def _main() -> None:
    """Write a check result with value-free reasons for invalid output."""
    try:
        output = json.loads(Path("candidate/output.json").read_text(encoding="utf-8"))
        data = output["data"]
        passed = (
            all(
                type(data[name]) in (int, float) and math.isfinite(data[name])
                for name in ("slope", "intercept", "fit_error")
            )
            and data["fit_error"] >= 0
        )
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        passed = False
    result = {
        "passed": passed,
        "reasons": [] if passed else ["missing or invalid line-fit numbers"],
    }
    Path(os.environ["SWARM_ARENA_RESULT"]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    _main()
