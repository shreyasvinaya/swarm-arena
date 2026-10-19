import json
import math
import os
from pathlib import Path


def _main() -> None:
    output = json.loads(Path("candidate/output.json").read_text(encoding="utf-8"))
    numbers = [value for value in output["data"].values() if type(value) in (int, float)]
    reasons = []
    if output["status"] != "ok":
        reasons.append("status not ok")
    if not numbers or not all(math.isfinite(value) for value in numbers):
        reasons.append("non-finite or missing numbers")
    Path(os.environ["SWARM_ARENA_RESULT"]).write_text(
        json.dumps({"passed": not reasons, "reasons": reasons}), encoding="utf-8"
    )


if __name__ == "__main__":
    _main()
