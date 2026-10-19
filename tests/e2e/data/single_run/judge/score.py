import csv
import json
import os
from pathlib import Path


def _main() -> None:
    output = json.loads(Path("candidate/output.json").read_text(encoding="utf-8"))
    a, b = output["data"]["a"], output["data"]["b"]
    with Path("judge/hidden/points.csv").open(encoding="utf-8", newline="") as stream:
        errors = [
            (a * float(row["x"]) + b - float(row["y"])) ** 2 for row in csv.DictReader(stream)
        ]
    Path(os.environ["SWARM_ARENA_RESULT"]).write_text(
        json.dumps({"value": sum(errors) / len(errors)}, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    _main()
