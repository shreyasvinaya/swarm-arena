import csv
import json
import os
from pathlib import Path


def _main() -> None:
    files = Path("candidate/files")
    a, b = map(float, (files / "fit.txt").read_text(encoding="utf-8").split())
    # Uploaded coefficients must agree with the candidate's effective edited files.
    assert a == float((files / "problem/model_a.txt").read_text(encoding="utf-8"))
    assert b == float((files / "problem/model_b.txt").read_text(encoding="utf-8"))
    with Path("judge/hidden/points.csv").open(encoding="utf-8", newline="") as stream:
        errors = [
            (a * float(row["x"]) + b - float(row["y"])) ** 2 for row in csv.DictReader(stream)
        ]
    Path(os.environ["SWARM_ARENA_RESULT"]).write_text(
        json.dumps({"value": sum(errors) / len(errors)}, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    _main()
