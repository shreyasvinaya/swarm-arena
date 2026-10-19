import json
import os
from pathlib import Path


def _main() -> None:
    a = float(Path("problem/model_a.txt").read_text(encoding="utf-8"))
    b = float(Path("problem/model_b.txt").read_text(encoding="utf-8"))
    files_seen = sorted(
        path.as_posix()
        for path in Path(".").rglob("*")
        if path.parts[0] not in {"swarm", "parents"} and path.is_file()
    )
    Path(os.environ["SWARM_ARENA_RESULT"]).write_text(
        json.dumps({"data": {"a": a, "b": b, "files_seen": files_seen}}, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    _main()
