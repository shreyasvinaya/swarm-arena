import json
import os
from pathlib import Path


def _main() -> None:
    a = float(Path("problem/model_a.txt").read_text(encoding="utf-8"))
    b = float(Path("problem/model_b.txt").read_text(encoding="utf-8"))
    Path("fit.txt").write_text(f"{a} {b}\n", encoding="utf-8")
    parent_files_present = {}
    for parent in sorted(Path("parents").glob("*")):
        data = json.loads((parent / "data.json").read_text(encoding="utf-8"))
        fit = parent / "fit.txt"
        model = parent / "problem/model_a.txt"
        parent_files_present[parent.name] = (
            fit.is_file()
            and model.is_file()
            and list(map(float, fit.read_text(encoding="utf-8").split())) == [data["a"], data["b"]]
            and float(model.read_text(encoding="utf-8")) == data["a"]
        )
    files_seen = sorted(
        path.as_posix()
        for path in Path(".").rglob("*")
        if path.parts[0] not in {"swarm", "parents"} and path.is_file()
    )
    Path(os.environ["SWARM_ARENA_RESULT"]).write_text(
        json.dumps(
            {
                "data": {
                    "a": a,
                    "b": b,
                    "files_seen": files_seen,
                    "parent_files_present": parent_files_present,
                },
                "files": ["fit.txt"],
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    _main()
