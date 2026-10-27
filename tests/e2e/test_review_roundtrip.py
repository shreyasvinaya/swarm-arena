"""The installed arena CLI supports a complete offline human review roundtrip."""

import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import anyio
import pytest
from _docs import docs_bindir, run_command
from _fakes.finished import FinishedRun, run_again
from _fakes.process_cleanup import ProcessIdentity, process_table, snapshot_process_tree

from swarm_arena._scoring.human_review import CSV_HEADER


def _cli(config_path: Path, *args: str) -> str:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [str(Path(sys.executable).parent / "swarm-arena"), *args, "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=environment,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _ids(output: str) -> list[str]:
    return [line.split("\t")[1] for line in output.splitlines() if "\t" in line]


async def test_review_roundtrip(finished_run: FinishedRun, tmp_path: Path) -> None:
    identifier = "2026-10-21T00-00-00Z_abcdef"
    await run_again(finished_run, run_id=identifier)
    cfg = finished_run.config_path
    assert "no scores for human:alice" in await anyio.to_thread.run_sync(
        _cli, cfg, "leaderboard", identifier, "--by", "human:alice"
    )
    directory = tmp_path / "rev"
    await anyio.to_thread.run_sync(
        _cli, cfg, "review", "export", identifier, "--out", str(directory)
    )
    path = directory / "review.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    exported = [row["id"] for row in rows]
    assert len(exported) > 1
    for index, row in enumerate(rows, 1):
        row.update(verdict="accept", score=str(index))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    await anyio.to_thread.run_sync(
        _cli, cfg, "review", "import", identifier, str(path), "--reviewer", "alice"
    )
    human = await anyio.to_thread.run_sync(
        _cli, cfg, "leaderboard", identifier, "--by", "human:alice"
    )
    ordinary = await anyio.to_thread.run_sync(_cli, cfg, "leaderboard", identifier)
    assert _ids(human) == list(reversed(exported))
    assert _ids(ordinary) == exported


# The README's run/view/share quickstart is covered by tests/test_docs.py::test_readme_quickstart.
@pytest.mark.parametrize(("document", "heading"), [("docs/arena.md", "Command line")])
def test_documented_arena_commands_run_as_written(
    tmp_path: Path, document: str, heading: str
) -> None:
    import re

    root = Path(__file__).resolve().parents[2]
    text = (root / document).read_text(encoding="utf-8")
    section = text.split(f"## {heading}\n", 1)[1]
    fence = re.search(r"^```(sh|bash)\n(.*?)^```", section, flags=re.MULTILINE | re.DOTALL)
    assert fence is not None, f"{document}: no sh or bash block under {heading}"
    block = fence.group(2)
    example = tmp_path / "examples" / "rosenbrock"
    shutil.copytree(root / "examples" / "rosenbrock", example)
    stdout, _ = run_command(
        ["bash", "-euo", "pipefail", "-c", block],
        source=document,
        line=text[: text.index(section) + fence.start()].count("\n") + 1,
        workdir=tmp_path,
        bindir=docs_bindir(),
        timeout_s=90,
    )
    identifier = re.search(r"^run (\S+) at ", stdout, flags=re.MULTILINE)
    assert identifier is not None
    reviews = tmp_path / ".swarm-arena" / f"review-{identifier[1]}" / "review.csv"
    with reviews.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(row["verdict"] == "accept" for row in rows)
    assert "scores as human:alice" in stdout
    assert "(human:alice," in stdout
    assert "human,pinned" in stdout


def test_arena_docs_timeout_reaps_separate_executor_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copytree = shutil.copytree
    communicate = subprocess.Popen.communicate
    captured: dict[int, ProcessIdentity] = {}
    held: list[int] = []
    forced = False

    def held_example(src: Path, dst: Path, *args: Any, **kwargs: Any) -> Path:
        result = copytree(src, dst, *args, **kwargs)
        module = Path(dst) / "rosen.py"
        if module.exists():
            code = module.read_text(encoding="utf-8")
            hold = (
                "    import os, time\n    from pathlib import Path\n"
                f"    Path({str(tmp_path)!r}, f'tool-ready-{{os.getpid()}}').touch()\n"
                "    time.sleep(300)\n"
            )
            assert "    x, y = x0, y0\n" in code
            module.write_text(
                code.replace("    x, y = x0, y0\n", hold + "    x, y = x0, y0\n"), encoding="utf-8"
            )
        return Path(result)

    def expire_after_tools_ready(
        self: subprocess.Popen[str], input: str | None = None, timeout: float | None = None
    ) -> tuple[str, str]:
        nonlocal forced
        if not forced and isinstance(self.args, list) and self.args[0] in {"sh", "bash"}:
            forced = True
            assert timeout == 90
            deadline = time.monotonic() + 60
            try:
                while len(list(tmp_path.glob("tool-ready-*"))) < 2:
                    assert self.poll() is None, "arena driver exited before tools were ready"
                    assert time.monotonic() < deadline, "executor tools did not become ready"
                    time.sleep(0.01)
            finally:
                captured.update(snapshot_process_tree(self.pid))
            held.extend(
                int(path.name.removeprefix("tool-ready-")) for path in tmp_path.glob("tool-ready-*")
            )
            assert len(held) == 2
            assert all(captured[pid].pgid == pid != self.pid for pid in held)
            return communicate(self, input=input, timeout=0.05)
        return communicate(self, input=input, timeout=timeout)

    monkeypatch.setattr(shutil, "copytree", held_example)
    monkeypatch.setattr(subprocess.Popen, "communicate", expire_after_tools_ready)
    try:
        with pytest.raises(AssertionError, match=r"docs/arena\.md:\d+ timed out after 90s"):
            test_documented_arena_commands_run_as_written(tmp_path, "docs/arena.md", "Command line")
        assert forced
        assert len(held) == 2
        current = process_table()
        assert {
            pid: record
            for pid, record in captured.items()
            if pid in current and current[pid].started == record.started
        } == {}
    finally:
        # Independent emergency cleanup runs only after the liveness assertion.
        current = process_table()
        for pid, record in reversed(captured.items()):
            if pid in current and current[pid].started == record.started:
                with suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 60
        while True:
            current = process_table()
            if not any(
                pid in current and current[pid].started == record.started
                for pid, record in captured.items()
            ):
                break
            assert time.monotonic() < deadline, "emergency executor cleanup did not finish"
            time.sleep(0.01)
