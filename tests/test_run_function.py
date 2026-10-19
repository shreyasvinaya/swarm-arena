import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _fakes import tools as fake_tools

from swarm_arena._model.tooling import ToolRequest
from swarm_arena._tooling.result_file import request_json

SECRET = "SECRET_SENTINEL_TASK18"


def run(tmp_path: Path, fn: str, params: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    (tmp_path / "swarm").mkdir(exist_ok=True)
    (tmp_path / "swarm/request.json").write_text(
        json.dumps(request_json(ToolRequest(inputs=params)))
    )
    return subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "swarm_arena.run_function",
            "--import-root",
            str(Path(fake_tools.__file__).resolve().parents[1]),
            fn,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_writes_result(tmp_path: Path) -> None:
    done = run(tmp_path, "_fakes.tools:add", {"a": 2, "b": 3})
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "swarm/result.json").read_text()) == {
        "schema_version": 1,
        "data": {"sum": 5},
    }
    assert sorted(p.name for p in (tmp_path / "swarm").iterdir()) == ["request.json", "result.json"]


def test_cwd_files_do_not_shadow_imports(tmp_path: Path) -> None:
    (tmp_path / "swarm_arena").mkdir()
    (tmp_path / "swarm_arena/__init__.py").write_text("open('shadow', 'w')")
    (tmp_path / "json.py").write_text("raise RuntimeError('shadowed')")
    done = run(tmp_path, "_fakes.tools:add", {"a": 2, "b": 3})
    assert done.returncode == 0
    assert not (tmp_path / "shadow").exists()
    assert json.loads((tmp_path / "swarm/result.json").read_text())["data"] == {"sum": 5}


@pytest.mark.parametrize(
    ("fn", "params", "stderr"),
    [
        ("_fakes.tools:add", {"a": "x"}, "validation error for add_inputs"),
        ("_fakes.tools:not_a_dict", {}, "JSON object"),
        ("_fakes.tools:boom", {}, "boom"),
        ("_fakes.tools:nan_result", {}, "finite"),
        ("_fakes.tools:non_json_result", {}, "finite"),
    ],
)
def test_failures_exit_1(tmp_path: Path, fn: str, params: dict[str, Any], stderr: str) -> None:
    done = run(tmp_path, fn, params)
    assert done.returncode == 1
    assert stderr in done.stderr
    assert not (tmp_path / "swarm/result.json").exists()


@pytest.mark.parametrize(
    ("fn", "params"),
    [
        ("_fakes.tools:add", {"a": SECRET}),
        ("_fakes.tools:add", {"a": 1, SECRET: SECRET}),
        ("_fakes.tools:nested_input", {"values": {SECRET: SECRET}}),
        ("_fakes.tools:secret_error", {"value": SECRET}),
    ],
)
def test_stderr_hides_secret_values(tmp_path: Path, fn: str, params: dict[str, Any]) -> None:
    done = run(tmp_path, fn, params)
    assert done.returncode == 1
    assert done.stderr
    assert SECRET not in done.stderr
    assert SECRET not in done.stdout
    assert not (tmp_path / "swarm/result.json").exists()


def test_main_atomically_replaces_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena import run_function

    monkeypatch.chdir(tmp_path)
    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/request.json").write_text(json.dumps({"params": {"a": 4}}))
    calls: list[Path] = []
    replace = os.replace

    def record(src: str | Path, dst: str | Path) -> None:
        assert Path(src).parent.resolve() == (tmp_path / "swarm")
        assert json.loads(Path(src).read_text()) == {"schema_version": 1, "data": {"sum": 5}}
        calls.append(Path(dst))
        replace(src, dst)

    monkeypatch.setattr(run_function.os, "replace", record)
    assert run_function.main(["_fakes.tools:add"]) == 0
    assert len(calls) == 1
    assert calls[0].name == "result.json"


def test_failed_replace_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from swarm_arena import run_function

    monkeypatch.chdir(tmp_path)
    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/request.json").write_text(json.dumps({"params": {"a": 4}}))

    def fail(src: str | Path, dst: str | Path) -> None:
        raise OSError(SECRET)

    monkeypatch.setattr(run_function.os, "replace", fail)
    assert run_function.main(["_fakes.tools:add"]) == 1
    assert SECRET not in capsys.readouterr().err
    assert [p.name for p in (tmp_path / "swarm").iterdir()] == ["request.json"]


@pytest.mark.parametrize("request_text", ["{SECRET_SENTINEL_TASK18", "[]", "{}"])
def test_bad_request_fails_without_leaking_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    request_text: str,
) -> None:
    from swarm_arena import run_function

    monkeypatch.chdir(tmp_path)
    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/request.json").write_text(request_text)
    assert run_function.main(["_fakes.tools:add"]) == 1
    assert SECRET not in capsys.readouterr().err
    assert not (tmp_path / "swarm/result.json").exists()


def test_bad_arguments_hide_values(capsys: pytest.CaptureFixture[str]) -> None:
    from swarm_arena import run_function

    assert run_function.main(["_fakes.tools:add", "--unexpected", SECRET]) == 1
    assert SECRET not in capsys.readouterr().err
