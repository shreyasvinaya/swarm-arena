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

SECRET = "b7c9cdbd45ac6961fae12985afab51de"


def run(
    tmp_path: Path,
    fn: str,
    params: dict[str, Any],
    *,
    import_dir: Path | None = None,
    pythonpath: Path | None = None,
) -> subprocess.CompletedProcess[str]:
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
            str(import_dir or Path(fake_tools.__file__).resolve().parents[1]),
            fn,
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(pythonpath)} if pythonpath is not None else None,
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


@pytest.mark.parametrize("function", ["_fakes.tools:add", "version_canary:solve"])
def test_newer_request_refused_before_user_import(tmp_path: Path, function: str) -> None:
    (tmp_path / "swarm").mkdir()
    wire = request_json(ToolRequest(inputs={"a": 2, "b": 3}))
    wire["schema_version"] = 99
    (tmp_path / "swarm/request.json").write_text(json.dumps(wire))
    root = tmp_path / "imports"
    root.mkdir()
    (root / "version_canary.py").write_text(
        "from pathlib import Path\n"
        "Path('imported').touch()\n"
        "def solve(a: int, b: int) -> dict[str, int]:\n"
        "    Path('called').touch()\n"
        "    return {'sum': a + b}\n"
    )
    import_root = (
        Path(fake_tools.__file__).resolve().parents[1] if function == "_fakes.tools:add" else root
    )
    done = subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "swarm_arena.run_function",
            "--import-root",
            str(import_root),
            function,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert done.returncode == 1, done.stderr
    assert "newer" in done.stderr
    assert not (tmp_path / "swarm/result.json").exists()
    assert not (tmp_path / "imported").exists()
    assert not (tmp_path / "called").exists()


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
        ("_fakes.tools:raises_error", {}, "RuntimeError: boom"),
        ("_fakes.tools:exits", {}, "function failed (SystemExit)"),
        ("_fakes.tools:nan_result", {}, "finite"),
        ("_fakes.tools:non_json_result", {}, "finite"),
    ],
)
def test_failures_exit_1(tmp_path: Path, fn: str, params: dict[str, Any], stderr: str) -> None:
    done = run(tmp_path, fn, params)
    assert done.returncode == 1
    assert stderr in done.stderr
    if fn == "_fakes.tools:exits":
        assert "Traceback (most recent call last):" in done.stderr
        assert "in exits" in done.stderr
        assert "SystemExit: 0" in done.stderr
    assert not (tmp_path / "swarm/result.json").exists()


@pytest.mark.parametrize(
    ("fn", "params"),
    [
        ("_fakes.tools:add", {"a": SECRET}),
        ("_fakes.tools:add", {"a": 1, SECRET: SECRET}),
        ("_fakes.tools:nested_input", {"values": {SECRET: SECRET}}),
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


@pytest.mark.parametrize("request_text", ["{b7c9cdbd45ac6961fae12985afab51de", "[]", "{}"])
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


@pytest.mark.parametrize(("params", "expected"), [({"a": "2", "b": "3"}, 5), ({"a": "2"}, 3)])
def test_calls_function_with_validated_values(
    tmp_path: Path, params: dict[str, Any], expected: int
) -> None:
    done = run(tmp_path, "_fakes.tools:add", params)
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "swarm/result.json").read_text())["data"] == {"sum": expected}


@pytest.mark.parametrize("has_swarm_dir", [False, True])
def test_result_stays_in_workspace_after_chdir(tmp_path: Path, has_swarm_dir: bool) -> None:
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    if has_swarm_dir:
        (destination / "swarm").mkdir()
    done = run(tmp_path, "_fakes.tools:changes_directory", {"directory": str(destination)})
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "swarm/result.json").read_text())["data"] == {"changed": True}
    assert not (destination / "swarm/result.json").exists()
    assert sorted(p.name for p in (tmp_path / "swarm").iterdir()) == ["request.json", "result.json"]


def test_user_traceback_is_written_to_own_stderr(tmp_path: Path) -> None:
    done = run(tmp_path, "_fakes.tools:secret_error", {"value": SECRET})
    assert done.returncode == 1
    assert "FunctionTool _fakes.tools:secret_error: function failed (RuntimeError)" in done.stderr
    assert "Traceback (most recent call last):" in done.stderr
    assert "in secret_error" in done.stderr
    assert f"RuntimeError: {SECRET}" in done.stderr
    assert not done.stdout
    assert not (tmp_path / "swarm/result.json").exists()


def test_import_root_precedes_pythonpath(tmp_path: Path) -> None:
    decoy = tmp_path / "decoy"
    (decoy / "_fakes").mkdir(parents=True)
    # Match the real _fakes namespace package so only sys.path order decides.
    (decoy / "_fakes/tools.py").write_text(
        'def add(a: int, b: int = 1) -> dict[str, int]:\n    return {"sum": -1}\n'
    )
    done = run(tmp_path, "_fakes.tools:add", {"a": 2, "b": 3}, pythonpath=decoy)
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "swarm/result.json").read_text())["data"] == {"sum": 5}


def test_runner_does_not_add_cwd_for_late_imports(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "late_import_function.py").write_text(
        "def solve() -> dict[str, bool]:\n"
        "    try:\n"
        "        import workspace_only_dependency\n"
        "    except ModuleNotFoundError:\n"
        "        return {'shadowed': False}\n"
        "    return {'shadowed': True}\n"
    )
    (tmp_path / "workspace_only_dependency.py").write_text("")
    done = run(tmp_path, "late_import_function:solve", {}, import_dir=root, pythonpath=root)
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "swarm/result.json").read_text())["data"] == {"shadowed": False}


@pytest.mark.parametrize(
    ("source", "error"),
    [
        (
            "import missing_function_dependency\n",
            "ModuleNotFoundError: No module named 'missing_function_dependency'",
        ),
        (f'raise RuntimeError("{SECRET}")\n', f"RuntimeError: {SECRET}"),
        ("raise SystemExit(0)\n", "SystemExit: 0"),
    ],
)
def test_import_failure_prints_user_traceback(tmp_path: Path, source: str, error: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "runner_import_failure.py").write_text(source)
    done = run(tmp_path, "runner_import_failure:solve", {}, import_dir=root)
    assert done.returncode == 1
    assert "cannot import runner_import_failure:solve" in done.stderr
    assert "Traceback (most recent call last):" in done.stderr
    assert "runner_import_failure.py" in done.stderr
    assert error in done.stderr
    assert not done.stdout
    assert not (tmp_path / "swarm/result.json").exists()
