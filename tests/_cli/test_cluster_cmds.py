import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Event, Thread, current_thread
from typing import Any

import anyio
import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._cli import cluster_cmds
from swarm_arena._cli.main import main


def test_submit_without_yes_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["submit", "run.toml", "plan.toml"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        'submit starts the driver unattended: check the estimate with "swarm-arena plan", '
        "then pass --yes\n"
    )


def test_submit_without_yes_never_imports_ray() -> None:
    code = (
        "import sys\nfrom swarm_arena._cli.main import main\n"
        "assert main(['submit', 'run.toml', 'plan.toml']) == 2\n"
        "assert not any(m == 'ray' or m.startswith('ray.') or "
        "m == 'swarm_arena.ray' or m.startswith('swarm_arena.ray.') for m in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr


def test_submit_prints_submission_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("swarm_arena.ray")
    from swarm_arena.ray import _submit

    calls: list[tuple[str | None, str, Mapping[str, Any] | None]] = []

    def submit(
        address: str | None, entrypoint: str, *, runtime_env: Mapping[str, Any] | None
    ) -> str:
        calls.append((address, entrypoint, runtime_env))
        return "raysubmit_9"

    monkeypatch.setattr(_submit, "submit_driver", submit)
    assert main(["submit", "--yes", "run.toml", "plan.toml"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "raysubmit_9"
    assert captured.err == ""
    assert calls == [(None, "swarm-arena run --yes -- run.toml plan.toml", None)]


def test_submit_forwards_explicit_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    pytest.importorskip("swarm_arena.ray")
    from swarm_arena.ray import _submit

    calls: list[tuple[str | None, str, Mapping[str, Any] | None]] = []

    def submit(
        address: str | None, entrypoint: str, *, runtime_env: Mapping[str, Any] | None
    ) -> str:
        calls.append((address, entrypoint, runtime_env))
        return "raysubmit_options"

    monkeypatch.setattr(_submit, "submit_driver", submit)
    monkeypatch.setenv("RAY_AUTH_TOKEN", "sentinel-secret")
    monkeypatch.chdir(tmp_path)
    for name in ["run.toml", "plans dir/a.toml", "b.toml"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert (
        main(
            [
                "submit",
                "--address",
                "http://head:8265",
                "--working-dir",
                ".",
                "--pip",
                "swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena",
                "--pip",
                "deepretro-swarm",
                "--yes",
                "run.toml",
                "plans dir/a.toml",
                "b.toml",
            ]
        )
        == 0
    )
    assert calls == [
        (
            "http://head:8265",
            "swarm-arena run --yes -- run.toml 'plans dir/a.toml' b.toml",
            {
                "working_dir": ".",
                "pip": [
                    "swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena",
                    "deepretro-swarm",
                ],
            },
        )
    ]
    captured = capsys.readouterr()
    assert captured.out == "raysubmit_options\n"
    assert captured.err == ""


def test_submit_without_yes_submits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("swarm_arena.ray")
    from swarm_arena.ray import _submit

    def unexpected(*args: Any, **kwargs: Any) -> str:
        pytest.fail("submission must require approval")

    monkeypatch.setattr(_submit, "submit_driver", unexpected)
    assert main(["submit", "run.toml", "plan.toml"]) == 2


@pytest.mark.parametrize("stage", ["client", "submission"])
def test_submit_ray_errors_preserve_safe_diagnostic_and_suppress_values(
    stage: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("swarm_arena.ray")
    from swarm_arena.ray import _submit

    class FailedClient:
        def __init__(self, address: str | None) -> None:
            if stage == "client":
                raise RuntimeError("sentinel-secret")

        def submit_job(self, **kwargs: Any) -> str:
            raise RuntimeError("sentinel-secret")

    monkeypatch.setattr(_submit, "JobSubmissionClient", FailedClient)
    assert main(["submit", "--yes", "run.toml", "plan.toml"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Ray job submission failed: RuntimeError\n"
    assert "sentinel-secret" not in captured.err


def test_help_loads_no_ray() -> None:
    code = (
        "import sys\nfrom swarm_arena._cli.main import main\n"
        "try: main(['submit', '--help'])\nexcept SystemExit as exc: assert exc.code == 0\n"
        "assert not any(m == 'ray' or m.startswith('ray.') or "
        "m == 'swarm_arena.ray' or m.startswith('swarm_arena.ray.') for m in sys.modules)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "--pip" in done.stdout


def test_submit_error_has_no_secret_on_exception_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("swarm_arena.ray")
    from swarm_arena.ray import _submit

    def failed(*args: Any, **kwargs: Any) -> str:
        error = RuntimeError("sentinel-secret")
        error.add_note("sentinel-secret")
        raise error from ValueError("sentinel-secret")

    monkeypatch.setattr(_submit, "submit_driver", failed)
    parser = argparse.ArgumentParser()
    cluster_cmds.register(parser.add_subparsers())
    args = parser.parse_args(["submit", "--yes", "run.toml", "plan.toml"])
    with pytest.raises(RuntimeError, match="Ray job submission failed: RuntimeError") as caught:
        args.handler(args)
    assert_secret_absent(caught.value, "sentinel-secret", sanitized=True)


def test_submit_requires_a_plan() -> None:
    with pytest.raises(SystemExit, match=r"^2$"):
        main(["submit", "--yes", "run.toml"])


@pytest.mark.parametrize("failure", ["missing-ray", "missing-other", "import-error"])
def test_approved_submit_import_diagnostics(failure: str) -> None:
    code = """
import builtins
import importlib.metadata
import sys
from swarm_arena._cli.main import main
original = builtins.__import__
def importing(name, *args, **kwargs):
    if name == "swarm_arena.ray":
        if sys.argv[1] == "missing-ray":
            raise ModuleNotFoundError("sentinel-secret", name="ray")
        if sys.argv[1] == "missing-other":
            raise ModuleNotFoundError("sentinel-secret", name="broken_dependency")
        raise ImportError("sentinel-secret")
    return original(name, *args, **kwargs)
builtins.__import__ = importing
raise SystemExit(main(["submit", "--yes", "run.toml", "plan.toml"]))
"""
    done = subprocess.run(
        [sys.executable, "-c", code, failure],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 1
    assert done.stdout == ""
    assert "sentinel-secret" not in done.stderr
    if failure == "missing-ray":
        assert done.stderr == (
            "submit needs the 'ray' extra: uv add "
            '"swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena"\n'
        )
    else:
        assert done.stderr == (
            "ModuleNotFoundError\n" if failure == "missing-other" else "ImportError\n"
        )


def test_approved_submit_without_installed_ray() -> None:
    code = """
import importlib.metadata
from swarm_arena._cli.main import main
original = importlib.metadata.version
def version(name):
    if name == "ray":
        raise importlib.metadata.PackageNotFoundError(name)
    return original(name)
importlib.metadata.version = version
raise SystemExit(main(["submit", "--yes", "run.toml", "plan.toml"]))
"""
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 1
    assert done.stdout == ""
    assert "submit needs the 'ray' extra: uv add" in done.stderr
    assert "swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena" in done.stderr


@pytest.mark.parametrize("case", ["missing-dir", "not-dir", "outside", "symlink", "missing-path"])
def test_invalid_working_directory_paths_fail_before_ray_import(case: str, tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "run.toml").touch()
    outside = tmp_path / "sentinel-secret.toml"
    outside.touch()
    (root / "link.toml").symlink_to(outside)
    directory = root
    plan = root / "run.toml"
    if case == "missing-dir":
        directory = root / "sentinel-secret"
    elif case == "not-dir":
        directory = outside
    elif case == "outside":
        plan = outside
    elif case == "symlink":
        plan = root / "link.toml"
    else:
        plan = root / "sentinel-secret.toml"
    code = """
import builtins
import sys
from swarm_arena._cli.main import main
original = builtins.__import__
def importing(name, *args, **kwargs):
    if name == "swarm_arena.ray":
        raise AssertionError("Ray import must follow local validation")
    return original(name, *args, **kwargs)
builtins.__import__ = importing
raise SystemExit(main(["submit", "--yes", "--working-dir", *sys.argv[1:]]))
"""
    done = subprocess.run(
        [sys.executable, "-c", code, str(directory), str(root / "run.toml"), str(plan)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 1
    assert done.stdout == ""
    expected = (
        "--working-dir must be an existing directory\n"
        if case in {"missing-dir", "not-dir"}
        else "run and plan paths must exist inside --working-dir\n"
    )
    assert done.stderr == expected
    assert "sentinel-secret" not in done.stderr


def test_working_directory_rewrites_paths_relative_to_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("swarm_arena.ray")
    from swarm_arena.ray import _submit

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "project"
    root.mkdir()
    (root / "run.toml").touch()
    (root / "-plan.toml").touch()
    calls: list[tuple[str, Mapping[str, Any] | None]] = []

    def submit(
        address: str | None, entrypoint: str, *, runtime_env: Mapping[str, Any] | None
    ) -> str:
        calls.append((entrypoint, runtime_env))
        return "id"

    monkeypatch.setattr(_submit, "submit_driver", submit)
    assert (
        main(
            [
                "submit",
                "--yes",
                "--working-dir",
                "project",
                "project/run.toml",
                "project/-plan.toml",
            ]
        )
        == 0
    )
    assert calls == [("swarm-arena run --yes -- run.toml -plan.toml", {"working_dir": "project"})]
    assert capsys.readouterr().out == "id\n"


@pytest.mark.parametrize(
    "spec",
    [
        "pkg @ git+https://user:sentinel-secret@host/repo.git",
        "https://sentinel-secret@host/pkg.whl",
        "git+ssh://git@host/repo.git",
    ],
)
def test_pip_url_userinfo_is_refused_before_ray_import(spec: str) -> None:
    code = """
import builtins
import sys
from swarm_arena._cli.main import main
original = builtins.__import__
def importing(name, *args, **kwargs):
    if name == "swarm_arena.ray":
        raise AssertionError("Ray import must follow local validation")
    return original(name, *args, **kwargs)
builtins.__import__ = importing
raise SystemExit(main(["submit", "--yes", "--pip", sys.argv[1], "run.toml", "plan.toml"]))
"""
    done = subprocess.run(
        [sys.executable, "-c", code, spec], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode == 1
    assert done.stdout == ""
    assert done.stderr == "--pip specifications must not contain URL userinfo\n"


@pytest.mark.parametrize("stage", ["client", "submission"])
def test_submission_timeout_abandons_one_wait_with_recoverable_id(
    stage: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("swarm_arena.ray")
    from swarm_arena.ray import _submit

    assert _submit._SUBMIT_TIMEOUT_S == 300
    entered, release = Event(), Event()
    workers: list[Thread] = []
    calls: list[dict[str, Any]] = []
    scopes: list[anyio.CancelScope] = []
    real_scope = anyio.move_on_after
    real_daemon = _submit.run_sync_daemon

    def scope(seconds: float) -> anyio.CancelScope:
        assert seconds == 300
        result = real_scope(seconds)
        assert result.deadline == pytest.approx(anyio.current_time() + 300, abs=1)
        scopes.append(result)
        return result

    def block() -> None:
        workers.append(current_thread())
        entered.set()
        assert release.wait(180)

    class Client:
        def __init__(self, address: str | None) -> None:
            if stage == "client":
                block()

        def submit_job(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            block()
            return kwargs["submission_id"]

    async def monitored(fn: Callable[[], str]) -> str:
        async def expire() -> None:
            assert await anyio.to_thread.run_sync(entered.wait, 60)
            scopes[0].deadline = float("-inf")

        async with anyio.create_task_group() as group:
            group.start_soon(expire)
            result = await real_daemon(fn)
        return result

    monkeypatch.setattr(_submit.anyio, "move_on_after", scope)
    monkeypatch.setattr(_submit, "run_sync_daemon", monitored)
    monkeypatch.setattr(_submit, "JobSubmissionClient", Client)
    try:
        assert main(["submit", "--yes", "run.toml", "plan.toml"]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""

        match = re.fullmatch(
            r"Ray job submission timed out: (swarm-arena-[0-9a-f]{32}); "
            r"check ray job status \1 before retrying\n",
            captured.err,
        )
        assert match is not None
        assert len(scopes) == 1
        if stage == "submission":
            assert calls[0]["submission_id"] == match.group(1)
        assert all(worker.daemon for worker in workers)
    finally:
        release.set()
        for worker in workers:
            worker.join(60)
            assert not worker.is_alive()
    if stage == "client":
        assert calls == []


def test_interrupt_prints_submission_id_and_status_instruction() -> None:
    pytest.importorskip("swarm_arena.ray")
    code = """
import os
import signal
import threading
from swarm_arena._cli.main import main
from swarm_arena.ray import _submit
class Client:
    def __init__(self, address):
        pass
    def submit_job(self, **kwargs):
        print(kwargs["submission_id"], flush=True)
        os.kill(os.getpid(), signal.SIGINT)
        threading.Event().wait()
_submit.JobSubmissionClient = Client
raise SystemExit(main(["submit", "--yes", "run.toml", "plan.toml"]))
"""
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert done.returncode != 0
    submission_id = done.stdout.strip()
    assert submission_id.startswith("swarm-arena-")
    assert done.stderr == (
        f"Ray job submission interrupted: {submission_id}; "
        f"check ray job status {submission_id} before retrying\n"
    )
