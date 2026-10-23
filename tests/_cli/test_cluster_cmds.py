import argparse
import subprocess
import sys
from collections.abc import Mapping
from typing import Any

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
    assert calls == [(None, "swarm-arena run --yes run.toml plan.toml", None)]


def test_submit_forwards_explicit_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    assert (
        main(
            [
                "submit",
                "--address",
                "http://head:8265",
                "--working-dir",
                ".",
                "--pip",
                "swarm-arena[ray]",
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
            "swarm-arena run --yes run.toml 'plans dir/a.toml' b.toml",
            {"working_dir": ".", "pip": ["swarm-arena[ray]", "deepretro-swarm"]},
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
