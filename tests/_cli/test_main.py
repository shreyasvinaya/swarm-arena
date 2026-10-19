import subprocess
import sys
from pathlib import Path

import pytest

import swarm_arena
from swarm_arena._cli.main import main


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"swarm-arena {swarm_arena.__version__}\n"


def test_no_arguments_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out.startswith("swarm-arena ")


def test_unknown_argument_is_usage_error() -> None:
    with pytest.raises(SystemExit, match=r"^2$") as exc:
        main(["--nope"])
    assert exc.value.code == 2


def test_console_script_installed() -> None:
    exe = Path(sys.executable).parent / "swarm-arena"
    done = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=False)
    assert (done.returncode, done.stdout) == (0, f"swarm-arena {swarm_arena.__version__}\n")
