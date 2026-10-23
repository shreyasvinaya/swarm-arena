import os
import resource
import subprocess
import sys
from pathlib import Path

import pytest
from _fakes.process_cleanup import owned_processes  # noqa: F401

from swarm_arena import _launch

pytestmark = pytest.mark.usefixtures("owned_processes")

# A workspace swarm_arena/__init__.py; -I keeps the cwd and PYTHONPATH off sys.path.
SHADOW = "open('shadow', 'w')"


def spawn(
    args: list[str], tmp_path: Path, env: dict[str, str] | None = None
) -> tuple[subprocess.Popen[bytes], int]:
    r, w = os.pipe()
    try:
        p = subprocess.Popen(
            [sys.executable, "-I", "-m", "swarm_arena._launch", "--go-fd", str(r), *args],
            pass_fds=(r,),
            cwd=tmp_path,
            env=env,
        )
    except BaseException:
        os.close(w)
        raise
    finally:
        os.close(r)
    return p, w


def go(w: int) -> None:
    os.write(w, b"go")
    os.close(w)


def test_execs_after_go(tmp_path: Path) -> None:
    p, w = spawn(
        ["--", sys.executable, "-c", "import os; open('pid', 'w').write(str(os.getpid()))"],
        tmp_path,
    )
    go(w)
    assert p.wait(60) == 0
    assert (tmp_path / "pid").read_text() == str(p.pid)


def test_exits_without_exec_when_pipe_closes(tmp_path: Path) -> None:
    p, w = spawn(["--", sys.executable, "-c", "open('m', 'w')"], tmp_path)
    os.close(w)
    assert p.wait(60) == 0
    assert not (tmp_path / "m").exists()


def test_exits_without_exec_on_other_bytes(tmp_path: Path) -> None:
    p, w = spawn(["--", sys.executable, "-c", "open('m', 'w')"], tmp_path)
    os.write(w, b"no")
    os.close(w)
    assert p.wait(60) == 0
    assert not (tmp_path / "m").exists()


@pytest.mark.parametrize("pythonpath", [None, "."])
def test_workspace_package_never_imported(tmp_path: Path, pythonpath: str | None) -> None:
    (tmp_path / "swarm_arena").mkdir()
    (tmp_path / "swarm_arena/__init__.py").write_text(SHADOW)
    env = (
        {**os.environ, "PYTHONPATH": pythonpath} if pythonpath else None
    )  # "." = an allowlisted PYTHONPATH
    p, w = spawn(["--", sys.executable, "-c", "pass"], tmp_path, env=env)
    os.close(w)  # no "go"
    assert p.wait(60) == 0
    assert not (tmp_path / "shadow").exists()


def test_exec_keeps_python_variables(tmp_path: Path) -> None:  # -I does not change os.environ
    p, w = spawn(
        ["--", sys.executable, "-c", "import os; open('pp', 'w').write(os.environ['PYTHONPATH'])"],
        tmp_path,
        env={**os.environ, "PYTHONPATH": "."},
    )
    go(w)
    assert p.wait(60) == 0
    assert (tmp_path / "pp").read_text() == "."


@pytest.mark.parametrize("command", ["no-such-binary-xyz", "./no-such-file"])
def test_command_not_found_exits_127(tmp_path: Path, command: str) -> None:
    p, w = spawn(["--", command], tmp_path)
    go(w)
    assert p.wait(60) == 127


def test_unexecutable_command_exits_126(tmp_path: Path) -> None:
    (tmp_path / "script.sh").write_text("echo hi > ran\n")  # no exec bit
    p, w = spawn(["--", "./script.sh"], tmp_path)
    go(w)
    assert p.wait(60) == 126
    assert not (tmp_path / "ran").exists()


def test_applies_rlimits(tmp_path: Path) -> None:
    code = (
        "import resource; "
        "open('nofile', 'w').write(str(resource.getrlimit(resource.RLIMIT_NOFILE)[0]))"
    )
    p, w = spawn(["--rlimit-nofile", "64", "--", sys.executable, "-c", code], tmp_path)
    go(w)
    assert p.wait(60) == 0
    assert (tmp_path / "nofile").read_text() == "64"


def test_exec_restores_default_sigpipe(tmp_path: Path) -> None:
    # Python ignores SIGPIPE, and an ignored signal survives exec. A direct spawn would not
    # ignore it, so `yes` must die quietly of SIGPIPE instead of reporting a broken pipe.
    p, w = spawn(["--", "/bin/sh", "-c", "yes 2> err.txt | head -c 1 > /dev/null"], tmp_path)
    go(w)
    assert p.wait(60) == 0
    assert (tmp_path / "err.txt").read_text() == ""


def test_refused_rlimit_is_reported_and_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(which: int, limits: tuple[int, int]) -> None:
        raise ValueError("not allowed")

    monkeypatch.setattr(resource, "setrlimit", refuse)
    r, w = os.pipe()
    os.close(w)  # no "go": main returns without exec
    assert _launch.main(["--go-fd", str(r), "--rlimit-cpu=5", "--rlimit-as=9", "--", "true"]) == 0
    err = capsys.readouterr().err
    assert "swarm-arena: rlimit cpu not supported: not allowed" in err
    assert "swarm-arena: rlimit as not supported: not allowed" in err
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(r)  # main closed the go descriptor


def test_command_is_required() -> None:
    r, w = os.pipe()
    os.close(w)
    try:
        with pytest.raises(SystemExit, match="2"):
            _launch.main(["--go-fd", str(r), "--"])
    finally:
        os.close(r)
