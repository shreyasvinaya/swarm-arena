from pathlib import Path

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import make_job

from swarm_arena._exec import env as exec_env
from swarm_arena._exec.env import BASE_ENV_NAMES, build_env, read_secrets_file, tool_env_vars
from swarm_arena.tool import Command


def test_read_secrets_file(tmp_path: Path) -> None:
    (tmp_path / "s.env").write_text("# comment\n\nA=1\n B = x=y \n")
    assert read_secrets_file(str(tmp_path / "s.env")) == {"A": "1", "B": "x=y"}


def test_bad_secrets_line_never_echoes_value(tmp_path: Path) -> None:
    (tmp_path / "s.env").write_text("A=1\n7d271c19-62ce-491c-90cb-c12c415563d1\n")
    with pytest.raises(ValueError, match="line 2") as exc:
        read_secrets_file(str(tmp_path / "s.env"))
    assert_secret_absent(exc.value, "7d271c19-62ce-491c-90cb-c12c415563d1")
    with pytest.raises(ValueError, match="secrets file not found"):
        read_secrets_file(str(tmp_path / "none.env"))


def test_tool_env_vars() -> None:
    assert tool_env_vars(
        run_id="r", goal_id="g", candidate_id="c", team="t", agent="a", round=2
    ) == {
        "SWARM_ARENA_RUN_ID": "r",
        "SWARM_ARENA_GOAL_ID": "g",
        "SWARM_ARENA_CANDIDATE_ID": "c",
        "SWARM_ARENA_TEAM": "t",
        "SWARM_ARENA_AGENT": "a",
        "SWARM_ARENA_ROUND": "2",
    }


def test_build_env_layers(tmp_path: Path) -> None:
    (tmp_path / "s.env").write_text("FROM_FILE=f\nAPI_KEY=file\n")
    node = {"PATH": "/bin", "HOME": "/h", "SECRET_X": "s", "OTHER": "o", "API_KEY": "node"}
    job = make_job(
        secret_env=("SECRET_X",),
        secrets_file=str(tmp_path / "s.env"),
        env={"SWARM_ARENA_RUN_ID": "r"},
        secret_values={"API_KEY": "virtual"},
    )
    env = build_env(
        job,
        Command(argv=["x"], env=["API_KEY", "FROM_FILE", "UNSET_NAME"]),
        workspace=tmp_path,
        node_env=node,
    )
    assert env == {
        "PATH": "/bin",
        "HOME": "/h",
        "SECRET_X": "s",
        "FROM_FILE": "f",
        "API_KEY": "virtual",
        "SWARM_ARENA_RUN_ID": "r",
        "SWARM_ARENA_RESULT": str(tmp_path / "swarm" / "result.json"),
    }
    plain = build_env(
        make_job(), Command(argv=["x"], env=["API_KEY"]), workspace=tmp_path, node_env=node
    )
    assert plain["API_KEY"] == "node"
    assert "OTHER" not in plain


@pytest.mark.parametrize("key", ["", "1KEY", "BAD-KEY", "BAD KEY", "é", "A\x00B"])
def test_bad_secrets_key_is_sanitized(tmp_path: Path, key: str) -> None:
    path = tmp_path / "s.env"
    path.write_text(f"# comment\n{key}=0996cfd4-7bda-48c7-938c-cda7b0dd532d\n")
    with pytest.raises(ValueError, match=r"line 2: expected KEY=VALUE") as exc:
        read_secrets_file(str(path))
    assert_secret_absent(exc.value, "0996cfd4-7bda-48c7-938c-cda7b0dd532d")


def test_secrets_file_preserves_values_and_last_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "s.env"
    path.write_text("  # comment\nA=old\nA=new\n_EMPTY=\nX=foo#bar\nY='quoted'\n")
    assert read_secrets_file(str(path)) == {
        "A": "new",
        "_EMPTY": "",
        "X": "foo#bar",
        "Y": "'quoted'",
    }


@pytest.mark.parametrize(
    "error",
    [
        OSError("0996cfd4-7bda-48c7-938c-cda7b0dd532d"),
        UnicodeError("0996cfd4-7bda-48c7-938c-cda7b0dd532d"),
    ],
)
def test_unreadable_secrets_file_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    import os

    def fail_read(fd: int, size: int) -> bytes:
        raise error

    path = tmp_path / "s.env"
    path.write_text("A=1")
    monkeypatch.setattr(os, "read", fail_read)
    with pytest.raises(ValueError, match="cannot read secrets file") as exc:
        read_secrets_file(str(path))
    assert_secret_absent(exc.value, "0996cfd4-7bda-48c7-938c-cda7b0dd532d")


def test_build_env_base_names_and_overrides(tmp_path: Path) -> None:
    assert BASE_ENV_NAMES == ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TZ")
    node = {name: name for name in BASE_ENV_NAMES} | {"OTHER": "hidden"}
    job = make_job(env={"PATH": "driver", "SWARM_ARENA_RESULT": "wrong"})
    env = build_env(job, Command(argv=["x"]), workspace=tmp_path, node_env=node)
    assert env == {
        **{name: name for name in BASE_ENV_NAMES},
        "PATH": "driver",
        "SWARM_ARENA_RESULT": str(tmp_path / "swarm/result.json"),
    }
    assert node["PATH"] == "PATH"
    assert job.env["SWARM_ARENA_RESULT"] == "wrong"
    final = build_env(
        make_job(secret_values={"SWARM_ARENA_RESULT": "final"}),
        Command(argv=["x"]),
        workspace=tmp_path,
        node_env={},
    )
    assert final == {"SWARM_ARENA_RESULT": "final"}


def test_build_env_reads_file_once_only_when_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[Path] = []

    def read(path: str) -> dict[str, str]:
        reads.append(Path(path))
        return {"FROM_FILE": "file", "SECOND": "two", "UNLISTED": "hidden", "NODE": "file"}

    monkeypatch.setattr(exec_env, "read_secrets_file", read)
    path = str(tmp_path / "s.env")
    job = make_job(secrets_file=path, secret_env=("NODE",))
    command = Command(argv=["x"], env=["NODE"])
    env = build_env(job, command, workspace=tmp_path, node_env={"NODE": ""})
    assert env["NODE"] == ""
    assert reads == []
    assert build_env(
        make_job(secrets_file=path), Command(argv=["x"]), workspace=tmp_path, node_env={}
    )
    assert reads == []
    env = build_env(
        make_job(secrets_file=path, secret_env=("SECOND", "MISSING")),
        Command(argv=["x"], env=["NODE", "FROM_FILE", "FROM_FILE"]),
        workspace=tmp_path,
        node_env={"NODE": "node"},
    )
    assert reads == [Path(path)]
    assert env == {
        "NODE": "node",
        "FROM_FILE": "file",
        "SECOND": "two",
        "SWARM_ARENA_RESULT": str(tmp_path / "swarm/result.json"),
    }


def test_build_env_reads_empty_file_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = 0

    def read(path: str) -> dict[str, str]:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(exec_env, "read_secrets_file", read)
    env = build_env(
        make_job(secrets_file="s.env", secret_env=("THIRD",)),
        Command(argv=["x"], env=["FIRST", "SECOND"]),
        workspace=tmp_path,
        node_env={},
    )
    assert reads == 1
    assert env == {"SWARM_ARENA_RESULT": str(tmp_path / "swarm/result.json")}


@pytest.mark.parametrize("source", ["fifo", "device"])
def test_nonregular_secrets_source_is_refused_promptly(tmp_path: Path, source: str) -> None:
    import os
    import subprocess
    import sys

    path = tmp_path / "s.env" if source == "fifo" else Path("/dev/null")
    if source == "fifo":
        os.mkfifo(path)
    script = """
import sys
from _fakes.secrecy import assert_secret_absent
from swarm_arena._exec.env import read_secrets_file
try:
    read_secrets_file(sys.argv[1])
except ValueError as error:
    assert 'regular file' in str(error)
    assert_secret_absent(error, 'bd1281fb-d6af-4260-b9c9-f806bef9dfeb')
else:
    raise AssertionError('non-regular source accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_secrets_file_read_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from _fakes.read_counts import count_reads

    secret = "bd1281fb-d6af-4260-b9c9-f806bef9dfeb"
    path = tmp_path / "s.env"
    path.write_bytes(f"KEY={secret}".encode() + b"x" * (2 * 2**20))
    counts: list[int] = []
    with count_reads(monkeypatch, counts), pytest.raises(ValueError, match="1 MiB") as exc:
        read_secrets_file(str(path))
    assert sum(counts) == 2**20 + 1
    assert_secret_absent(exc.value, secret)


def test_secrets_file_at_byte_limit_and_regular_symlink(tmp_path: Path) -> None:
    path = tmp_path / "s.env"
    value = "é" * ((2**20 - 4) // 2)
    path.write_bytes(f"KEY={value}".encode())
    assert path.stat().st_size == 2**20
    assert read_secrets_file(str(path)) == {"KEY": value}
    link = tmp_path / "link.env"
    link.symlink_to(path)
    assert read_secrets_file(str(link)) == {"KEY": value}


def test_invalid_utf8_secrets_is_sanitized(tmp_path: Path) -> None:
    secret = "bd1281fb-d6af-4260-b9c9-f806bef9dfeb"
    path = tmp_path / "s.env"
    path.write_bytes(f"KEY={secret}".encode() + b"\xff")
    with pytest.raises(ValueError, match="cannot read secrets file") as exc:
        read_secrets_file(str(path))
    assert_secret_absent(exc.value, secret)
