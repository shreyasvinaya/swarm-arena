import traceback
from pathlib import Path

import pytest
from _fakes.tools import make_job

from swarm_arena._exec.env import BASE_ENV_NAMES, build_env, read_secrets_file, tool_env_vars
from swarm_arena.tool import Command


def test_read_secrets_file(tmp_path: Path) -> None:
    (tmp_path / "s.env").write_text("# comment\n\nA=1\n B = x=y \n")
    assert read_secrets_file(str(tmp_path / "s.env")) == {"A": "1", "B": "x=y"}


def test_bad_secrets_line_never_echoes_value(tmp_path: Path) -> None:
    (tmp_path / "s.env").write_text("A=1\nSECRETVALUE\n")
    with pytest.raises(ValueError, match="line 2") as exc:
        read_secrets_file(str(tmp_path / "s.env"))
    assert "SECRETVALUE" not in str(exc.value)
    assert "SECRETVALUE" not in "".join(traceback.format_exception(exc.value))
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
    path.write_text(f"# comment\n{key}=sentinel-secret\n")
    with pytest.raises(ValueError, match=r"line 2: expected KEY=VALUE") as exc:
        read_secrets_file(str(path))
    assert "sentinel-secret" not in "".join(traceback.format_exception(exc.value))


def test_secrets_file_preserves_values_and_last_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "s.env"
    path.write_text("  # comment\nA=old\nA=new\n_EMPTY=\nX=foo#bar\nY='quoted'\n")
    assert read_secrets_file(str(path)) == {
        "A": "new",
        "_EMPTY": "",
        "X": "foo#bar",
        "Y": "'quoted'",
    }


@pytest.mark.parametrize("error", [OSError("sentinel-secret"), UnicodeError("sentinel-secret")])
def test_unreadable_secrets_file_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail_read(self: Path, *, encoding: str) -> str:
        raise error

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(ValueError, match="cannot read secrets file") as exc:
        read_secrets_file("s.env")
    assert "sentinel-secret" not in "".join(traceback.format_exception(exc.value))


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

    def read(self: Path, *, encoding: str) -> str:
        reads.append(self)
        return "FROM_FILE=file\nSECOND=two\nUNLISTED=hidden\nNODE=file\n"

    monkeypatch.setattr(Path, "read_text", read)
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

    def read(self: Path, *, encoding: str) -> str:
        nonlocal reads
        reads += 1
        return ""

    monkeypatch.setattr(Path, "read_text", read)
    env = build_env(
        make_job(secrets_file="s.env", secret_env=("THIRD",)),
        Command(argv=["x"], env=["FIRST", "SECOND"]),
        workspace=tmp_path,
        node_env={},
    )
    assert reads == 1
    assert env == {"SWARM_ARENA_RESULT": str(tmp_path / "swarm/result.json")}
