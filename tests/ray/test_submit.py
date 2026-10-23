import importlib.metadata
import re
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

try:
    importlib.metadata.version("ray")
except importlib.metadata.PackageNotFoundError:
    pytest.skip("Ray submission tests require the ray extra", allow_module_level=True)

# isort: split
from swarm_arena.ray._submit import driver_entrypoint, submit_driver, submit_runtime_env


class _FakeClient:
    def __init__(self, *, returns: str) -> None:
        self.returns = returns
        self.address: str | None = None
        self.kwargs: dict[str, Any] = {}

    def __call__(self, address: str | None) -> "_FakeClient":
        self.address = address
        return self

    def submit_job(self, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return self.returns


def test_driver_entrypoint_quotes_paths() -> None:
    assert driver_entrypoint(["run.toml", "plans dir/a.toml"]) == (
        "swarm-arena run --yes -- run.toml 'plans dir/a.toml'"
    )


def test_driver_entrypoint_preserves_shell_metacharacters() -> None:
    files = ["run 'one'.toml", "$(touch marker); plan.toml", "a\nb.toml"]
    assert shlex.split(driver_entrypoint(files)) == ["swarm-arena", "run", "--yes", "--", *files]


def test_submit_runtime_env_never_has_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAY_AUTH_TOKEN", "sentinel-secret")
    assert submit_runtime_env(working_dir=".", pip=["deepretro-swarm"]) == {
        "working_dir": ".",
        "pip": ["deepretro-swarm"],
    }
    assert submit_runtime_env(working_dir=None, pip=[]) is None


@pytest.mark.parametrize(
    ("working_dir", "pip", "expected"),
    [("", [], None), (".", [], {"working_dir": "."}), (None, ["pkg"], {"pip": ["pkg"]})],
)
def test_submit_runtime_env_omits_empty_keys(
    working_dir: str | None, pip: list[str], expected: dict[str, Any] | None
) -> None:
    assert submit_runtime_env(working_dir=working_dir, pip=pip) == expected


def test_submit_runtime_env_copies_pip() -> None:
    specs = ["connector"]
    env = submit_runtime_env(working_dir=None, pip=specs)
    specs.append("later")
    assert env == {"pip": ["connector"]}


@pytest.mark.parametrize("address", ["http://head:8265", None])
@pytest.mark.parametrize("runtime_env", [None, {"working_dir": ".", "pip": ["connector"]}])
def test_submit_driver_calls_job_submission_client(
    monkeypatch: pytest.MonkeyPatch,
    address: str | None,
    runtime_env: Mapping[str, Any] | None,
) -> None:
    fake = _FakeClient(returns="raysubmit_123")
    monkeypatch.setattr("swarm_arena.ray._submit.JobSubmissionClient", fake)
    assert (
        submit_driver(address, "swarm-arena run --yes r.toml p.toml", runtime_env=runtime_env)
        == "raysubmit_123"
    )
    assert fake.address == address
    submission_id = fake.kwargs.pop("submission_id")
    assert re.fullmatch(r"swarm-arena-[0-9a-f]{32}", submission_id)
    assert fake.kwargs == {
        "entrypoint": "swarm-arena run --yes r.toml p.toml",
        "runtime_env": runtime_env,
    }


@pytest.mark.parametrize("plan", ["-plan.toml", "--help", "plans dir/a.toml"])
def test_submit_to_run_parser_round_trip(plan: str) -> None:
    from swarm_arena._cli.main import _parser

    args = _parser().parse_args(["submit", "--yes", "--", "run.toml", plan])
    argv = shlex.split(driver_entrypoint([args.run_toml, *args.plans]))
    run = _parser().parse_args(argv[1:])
    assert run.command == "run"
    assert run.yes
    assert run.run_toml == Path("run.toml")
    assert run.plans == [Path(plan)]


@pytest.mark.parametrize("variable", ["RAY_ADDRESS", "RAY_API_SERVER_ADDRESS"])
def test_explicit_address_refuses_conflicting_environment(
    variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.secrecy import assert_secret_absent

    def unexpected(address: str | None) -> None:
        pytest.fail("address conflicts must be refused before client construction")

    monkeypatch.setattr("swarm_arena.ray._submit.JobSubmissionClient", unexpected)
    monkeypatch.setenv(variable, "http://sentinel-secret:8265")
    with pytest.raises(
        RuntimeError, match="--address conflicts with RAY_ADDRESS/RAY_API_SERVER_ADDRESS"
    ) as caught:
        submit_driver("http://explicit:8265", "command", runtime_env=None)
    assert_secret_absent(caught.value, "sentinel-secret", sanitized=True)


@pytest.mark.parametrize("address", [None, "http://same:8265"])
def test_environment_fallback_or_matching_explicit_address(
    address: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAY_ADDRESS", "http://same:8265")
    monkeypatch.setenv("RAY_API_SERVER_ADDRESS", "http://same:8265")
    fake = _FakeClient(returns="id")
    monkeypatch.setattr("swarm_arena.ray._submit.JobSubmissionClient", fake)
    assert submit_driver(address, "command", runtime_env=None) == "id"
    assert fake.address == address
