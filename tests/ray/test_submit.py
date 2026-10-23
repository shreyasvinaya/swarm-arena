import importlib.metadata
import shlex
from collections.abc import Mapping
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
        "swarm-arena run --yes run.toml 'plans dir/a.toml'"
    )


def test_driver_entrypoint_preserves_shell_metacharacters() -> None:
    files = ["run 'one'.toml", "$(touch marker); plan.toml", "a\nb.toml"]
    assert shlex.split(driver_entrypoint(files)) == ["swarm-arena", "run", "--yes", *files]


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
    assert fake.kwargs == {
        "entrypoint": "swarm-arena run --yes r.toml p.toml",
        "runtime_env": runtime_env,
    }
