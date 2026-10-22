"""Shared fixtures, and collection rules for test paths that need an optional extra."""

import importlib
import importlib.metadata
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import anyio
import pytest
from _fakes.finished import FinishedRun, run_again

from swarm_arena._exec.local import LocalExecutor
from swarm_arena._model.run_config import RunConfig
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._store.local import LocalStore


def _missing(dist: str) -> bool:
    """Return True when the distribution ``dist`` is not installed."""
    try:
        importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return True
    return False


# By distribution, never find_spec: pythonpath = ["tests"] makes tests/ray/ a namespace package.
# -m deselection happens only after import, so a path that imports an extra must not be collected.
collect_ignore: list[str] = []
if _missing("openai-agents"):
    collect_ignore += [
        "llm",
        "testing/test_fake_model.py",
        "e2e/test_rosenbrock_llm.py",
        "e2e/test_autonomous_replan.py",
        "e2e/test_llm_plan_file.py",
        "e2e/test_llm_final_slice.py",
    ]
if _missing("ray"):
    collect_ignore += [
        "ray",
        "e2e/test_rosenbrock_ray.py",
        "e2e/test_ray_cancel.py",
        "e2e/test_ray_dispatch_crash.py",
    ]
if _missing("s3fs"):
    collect_ignore.append("_store/test_s3.py")

# Script fixtures and test helpers are never imported as doctest modules.
collect_ignore_glob = ["*/data/*", "_fakes/*"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def artifact_root(tmp_path: Path) -> str:
    return str(tmp_path / "artifacts")


@pytest.fixture
def artifacts(artifact_root: str) -> ArtifactStore:
    return ArtifactStore(open_store(artifact_root))


@pytest.fixture
def local_executor(tmp_path: Path) -> LocalExecutor:
    return LocalExecutor(workspace_root=str(tmp_path / "ws"))


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / "run")


@pytest.fixture
def run_config(tmp_path: Path, artifact_root: str) -> RunConfig:
    return RunConfig(
        run_root=str(tmp_path / "runs"),
        arena_root=str(tmp_path / "arena"),
        artifact_root=artifact_root,
        lease_ttl_s=60,
    )


def _s3_dependency(module: str) -> Any:
    if os.environ.get("CI"):
        try:
            return importlib.import_module(module)
        except ImportError:
            pytest.fail(f"S3 tests require {module} in CI")
    return pytest.importorskip(module)


@pytest.fixture(scope="session")
def _s3_endpoint() -> Iterator[str]:
    endpoint = os.environ.get("SWARM_ARENA_TEST_S3_ENDPOINT")
    if endpoint:
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username:
            pytest.fail("S3 tests require an http endpoint on 127.0.0.1")
        yield endpoint
        return
    server = _s3_dependency("moto.server").ThreadedMotoServer(
        ip_address="127.0.0.1", port=0, verbose=False
    )
    server.start()
    try:
        host, port = server.get_host_and_port()
        yield f"http://{host}:{port}"
    finally:
        server.stop()


@pytest.fixture
def s3_url(monkeypatch: pytest.MonkeyPatch, _s3_endpoint: str) -> Iterator[str]:
    s3fs = _s3_dependency("s3fs")
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_ENDPOINT_URL_S3": _s3_endpoint,
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_IGNORE_CONFIGURED_ENDPOINT_URLS": "false",
    }.items():
        monkeypatch.setenv(key, value)
    for key in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    fs = s3fs.S3FileSystem(skip_instance_cache=True, use_listings_cache=False)
    try:
        assert fs.s3.meta.endpoint_url == _s3_endpoint
        fs.call_s3("create_bucket", Bucket="swarm-arena-test")
        yield f"s3://swarm-arena-test/{uuid4().hex}"
    finally:
        type(fs).close_session(fs.loop, fs._s3creator)


@pytest.fixture(autouse=True)
def isolated_board_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every default board cache inside its test's temporary root."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))


@pytest.fixture(autouse=True)
def isolated_llm_process_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no process-wide LiteLLM response cache or call logger."""
    # Only modules a test already imported are reset, so core-only tests import nothing.
    litellm = sys.modules.get("litellm")
    if litellm is not None:
        monkeypatch.setattr(litellm, "cache", None)
        monkeypatch.setattr(litellm, "callbacks", [])
    logger = sys.modules.get("swarm_arena.llm._logger")
    if logger is not None:
        monkeypatch.setattr(logger, "_installed_policy", None)


@pytest.fixture
async def finished_run(tmp_path: Path, anyio_backend: str) -> "FinishedRun":
    """Finish the two-round public Rosenbrock example with posted learnings."""
    from swarm_arena import Plan, RoundSpec

    example = (
        (await anyio.to_thread.run_sync(Path(__file__).resolve)).parents[1]
        / "examples"
        / "rosenbrock"
    )
    plan = Plan.load(example / "plan.toml")
    plan = plan.model_copy(deep=True, update={"rounds": [RoundSpec(), RoundSpec()]})
    for team in plan.teams:
        for spec in team.agents:
            spec.manager.config["post_learnings"] = True
    config = RunConfig(
        run_root=str(tmp_path / "runs"),
        arena_root=str(tmp_path / "arena"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    path = tmp_path / "run.toml"
    path.write_text(config.dumps())
    run = FinishedRun(
        config=config, config_path=path, plan=plan, run_id="first", namespace=plan.goal.namespace
    )
    await run_again(run, run_id=run.run_id)
    return run


@pytest.fixture(autouse=True)
def no_cwd_disk_cache() -> Iterator[None]:
    """Reject LiteLLM's default disk directory without importing optional extras."""
    working_dir = Path.cwd()
    yield
    for directory in {working_dir, Path.cwd()}:
        assert not (directory / ".litellm_cache").exists(), (
            "LiteLLM disk cache in cwd: set CachePolicy.disk_dir under tmp_path"
        )
