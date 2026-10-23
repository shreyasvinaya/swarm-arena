import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import anyio
import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    Goal,
    Plan,
    Ref,
    Resources,
    RoundSpec,
    RunConfig,
    TeamSpec,
)
from swarm_arena._engine import manifest as manifest_mod
from swarm_arena._engine.manifest import RunManifest, build_manifest, build_manifest_async
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.catalog import CatalogEntry
from swarm_arena.config import SecretsPolicy


def _plan() -> Plan:
    return Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[
            TeamSpec(
                name="t",
                rationale="Explore",
                agents=[
                    AgentSpec(
                        name="a",
                        rationale="Explore",
                        tool=Ref(path="_fakes.tools:CountingTool"),
                        manager=Ref(path="_fakes.tools:NoopManager"),
                    )
                ],
            )
        ],
        judges=[],
    )


def test_manifest_keys_and_no_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-123")
    m = build_manifest(
        run_id="r",
        config=RunConfig(secrets=SecretsPolicy(allow=["ANTHROPIC_API_KEY"])),
        plans=[_plan()],
        capacity=None,
    )
    assert set(m) == {
        "schema_version",
        "run_id",
        "created_at",
        "versions",
        "host",
        "executor_capacity",
        "run_config_sha256",
        "plans",
        "tools",
        "llm_gateway",
    }
    assert b"sk-secret-123" not in canonical_json(m)
    assert RunManifest.model_validate_json(canonical_json(m)).schema_version == 1


def test_manifest_hashes_capacity_versions_and_base_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, config = _plan(), RunConfig()
    plan.rounds.append(RoundSpec(tool_overrides={"k": 2}))
    plan = plan.model_copy(
        update={
            "converge": ConvergeSpec(
                mode="team", team=plan.teams[0].model_copy(update={"name": "join"})
            )
        }
    )
    monkeypatch.setattr(manifest_mod, "host_id", lambda: "host:boot")
    m = build_manifest(run_id="r", config=config, plans=[plan], capacity=Resources(cpus=2))
    assert m["host"] == {"host_id": "host:boot", "cpu_count": manifest_mod.os.cpu_count()}
    assert set(m["versions"]) == {"swarm_arena", "python", "platform"}
    assert m["executor_capacity"] == Resources(cpus=2).model_dump(mode="json")
    assert m["run_config_sha256"] == sha256_hex(config.dumps().encode())
    assert m["plans"] == [
        {
            "goal_id": "g",
            "revision": 0,
            "sha256": sha256_hex(plan.dumps().encode()),
            "files_sha256": plan.goal.key().files_sha256,
        }
    ]
    from swarm_arena._tooling.staging import tool_identity

    assert len(m["tools"]) == 1
    assert m["tools"][0]["identity"] == tool_identity(plan.teams[0].agents[0].tool, None)


def test_git_commit_and_no_repo(tmp_path: Path) -> None:
    plan = _plan()
    m = build_manifest(run_id="r", config=RunConfig(), plans=[plan], capacity=None)
    done = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    assert m["tools"][0]["git_commit"] == done.stdout.strip()
    plan = Plan.loads(plan.dumps(), base_dir=tmp_path)
    assert (
        build_manifest(run_id="r", config=RunConfig(), plans=[plan], capacity=None)["tools"][0][
            "git_commit"
        ]
        is None
    )


@pytest.mark.parametrize(
    "failure", [OSError("GIT_SECRET"), subprocess.TimeoutExpired("GIT_SECRET", 1)]
)
def test_git_failures_are_bounded_and_private(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def run(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert 0 < kwargs["timeout"] <= 10
        raise failure

    monkeypatch.setattr(manifest_mod, "host_id", lambda: "host:boot")
    monkeypatch.setattr(manifest_mod.subprocess, "run", run)
    m = build_manifest(run_id="r", config=RunConfig(), plans=[_plan()], capacity=None)
    assert m["tools"][0]["git_commit"] is None
    assert b"GIT_SECRET" not in canonical_json(m)


@pytest.mark.parametrize("field", ["versions", "host", "executor_capacity", "plans", "tools"])
@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), {1: "value"}, {"x"}, b"bytes", "\ud800"]
)
def test_manifest_rejects_non_json_subtrees(field: str, bad: Any) -> None:
    raw = build_manifest(run_id="r", config=RunConfig(), plans=[], capacity=None)
    value = {"MANIFEST_SECRET": [bad]}
    raw[field] = [value] if field in ("plans", "tools") else value
    with pytest.raises(ValueError, match=r"JSON|UTF-8|validation") as caught:
        RunManifest.model_validate(raw)
    assert_secret_absent(caught.value, "MANIFEST_SECRET")


def test_manifest_refuses_newer_version() -> None:
    with pytest.raises(ValueError, match="newer"):
        RunManifest.model_validate_json('{"schema_version": 2}')


def test_manifest_tool_identity_excludes_catalog_settable_fields() -> None:
    from swarm_arena._tooling.staging import tool_identity
    from swarm_arena.config import CatalogEntry, PlannerCatalog

    entry = CatalogEntry(
        name="counter",
        kind="tool",
        ref="_fakes.tools:CountingTool",
        description="Count",
        settable={"k": {"type": "integer"}},
    )
    plan = _plan()
    agent = plan.teams[0].agents[0]
    plan.teams[0].agents.append(
        agent.model_copy(
            update={"name": "other", "tool": Ref(path=agent.tool.path, config={"k": 3})}
        )
    )
    config = RunConfig(planner_catalog=PlannerCatalog(entries=[entry]))
    tools = build_manifest(run_id="r", config=config, plans=[plan], capacity=None)["tools"]
    assert len(tools) == 1
    assert tools[0]["identity"] == tool_identity(agent.tool, entry)


def test_manifest_invalid_json_hides_input() -> None:
    malformed = '{"MANIFEST_SECRET":'
    with pytest.raises(ValueError, match="json_invalid") as caught:
        RunManifest.model_validate_json(malformed)
    assert_secret_absent(caught.value, "MANIFEST_SECRET")


@pytest.mark.parametrize("capacity", [None, Resources(cpus=2)])
async def test_async_manifest_preserves_output_and_worker_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capacity: Resources | None
) -> None:
    from swarm_arena.config import PlannerCatalog

    plan = _plan()
    plan.rounds.append(RoundSpec(tool_overrides={"k": 2}))
    plan = plan.model_copy(
        update={
            "converge": ConvergeSpec(
                mode="team", team=plan.teams[0].model_copy(update={"name": "join"})
            )
        }
    )
    plans = [plan, Plan.loads(plan.dumps(), base_dir=tmp_path)]
    config = RunConfig(
        planner_catalog=PlannerCatalog(
            entries=[
                CatalogEntry(
                    name="counter",
                    kind="tool",
                    ref="_fakes.tools:CountingTool",
                    description="Count",
                    settable={"k": {"type": "integer"}},
                )
            ]
        )
    )
    now = manifest_mod.utc_now()
    monkeypatch.setattr(manifest_mod, "utc_now", lambda: now)
    expected = build_manifest(run_id="r", config=config, plans=plans, capacity=capacity)
    loop_thread = threading.get_ident()
    phases: list[str] = []
    real_identity = manifest_mod.tool_identity
    real_git = manifest_mod._git_commit
    real_host = manifest_mod.host_id

    def identity(ref: Ref, entry: CatalogEntry | None) -> str:
        assert threading.current_thread().daemon
        assert threading.get_ident() != loop_thread
        phases.append("identity")
        return real_identity(ref, entry)

    def git(base_dir: Path) -> str | None:
        assert anyio.from_thread.run_sync(threading.get_ident) == loop_thread
        assert not threading.current_thread().daemon
        phases.append("git")
        return real_git(base_dir)

    def host() -> str:
        assert anyio.from_thread.run_sync(threading.get_ident) == loop_thread
        phases.append("host")
        return real_host()

    monkeypatch.setattr(manifest_mod, "tool_identity", identity)
    monkeypatch.setattr(manifest_mod, "_git_commit", git)
    monkeypatch.setattr(manifest_mod, "host_id", host)
    actual = await build_manifest_async(run_id="r", config=config, plans=plans, capacity=capacity)
    assert actual == expected
    assert len(actual["tools"]) == 2
    assert phases == ["identity"] * 4 + ["git", "git", "host"]


@pytest.mark.parametrize("hook", ["validator", "serializer"])
def test_cancelled_manifest_identity_allows_process_exit(tmp_path: Path, hook: str) -> None:
    script = r"""
import threading
from typing import Self

import anyio
from pydantic import field_serializer, model_validator
from _fakes.tools import CountingTool, RecordingExecutor
from swarm_arena import AgentSpec, Goal, Plan, Ref, RunConfig, TeamSpec
from swarm_arena._engine.manifest import build_manifest_async
from swarm_arena._engine.preflight import run_preflight

block = False
entered = threading.Event()

def block_hook(name: str) -> None:
    if block and name == HOOK:
        print(f"{name}_entered daemon={threading.current_thread().daemon}", flush=True)
        entered.set()
        threading.Event().wait()

class BlockingTool(CountingTool):
    class Config(CountingTool.Config):
        @model_validator(mode="after")
        def validate_prerequisites(self) -> Self:
            block_hook("validator")
            return self

        @field_serializer("k")
        def serialize_k(self, value: int) -> int:
            block_hook("serializer")
            return value

async def main() -> None:
    global block
    plan = Plan(
        goal=Goal(id="g", namespace="n"),
        teams=[TeamSpec(name="t", rationale="Explore", agents=[
            AgentSpec(name="a", rationale="Explore",
                      tool=Ref(path="__main__:BlockingTool"),
                      manager=Ref(path="_fakes.tools:NoopManager"))
        ])],
        judges=[],
    )
    config = RunConfig()
    await run_preflight([plan], config, RecordingExecutor())
    print("preflight_passed", flush=True)
    block = True
    scope = anyio.CancelScope()

    async def build() -> None:
        with scope:
            await build_manifest_async(
                run_id="r", config=config, plans=[plan], capacity=None
            )
        print(f"manifest_returned cancel_called={scope.cancel_called}", flush=True)

    async with anyio.create_task_group() as group:
        group.start_soon(build)
        with anyio.fail_after(2):
            while not entered.is_set():
                await anyio.sleep(0.01)
        scope.cancel()

anyio.run(main)
print("interpreter_exiting", flush=True)
"""
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")]),
    }
    try:
        done = subprocess.run(
            [sys.executable, "-B", "-c", f"HOOK = {hook!r}\n" + script],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"manifest cancellation blocked process exit: {exc.stdout!r}")
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    assert done.stdout.splitlines() == [
        "preflight_passed",
        f"{hook}_entered daemon=True",
        "manifest_returned cancel_called=True",
        "interpreter_exiting",
    ]


def _gateway_block() -> dict[str, Any]:
    return {
        "url": "https://gateway.example",
        "cap": "soft",
        "verified": False,
        "failures": ["no maximum request duration is enforced"],
        "terminal_accounting": False,
    }


async def test_manifest_records_gateway_cap() -> None:
    block = _gateway_block()
    for result in (
        build_manifest(run_id="r", config=RunConfig(), plans=[], capacity=None, llm_gateway=block),
        await build_manifest_async(
            run_id="r", config=RunConfig(), plans=[], capacity=None, llm_gateway=block
        ),
    ):
        assert result["llm_gateway"] == block
        assert RunManifest.model_validate_json(canonical_json(result)).llm_gateway is not None


def test_manifest_without_gateway_is_null() -> None:
    raw = build_manifest(run_id="r", config=RunConfig(), plans=[], capacity=None)
    assert raw["llm_gateway"] is None
    del raw["llm_gateway"]
    assert RunManifest.model_validate(raw).llm_gateway is None


@pytest.mark.parametrize(
    "patch",
    [
        {"url": "https://user:MANIFEST_SECRET@example.test"},
        {"url": "https://example.test?key=MANIFEST_SECRET"},
        {"url": "https://example.test#MANIFEST_SECRET"},
        {"url": "MANIFEST_SECRET"},
        {"cap": "MANIFEST_SECRET"},
        {"cap": "hard"},
        {"verified": True},
        {"verified": "MANIFEST_SECRET"},
        {"failures": []},
        {"failures": [123]},
        {"terminal_accounting": "MANIFEST_SECRET"},
        {"master_key": "MANIFEST_SECRET"},
    ],
)
def test_manifest_gateway_rejects_invalid_evidence_without_values(patch: dict[str, Any]) -> None:
    raw = build_manifest(run_id="r", config=RunConfig(), plans=[], capacity=None)
    raw["llm_gateway"] = {**_gateway_block(), **patch}
    with pytest.raises(ValueError, match="validation") as caught:
        RunManifest.model_validate(raw)
    assert_secret_absent(caught.value, "MANIFEST_SECRET")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", "https://u:MANIFEST_SECRET@example.test"),
        ("failures", [123]),
        ("terminal_accounting", "MANIFEST_SECRET"),
    ],
)
def test_manifest_gateway_revalidates_copied_models(field: str, value: Any) -> None:
    from swarm_arena._engine.manifest import _GatewayManifest

    block = _GatewayManifest.model_validate(_gateway_block()).model_copy(update={field: value})
    raw = build_manifest(run_id="r", config=RunConfig(), plans=[], capacity=None)
    raw["llm_gateway"] = block
    with pytest.raises(ValueError, match="validation") as caught:
        RunManifest.model_validate(raw)
    assert_secret_absent(caught.value, "MANIFEST_SECRET")


async def test_manifest_gateway_failure_bounds_survive_both_builders() -> None:
    block = {**_gateway_block(), "failures": ["unsafe\n\x1b" + "x" * 5000] * 15}
    for raw in (
        build_manifest(run_id="r", config=RunConfig(), plans=[], capacity=None, llm_gateway=block),
        await build_manifest_async(
            run_id="r", config=RunConfig(), plans=[], capacity=None, llm_gateway=block
        ),
    ):
        assert raw["llm_gateway"]["failures"] == ["<unknown>"] * 10 + ["and 5 more"]
        assert RunManifest.model_validate_json(canonical_json(raw)).model_dump(mode="json") == raw
