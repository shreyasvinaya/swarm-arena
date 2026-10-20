import subprocess
from pathlib import Path
from typing import Any

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
from swarm_arena._engine.manifest import RunManifest, build_manifest
from swarm_arena._model.canonical import canonical_json, sha256_hex
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
