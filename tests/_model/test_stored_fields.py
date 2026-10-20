"""Pin stored field sets to their format versions, including nested models."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from swarm_arena import Plan, RunConfig
from swarm_arena._engine.manifest import RunManifest, build_manifest
from swarm_arena._engine.solution import SolutionDocument
from swarm_arena._exec.executor import Job
from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.lease import OwnerDoc
from swarm_arena._journal.segment import SegmentLink
from swarm_arena._model import migrate
from swarm_arena._model.base import StoredModel
from swarm_arena._model.board_types import Frontier
from swarm_arena._model.migrate import CURRENT, migration
from swarm_arena._model.results import Candidate, Learning, LLMCall
from swarm_arena._model.tooling import ToolOutput, ToolRequest
from swarm_arena._tooling.result_file import ToolResultFile, request_json
from swarm_arena._tooling.staging import plan_script_stage

FIELD_SETS = {
    ("manifest", 1): "1d2a6841f6bef3f68d79b19eba61c50bce6f00f834a7d5b58291a2a6be64fd95",
    ("solution", 1): "a325bf2c0a9df566ae25ad655f29491c59ec671bead59f34048deb4808a1d449",
    ("owner", 1): "84cf8f7b6d0352d6a9dabbb2b5c6d692b9992d8ca6dc367ce61c973f22904faf",
    ("frontier", 1): "42d1364c8cbcf941ba14fc9266d59903ec337f527db13f0fd491893738056dee",
    ("event", 1): "113ce06fe87ea705e6772c41d333ac3166115b8211ef70f14936f234bcaff4b5",
    ("segment", 1): "24a9c075f402a838718212851482617251b3bab08bafa51018f9e4e6d0426f09",
    ("candidate", 1): "5cd32e071d3d68671ff5389518f120123539420ec264eca3e27c8f81ecadbc25",
    ("learning", 1): "c5e234505180affe880f7822eb49708b48bd501ff90903528fc4eb0b7bb478b1",
    ("llm_call", 1): "a1173597b9e95c35e761ccdf608c93809cb0d8164767effc5dba90e58c297c3f",
    ("plan", 1): "72cf827e6a89feaed4f75dd19867c98dbecf9b57ba87cb45abeee78f53e39f89",
    ("run_config", 1): "4e8be36b4a984eb29c94074399a9415d8a0015c236c8493ecaab34b0b79fcc52",
    ("job_started", 1): "9014b7001623ba750fd0c7bd48434c3cb80ff11fbdb669a627d8188ec3257a80",
    ("tool_result", 1): "b0f03da03916731b82f5d6e1a2c6d7e844a1cb0e207c8b29d89bc34dea07841f",
}


@pytest.mark.parametrize(
    "model",
    [
        Candidate,
        Learning,
        LLMCall,
        Plan,
        RunConfig,
        Job.Started,
        ToolResultFile,
        Event,
        SegmentLink,
        OwnerDoc,
        Frontier,
        SolutionDocument,
        RunManifest,
    ],
)
def test_stored_field_sets_require_a_version_bump(model: type[StoredModel]) -> None:
    schema = model.model_json_schema()
    fields = {"root": sorted(schema["properties"])}
    fields.update(
        {
            name: sorted(node["properties"])
            for name, node in schema.get("$defs", {}).items()
            if "properties" in node
        }
    )
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    kind = model.schema_kind
    assert digest == FIELD_SETS[(kind, CURRENT[kind])], (
        "stored fields changed: bump schema_version and add a migration"
    )


def test_workspace_request_field_sets_require_a_version_bump() -> None:
    request = ToolRequest()
    candidate = Candidate(
        id="c",
        run_id="r",
        goal_id="g",
        team="t",
        agent="a",
        round=0,
        request=request,
        output=ToolOutput(status="ok"),
    )
    stage = plan_script_stage(own_files={}, candidate=candidate, effective_files={})
    assert stage.candidate is not None
    snapshots = {
        ("request", 1): {"schema_version", "params", "round", "round_overrides", "replicate"},
        ("candidate_request", 1): {
            "schema_version",
            "params",
            "round",
            "round_overrides",
            "replicate",
            "parents",
        },
        ("candidate_output", 1): {"schema_version", "status", "partial", "data", "usage", "files"},
    }
    for kind, actual in (
        ("request", request_json(request)),
        ("candidate_request", stage.candidate.request),
        ("candidate_output", stage.candidate.output),
    ):
        assert set(actual) == snapshots[(kind, CURRENT[kind])]


@pytest.mark.usefixtures("isolated_migrations")
@pytest.mark.parametrize("model", [Event, SegmentLink])
def test_journal_stored_models_apply_registered_migration(
    monkeypatch: pytest.MonkeyPatch, model: type[Event] | type[SegmentLink]
) -> None:
    raw: dict[str, Any]
    if model is Event:
        raw = Event(
            epoch=1,
            seq=1,
            ts=datetime(2026, 10, 18, tzinfo=UTC),
            run_id="r",
            context_id="r",
            type=EventType.RUN_STARTED,
        ).model_dump(mode="json")
    else:
        raw = {"schema_version": 1, "link": None}
    kind = model.schema_kind
    monkeypatch.setattr(migrate, "CURRENT", {**migrate.CURRENT, kind: 2})
    calls: list[dict[str, Any]] = []

    @migration(kind, 1)
    def step(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(data.copy())
        if kind == "event":
            return {**data, "payload": {"migrated": True}}
        return {**data, "link": "journal/e1-1.jsonl"}

    upgraded = model.model_validate_json(json.dumps(raw))
    assert upgraded.schema_version == 2
    assert calls == [raw]
    if isinstance(upgraded, Event):
        assert upgraded.payload == {"migrated": True}
    else:
        assert upgraded.link == "journal/e1-1.jsonl"
    assert raw["schema_version"] == 1


@pytest.mark.usefixtures("isolated_migrations")
def test_owner_applies_registered_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {
        "schema_version": 1,
        "version": 3,
        "epoch": 2,
        "owner_id": "o1",
        "host": "h",
        "expires_at": "2026-10-18T13:00:00.000Z",
        "head": {"epoch": 1, "seq": 3, "segment": "journal/e1-3.jsonl"},
    }
    monkeypatch.setattr(migrate, "CURRENT", {**migrate.CURRENT, "owner": 2})
    calls: list[dict[str, Any]] = []

    @migration("owner", 1)
    def step(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(data.copy())
        return {**data, "host": "migrated-host"}

    upgraded = OwnerDoc.model_validate_json(json.dumps(raw))
    assert upgraded.schema_version == 2
    assert upgraded.host == "migrated-host"
    assert upgraded.version == 3
    assert upgraded.epoch == 2
    assert calls == [raw]
    assert raw["schema_version"] == 1


@pytest.mark.usefixtures("isolated_migrations")
def test_frontier_applies_registered_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = Frontier(reviews=["old"]).model_dump(mode="json")
    monkeypatch.setattr(migrate, "CURRENT", {**migrate.CURRENT, "frontier": 2})
    calls: list[dict[str, Any]] = []

    @migration("frontier", 1)
    def step(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(data.copy())
        return {**data, "reviews": ["migrated"]}

    upgraded = Frontier.model_validate_json(json.dumps(raw))
    assert upgraded.schema_version == 2
    assert upgraded.reviews == ["migrated"]
    assert calls == [raw]
    assert raw["schema_version"] == 1
    assert raw["reviews"] == ["old"]


def test_frontier_refuses_newer_version() -> None:
    with pytest.raises(ValueError, match="newer"):
        Frontier.model_validate_json('{"schema_version": 2}')


@pytest.mark.usefixtures("isolated_migrations")
def test_solution_applies_registered_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {
        "schema_version": 1,
        "run_id": "r",
        "goal_id": "g",
        "status": "no_result",
        "reason": None,
        "finalists": [],
        "parts": [],
        "replicate_stats": [],
        "best_of_n": None,
        "plan_revisions": [],
        "manifest": {"git_commit": None},
        "source_head": {"epoch": 1, "seq": 1, "segment": "journal/e1-1.jsonl"},
    }
    monkeypatch.setattr(migrate, "CURRENT", {**migrate.CURRENT, "solution": 2})
    calls: list[dict[str, Any]] = []

    @migration("solution", 1)
    def step(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(data.copy())
        return {**data, "reason": "migrated"}

    upgraded = SolutionDocument.model_validate_json(json.dumps(raw))
    assert upgraded.schema_version == 2
    assert upgraded.reason == "migrated"
    assert upgraded.manifest == {"git_commit": None}
    assert calls == [raw]
    assert raw["schema_version"] == 1


@pytest.mark.usefixtures("isolated_migrations")
def test_manifest_applies_registered_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = build_manifest(run_id="r", config=RunConfig(), plans=[], capacity=None)
    monkeypatch.setattr(migrate, "CURRENT", {**migrate.CURRENT, "manifest": 2})

    @migration("manifest", 1)
    def step(data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "run_id": "migrated"}

    upgraded = RunManifest.model_validate_json(json.dumps(raw))
    assert upgraded.schema_version == 2
    assert upgraded.run_id == "migrated"
    assert raw["schema_version"] == 1
