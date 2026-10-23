"""Pin stored field sets to their format versions, including nested models."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from swarm_arena import Plan, RunConfig
from swarm_arena._board.human_objects import HumanObject
from swarm_arena._board.pointers import RunPointer
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
from swarm_arena._scoring.script_check import _CheckResultFile
from swarm_arena._scoring.script_judge import _JudgeResultFile
from swarm_arena._tooling.result_file import (
    CandidateOutputFile,
    CandidateRequestFile,
    RequestFile,
    ToolResultFile,
    request_json,
)
from swarm_arena._tooling.staging import plan_script_stage

FIELD_SETS = {
    ("human_object", 1): "31bace63e96bd4af455f20385d1f22643640dffc672fe0d047d0330337a812be",
    ("run_pointer", 1): "4505b29fdc14a4b2eacd9558c7518f4d0784f21a8573ad56b808ede57eabc01f",
    ("request", 1): "9e0727fd1588041cb6f5f28c1f41d72031aecd71edc393856d184bcd11924e97",
    ("candidate_request", 1): "9e0727fd1588041cb6f5f28c1f41d72031aecd71edc393856d184bcd11924e97",
    ("candidate_output", 1): "71060b6a928593fec41a4c0c5b316e670225549db0f650d324f6c8cf22de4e39",
    ("check_result", 1): "ab48eb85724967a6c6862ddc3ae3e09da29dd534ed83de29811007dba709443f",
    ("judge_result", 1): "655351cea722865ef47b488b51a1adfde5446c0dd3548d9aea8d22c6ff454770",
    ("manifest", 1): "937d60c769ad19cdceb6cd56f0a4dcd1339ef21391823bf253428c5c989f5689",
    ("solution", 1): "a325bf2c0a9df566ae25ad655f29491c59ec671bead59f34048deb4808a1d449",
    ("owner", 1): "84cf8f7b6d0352d6a9dabbb2b5c6d692b9992d8ca6dc367ce61c973f22904faf",
    ("frontier", 1): "42d1364c8cbcf941ba14fc9266d59903ec337f527db13f0fd491893738056dee",
    ("event", 1): "113ce06fe87ea705e6772c41d333ac3166115b8211ef70f14936f234bcaff4b5",
    ("segment", 1): "24a9c075f402a838718212851482617251b3bab08bafa51018f9e4e6d0426f09",
    ("candidate", 1): "c9f2c8fc941f5450e39e72dae59f69f9c14481e30c88230971ff030fff6caf09",
    ("learning", 1): "c5e234505180affe880f7822eb49708b48bd501ff90903528fc4eb0b7bb478b1",
    ("llm_call", 1): "a1173597b9e95c35e761ccdf608c93809cb0d8164767effc5dba90e58c297c3f",
    ("plan", 1): "72cf827e6a89feaed4f75dd19867c98dbecf9b57ba87cb45abeee78f53e39f89",
    ("run_config", 1): "4e8be36b4a984eb29c94074399a9415d8a0015c236c8493ecaab34b0b79fcc52",
    ("job_started", 1): "237fdb12287452b9309f010994d452813755353410e771cac98d04ad65f79fe9",
    ("tool_result", 1): "b0f03da03916731b82f5d6e1a2c6d7e844a1cb0e207c8b29d89bc34dea07841f",
}


@pytest.mark.parametrize(
    "model",
    [
        RunPointer,
        HumanObject,
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
        _CheckResultFile,
        _JudgeResultFile,
        RequestFile,
        CandidateRequestFile,
        CandidateOutputFile,
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
@pytest.mark.parametrize("model", [RequestFile, CandidateRequestFile, CandidateOutputFile])
@pytest.mark.parametrize("explicit_version", [False, True])
def test_workspace_envelopes_apply_migration_in_every_reader(
    monkeypatch: pytest.MonkeyPatch, model: type[StoredModel], explicit_version: bool
) -> None:
    from swarm_arena._tooling.result_file import _workspace_envelope

    kind = model.schema_kind
    field = "data" if kind == "candidate_output" else "params"
    raw: dict[str, Any] = {field: {"original": None}}
    if explicit_version:
        raw["schema_version"] = 1
    monkeypatch.setattr(migrate, "CURRENT", {**migrate.CURRENT, kind: 2})
    calls: list[dict[str, Any]] = []

    @migration(kind, 1)
    def step(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(data.copy())
        return {**data, field: {"migrated": None}}

    loaded = model.model_validate_json(json.dumps(raw))
    assert loaded.schema_version == 2
    upgraded = _workspace_envelope(model, raw)
    assert upgraded == {"schema_version": 2, field: {"migrated": None}}
    if kind == "request":
        assert Job.Stage(request_json=raw).request_json == upgraded
    else:
        staged = Job.CandidateFiles(
            request=raw if kind == "candidate_request" else {},
            output=raw if kind == "candidate_output" else {},
            files={},
        )
        assert (staged.output if kind == "candidate_output" else staged.request) == upgraded
    assert calls == [raw, raw, raw]
    assert raw[field] == {"original": None}


@pytest.mark.usefixtures("isolated_migrations")
def test_function_runner_applies_request_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena.run_function import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / "swarm").mkdir()
    raw = {"schema_version": 1, "params": {"a": 2, "b": 3}}
    (tmp_path / "swarm/request.json").write_text(json.dumps(raw))
    monkeypatch.setattr(migrate, "CURRENT", {**migrate.CURRENT, "request": 2})
    calls: list[dict[str, Any]] = []

    @migration("request", 1)
    def step(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(data.copy())
        return {**data, "params": {"a": 20, "b": 3}}

    assert main(["_fakes.tools:add"]) == 0
    assert json.loads((tmp_path / "swarm/result.json").read_bytes())["data"] == {"sum": 23}
    assert calls == [raw]


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
