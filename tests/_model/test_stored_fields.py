"""Pin stored field sets to their format versions, including nested models."""

import hashlib
import json

import pytest

from swarm_arena import Plan, RunConfig
from swarm_arena._exec.executor import Job
from swarm_arena._model.base import StoredModel
from swarm_arena._model.migrate import CURRENT
from swarm_arena._model.results import Candidate, Learning, LLMCall
from swarm_arena._model.tooling import ToolOutput, ToolRequest
from swarm_arena._tooling.result_file import ToolResultFile, request_json
from swarm_arena._tooling.staging import plan_script_stage

FIELD_SETS = {
    ("candidate", 1): "5cd32e071d3d68671ff5389518f120123539420ec264eca3e27c8f81ecadbc25",
    ("learning", 1): "c5e234505180affe880f7822eb49708b48bd501ff90903528fc4eb0b7bb478b1",
    ("llm_call", 1): "a1173597b9e95c35e761ccdf608c93809cb0d8164767effc5dba90e58c297c3f",
    ("plan", 1): "72cf827e6a89feaed4f75dd19867c98dbecf9b57ba87cb45abeee78f53e39f89",
    ("run_config", 1): "4e8be36b4a984eb29c94074399a9415d8a0015c236c8493ecaab34b0b79fcc52",
    ("job_started", 1): "9014b7001623ba750fd0c7bd48434c3cb80ff11fbdb669a627d8188ec3257a80",
    ("tool_result", 1): "b0f03da03916731b82f5d6e1a2c6d7e844a1cb0e207c8b29d89bc34dea07841f",
}


@pytest.mark.parametrize(
    "model", [Candidate, Learning, LLMCall, Plan, RunConfig, Job.Started, ToolResultFile]
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
