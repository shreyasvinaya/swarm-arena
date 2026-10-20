"""Require portable data and limits before serialization or execution."""

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena import Goal, Plan, Ref, RunConfig
from swarm_arena._exec.executor import Job
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Resources, RoundSpec
from swarm_arena._model.results import CheckResult
from swarm_arena._model.schema import compile_object_schema
from swarm_arena._model.settings import GatewayConfig
from swarm_arena._model.tooling import Estimate, ParentRef, ToolOutput, ToolRequest, Usage

SECRET = "e70ac96c-3fcb-49c4-83bd-d89c203f1574"


@pytest.mark.parametrize("bad", [None, float("inf"), float("nan"), datetime.now(UTC), {1}, b"x"])
@pytest.mark.parametrize("kind", ["inputs", "metadata", "config", "overrides", "output"])
def test_stored_freeform_data_is_representable(kind: str, bad: Any) -> None:
    data = {SECRET: [{"value": bad}]}
    if kind == "output" and bad is None:
        output = ToolOutput(status="ok", data=data)
        assert ToolOutput.model_validate_json(output.model_dump_json()) == output
        return

    def action() -> None:
        if kind in {"inputs", "metadata"}:
            Goal.model_validate({"id": "g", "namespace": "n", kind: data})
        elif kind == "config":
            Ref(path="builtins:str", config=data)
        elif kind == "overrides":
            RoundSpec(tool_overrides=data)
        else:
            ToolOutput(status="ok", data=data)

    with pytest.raises(ValidationError) as caught:
        action()
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {"x": object()}])
def test_check_details_require_finite_json(bad: Any) -> None:
    with pytest.raises(ValidationError):
        CheckResult(check="c", passed=True, details={SECRET: bad})


@pytest.mark.parametrize("kind", ["edits", "inputs", "overrides", "output", "parent", "compiled"])
def test_json_bound_strings_require_utf8(kind: str) -> None:
    bad = SECRET + "\ud800"

    def action() -> None:
        if kind == "edits":
            ToolRequest(edits={"a.txt": bad})
        elif kind in {"inputs", "overrides"}:
            ToolRequest.model_validate(
                {"inputs" if kind == "inputs" else "round_overrides": {"x": bad}}
            )
        elif kind == "output":
            ToolOutput(status="ok", data={"x": bad})
        elif kind == "parent":
            ParentRef(candidate_id="c", run_id="r", tool_identity="t", data={"x": bad})
        else:
            model = compile_object_schema("Params", {"x": {"type": "string"}})
            model.model_validate({"x": bad})

    with pytest.raises(ValidationError) as caught:
        action()
    assert_secret_absent(caught.value, SECRET, sanitized=True)


def test_output_nesting_is_bounded_before_dumping() -> None:
    value: Any = 0
    for _ in range(300):
        value = [value]
    with pytest.raises(ValidationError):
        ToolOutput(status="ok", data={"x": value})
    value = {"x": [1]}
    for _ in range(90):
        value = {"x": value}
    out = ToolOutput(status="ok", data=value)
    assert ToolOutput.model_validate_json(out.model_dump_json()) == out


@pytest.mark.parametrize("field", list(Usage.model_fields))
def test_usage_rejects_negative_amounts(field: str) -> None:
    with pytest.raises(ValidationError):
        Usage(**{field: -1})


@pytest.mark.parametrize("field", list(Estimate.model_fields))
@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan")])
def test_estimates_are_finite_nonnegative(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        Estimate(**{field: bad})


@pytest.mark.parametrize(
    "field", ["schedule_timeout_s", "sync_interval_s", "lease_ttl_s", "settle_timeout_s"]
)
@pytest.mark.parametrize("bad", [-1, 0, float("inf"), float("nan")])
def test_run_timers_are_finite_positive(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({field: bad})


@pytest.mark.parametrize(
    "field", ["request_timeout_s", "key_duration_s", "default_estimated_output_tokens"]
)
@pytest.mark.parametrize("bad", [-1, 0, float("inf"), float("nan")])
def test_gateway_limits_are_finite_positive(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate({"url": "http://localhost", field: bad})


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_safe_base_refuses_nonfinite_resource_and_budget_floats(bad: float) -> None:
    with pytest.raises(ValidationError):
        Resources(memory_gib=bad)
    with pytest.raises(ValidationError):
        RunConfig(limits={"max_usd": bad})


@pytest.mark.parametrize("kind", ["root", "candidate", "manifest", "overlay"])
def test_all_manifests_validate_hashes(kind: str) -> None:
    data = {"a.txt": SECRET}

    def action() -> None:
        if kind == "root":
            Job.Stage(root=data)
        elif kind == "candidate":
            Job.CandidateFiles(request={}, output={}, files=data)
        elif kind == "manifest":
            FileSet(manifest=data)
        else:
            ParentRef(candidate_id="c", run_id="r", tool_identity="t", overlay=data)

    with pytest.raises(ValidationError) as caught:
        action()
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize("seed", range(50))
def test_generated_plan_and_config_roundtrip_exactly(seed: int) -> None:
    rng = random.Random(seed)

    def value(depth: int) -> Any:
        leaves: list[Any] = [
            False,
            True,
            rng.randint(-(2**63), 2**63 - 1),
            rng.uniform(-1e10, 1e10),
            "λ\n\u007f",
        ]
        if depth:
            leaves.extend([[value(depth - 1), value(depth - 1)], {"a.b": value(depth - 1)}])
        return rng.choice(leaves)

    data = {"nested": value(3)}
    ref = Ref(path="builtins:str", config=data, name="score")
    plan = Plan(
        goal=Goal(id="g", namespace="n", inputs=data, metadata=data),
        teams=[],
        judges=[ref],
        rounds=[RoundSpec(tool_overrides=data)],
    )
    loaded = Plan.loads(plan.dumps(), base_dir=plan.base_dir)
    assert loaded == plan
    assert loaded.goal.key() == plan.goal.key()
    cfg = RunConfig(executor=ref)
    assert RunConfig.loads(cfg.dumps()) == cfg
    assert json.loads(cfg.model_dump_json()) == json.loads(
        RunConfig.loads(cfg.dumps()).model_dump_json()
    )


@pytest.mark.parametrize("field", list(Usage.model_fields))
def test_negative_usage_in_result_file_is_a_run_error(tmp_path: Path, field: str) -> None:
    from swarm_arena._model.tooling import CompletedRun
    from swarm_arena._tooling.result_file import tool_output_from_result

    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/result.json").write_text(json.dumps({"data": {}, "usage": {field: -1}}))
    now = datetime.now(UTC)
    run = CompletedRun(
        returncode=0,
        status="ok",
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        started_at=now,
        ended_at=now,
    )
    output = tool_output_from_result(tmp_path, run)
    assert output.status == "error"
    note = run.stderr_path.read_text()
    assert field in note
    assert "greater_than_equal" in note


def test_shared_json_helper_refuses_cycles_and_preserves_values() -> None:
    from swarm_arena._model.data import finite_json

    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="nesting"):
        finite_json(cycle)
    value = {"ok": [None, 2.5, "λ"]}
    assert finite_json(value) is value
