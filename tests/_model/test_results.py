import pickle
from typing import Any

import pytest
from _fakes.tools import cand
from pydantic import ValidationError

from swarm_arena._model.results import (
    AgentStatus,
    Candidate,
    CheckResult,
    Learning,
    LLMCall,
    Score,
    _finite_key,
)
from swarm_arena._model.tooling import Usage


def test_gated_out_is_derived_and_never_stored() -> None:
    c = cand(checks=[CheckResult(check="a", passed=True), CheckResult(check="b", passed=False)])
    assert c.gated_out
    assert not cand().gated_out
    assert "gated_out" not in c.model_dump()
    assert Candidate.model_validate_json(c.model_dump_json()) == c


def test_candidate_newer_schema_refused() -> None:
    with pytest.raises(ValidationError, match="newer"):
        Candidate.model_validate({**cand().model_dump(mode="json"), "schema_version": 2})


def test_candidate_overlay_paths_checked() -> None:
    for overlay in ({"/abs": "a" * 64}, {"../x": "a" * 64}, {"swarm/x": "a" * 64}):
        with pytest.raises(ValidationError):
            cand(overlay=overlay)


def test_score_helpers() -> None:
    assert (Score.minimize(0.25).key, Score.minimize(0.25).value) == ((-0.25,), 0.25)
    assert Score.maximize(3.0, verdict="good", notes="n").key == (3.0,)
    assert Score(key=(1.0,), judge="j", rubric_hash="r", panel_hash="p").identity == ("j", "r", "p")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_score_rejects_nonfinite(bad: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        Score(key=(1.0, bad))
    with pytest.raises(ValidationError, match="finite"):
        Score.minimize(bad)


def test_score_needs_a_key_value() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        Score(key=())


def test_learning_limits() -> None:
    ok = {
        "id": "L1",
        "namespace": "n",
        "run_id": "r",
        "goal_id": "g",
        "round": 1,
        "author": "t/a",
        "title": "x" * 120,
        "text": "y" * 1500,
        "evidence": ["c1"],
    }
    assert Learning.model_validate(ok).title == "x" * 120
    for change in (
        {"title": "x" * 121},
        {"title": "a\nb"},
        {"title": "  "},
        {"text": "y" * 1501},
        {"evidence": []},
    ):
        with pytest.raises(ValidationError):
            Learning.model_validate({**ok, **change})
    human = Learning.model_validate({**ok, "author": "human:alice", "evidence": []})
    assert human.from_human
    assert not Learning.model_validate(ok).from_human


def test_agent_status_and_llm_call_defaults() -> None:
    assert [s.value for s in AgentStatus] == [
        "active",
        "finished",
        "starved",
        "pruned",
        "stopped",
        "watchdog_stopped",
    ]
    call = LLMCall(
        id="llm1", run_id="r", goal_id="g", role="judge", model="openai/gpt-6", provider="openai"
    )
    assert (call.status, call.cost_usd, call.cache_hit, call.usage) == ("ok", 0.0, False, Usage())


@pytest.mark.parametrize("key", [(), (float("nan"),), (float("inf"),), (float("-inf"),)])
def test_finite_key_rejects_invalid_keys(key: tuple[float, ...]) -> None:
    assert not _finite_key(key)


def test_score_helper_fields_and_roundtrip() -> None:
    low = Score.minimize(-2.0, verdict="best", notes="lower")
    high = Score.maximize(2.0, verdict="best", notes="higher")
    assert (low.key, low.value, low.verdict, low.notes) == ((2.0,), -2.0, "best", "lower")
    assert (high.key, high.value, high.verdict, high.notes) == ((2.0,), 2.0, "best", "higher")
    assert _finite_key((0.0, -1.0, 1.0))
    assert Score.model_validate_json(low.model_dump_json()) == low
    assert pickle.loads(pickle.dumps(high)) == high


@pytest.mark.parametrize(
    "overlay",
    [{"./x": "a" * 64}, {"a//b": "a" * 64}, {"a": "A" * 64}, {"a": "a" * 63}, {"a": "g" * 64}],
)
def test_candidate_overlay_rejects_noncanonical_keys_and_bad_hashes(
    overlay: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        cand(overlay=overlay)


def test_candidate_fake_overrides_and_default_isolation() -> None:
    c = cand(id="other", data={"score": 3}, overlay={"problem/x": "a" * 64})
    assert c.id == "other"
    assert c.output.data == {"score": 3}
    assert c.schema_kind == "candidate"
    assert c.schema_version == 1
    c.checks.append(CheckResult(check="x", passed=False))
    assert not cand().gated_out
    assert pickle.loads(pickle.dumps(c)) == c
    assert Candidate.model_validate_json(c.model_dump_json()) == c


def test_learning_normalizes_title_and_roundtrips() -> None:
    learning = Learning(
        id="l",
        namespace="n",
        run_id="r",
        goal_id="g",
        round=1,
        author="human:a",
        title="  useful  ",
        text="text",
    )
    assert learning.title == "useful"
    assert "from_human" not in learning.model_dump()
    assert Learning.model_validate_json(learning.model_dump_json()) == learning
    assert learning.schema_kind == "learning"
    assert pickle.loads(pickle.dumps(learning)) == learning
    with pytest.raises(ValidationError):
        Learning.model_validate({**learning.model_dump(), "title": "a\rb"})


def test_llm_call_roundtrip_and_usage_isolation() -> None:
    call = LLMCall(
        id="llm",
        run_id="r",
        goal_id="g",
        role="tool",
        model="local/m",
        provider="local",
        tool_calls=[{"id": "tc1", "name": "run"}],
        usage=Usage(input_tokens=2),
        litellm_call_id="gateway-id",
        status="error",
        error="failed",
    )
    assert call.schema_kind == "llm_call"
    assert LLMCall.model_validate_json(call.model_dump_json()) == call
    assert pickle.loads(pickle.dumps(call)) == call


@pytest.mark.parametrize(
    "fields",
    [
        {"overlay": {"../SENTINEL_SECRET": "a" * 64}},
        {"overlay": {"SENTINEL_SECRET": "invalid"}},
        {"SENTINEL_SECRET": "value"},
        {"checks": [{"check": "x", "passed": "SENTINEL_SECRET"}]},
        {
            "request": {
                "parents": [
                    {"candidate_id": "../SENTINEL_SECRET", "run_id": "r", "tool_identity": "t"}
                ]
            }
        },
        {"output": {"status": "ok", "usage": {"SENTINEL_SECRET": 1}}},
    ],
)
def test_candidate_errors_never_expose_values_or_mapping_keys(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError) as caught:
        cand(**fields)
    assert "SENTINEL_SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (Score, {"key": (1.0, "SENTINEL_SECRET")}),
        (CheckResult, {"check": "x", "passed": "SENTINEL_SECRET"}),
        (
            Learning,
            {
                "id": "l",
                "namespace": "n",
                "run_id": "r",
                "goal_id": "g",
                "round": 1,
                "author": "human:a",
                "title": "SENTINEL_SECRET\nb",
                "text": "t",
            },
        ),
        (
            LLMCall,
            {
                "id": "llm",
                "run_id": "r",
                "goal_id": "g",
                "role": "SENTINEL_SECRET",
                "model": "m",
                "provider": "p",
            },
        ),
    ],
)
def test_result_errors_never_expose_values(
    model: type[Score | CheckResult | Learning | LLMCall], fields: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate(fields)
    assert "SENTINEL_SECRET" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_result_error_locations_keep_schema_fields_only() -> None:
    from swarm_arena._model.base import _safe_location

    assert _safe_location(Candidate, ("output", "usage", "input_tokens")) == (
        "output",
        "usage",
        "input_tokens",
    )
    assert _safe_location(Candidate, ("checks", 0, "details", "passed")) == (
        "checks",
        0,
        "details",
        "<…>",
    )
    assert _safe_location(Candidate, ("overlay", 123456789, "[key]")) == ("overlay", "<…>", "<…>")
    assert _safe_location(Candidate, ("request", "inputs", "SENTINEL_SECRET")) == (
        "request",
        "inputs",
        "<…>",
    )


def test_stored_results_reject_newer_schema_and_unknown_fields() -> None:
    learning = Learning(
        id="l",
        namespace="n",
        run_id="r",
        goal_id="g",
        round=1,
        author="human:a",
        title="title",
        text="text",
    )
    call = LLMCall(id="llm", run_id="r", goal_id="g", role="tool", model="m", provider="p")
    for record in (cand(), learning, call):
        for update in ({"schema_version": 2}, {"SENTINEL_SECRET": "secret"}):
            with pytest.raises(ValidationError) as caught:
                type(record).model_validate({**record.model_dump(), **update})
            assert "SENTINEL_SECRET" not in str(caught.value)
        field_name = "id"
        with pytest.raises(ValidationError, match="frozen_instance"):
            setattr(record, field_name, "other")
