"""Per-call settings and authoritative adapter merges."""

from pathlib import Path
from typing import Any

import litellm
import pytest
from _fakes.llm_run import open_llm_run
from agents import ModelSettings
from openai.types.shared import Reasoning

from swarm_arena._budget.llm_scope import LLMScope
from swarm_arena._model.base import LLMRole
from swarm_arena.config import CachePolicy, ModelSpec
from swarm_arena.llm._cache import bypass_response_cache, cache_switches
from swarm_arena.llm._settings import injection_points, merge_model_settings, model_settings
from tests.llm.test_upstream_facts import _invoke_adapter

pytestmark = pytest.mark.filterwarnings(
    "ignore:Items? .*TypedDict class .*ReadOnly.*:UserWarning:pydantic._internal._generate_schema"
)


OPUS = ModelSpec(model="anthropic/claude-opus-5-5", roles=["manager"], max_tokens=8000)


def settings(scope: LLMScope, **overrides: Any) -> ModelSettings:
    return model_settings(
        OPUS,
        alias="opus",
        scope=scope,
        reservation_id="r-1",
        call_id="llm-1",
        namespace="physika",
        **{"read": False, "write": False, **overrides},
    )


async def test_metadata_carries_tracking_ids(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        s = settings(run.scope("manager"))
        assert s.metadata == {
            "run_id": run.run_id,
            "goal_id": "g",
            "team": "t",
            "agent": "a",
            "round": "1",
            "role": "manager",
            "reservation_id": "r-1",
            "call_id": "llm-1",
            "model_alias": "opus",
        }
        assert all(type(v) is str for v in s.metadata.values())
        assert s.include_usage is True
        assert s.max_tokens == 8000


async def test_max_tokens_override_capped_by_catalog(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("manager")
        assert settings(scope, max_tokens=100).max_tokens == 100
        assert settings(scope, max_tokens=10_000).max_tokens == 8000


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-opus-5-5", "bedrock/anthropic.claude-opus-5-5", "vertex_ai/claude-opus-5-5"],
)
def test_claude_injection_points_default_ttl(model: str) -> None:
    assert injection_points(ModelSpec(model=model, roles=["manager"])) == [
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1},
    ]


def test_claude_injection_points_1h_ttl() -> None:
    pts = injection_points(
        OPUS.model_copy(update={"prompt_cache": ModelSpec.PromptCache(ttl="1h")})
    )
    assert pts is not None
    assert all(p["control"] == {"type": "ephemeral", "ttl": "1h"} for p in pts)


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-6", "gemini/gemini-3-flash", "deepseek/deepseek-chat", "bedrock/meta.llama4-70b"],
)
def test_no_injection_points_for_other_models(model: str) -> None:
    assert injection_points(ModelSpec(model=model, roles=["manager"])) is None


def test_prompt_cache_disabled_means_no_points() -> None:
    assert (
        injection_points(
            OPUS.model_copy(update={"prompt_cache": ModelSpec.PromptCache(enabled=False)})
        )
        is None
    )


async def test_cache_dict_from_switches(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("manager")
        assert (settings(scope, read=True, write=True).extra_args or {})["cache"] == {
            "use-cache": True,
            "no-cache": False,
            "no-store": False,
            "namespace": "physika",
        }
        assert (settings(scope, write=True).extra_args or {})["cache"]["no-cache"] is True
        assert (settings(scope, read=True).extra_args or {})["cache"]["no-store"] is True
        assert "cache" not in (settings(scope).extra_args or {})


async def test_temperature_comes_with_drop_params(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("manager")
        assert (settings(scope, temperature=0).extra_args or {})["drop_params"] is True
        assert "drop_params" not in (settings(scope, temperature=None).extra_args or {})


async def test_merge_protects_tracking_and_cache_controls(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope("manager")
        caller = ModelSettings(
            metadata={"reservation_id": "wrong", "tag": "kept"},
            extra_args={
                "metadata": {"call_id": "wrong", "model_alias": "wrong"},
                "litellm_metadata": {"reservation_id": "wrong"},
                "cache": {"namespace": "wrong"},
                "caching": True,
                "preset_cache_key": "wrong",
                "cache_key": "wrong",
                "cache_control_injection_points": [],
                "drop_params": False,
                "seed": 17,
            },
        )
        per_call = settings(scope, read=False, write=True, temperature=0)
        merged = merge_model_settings(caller, per_call)
        assert merged.metadata == {**(per_call.metadata or {}), "tag": "kept"}
        extra = merged.extra_args or {}
        assert extra["cache"] == (per_call.extra_args or {})["cache"]
        assert extra["seed"] == 17
        assert extra["drop_params"] is True
        assert extra["cache_control_injection_points"] == injection_points(OPUS)
        assert not (
            {"preset_cache_key", "cache_key", "metadata", "litellm_metadata", "caching"}
            & extra.keys()
        )
        disabled = merge_model_settings(caller, settings(scope))
        assert "cache" not in (disabled.extra_args or {})
        assert caller.extra_args is not None
        assert caller.extra_args["caching"] is True


@pytest.mark.parametrize(
    "caller",
    [
        ModelSettings(reasoning=Reasoning(effort="high")),
        ModelSettings(extra_args={"thinking": {"type": "enabled", "budget_tokens": 1024}}),
        ModelSettings(extra_args={"reasoning_effort": "high"}),
        ModelSettings(extra_body={"reasoning_effort": "high"}),
        ModelSettings(extra_body={"thinking": {"type": "adaptive"}}),
        ModelSettings(extra_args={"extra_body": {"thinking": {"type": "enabled"}}}),
        ModelSettings(extra_args={"stream": True}),
    ],
)
async def test_merge_detects_streamed_and_thinking_channels(
    tmp_path: Path, caller: ModelSettings
) -> None:
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(caller, settings(run.scope("manager"), read=True, write=True))
        assert "cache" not in (merged.extra_args or {})


@pytest.mark.parametrize("role", ["judge", "manager"])
async def test_probe_bypass_is_complete_and_restored(tmp_path: Path, role: LLMRole) -> None:
    async with open_llm_run(tmp_path) as run:
        scope = run.scope(role, force_no_cache_read=role == "judge")
        policy = CachePolicy(
            backend="disk", roles={role: CachePolicy.RoleSwitch(read=True, write=True)}
        )
        assert cache_switches(
            policy, role, force_no_read=False, streamed=False, thinking=False
        ) == (True, True)
        assert (settings(scope, read=True, write=True).extra_args or {})["cache"]["no-cache"] is (
            role == "judge"
        )
        with (  # noqa: PT012 - verify restoration on failure
            pytest.raises(RuntimeError, match="probe"),
            bypass_response_cache(),
        ):
            assert "cache" not in (settings(scope, read=True, write=True).extra_args or {})
            assert cache_switches(
                policy, role, force_no_read=False, streamed=False, thinking=False
            ) == (False, False)
            raise RuntimeError("probe")
        assert "cache" in (settings(scope, read=True, write=True).extra_args or {})
        per_call = settings(scope, read=True, write=True)
        with bypass_response_cache():
            assert "cache" not in (merge_model_settings(ModelSettings(), per_call).extra_args or {})
        for flags in ({"streamed": True}, {"thinking": True}):
            assert "cache" not in (
                merge_model_settings(ModelSettings(), per_call, **flags).extra_args or {}
            )


async def test_final_settings_follow_real_adapter_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def completion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return litellm.ModelResponse()

    monkeypatch.setattr(litellm, "acompletion", completion)
    async with open_llm_run(tmp_path) as run:
        per_call = settings(run.scope("manager"), read=False, write=True, temperature=0)
        caller = ModelSettings(extra_args={"metadata": {"call_id": "wrong"}, "caching": True})
        await _invoke_adapter(merge_model_settings(caller, per_call))
        assert captured["metadata"] == per_call.metadata
        assert captured["cache"] == (per_call.extra_args or {})["cache"]
        assert captured["cache_control_injection_points"] == injection_points(OPUS)
        assert captured["drop_params"] is True
        assert captured["temperature"] == 0
        assert "caching" not in captured


async def test_merged_caller_temperature_also_drops_unsupported_params(tmp_path: Path) -> None:
    async with open_llm_run(tmp_path) as run:
        merged = merge_model_settings(ModelSettings(temperature=0), settings(run.scope("manager")))
        assert merged.temperature == 0
        assert (merged.extra_args or {})["drop_params"] is True
