# ruff: noqa: E501 -- preserve the specification TOML example verbatim.
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import BaseModel, ValidationError

from swarm_arena._model.limits import Limits
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.settings import CachePolicy, GatewayConfig, SecretsPolicy, TracingConfig

CACHE_TOML = """[cache]
backend = "s3"                       # "disk" | "s3" | "redis"
s3_path = "s3://bucket/swarm-arena/llm-cache"
# namespace = "physika"               # optional override; default: each call uses its goal's Goal.namespace
ttl_s = 2592000                       # 30 days

[cache.roles]                         # separate read and write switches per role
judge   = { read = true,  write = true  }   # repeat judge requests (for example after resume) cost nothing; anchors and the final judge never read (see below)
planner = { read = false, write = true  }
manager = { read = false, write = false }   # managers should explore, not replay
curator = { read = true,  write = true  }
"""


def test_spec_cache_policy(tmp_path: Path) -> None:
    (tmp_path / "run.toml").write_text(CACHE_TOML)
    cache = RunConfig.load(tmp_path / "run.toml").cache
    assert cache is not None
    assert (cache.backend, cache.ttl_s) == ("s3", 2_592_000)
    assert (
        cache.roles
        == CachePolicy(backend="disk").roles
        == {
            "judge": CachePolicy.RoleSwitch(read=True, write=True),
            "planner": CachePolicy.RoleSwitch(read=False, write=True),
            "manager": CachePolicy.RoleSwitch(read=False, write=False),
            "curator": CachePolicy.RoleSwitch(read=True, write=True),
        }
    )


def test_cache_backend_requirements() -> None:
    with pytest.raises(ValidationError, match="s3_path"):
        CachePolicy(backend="s3")
    with pytest.raises(ValidationError, match="redis_url_env"):
        CachePolicy(backend="redis")


def test_gateway() -> None:
    gw = GatewayConfig(url="http://gw:4000")
    assert (
        gw.master_key_env,
        gw.request_timeout_s,
        gw.key_duration_s,
        gw.default_estimated_output_tokens,
    ) == ("LITELLM_MASTER_KEY", 600.0, 86_400.0, 4096)
    assert gw.hard_cap_required(Limits(max_usd=5.0))
    assert not gw.hard_cap_required(Limits())
    assert not GatewayConfig(url="http://gw:4000", require_hard_cap=False).hard_cap_required(
        Limits(max_usd=5.0)
    )
    with pytest.raises(ValidationError):
        GatewayConfig(url="gw:4000")


def test_secrets_allow_names() -> None:
    assert SecretsPolicy(allow=["ANTHROPIC_API_KEY"]).file is None
    with pytest.raises(ValidationError):
        SecretsPolicy(allow=["NOT-A-NAME"])


def test_settings_boundaries_and_defaults() -> None:
    assert TracingConfig().model_dump() == {"store_payloads": True, "langfuse": False}
    assert SecretsPolicy().model_dump() == {"file": None, "allow": []}
    assert GatewayConfig(url="https://example", require_hard_cap=True).hard_cap_required(Limits())
    assert CachePolicy(backend="redis", redis_url_env="REDIS_URL", ttl_s=1).ttl_s == 1
    with pytest.raises(ValidationError):
        CachePolicy(backend="disk", ttl_s=0)
    first = CachePolicy(backend="disk")
    first.roles.clear()
    assert len(CachePolicy(backend="disk").roles) == 4


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (SecretsPolicy, {"allow": ["a32343e2-f0ae-535d-a6a0-60742e07d275"]}),
        (TracingConfig, {"store_paylods": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4"}),
        (GatewayConfig, {"url": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4"}),
        (CachePolicy, {"backend": "s3", "namespace": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4"}),
        (CachePolicy, {"backend": "redis", "namespace": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4"}),
        (
            CachePolicy,
            {
                "backend": "disk",
                "roles": {"f31a4604-550d-5b3d-b9b3-65a6a2254cd4": {"read": 1, "write": 1}},
            },
        ),
        (CachePolicy.RoleSwitch, {"read": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4", "write": True}),
    ],
)
def test_settings_errors_hide_secret_values(
    model: type[BaseModel],
    data: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    assert_secret_absent(caught.value, "f31a4604-550d-5b3d-b9b3-65a6a2254cd4")


@pytest.mark.parametrize(
    ("model", "data", "key"),
    [
        (SecretsPolicy, {}, "alow"),
        (TracingConfig, {}, "store_paylods"),
        (GatewayConfig, {"url": "https://gateway.example"}, "request_timout_s"),
        (CachePolicy, {"backend": "disk"}, "ttl_sec"),
        (CachePolicy.RoleSwitch, {"read": True, "write": True}, "reed"),
    ],
)
def test_unknown_settings_fields_show_identifiers_without_values(
    model: type[BaseModel], data: dict[str, Any], key: str
) -> None:
    data = data | {key: "f31a4604-550d-5b3d-b9b3-65a6a2254cd4"}
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    assert key in str(caught.value)
    assert "extra_forbidden" in str(caught.value)
    assert_secret_absent(caught.value, "f31a4604-550d-5b3d-b9b3-65a6a2254cd4")
