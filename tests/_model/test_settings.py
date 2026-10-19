# ruff: noqa: E501 -- preserve the specification TOML example verbatim.
import traceback
from pathlib import Path
from typing import Any

import pytest
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
        (SecretsPolicy, {"allow": ["SECRET_SENTINEL-INVALID"]}),
        (TracingConfig, {"SECRET_SENTINEL": True}),
        (GatewayConfig, {"url": "SECRET_SENTINEL"}),
        (CachePolicy, {"backend": "s3", "namespace": "SECRET_SENTINEL"}),
        (CachePolicy, {"backend": "redis", "namespace": "SECRET_SENTINEL"}),
        (CachePolicy, {"backend": "disk", "roles": {"SECRET_SENTINEL": {"read": 1, "write": 1}}}),
        (CachePolicy.RoleSwitch, {"read": "SECRET_SENTINEL", "write": True}),
    ],
)
def test_settings_errors_hide_secret_values(
    model: type[BaseModel],
    data: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    assert "SECRET_SENTINEL" not in "".join(traceback.format_exception(caught.value))
    assert "SECRET_SENTINEL" not in caught.value.json()
