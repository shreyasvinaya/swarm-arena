import tomllib
import traceback
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from swarm_arena._errors import ConfigError
from swarm_arena._model.catalog import ModelCatalog, ModelSpec
from swarm_arena._model.limits import Limits
from swarm_arena._model.ref import Ref
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.settings import SecretsPolicy
from swarm_arena._model.toml_io import dumps_toml

MODELS_TOML = """[models.opus]
model = "anthropic/claude-opus-5-5"     # LiteLLM "provider/model" name
max_tokens = 8000
roles = ["manager", "judge", "planner"] # what it may be used for
prompt_cache = { ttl = "5m" }           # optional override

[models.gpt]
model = "openai/gpt-6"
roles = ["manager", "judge"]

[models.flash]
model = "gemini/gemini-3-flash"
roles = ["judge", "curator"]
prices = { input = 0.3, output = 2.5 }  # optional USD per 1M tokens; overrides LiteLLM's price map
"""


def test_run_config_defaults() -> None:
    c = RunConfig()
    assert (c.reserve_fraction, c.board, c.curator, c.llm_gateway, c.cache) == (
        0.15,
        None,
        None,
        None,
        None,
    )
    assert c.executor == Ref(path="swarm_arena.executor:LocalExecutor")
    assert (c.run_root, c.arena_root, c.artifact_root) == (
        ".swarm-arena/runs",
        ".swarm-arena/arena",
        ".swarm-arena/artifacts",
    )
    assert (c.schedule_timeout_s, c.sync_interval_s, c.lease_ttl_s, c.settle_timeout_s) == (
        600.0,
        60.0,
        60.0,
        60.0,
    )
    assert c.tracing.store_payloads
    assert c.models == ModelCatalog({})
    assert c.limits == Limits()


def test_spec_model_catalog(tmp_path: Path) -> None:
    (tmp_path / "run.toml").write_text(MODELS_TOML)
    models = RunConfig.load(tmp_path / "run.toml").models
    opus = models.resolve("opus", "judge")
    assert (opus.model, opus.max_tokens, opus.prompt_cache.ttl, opus.provider) == (
        "anthropic/claude-opus-5-5",
        8000,
        "5m",
        "anthropic",
    )
    assert models.resolve("gpt", "manager").max_tokens == 4096
    assert models.resolve("flash", "curator").prices == ModelSpec.Prices(input=0.3, output=2.5)
    with pytest.raises(ValueError, match="role 'planner'"):
        models.resolve("gpt", "planner")
    with pytest.raises(ValueError, match="unknown model alias 'nope'"):
        models.resolve("nope", "judge")


def test_run_config_dumps_roundtrip_and_errors(tmp_path: Path) -> None:
    cfg = RunConfig(
        secrets=SecretsPolicy(file="/etc/sa.env", allow=["ANTHROPIC_API_KEY"]),
        models=ModelCatalog({"gpt": ModelSpec(model="openai/gpt-6", roles=["judge"])}),
    )
    assert RunConfig.model_validate(tomllib.loads(cfg.dumps())) == cfg
    (tmp_path / "r.toml").write_text("reserve_fraction = 1.5\n")
    with pytest.raises(ConfigError, match=r"(?s)r\.toml.*reserve_fraction"):
        RunConfig.load(tmp_path / "r.toml")


@pytest.mark.parametrize(
    "data",
    [
        {"reserve_fraction": -0.01},
        {"reserve_fraction": 1.0},
        {"goal_weights": {"g": 0}},
        {"goal_weights": {"g": -1}},
        {"schema_version": 2},
        {"unknown": "value"},
    ],
)
def test_run_config_rejects_invalid_fields(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


@pytest.mark.parametrize(
    ("table", "key", "defaults", "location"),
    [
        ("", "settle_timout_s", {}, "settle_timout_s"),
        ("limits", "max_teamz", {}, "limits.max_teamz"),
        ("secrets", "alow", {}, "secrets.alow"),
        ("tracing", "store_paylods", {}, "tracing.store_paylods"),
        (
            "llm_gateway",
            "request_timout_s",
            {"url": "https://gateway.example"},
            "llm_gateway.request_timout_s",
        ),
        ("cache", "ttl_sec", {"backend": "disk"}, "cache.ttl_sec"),
        ("cache.roles.judge", "reed", {"read": True, "write": True}, "cache.roles.<key>.reed"),
    ],
)
def test_unknown_run_config_fields_show_identifiers_without_values(
    tmp_path: Path, table: str, key: str, defaults: dict[str, Any], location: str
) -> None:
    data: dict[str, Any] = {"cache": {"backend": "disk"}} if table.startswith("cache") else {}
    target = data
    for segment in table.split(".") if table else []:
        target = target.setdefault(segment, {})
    target.update(defaults)
    target[key] = "SECRET_SENTINEL"
    with pytest.raises(ValidationError) as caught:
        RunConfig.model_validate(data)
    assert location in str(caught.value)
    assert "extra_forbidden" in str(caught.value)
    assert "SECRET_SENTINEL" not in str(caught.value)
    assert "SECRET_SENTINEL" not in repr(caught.value.args)
    assert "SECRET_SENTINEL" not in "".join(traceback.format_exception(caught.value))
    assert "SECRET_SENTINEL" not in caught.value.json()

    path = tmp_path / "typo.toml"
    path.write_text(dumps_toml(data))
    with pytest.raises(ConfigError, match=r"typo\.toml") as loaded:
        RunConfig.load(path)
    assert location in str(loaded.value)
    assert "extra_forbidden" in str(loaded.value)
    assert "SECRET_SENTINEL" not in str(loaded.value)
    assert "SECRET_SENTINEL" not in repr(loaded.value.args)
    assert "SECRET_SENTINEL" not in "".join(traceback.format_exception(loaded.value))


def test_run_config_refs_roots_and_defaults_are_independent(tmp_path: Path) -> None:
    from _fakes.tools import Widget

    cfg = RunConfig.model_validate(
        {"executor": Widget, "board": Widget({"size": 2}), "curator": Widget}
    )
    assert cfg.executor == Ref.of(Widget)
    assert cfg.board == Ref.of(Widget({"size": 2}))
    assert cfg.curator == Ref.of(Widget)
    path = tmp_path / "roots.toml"
    path.write_text('run_root="./runs"\narena_root="../arena"\nartifact_root="artifacts"\n')
    loaded = RunConfig.load(str(path))
    assert (loaded.run_root, loaded.arena_root, loaded.artifact_root) == (
        "./runs",
        "../arena",
        "artifacts",
    )
    assert RunConfig(reserve_fraction=0, goal_weights={"g": 1}).reserve_fraction == 0
    cfg.models.root["test"] = ModelSpec(model="a/b", roles=["judge"])
    cfg.secrets.allow.append("KEY")
    assert RunConfig().models.root == {}
    assert RunConfig().secrets.allow == []
    with pytest.raises(ValidationError, match="frozen_instance"):
        cfg.reserve_fraction = 0.5  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "text",
    [
        'reserve_fraction="SECRET_SENTINEL"',
        'settle_timout_s="SECRET_SENTINEL"',
        '[goal_weights]\nSECRET_SENTINEL="bad"',
        '[models.SECRET_SENTINEL]\nmodel="a/b"\nroles=["SECRET_SENTINEL"]',
        '[executor]\npath="SECRET_SENTINEL"',
        '[cache]\nbackend="s3"\nnamespace="SECRET_SENTINEL"',
        '[[planner_catalog.entries]]\nname="w"\nkind="tool"\nref="a:B"\ndescription="SECRET_SENTINEL"\n'
        '[planner_catalog.entries.settable.x]\ntype="SECRET_SENTINEL"',
        'value="SECRET_SENTINEL"\nvalue=2',
        '"SECRET_SENTINEL"=[',
    ],
)
def test_load_errors_hide_values_and_data_keys(tmp_path: Path, text: str) -> None:
    path = tmp_path / "secret.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=r"secret\.toml") as caught:
        RunConfig.load(path)
    assert "SECRET_SENTINEL" not in str(caught.value)
    assert "SECRET_SENTINEL" not in repr(caught.value.args)
    assert "SECRET_SENTINEL" not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_load_read_errors_are_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"missing\.toml"):
        RunConfig.load(tmp_path / "missing.toml")
    path = tmp_path / "invalid.toml"
    path.write_bytes(b'key="SECRET_SENTINEL\xff"')
    with pytest.raises(ConfigError, match=r"invalid\.toml") as caught:
        RunConfig.load(path)
    assert "SECRET_SENTINEL" not in "".join(traceback.format_exception(caught.value))


def test_dumps_deterministic_populated_config(tmp_path: Path) -> None:
    cfg = RunConfig.model_validate(
        {
            "goal_weights": {"g": 2},
            "board": {"path": "_fakes.tools:Widget", "config": {"size": 2}},
            "llm_gateway": {"url": "https://gateway.example", "require_hard_cap": True},
            "cache": {"backend": "disk", "disk_dir": "./cache"},
            "planner_catalog": {
                "entries": [
                    {
                        "name": "w",
                        "kind": "tool",
                        "ref": "_fakes.tools:Widget",
                        "description": "Widget",
                        "config": {"size": 1},
                        "settable": {"label": {"type": "string", "default": "ok"}},
                    }
                ]
            },
        }
    )
    path = tmp_path / "run.toml"
    path.write_text(cfg.dumps())
    loaded = RunConfig.load(path)
    assert loaded == cfg
    assert loaded.dumps() == path.read_text()
    assert "schema_version = 1" in cfg.dumps()
    assert cfg.schema_kind == "run_config"
