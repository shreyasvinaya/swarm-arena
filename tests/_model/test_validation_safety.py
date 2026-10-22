"""Exercise the shared validation boundary and stable diagnostics."""

import ast
import importlib
import importlib.metadata
import json
import pkgutil
from collections.abc import Iterable
from importlib.machinery import PathFinder
from pathlib import Path
from types import SimpleNamespace
from typing import Any, LiteralString, Self, cast
from uuid import NAMESPACE_OID, uuid5

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import GOAL
from pydantic import AliasChoices, AliasPath, BaseModel, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError

import swarm_arena
from swarm_arena._errors import ConfigError
from swarm_arena._exec.executor import Job
from swarm_arena._model.base import _SafeModel, _sanitize_validation_error
from swarm_arena._model.catalog import CatalogEntry, ModelCatalog
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import Plan
from swarm_arena._model.ref import Configurable, Ref
from swarm_arena._model.results import CheckResult
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.schema import compile_object_schema
from swarm_arena._model.settings import TracingConfig
from swarm_arena._tooling.function_tool import function_input_model
from swarm_arena._tooling.script_tool import ScriptTool

# Stable user-visible diagnostic text.
BASELINE_TEXT = {
    "Plan": "1 validation error for Plan\nrevision\n  int_parsing [type=int_parsing]",
    "RunConfig": (
        "1 validation error for RunConfig\nreserve_fraction\n  float_parsing [type=arena_config]"
    ),
    "Limits": "1 validation error for Limits\nmax_teams\n  int_parsing [type=int_parsing]",
    "TracingConfig": (
        "1 validation error for TracingConfig\nlangfuse\n  bool_parsing [type=arena_config]"
    ),
    "Stage": "1 validation error for Stage\nroot\n  dict_type [type=dict_type]",
    "CheckResult": (
        "1 validation error for CheckResult\n"
        "  Value error, ('passed',): bool_parsing [type=value_error]\n"
        "    For further information visit https://errors.pydantic.dev/2.14/v/value_error"
    ),
    "Ref.build": "_fakes.tools:Widget: size: int_parsing",
}

SNAPSHOT_CASES = [
    (
        Plan,
        {
            "goal": GOAL,
            "teams": [],
            "judges": [],
            "revision": "4bc403e3-1d36-5015-9b9f-9a62ef7caed5",
        },
    ),
    (RunConfig, {"reserve_fraction": "4bc403e3-1d36-5015-9b9f-9a62ef7caed5"}),
    (Limits, {"max_teams": "4bc403e3-1d36-5015-9b9f-9a62ef7caed5"}),
    (TracingConfig, {"langfuse": "4bc403e3-1d36-5015-9b9f-9a62ef7caed5"}),
    (Job.Stage, {"root": "4bc403e3-1d36-5015-9b9f-9a62ef7caed5"}),
    (CheckResult, {"check": "c", "passed": "4bc403e3-1d36-5015-9b9f-9a62ef7caed5"}),
]


@pytest.mark.parametrize(
    ("model", "data"), SNAPSHOT_CASES, ids=lambda v: getattr(v, "__name__", "")
)
def test_validation_text_snapshot(model: type[BaseModel], data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    expected = BASELINE_TEXT[model.__name__]
    # Intentional changes mandated by the unified built-in-code/structured-detail rule.
    if model is RunConfig:
        expected = expected.replace("type=arena_config", "type=float_parsing")
    elif model is TracingConfig:
        expected = expected.replace("type=arena_config", "type=bool_parsing")
    elif model is CheckResult:
        expected = "1 validation error for CheckResult\npassed\n  bool_parsing [type=bool_parsing]"
    assert str(caught.value) == expected


def test_ref_build_text_snapshot() -> None:
    with pytest.raises(ConfigError) as caught:
        Ref(
            path="_fakes.tools:Widget", config={"size": "4bc403e3-1d36-5015-9b9f-9a62ef7caed5"}
        ).build(Configurable)
    assert str(caught.value) == BASELINE_TEXT["Ref.build"]


# Cross-check source declarations with imported runtime classes and dynamic factories.
EXTERNAL_MODELS: set[tuple[str, str]] = set()


def _module_available(module: str) -> bool:
    optional = {
        "swarm_arena.llm": ("openai-agents", "litellm"),
        "swarm_arena.testing._fake_model": ("openai-agents", "litellm"),
        "swarm_arena.ray": ("ray",),
        "swarm_arena._store.s3": ("s3fs",),
    }
    for package, distributions in optional.items():
        if module == package or module.startswith(package + "."):
            for distribution in distributions:
                try:
                    importlib.metadata.version(distribution)
                except importlib.metadata.PackageNotFoundError:
                    return False
    return True


def _import_modules(path: Iterable[str], prefix: str) -> None:
    # walk_packages imports packages before recursion; prune absent extras first.
    for module in pkgutil.iter_modules(path, prefix):
        if not _module_available(module.name):
            continue
        imported = importlib.import_module(module.name)
        if module.ispkg:
            _import_modules(imported.__path__, module.name + ".")


def _source_models() -> list[tuple[str, str]]:
    candidates: list[tuple[str, str, list[str]]] = []
    root = Path(__file__).parents[2] / "src" / "swarm_arena"

    def visit(nodes: list[ast.stmt], module: str, parent: str = "") -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                name = parent + node.name
                bases = [
                    base.value if isinstance(base, ast.Subscript) else base for base in node.bases
                ]
                candidates.append(
                    (module, name, [base.id for base in bases if isinstance(base, ast.Name)])
                )
                visit(node.body, module, name + ".")

    for path in sorted(root.rglob("*.py")):
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        if _module_available("swarm_arena." + module):
            visit(ast.parse(path.read_text()).body, module)
    names = {"BaseModel", "RootModel"}
    found: set[tuple[str, str]] = set()
    while True:
        matches = [
            (module, name) for module, name, bases in candidates if names.intersection(bases)
        ]
        if set(matches) == found:
            return sorted(found)
        found.update(matches)
        names.update(name.rsplit(".", 1)[-1] for _, name in matches)


def _model_classes() -> list[type[BaseModel]]:
    found: list[type[BaseModel]] = []
    for module, name in _source_models():
        if (module, name) in EXTERNAL_MODELS:
            continue
        value: Any = importlib.import_module("swarm_arena." + module)
        for part in name.split("."):
            value = getattr(value, part)
        found.append(value)
    # Exercise each dynamic factory as well as source declarations.
    found.append(
        CatalogEntry(
            name="x",
            kind="tool",
            ref="demo:Tool",
            description="d",
            settable={"count": {"type": "integer"}},
        ).settable_model()
    )
    found.extend(
        [
            compile_object_schema("InventoryParams", {"count": {"type": "integer"}}),
            ScriptTool(command=["x"], params={"count": {"type": "integer"}}).get_input_model(),
            function_input_model(_inventory_function),
        ]
    )
    _import_modules(swarm_arena.__path__, "swarm_arena.")
    found.extend(_runtime_models())
    return list(dict.fromkeys(found))


def _inventory_function(count: int) -> int:
    return count


def _runtime_models() -> set[type[BaseModel]]:
    found: set[type[BaseModel]] = set()
    pending = list(BaseModel.__subclasses__())
    seen: set[type[BaseModel]] = set()
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        pending.extend(model.__subclasses__())
        if model.__module__ == "swarm_arena" or model.__module__.startswith("swarm_arena."):
            found.add(model)
    return found


MODELS = _model_classes()
SHOW_MODELS = {
    "Limits",
    "Plan",
    "Resources",
    "PartSpec",
    "AgentSpec",
    "TeamSpec",
    "RoundSpec",
    "ConvergeSpec",
    "FlowSpec",
    "RunConfig",
    "_HumanConfigModel",
    "SecretsPolicy",
    "TracingConfig",
    "GatewayConfig",
    "CachePolicy",
    "CachePolicy.RoleSwitch",
}


@pytest.mark.parametrize("missing", [None, "openai-agents", "litellm"])
def test_inventory_prunes_missing_extras_before_importing(
    missing: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported: list[str] = []

    def version(distribution: str) -> str:
        if distribution == missing:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return "1"

    def modules(path: Iterable[str], prefix: str) -> list[pkgutil.ModuleInfo]:
        children = {
            "swarm_arena.": [("llm", True), ("_model", True)],
            "swarm_arena.llm.": [("_config", False)],
            "swarm_arena._model.": [("base", False)],
        }
        return [
            pkgutil.ModuleInfo(PathFinder, prefix + name, package)
            for name, package in children[prefix]
        ]

    def importing(module: str) -> Any:
        imported.append(module)
        return SimpleNamespace(__path__=[])

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", version)
        patch.setattr(pkgutil, "iter_modules", modules)
        patch.setattr(importlib, "import_module", importing)
        _import_modules([], "swarm_arena.")

    expected = ["swarm_arena._model", "swarm_arena._model.base"]
    if missing is None:
        expected = ["swarm_arena.llm", "swarm_arena.llm._config", *expected]
    assert imported == expected


def test_source_inventory_has_explicit_ownership() -> None:
    assert not EXTERNAL_MODELS
    assert all(model in MODELS or issubclass(model, _SafeModel) for model in _runtime_models())
    assert len(MODELS) >= len(_source_models())
    assert all(issubclass(model, _SafeModel) for model in MODELS)


def _forbids_unknown(model: type[BaseModel]) -> bool:
    return model.model_config.get("extra") == "forbid" or bool(model.__pydantic_root_model__)


def _validate_bad_input(model: type[BaseModel], surface: str, secret: str) -> object:
    if surface == "json-invalid":
        return model.model_validate_json('{"value":"' + secret)
    if surface == "strings":
        return model.model_validate_strings({"misspelled_field": secret})
    if surface == "extra-value":
        return model.model_validate({"misspelled_field": secret})
    if surface == "bad-field":
        # Wrong type in a real field, including stored-version migration errors.
        fields = {
            key: [{"untrusted": secret}] for key in model.model_fields or ["misspelled_field"]
        }
        return model.model_validate(fields)
    return model.model_validate(secret)


@pytest.mark.parametrize("model", MODELS, ids=lambda cls: cls.__module__ + "." + cls.__qualname__)
@pytest.mark.parametrize(
    "surface", ["bad-value", "bad-field", "extra-value", "json-invalid", "strings"]
)
def test_every_model_rejects_secret_safely(model: type[BaseModel], surface: str) -> None:
    secret = str(
        uuid5(NAMESPACE_OID, "sentinel_" + model.__qualname__.replace(".", "_") + "_" + surface)
    )
    if surface in {"extra-value", "strings"} and not _forbids_unknown(model):
        # A model that ignores unknown keys may accept them; it must not keep their values.
        try:
            accepted = _validate_bad_input(model, surface, secret)
        except ValidationError as error:
            assert_secret_absent(error, secret, sanitized=True)
        else:
            assert isinstance(accepted, BaseModel)
            assert secret not in repr(accepted)
            assert secret not in repr(accepted.model_dump())
            assert secret not in repr(accepted.__dict__)
            assert secret not in repr(accepted.__pydantic_extra__)
            assert secret not in repr(accepted.__pydantic_private__)
        return
    with pytest.raises(ValidationError) as caught:
        _validate_bad_input(model, surface, secret)
    assert_secret_absent(caught.value, secret, sanitized=True)


@pytest.mark.parametrize("model", MODELS, ids=lambda cls: cls.__module__ + "." + cls.__qualname__)
def test_unknown_key_policy(model: type[BaseModel]) -> None:
    # Root mappings have data keys, never model-field typo identifiers.
    expected_policy = "show" if model.__qualname__ in SHOW_MODELS else "hide"
    assert getattr(model, "_unknown_fields", None) == expected_policy
    if model.__pydantic_root_model__ or not _forbids_unknown(model):
        return
    secret = str(uuid5(NAMESPACE_OID, "sentinel_" + model.__qualname__ + "_unknown_value"))
    with pytest.raises(ValidationError) as caught:
        model.model_validate({"misspelled_field": secret})
    extras = [item for item in caught.value.errors() if item["type"] == "extra_forbidden"]
    # StoredModel itself intentionally has no schema_kind and fails before fields.
    if extras:
        expected = "misspelled_field" if expected_policy == "show" else "<field>"
        assert [item["loc"] for item in extras] == [(expected,)]
        assert ("misspelled_field" in str(caught.value)) == (expected != "<field>")
    assert_secret_absent(caught.value, secret, sanitized=True)


@pytest.mark.parametrize("kind", ["custom-code", "custom-builtin", "value-error", "type-error"])
@pytest.mark.parametrize("entry", ["constructor", "python", "json", "strings"])
def test_subclass_after_validator_is_inside_shared_boundary(kind: str, entry: str) -> None:
    secret = str(uuid5(NAMESPACE_OID, "after_validator_" + kind + "_sentinel"))

    class Untrusted(_SafeModel):
        value: str

        @model_validator(mode="after")
        def reject(self) -> Self:
            if kind == "custom-code":
                raise PydanticCustomError(
                    cast(LiteralString, secret), "{value}", {"value": self.value}
                )
            if kind == "custom-builtin":
                raise PydanticCustomError("int_parsing", "{value}", {"value": self.value})
            if kind == "value-error":
                raise ValueError(self.value)
            raise TypeError(self.value)

    data = {"value": secret}
    validate = {
        "constructor": lambda: Untrusted(**data),
        "python": lambda: Untrusted.model_validate(data),
        "json": lambda: Untrusted.model_validate_json(json.dumps(data)),
        "strings": lambda: Untrusted.model_validate_strings(data),
    }[entry]
    with pytest.raises(ValidationError) as caught:
        validate()
    assert_secret_absent(caught.value, secret, sanitized=True)
    expected_code = {
        "custom-code": "invalid",
        "custom-builtin": "int_parsing",
        "value-error": "value_error",
        "type-error": "invalid",
    }[kind]
    assert caught.value.errors()[0]["type"] == expected_code
    if kind == "type-error":
        assert caught.value.errors()[0]["msg"] == "invalid: TypeError"


@pytest.mark.parametrize(
    "alias", ["amount", AliasChoices("amount", "count"), AliasPath("items", 0)]
)
def test_shared_location_handles_aliases(alias: str | AliasChoices | AliasPath) -> None:
    class Aliased(_SafeModel):
        count: int = Field(validation_alias=alias)

    secret = str(uuid5(NAMESPACE_OID, "alias_" + type(alias).__name__ + "_sentinel"))
    data = {"items": [secret]} if isinstance(alias, AliasPath) else {"amount": secret}
    with pytest.raises(ValidationError) as caught:
        Aliased.model_validate(data)
    assert_secret_absent(caught.value, secret, sanitized=True)
    expected = ("items", 0) if isinstance(alias, AliasPath) else ("amount",)
    assert caught.value.errors()[0]["loc"] == expected


def test_core_is_idempotent_and_defaults_to_hidden_unknown_keys() -> None:
    secret = str(uuid5(NAMESPACE_OID, "core_" + "sentinel"))
    with pytest.raises(ValidationError) as caught:
        _SafeModel.model_validate(secret)
    once = _sanitize_validation_error(_SafeModel, caught.value)
    twice = _sanitize_validation_error(_SafeModel, once)
    assert once.errors() == twice.errors()
    assert_secret_absent(twice, secret, sanitized=True)


def test_frozen_assignment_and_deletion_are_safe() -> None:
    model = Limits()
    secret = str(uuid5(NAMESPACE_OID, "assignment_" + "sentinel"))
    field_name = "max_teams"
    with pytest.raises(ValidationError) as caught:
        setattr(model, field_name, secret)
    assert_secret_absent(caught.value, secret, sanitized=True)
    assert caught.value.errors()[0]["type"] == "frozen_instance"
    with pytest.raises(ValidationError) as deleted:
        del model.max_teams
    assert_secret_absent(deleted.value, secret, sanitized=True)


def test_root_catalog_redacts_data_keys_colliding_with_schema_names() -> None:
    secret = str(uuid5(NAMESPACE_OID, "root_" + "sentinel"))
    with pytest.raises(ValidationError) as caught:
        ModelCatalog.model_validate({secret: {"model": secret}, "root": {"model": secret}})
    assert_secret_absent(caught.value, secret, sanitized=True)
    # RootModel.root is a serialization field, not an input mapping key identifier.
    assert all(item["loc"][0] == "<…>" for item in caught.value.errors())
