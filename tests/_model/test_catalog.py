from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import BaseModel, ValidationError

from swarm_arena._model.catalog import CatalogEntry, ModelCatalog, ModelSpec, PlannerCatalog
from swarm_arena._model.ref import Ref


def test_model_spec_rules() -> None:
    for bad in (
        {"model": "noslash", "roles": ["judge"]},
        {"model": "a/b", "roles": []},
        {"model": "a/b", "roles": ["judge", "judge"]},
    ):
        with pytest.raises(ValidationError):
            ModelSpec.model_validate(bad)
    with pytest.raises(ValidationError):
        ModelCatalog.model_validate({"bad alias": {"model": "a/b", "roles": ["judge"]}})


def test_catalog_settable_model() -> None:
    e = CatalogEntry(
        name="m",
        kind="manager",
        ref="_fakes.tools:Widget",
        description="d",
        settable={"label": {"type": "string", "maxLength": 10}},
    )
    M = e.settable_model()
    assert M().model_dump() == {"label": None}
    assert M.model_validate({"label": "x"}).model_dump() == {"label": "x"}
    for bad in ({"label": "x" * 11}, {"size": 1}):
        with pytest.raises(ValidationError):
            M.model_validate(bad)


def test_catalog_rejects_human_only_settable() -> None:
    entry = {
        "name": "s",
        "kind": "tool",
        "ref": "_fakes.tools:HumanOnlyWidget",
        "description": "d",
        "settable": {"command": {"type": "array"}},
    }
    with pytest.raises(ValidationError, match="human-only field 'command'"):
        PlannerCatalog.model_validate({"entries": [entry]})
    PlannerCatalog.model_validate(
        {"entries": [{**entry, "settable": {"timeout_s": {"type": "number"}}}]}
    )  # allowed


def test_catalog_get_and_match() -> None:
    cat = PlannerCatalog(
        entries=[
            CatalogEntry(
                name="w",
                kind="manager",
                ref="_fakes.tools:Widget",
                description="d",
                config={"size": 1},
                settable={"label": {"type": "string"}},
            )
        ]
    )
    assert cat.get("w").name == "w"
    with pytest.raises(ValueError, match="unknown catalog entry 'x'"):
        cat.get("x")
    assert (
        cat.match(Ref(path="_fakes.tools:Widget", config={"size": 1, "label": "hi"}), "manager")
        is cat.entries[0]
    )
    assert cat.match(Ref(path="_fakes.tools:Widget", config={"size": 2}), "manager") is None
    assert cat.match(Ref(path="_fakes.tools:Widget", config={"size": 1}), "tool") is None


@pytest.mark.parametrize(
    ("schema", "valid", "invalid"),
    [
        ({"type": "number"}, 1.5, "secret-value"),
        ({"type": "string"}, "ok", 1),
        ({"type": "integer", "minimum": 1, "maximum": 3, "default": 2}, 1, 4),
        ({"type": "array", "items": {"type": "integer"}}, [1, 2], ["secret-value"]),
        ({"type": "string", "enum": ["yes", "no"]}, "yes", "secret-value"),
    ],
)
def test_settable_optional_strict_properties(
    schema: dict[str, Any],
    valid: Any,
    invalid: Any,
) -> None:
    entry = CatalogEntry(
        name="w",
        kind="tool",
        ref="_fakes.tools:Widget",
        description="d",
        settable={"value": schema},
    )
    model = entry.settable_model()
    assert model().model_dump() == {"value": None}
    assert model.model_validate({"value": valid}).model_dump()["value"] == valid
    for value in (invalid, None):
        with pytest.raises(ValidationError):
            model.model_validate({"value": value})
    with pytest.raises(ValidationError, match="frozen_instance"):
        model.model_validate({"value": valid}).value = valid  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4"},
        {"type": "string", "default": 123},
        {"type": "object"},
        {"type": "string", "pattern": "["},
        {"type": "string", "minimum": 1},
        {"f31a4604-550d-5b3d-b9b3-65a6a2254cd4": True},
        "f31a4604-550d-5b3d-b9b3-65a6a2254cd4",
    ],
)
def test_catalog_rejects_bad_schema_at_load(schema: Any) -> None:
    with pytest.raises(ValidationError, match="invalid settable schema") as caught:
        CatalogEntry(
            name="w",
            kind="tool",
            ref="_fakes.tools:Widget",
            description="d",
            settable={"value": schema},
        )
    assert_secret_absent(caught.value, "f31a4604-550d-5b3d-b9b3-65a6a2254cd4")


def test_catalog_duplicates_and_first_match() -> None:
    entry = CatalogEntry(name="w", kind="tool", ref="_fakes.tools:Widget", description="d")
    other = entry.model_copy(update={"name": "other"})
    with pytest.raises(ValidationError, match="duplicate catalog entry names"):
        PlannerCatalog(entries=[entry, entry])
    cat = PlannerCatalog(entries=[entry, other])
    assert cat.match(Ref(path=entry.ref), "tool") is entry
    assert cat.match(Ref(path="builtins:str"), "tool") is None


def test_catalog_import_failure_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: str) -> object:
        raise RuntimeError("f31a4604-550d-5b3d-b9b3-65a6a2254cd4")

    monkeypatch.setattr("swarm_arena._model.ref.importlib.import_module", fail)
    entry = CatalogEntry(
        name="w",
        kind="tool",
        ref="unavailable_catalog:Component",
        description="d",
        settable={"label": {"type": "string"}},
    )
    with pytest.raises(ValidationError, match="value_error") as caught:
        PlannerCatalog(entries=[entry])
    assert_secret_absent(caught.value, "f31a4604-550d-5b3d-b9b3-65a6a2254cd4")
    assert PlannerCatalog(entries=[entry.model_copy(update={"settable": {}})]).entries


@pytest.mark.parametrize(
    "data",
    [
        {"model": "a/b", "roles": ["judge"], "max_tokens": 0},
        {"model": "a/b", "roles": ["tool"]},
        {"model": "a/b", "roles": ["judge"], "prices": {"input": -1, "output": 0}},
        {"model": "a/b", "roles": ["judge"], "prices": {"input": 0, "output": 0, "cache_read": -1}},
        {
            "model": "a/b",
            "roles": ["judge"],
            "prices": {"input": 0, "output": 0, "cache_write": -1},
        },
        {"model": "a/b", "roles": ["judge"], "prompt_cache": {"ttl": "2h"}},
    ],
)
def test_model_spec_boundaries(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(data)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (ModelSpec, {"model": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4", "roles": ["judge"]}),
        (ModelSpec.Prices, {"input": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4", "output": 0}),
        (ModelSpec.PromptCache, {"ttl": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4"}),
        (
            ModelCatalog,
            {
                "f31a4604-550d-5b3d-b9b3-65a6a2254cd4": {
                    "model": "a/b",
                    "roles": ["f31a4604-550d-5b3d-b9b3-65a6a2254cd4"],
                }
            },
        ),
        (
            CatalogEntry,
            {
                "name": "w",
                "kind": "tool",
                "ref": "f31a4604-550d-5b3d-b9b3-65a6a2254cd4",
                "description": "d",
            },
        ),
        (
            CatalogEntry,
            {
                "name": "w",
                "kind": "tool",
                "ref": "a:B",
                "description": "d",
                "f31a4604-550d-5b3d-b9b3-65a6a2254cd4": 1,
            },
        ),
    ],
)
def test_catalog_errors_hide_values_and_mapping_keys(
    model: type[BaseModel],
    data: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError) as caught:
        model.model_validate(data)
    assert_secret_absent(caught.value, "f31a4604-550d-5b3d-b9b3-65a6a2254cd4")
