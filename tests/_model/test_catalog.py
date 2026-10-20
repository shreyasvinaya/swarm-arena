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


@pytest.mark.parametrize(
    ("fixed", "offered"),
    [(True, 1), (True, 1.0), (1, 1.0), (False, 0), (False, 0.0), (0, 0.0)],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shape", ["scalar", "list", "mapping", "nested"])
def test_catalog_match_distinguishes_json_scalar_types(
    fixed: Any, offered: Any, reverse: bool, shape: str
) -> None:
    if reverse:
        fixed, offered = offered, fixed
    if shape == "list":
        fixed, offered = [fixed], [offered]
    elif shape == "mapping":
        fixed, offered = {"x": fixed}, {"x": offered}
    elif shape == "nested":
        fixed, offered = {"x": [{"y": fixed}]}, {"x": [{"y": offered}]}
    entry = CatalogEntry(
        name="demo", kind="tool", ref="demo:Tool", description="", config={"value": fixed}
    )
    catalog = PlannerCatalog(entries=[entry])
    assert catalog.match(Ref(path=entry.ref, config={"value": offered}), "tool") is None


@pytest.mark.parametrize(
    ("fixed", "offered", "matches"),
    [
        (True, True, True),
        (1, 1, True),
        (1.0, 1.0, True),
        (0.0, -0.0, True),
        ("1", 1, False),
        ([], {}, False),
        ([1, 2], [2, 1], False),
        ([1], [1, 2], False),
        ([1, 2], [1], False),
        ({"x": 1}, {"y": 1}, False),
        ({"x": 1}, {"x": 1, "y": 2}, False),
        (
            {"x": [True, 1, 1.0, "é"], "y": {"a": 2, "b": 3}},
            {"y": {"b": 3, "a": 2}, "x": [True, 1, 1.0, "é"]},
            True,
        ),
    ],
)
def test_catalog_match_preserves_json_value_and_container_rules(
    fixed: Any, offered: Any, matches: bool
) -> None:
    entry = CatalogEntry(
        name="demo", kind="tool", ref="demo:Tool", description="", config={"value": fixed}
    )
    result = PlannerCatalog(entries=[entry]).match(
        Ref(path=entry.ref, config={"value": offered}), "tool"
    )
    assert result is (entry if matches else None)


def test_catalog_match_only_removes_top_level_settable_fields_without_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = CatalogEntry(
        name="demo",
        kind="tool",
        ref="_fakes.tools:Widget",
        description="",
        config={"nested": {"label": True}},
        settable={"label": {"type": "string"}},
    )
    catalog = PlannerCatalog(entries=[entry])
    good = Ref(path=entry.ref, config={"nested": {"label": True}, "label": 42})
    bad = Ref(path=entry.ref, config={"nested": {"label": 1}, "label": "allowed"})
    wrong_path = Ref(path="different:Tool", config=good.config)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("catalog matching must not import, validate components, or perform I/O")

    with monkeypatch.context() as patch:
        patch.setattr("swarm_arena._model.catalog.load_object", unexpected)
        patch.setattr("builtins.open", unexpected)
        patch.setattr("io.open", unexpected)
        patch.setattr("os.open", unexpected)
        patch.setattr(Ref, "_resolve_config", unexpected)
        patch.setattr("importlib.import_module", unexpected)
        assert catalog.match(good, "tool") is entry
        assert catalog.match(bad, "tool") is None
        assert catalog.match(wrong_path, "tool") is None
        assert catalog.match(good, "manager") is None


def test_catalog_match_uses_shared_json_equal_and_first_typed_match() -> None:
    from swarm_arena._model import catalog, data

    assert catalog.json_equal is data.json_equal
    entries = [
        CatalogEntry(
            name=name,
            kind="tool",
            ref="demo:Tool",
            description="",
            config={"x": [{"value": value}]},
        )
        for name, value in [("bool", True), ("int", 1), ("float", 1.0), ("int-again", 1)]
    ]
    matched = PlannerCatalog(entries=entries).match(
        Ref(path="demo:Tool", config={"x": [{"value": 1}]}), "tool"
    )
    assert matched is entries[1]
