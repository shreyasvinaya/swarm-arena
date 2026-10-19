import importlib
from collections.abc import Mapping
from traceback import format_exception
from types import MappingProxyType, ModuleType
from typing import Annotated, Any, Literal

import pytest
from _fakes.tools import NoNameWidget, Widget
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    Field,
    RootModel,
    ValidationError,
    create_model,
    field_validator,
)

from swarm_arena._errors import ConfigError
from swarm_arena._model.base import FrozenModel
from swarm_arena._model.ref import (
    Configurable,
    EmptyConfig,
    Ref,
    RefField,
    coerce_ref,
    config_or_fields,
    import_path,
    load_object,
    unique_names,
)


def test_path_format() -> None:
    assert Ref(path="a.b:C.D").path == "a.b:C.D"
    with pytest.raises(ValidationError):
        Ref(path="nocolon")


def test_of_instance_and_class() -> None:
    assert Ref.of(Widget(Widget.Config(size=3))) == Ref(
        path="_fakes.tools:Widget", config={"size": 3, "label": ""}
    )
    assert Ref.of(Widget) == Ref(path="_fakes.tools:Widget")


def test_of_rejects_lambda_local_main_and_non_components() -> None:
    with pytest.raises(TypeError, match="lambda"):
        import_path(lambda: None)

    class Local(Configurable): ...

    with pytest.raises(TypeError, match="local"):
        Ref.of(Local)
    main_component = type("MainComponent", (Configurable,), {"__module__": "__main__"})
    with pytest.raises(TypeError, match="__main__"):
        Ref.of(main_component)
    with pytest.raises(TypeError, match="component"):
        Ref.of(object())


def test_build_validates_config() -> None:
    w = Ref(path="_fakes.tools:Widget", config={"size": 2}).build(Configurable)
    assert isinstance(w, Widget)
    assert w.config == Widget.Config(size=2)


def test_build_rejects_misspelled_key() -> None:
    with pytest.raises(ConfigError, match=r"(?s)_fakes\.tools:Widget.*<key>.*extra_forbidden"):
        Ref(path="_fakes.tools:Widget", config={"sise": 2}).build(Configurable)


def test_build_config_error_omits_input_values() -> None:
    secret = "review-only-secret-sentinel"
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"size": secret, "sise": secret}).build(Configurable)

    assert secret not in str(exc.value)
    assert secret not in repr(exc.value.args)
    assert secret not in "".join(format_exception(exc.value))
    for detail in ("_fakes.tools:Widget", "size", "<key>", "int_parsing", "extra_forbidden"):
        assert detail in str(exc.value)


@pytest.mark.parametrize(
    ("annotation", "credentials", "location"),
    [
        (dict[str, int], {"review-only-secret-sentinel": "invalid"}, ("credentials", "<key>")),
        (
            dict[str, dict[str, int]],
            {"outer": {"review-only-secret-sentinel": "invalid"}},
            ("credentials", "<key>", "<key>"),
        ),
        (list[int], ["review-only-secret-sentinel"], ("credentials", 0)),
        (dict[int, int], {739201: "review-only-secret-sentinel"}, ("credentials", "<key>")),
        (
            list[dict[str, int]],
            [{"review-only-secret-sentinel": "invalid"}],
            ("credentials", 0, "<key>"),
        ),
        (
            dict[str, list[int]],
            {"review-only-secret-sentinel": ["invalid"]},
            ("credentials", "<key>", 0),
        ),
        (dict[str, int], {"size": "review-only-secret-sentinel"}, ("credentials", "<key>")),
    ],
)
def test_build_config_error_redacts_data_locations(
    monkeypatch: pytest.MonkeyPatch,
    annotation: Any,
    credentials: object,
    location: tuple[str | int, ...],
) -> None:
    config_model = create_model("Config", __base__=Widget.Config, credentials=(annotation, ...))
    monkeypatch.setattr(Widget, "Config", config_model)
    secret = "review-only-secret-sentinel"
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"credentials": credentials}).build(Configurable)

    for diagnostic in (str(exc.value), repr(exc.value.args), "".join(format_exception(exc.value))):
        assert secret not in diagnostic
        assert "739201" not in diagnostic
        assert "credentials" in diagnostic
    assert f"{location}: int_parsing" in str(exc.value)
    assert exc.value.__suppress_context__


def test_build_config_error_redacts_unknown_field_names() -> None:
    secret = "review-only-secret-sentinel"
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={secret: 1}).build(Configurable)

    for diagnostic in (str(exc.value), repr(exc.value.args), "".join(format_exception(exc.value))):
        assert secret not in diagnostic
    assert "('<key>',): extra_forbidden" in str(exc.value)


@pytest.mark.parametrize("alias", ["count", AliasChoices("total", "count"), AliasPath("counts", 0)])
def test_build_config_error_preserves_nested_schema_locations(
    monkeypatch: pytest.MonkeyPatch, alias: str | AliasChoices | AliasPath
) -> None:
    class Entry(BaseModel):
        amount: int = Field(validation_alias=alias)

    class Config(Widget.Config):
        credentials: list[Entry] | None

    monkeypatch.setattr(Widget, "Config", Config)
    secret = "review-only-secret-sentinel"
    entry = {"counts": [secret]} if isinstance(alias, AliasPath) else {"count": secret}
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"credentials": [entry]}).build(Configurable)

    for diagnostic in (str(exc.value), repr(exc.value.args), "".join(format_exception(exc.value))):
        assert secret not in diagnostic
    location = (
        ("credentials", 0, "counts", 0)
        if isinstance(alias, AliasPath)
        else ("credentials", 0, "count")
    )
    assert f"{location}: int_parsing" in str(exc.value)


@pytest.mark.parametrize("container", ["model", "mapping", "root", "annotated"])
def test_build_config_error_follows_nested_models(
    monkeypatch: pytest.MonkeyPatch, container: str
) -> None:
    class Entry(BaseModel):
        amount: int = Field(alias="count")

    secret = "review-only-secret-sentinel"
    annotation: Any = Entry
    credentials: object = {"count": secret}
    location: tuple[str | int, ...] = ("credentials", "count")
    if container == "mapping":
        annotation = dict[str, Entry]
        credentials = {secret: credentials}
        location = ("credentials", "<key>", "count")
    elif container == "root":
        annotation = RootModel[list[Entry]]
        credentials = [credentials]
        location = ("credentials", 0, "count")
    elif container == "annotated":
        annotation = list[Annotated[Entry, Field(description="Nested credential")]]
        credentials = [credentials]
        location = ("credentials", 0, "count")
    config_model = create_model("Config", __base__=Widget.Config, credentials=(annotation, ...))
    monkeypatch.setattr(Widget, "Config", config_model)
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"credentials": credentials}).build(Configurable)

    for diagnostic in (str(exc.value), repr(exc.value.args), "".join(format_exception(exc.value))):
        assert secret not in diagnostic
    assert f"{location}: int_parsing" in str(exc.value)


def test_build_config_error_redacts_union_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    class First(BaseModel):
        kind: Literal["review-only-secret-sentinel"]
        amount: int

    class Second(BaseModel):
        kind: Literal["second"]
        amount: int

    class Config(Widget.Config):
        credentials: Annotated[First | Second, Field(discriminator="kind")]

    monkeypatch.setattr(Widget, "Config", Config)
    secret = "review-only-secret-sentinel"
    with pytest.raises(ConfigError) as exc:
        Ref(
            path="_fakes.tools:Widget",
            config={"credentials": {"kind": secret, "amount": "invalid"}},
        ).build(Configurable)

    for diagnostic in (str(exc.value), repr(exc.value.args), "".join(format_exception(exc.value))):
        assert secret not in diagnostic
        assert "credentials" in diagnostic
    assert "<key>" in str(exc.value)


def test_build_config_error_omits_validator_message_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config(Widget.Config):
        @field_validator("size", mode="before")
        @classmethod
        def reject_size(cls, value: object) -> object:
            raise ValueError(f"invalid size: {value}")

    monkeypatch.setattr(Widget, "Config", Config)
    secret = "validator-only-secret-sentinel"
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"size": secret}).build(Configurable)

    assert secret not in str(exc.value)
    assert secret not in repr(exc.value.args)
    assert secret not in "".join(format_exception(exc.value))
    for detail in ("_fakes.tools:Widget", "size", "value_error"):
        assert detail in str(exc.value)


def test_build_wrong_base_and_load_errors() -> None:
    with pytest.raises(TypeError, match="not a subclass of Widget"):
        Ref(path="_fakes.tools:NoNameWidget").build(Widget)
    with pytest.raises(ValueError, match="cannot import no_such_mod:X"):
        Ref(path="no_such_mod:X").load()
    with pytest.raises(TypeError, match="not a class"):
        Ref(path="json:dumps").load()


def test_load_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    load_object.cache_clear()
    calls: list[str] = []
    original_import = importlib.import_module

    def track_import(name: str) -> ModuleType:
        calls.append(name)
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", track_import)
    try:
        assert Ref(path="_fakes.tools:Widget").load() is Widget
        assert Ref(path="_fakes.tools:Widget").load() is Widget
        assert calls == ["_fakes.tools"]
    finally:
        load_object.cache_clear()


def test_component_name() -> None:
    assert Ref(path="_fakes.tools:Widget", name="w2").component_name() == "w2"
    assert Ref(path="_fakes.tools:Widget").component_name() == "widget"
    with pytest.raises(TypeError, match="name"):
        Ref(path="_fakes.tools:NoNameWidget").component_name()


def test_unique_names() -> None:
    assert unique_names(
        [Ref(path="_fakes.tools:Widget"), Ref(path="_fakes.tools:Widget", name="b")]
    ) == ["widget", "b"]
    with pytest.raises(ValueError, match="duplicate component name 'widget'"):
        unique_names([Ref(path="_fakes.tools:Widget"), Ref(path="_fakes.tools:Widget")])


def test_ref_field_coerces_instances() -> None:
    class Holder(FrozenModel):
        ref: RefField

    assert Holder.model_validate({"ref": Widget()}).ref == Ref.of(Widget())
    assert Holder.model_validate({"ref": Widget}).ref == Ref.of(Widget)
    assert Holder.model_validate({"ref": {"path": "_fakes.tools:Widget"}}).ref == Ref.of(Widget)


def test_configurable_config_forms() -> None:
    assert Configurable().config == EmptyConfig()
    assert Widget({"size": 5}).config.size == 5
    assert Widget(Widget.Config(size=6)).config.size == 6
    with pytest.raises(ValidationError):
        Configurable({"x": 1})
    assert Widget().config == Widget.Config()
    assert Widget(MappingProxyType({"size": 7})).config.size == 7


@pytest.mark.parametrize(
    ("obj", "attribute", "value", "error", "message"),
    [
        (Widget(), "config", Widget.Config(size=2), AttributeError, "config"),
        (Widget().config, "size", 2, ValidationError, "frozen"),
        (Ref(path="_fakes.tools:Widget"), "name", "other", ValidationError, "frozen"),
    ],
)
def test_config_is_read_only_and_frozen(
    obj: object, attribute: str, value: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        setattr(obj, attribute, value)


def test_nested_import_and_missing_attribute() -> None:
    assert import_path(Widget.Config) == "_fakes.tools:Widget.Config"
    assert load_object("_fakes.tools:Widget.Config") is Widget.Config
    with pytest.raises(ValueError, match=r"cannot import _fakes\.tools:Widget\.Missing"):
        load_object("_fakes.tools:Widget.Missing")


def test_coerce_ref_preserves_other_values() -> None:
    for value in (Ref.of(Widget), {"path": "_fakes.tools:Widget"}, object(), int, None):
        assert coerce_ref(value) is value


def test_component_name_requires_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NoNameWidget, "name", 1, raising=False)
    with pytest.raises(TypeError, match=r"has no name; set Ref\.name"):
        Ref.of(NoNameWidget).component_name()


def test_construction_does_not_import_referenced_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbid_import(name: str) -> ModuleType:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", forbid_import)
    assert Ref(path="no_such_mod:X", name="explicit").component_name() == "explicit"
    assert Widget().config == Widget.Config()


def test_config_or_fields() -> None:
    config = Widget.Config(size=3)
    fields: Mapping[str, Any] = {"size": 4}
    assert config_or_fields(config, {}) is config
    assert config_or_fields(None, fields) is fields
    assert config_or_fields(None, {}) == {}
    assert config_or_fields({}, {}) == {}
    with pytest.raises(TypeError, match="pass a config or fields, not both"):
        config_or_fields(config, fields)
    with pytest.raises(TypeError, match="pass a config or fields, not both"):
        config_or_fields({}, fields)
