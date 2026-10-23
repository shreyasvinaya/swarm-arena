import importlib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Annotated, Any, Literal

import pytest
from _fakes.secrecy import assert_secret_absent
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
    model_validator,
)
from pydantic_core import PydanticCustomError

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


@pytest.mark.parametrize("kind", ["terminal", "limit", "provider"])
@pytest.mark.parametrize("nested", [False, True])
def test_validate_config_control_disposition(kind: str, nested: bool) -> None:
    from swarm_arena._errors import (
        ProviderFailed,
        _core_category,
        _core_error,
        _core_signal,
        _SignalCategory,
        _stop_cleanup_error,
    )
    from swarm_arena._model.ref import _validate_config
    from swarm_arena.manager import LimitReached

    class Input(BaseModel):
        @model_validator(mode="before")
        @classmethod
        def terminal(cls, value: Any) -> Any:
            error: BaseException = (
                _stop_cleanup_error()
                if kind == "terminal"
                else _core_signal(LimitReached("stop", "the round is stopping"))
                if kind == "limit"
                else _core_error(ProviderFailed, "provider")
            )
            if nested:
                raise BaseExceptionGroup("config", [BaseExceptionGroup("validator", [error])])
            raise error

    with pytest.raises(RuntimeError if kind == "terminal" else ConfigError) as caught:
        _validate_config(Input, {}, where="component")
    assert _core_category(caught.value) is (
        _SignalCategory.TERMINAL if kind == "terminal" else _SignalCategory.DIAGNOSTIC
    )


def test_path_format() -> None:
    assert Ref(path="a.b:C.D").path == "a.b:C.D"
    with pytest.raises(ValidationError):
        Ref(path="nocolon")


def test_is_ref_path_agrees_with_ref_validation() -> None:
    from swarm_arena._model.ref import is_ref_path

    samples = [
        "demo:Thing",
        "demo.sub:Outer.Inner",
        "_demo:_Thing",
        "aé:bλ",
        "review_a\u0301:Policy",
        "demo:Poli\u0301cy",
        "demo.su\u0301b:Outer.Inne\u0301r",
        "a\u200c:b\u2160",
        "",
        "demo",
        ":Thing",
        "demo:",
        "demo:Thing:Other",
        "demo:Thing/other",
        "0demo:Thing",
        "demo.0thing:Thing",
        "demo:0Thing",
        "demo:Outer.0Inner",
        "é:Thing",
        "demo:\u0301Thing",
        "demo..sub:Thing",
        "demo:Outer..Inner",
    ]
    for path in samples:
        for candidate in (path, path + "\n", path + "\r\n", " " + path, path + "\x00"):
            try:
                Ref(path=candidate)
            except ValidationError:
                accepted = False
            else:
                accepted = True
            assert is_ref_path(candidate) is accepted, ascii(candidate)


@pytest.mark.parametrize("value", [None, 1, True, b"demo:Thing", [], {}])
def test_is_ref_path_requires_a_string(value: object) -> None:
    from swarm_arena._model.ref import is_ref_path

    assert not is_ref_path(value)


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
    with pytest.raises(ConfigError, match=r"(?s)_fakes\.tools:Widget.*<field>.*extra_forbidden"):
        Ref(path="_fakes.tools:Widget", config={"sise": 2}).build(Configurable)


def test_build_config_error_omits_input_values() -> None:
    secret = "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"size": secret, "sise": secret}).build(Configurable)

    assert_secret_absent(exc.value, secret)
    for detail in ("_fakes.tools:Widget", "size", "<field>", "int_parsing", "extra_forbidden"):
        assert detail in str(exc.value)


@pytest.mark.parametrize(
    ("annotation", "credentials", "location"),
    [
        (dict[str, int], {"7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356": "invalid"}, "credentials.<…>"),
        (
            dict[str, dict[str, int]],
            {"outer": {"7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356": "invalid"}},
            "credentials.<…>.<…>",
        ),
        (list[int], ["7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"], "credentials.0"),
        (dict[int, int], {"739201": "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"}, "credentials.<…>"),
        (
            list[dict[str, int]],
            [{"7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356": "invalid"}],
            "credentials.0.<…>",
        ),
        (
            dict[str, list[int]],
            {"7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356": ["invalid"]},
            "credentials.<…>.0",
        ),
        (dict[str, int], {"size": "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"}, "credentials.<…>"),
    ],
)
def test_build_config_error_redacts_data_locations(
    monkeypatch: pytest.MonkeyPatch,
    annotation: Any,
    credentials: object,
    location: str,
) -> None:
    config_model = create_model("Config", __base__=Widget.Config, credentials=(annotation, ...))
    monkeypatch.setattr(Widget, "Config", config_model)
    secret = "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"credentials": credentials}).build(Configurable)

    assert_secret_absent(exc.value, secret, sanitized=True)
    assert_secret_absent(exc.value, "739201", sanitized=True)
    assert "credentials" in str(exc.value)
    assert str(exc.value) == f"_fakes.tools:Widget: {location}: int_parsing"


def test_build_config_error_redacts_unknown_field_names() -> None:
    secret = "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={secret: 1}).build(Configurable)

    assert_secret_absent(exc.value, secret)
    assert str(exc.value) == "_fakes.tools:Widget: <field>: extra_forbidden"


@pytest.mark.parametrize(
    "alias",
    [
        "count",
        AliasChoices("total", "count"),
        AliasPath("counts", 0),
        "declared_count",
    ],
)
def test_build_config_error_redacts_nested_schema_locations(
    monkeypatch: pytest.MonkeyPatch, alias: str | AliasChoices | AliasPath
) -> None:
    class Entry(BaseModel):
        amount: int = Field(validation_alias=alias)

    class Config(Widget.Config):
        credentials: list[Entry] | None

    monkeypatch.setattr(Widget, "Config", Config)
    secret = "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"
    entry = (
        {"counts": [secret]}
        if isinstance(alias, AliasPath)
        else {alias if isinstance(alias, str) else "count": secret}
    )
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"credentials": [entry]}).build(Configurable)

    assert_secret_absent(exc.value, secret)
    assert "credentials" in str(exc.value)
    location = (
        "counts.0"
        if isinstance(alias, AliasPath)
        else (alias if isinstance(alias, str) else "count")
    )
    assert str(exc.value) == f"_fakes.tools:Widget: credentials.0.{location}: int_parsing"


@pytest.mark.parametrize("container", ["model", "mapping", "root", "annotated"])
def test_build_config_error_redacts_nested_models(
    monkeypatch: pytest.MonkeyPatch, container: str
) -> None:
    class Entry(BaseModel):
        amount: int = Field(alias="count")

    secret = "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"
    annotation: Any = Entry
    credentials: object = {"count": secret}
    if container == "mapping":
        annotation = dict[str, Entry]
        credentials = {secret: credentials}
    elif container == "root":
        annotation = RootModel[list[Entry]]
        credentials = [credentials]
    elif container == "annotated":
        annotation = list[Annotated[Entry, Field(description="Nested credential")]]
        credentials = [credentials]
    config_model = create_model("Config", __base__=Widget.Config, credentials=(annotation, ...))
    monkeypatch.setattr(Widget, "Config", config_model)
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"credentials": credentials}).build(Configurable)

    assert_secret_absent(exc.value, secret)
    assert "credentials" in str(exc.value)
    location = {
        "model": "count",
        "mapping": "<…>.count",
        "root": "0.count",
        "annotated": "0.count",
    }[container]
    assert str(exc.value) == f"_fakes.tools:Widget: credentials.{location}: int_parsing"


def test_build_config_error_redacts_union_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    class First(BaseModel):
        kind: Literal["7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"]
        amount: int

    class Second(BaseModel):
        kind: Literal["second"]
        amount: int

    class Config(Widget.Config):
        credentials: Annotated[First | Second, Field(discriminator="kind")]

    monkeypatch.setattr(Widget, "Config", Config)
    secret = "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"
    with pytest.raises(ConfigError) as exc:
        Ref(
            path="_fakes.tools:Widget",
            config={"credentials": {"kind": secret, "amount": "invalid"}},
        ).build(Configurable)

    assert_secret_absent(exc.value, secret)
    assert "credentials" in str(exc.value)
    assert str(exc.value) == "_fakes.tools:Widget: credentials.<…>.amount: int_parsing"


@pytest.mark.parametrize("value", ["invalid", "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"])
def test_build_config_error_redacts_alias_collision(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    class Config(Widget.Config):
        credentials: list[int] = Field(alias="items")
        lookup: dict[int, int] = Field(alias="credentials")

    monkeypatch.setattr(Widget, "Config", Config)
    config = {"items": [], "credentials": {"739201": value}}
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config=config).build(Configurable)

    assert_secret_absent(exc.value, "739201", sanitized=True)
    assert_secret_absent(exc.value, value, sanitized=True)
    assert "credentials" in str(exc.value)
    assert str(exc.value) == "_fakes.tools:Widget: credentials.<…>: int_parsing"


@pytest.mark.parametrize(
    "alias", ["items", AliasChoices("entries", "items"), AliasPath("items", 0)]
)
def test_build_config_error_preserves_top_level_alias(
    monkeypatch: pytest.MonkeyPatch, alias: str | AliasChoices | AliasPath
) -> None:
    class Config(Widget.Config):
        credentials: int = Field(validation_alias=alias)

    monkeypatch.setattr(Widget, "Config", Config)
    secret = "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"
    value = [secret] if isinstance(alias, AliasPath) else secret
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"items": value}).build(Configurable)

    assert_secret_absent(exc.value, secret)
    assert "items" in str(exc.value)
    location = "items.0" if isinstance(alias, AliasPath) else "items"
    assert str(exc.value) == f"_fakes.tools:Widget: {location}: int_parsing"


def test_build_config_error_preserves_missing_top_level_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config(Widget.Config):
        credentials: int

    monkeypatch.setattr(Widget, "Config", Config)
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget").build(Configurable)

    assert str(exc.value) == "_fakes.tools:Widget: credentials: missing"


def test_build_config_error_redacts_model_level_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Config(Widget.Config):
        @model_validator(mode="before")
        @classmethod
        def reject_config(cls, value: object) -> object:
            raise ValueError(f"invalid config: {value}")

    monkeypatch.setattr(Widget, "Config", Config)
    secret = "7f3d895c-dbf6-5bd5-b3bc-c1503dcc8356"
    with pytest.raises(ConfigError) as exc:
        Ref(path="_fakes.tools:Widget", config={"label": secret}).build(Configurable)

    assert_secret_absent(exc.value, secret, sanitized=True)
    assert str(exc.value) == "_fakes.tools:Widget: value_error"


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

    assert_secret_absent(exc.value, secret)
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


class _ValidatorFailure(Exception):
    pass


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TypeError("validator_secret_sentinel"), "invalid"),
        (ValueError("validator_secret_sentinel"), "size: value_error"),
        (AssertionError("validator_secret_sentinel"), "size: assertion_error"),
        (_ValidatorFailure("validator_secret_sentinel"), "invalid"),
        (
            PydanticCustomError(
                "validator_secret_sentinel",
                "bad input: {value}",
                {"value": "validator_secret_sentinel"},
            ),
            "size: invalid",
        ),
        (PydanticCustomError("unregistered_type", "validator_secret_sentinel"), "size: invalid"),
        (None, "size: int_parsing"),
    ],
)
def test_build_config_validation_failures_are_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception | None,
    expected: str,
) -> None:
    class Config(Widget.Config):
        @field_validator("size", mode="before")
        @classmethod
        def reject_size(cls, value: object) -> object:
            if error is not None:
                raise error
            return value

    monkeypatch.setattr(Widget, "Config", Config)
    secret = "validator_secret_sentinel"
    ref = Ref(path="_fakes.tools:Widget", config={"size": secret})
    with pytest.raises(ConfigError) as exc:
        ref.build(Configurable)

    assert_secret_absent(exc.value, secret, sanitized=True)
    assert str(exc.value) == f"_fakes.tools:Widget: {expected}"


@pytest.mark.parametrize("method", ["object", "load", "build", "name"])
@pytest.mark.parametrize("error_type", [ImportError, AttributeError, ValueError, RuntimeError])
def test_import_failures_have_one_safe_boundary(
    monkeypatch: pytest.MonkeyPatch, method: str, error_type: type[Exception]
) -> None:
    import swarm_arena._model.ref as module

    secret = "f83b6f85-3f20-4880-a9d2-835483f36e96"

    def fail(name: str) -> object:
        raise error_type(secret)

    monkeypatch.setattr(module.importlib, "import_module", fail)
    ref = Ref(path="missing_component:Tool")

    def action() -> None:
        if method == "object":
            module.load_object(ref.path)
        elif method == "load":
            ref.load()
        elif method == "build":
            ref.build(Configurable)
        else:
            ref.component_name()

    with pytest.raises(ValueError, match="cannot import missing_component:Tool") as caught:
        action()
    assert_secret_absent(caught.value, secret, sanitized=True)


def test_reference_of_omits_absent_typed_defaults() -> None:
    from swarm_arena.executor import LocalExecutor

    ref = Ref.of(LocalExecutor())
    assert ref.build(LocalExecutor).config == LocalExecutor().config


def test_untrusted_missing_llm_import_keeps_values_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "untrusted-module-secret-sentinel"

    def fail(name: str) -> object:
        raise ModuleNotFoundError(secret, name="agents")

    # An earlier test in the same process may have cached the real lookup.
    load_object.cache_clear()
    monkeypatch.setattr(importlib, "import_module", fail)
    try:
        with pytest.raises(ValueError, match=r"cannot import swarm_arena\.llm:LLMJudge") as caught:
            load_object("swarm_arena.llm:LLMJudge")
    finally:
        load_object.cache_clear()
    assert_secret_absent(caught.value, secret, sanitized=True)


def test_reference_of_never_drops_explicit_null_with_nonnull_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config(Widget.Config):
        timeout_s: float | None = 600.0

    monkeypatch.setattr(Widget, "Config", Config)
    with pytest.raises(ValueError, match="null"):
        Ref.of(Widget(Config(timeout_s=None)))


def test_sync_import_preserves_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._model.ref import load_object

    (tmp_path / "pr1_r3_interrupt.py").write_text("raise KeyboardInterrupt\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        load_object("pr1_r3_interrupt:Tool")
