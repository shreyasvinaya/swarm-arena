"""Validate sanitizer totality, schema-index cost and boundary ownership."""

import json
from collections.abc import Callable
from contextvars import Context, copy_context
from functools import partial
from threading import Thread
from typing import Annotated, Any, ClassVar, Self, cast

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PydanticSchemaGenerationError,
    PydanticUserError,
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    create_model,
    field_validator,
    model_validator,
)
from pydantic.errors import PydanticImportError, PydanticUndefinedAnnotation

from swarm_arena._errors import ConfigError
from swarm_arena._model import base, locations, ref, safe
from swarm_arena._model.base import _SafeModel
from swarm_arena._model.ref import _validate_config

SECRET = "72637a4e-d59e-5316-bb5e-d65f326c5edf"


@pytest.mark.parametrize("kind", ["terminal", "limit", "provider"])
@pytest.mark.parametrize("nested", [False, True])
def test_validation_wrapper_control_disposition(kind: str, nested: bool) -> None:
    from swarm_arena._errors import (
        ProviderFailed,
        _core_category,
        _core_error,
        _core_signal,
        _SignalCategory,
        _stop_cleanup_error,
    )
    from swarm_arena._model.base import FrozenModel
    from swarm_arena.manager import LimitReached

    class Input(FrozenModel):
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
                raise BaseExceptionGroup("model", [BaseExceptionGroup("validator", [error])])
            raise error

    with pytest.raises(RuntimeError if kind == "terminal" else ValidationError) as caught:
        Input()
    assert _core_category(caught.value) is (
        _SignalCategory.TERMINAL if kind == "terminal" else None
    )


def validate(model: type[BaseModel], entry: str, data: dict[str, Any]) -> BaseModel:
    if entry == "constructor":
        return model(**data)
    if entry == "json":
        return model.model_validate_json(json.dumps(data))
    if entry == "strings":
        return model.model_validate_strings(data)
    return model.model_validate(data)


def wrapped(value: Any, handler: ValidatorFunctionWrapHandler, *, limit: int = 3) -> Any:
    return handler(value)


class CallableWrap:
    def __call__(self, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        return handler(value)


@pytest.mark.parametrize("fn", [partial(wrapped, limit=3), CallableWrap()])
@pytest.mark.parametrize("boundary", ["safe", "config"])
def test_callable_union_never_crashes_or_leaks(fn: Any, boundary: str) -> None:
    parent = _SafeModel if boundary == "safe" else BaseModel
    model = create_model(
        "CallableUnion", __base__=parent, value=(Annotated[int, WrapValidator(fn)] | str, ...)
    )
    action = (
        (lambda: model.model_validate({"value": {SECRET: SECRET}}))
        if boundary == "safe"
        else (lambda: _validate_config(model, {"value": {SECRET: SECRET}}, where="m:T"))
    )
    with pytest.raises(ValidationError if boundary == "safe" else ConfigError) as caught:
        action()
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize("boundary", ["safe", "config"])
def test_boundary_falls_back_when_sanitizer_crashes(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    Config = create_model(
        "Config", __base__=_SafeModel if boundary == "safe" else BaseModel, count=(int, ...)
    )

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(SECRET)

    monkeypatch.setattr(safe if boundary == "safe" else ref, "_sanitize_validation_error", crash)
    action = (
        (lambda: Config(**{"count": SECRET}))
        if boundary == "safe"
        else (lambda: _validate_config(Config, {"count": SECRET}, where="m:T"))
    )
    with pytest.raises(ValidationError if boundary == "safe" else ConfigError) as caught:
        action()
    assert_secret_absent(caught.value, SECRET, sanitized=True)
    if isinstance(caught.value, ValidationError):
        assert caught.value.errors() == [
            {"type": "invalid", "loc": (), "msg": "invalid", "input": None}
        ]
    else:
        assert str(caught.value) == "m:T: invalid"


def test_location_walker_itself_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Config(_SafeModel):
        count: int

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(SECRET)

    monkeypatch.setattr(locations, "_location_parts", crash)
    assert locations._safe_location(Config, ("count", SECRET)) == ("<…>", "<…>")


def test_schema_index_built_once_and_rebuilt_with_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    class Config(_SafeModel):
        items: list[int]

    calls = 0
    walks = 0
    original = locations._schema_definitions
    original_walk = locations._location_parts

    def walk_spy(*args: Any, **kwargs: Any) -> tuple[str | int, ...]:
        nonlocal walks
        walks += 1
        return original_walk(*args, **kwargs)

    def spy(schema: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(schema)

    monkeypatch.setattr(locations, "_schema_definitions", spy)
    monkeypatch.setattr(locations, "_location_parts", walk_spy)
    for _ in range(2):
        with pytest.raises(ValidationError) as caught:
            Config(items=[SECRET] * 1000)
        assert caught.value.error_count() == 21
    assert calls == 1
    assert walks < 20  # Two thousand errors must not re-walk even the root fields.
    Config.model_rebuild(force=True)
    with pytest.raises(ValidationError):
        Config(items=[SECRET] * 1000)
    assert calls == 2
    assert walks < 40


def test_copied_context_validates_safely_in_another_thread() -> None:
    errors: list[Exception] = []

    class Inner(_SafeModel):
        value: int

        @model_validator(mode="before")
        @classmethod
        def reject(cls, value: Any) -> Any:
            raise ValueError(SECRET)

    class Outer(_SafeModel):
        @model_validator(mode="after")
        def run_copied_context(self) -> Self:
            context = copy_context()

            def worker() -> None:
                try:
                    # Core validation checks the inherited owner, independently
                    # of explicit model_validate/constructor entry points.
                    context.run(Inner.__pydantic_validator__.validate_python, {"value": SECRET})
                except Exception as error:
                    errors.append(error)

            thread = Thread(target=worker)
            thread.start()
            thread.join(timeout=60)
            assert not thread.is_alive()
            return self

    Outer()
    assert len(errors) == 1
    assert isinstance(errors[0], ValidationError)
    assert_secret_absent(errors[0], SECRET, sanitized=True)


@pytest.mark.parametrize("entry", ["constructor", "python", "json", "strings"])
@pytest.mark.parametrize("outer_policy", ["show", "hide"])
def test_explicit_nested_validation_owns_its_policy(entry: str, outer_policy: str) -> None:
    errors: list[Exception] = []

    class Inner(_SafeModel):
        _unknown_fields: ClassVar = "hide" if outer_policy == "show" else "show"
        count: int

    class Outer(_SafeModel):
        _unknown_fields: ClassVar = outer_policy

        @model_validator(mode="after")
        def validate_explicitly(self) -> Self:
            try:
                validate(Inner, entry, {"count": "1", "typo": SECRET})
            except ValidationError as error:
                errors.append(error)
            return self

    Outer()
    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, ValidationError)
    assert error.errors()[0]["loc"] == ("<field>" if outer_policy == "show" else "typo",)
    assert_secret_absent(error, SECRET, sanitized=True)


@pytest.mark.parametrize("entry", ["constructor", "python", "json", "strings"])
def test_developer_api_errors_propagate(entry: str) -> None:
    class Config(_SafeModel):
        value: int

    if entry == "constructor":
        Incomplete = create_model(
            "Incomplete", __base__=_SafeModel, value=(cast(Any, "NotDefinedYet"), ...)
        )

        with pytest.raises(PydanticUserError, match="not fully defined"):
            Incomplete(value=1)
    else:
        method = getattr(
            Config,
            {
                "python": "model_validate",
                "json": "model_validate_json",
                "strings": "model_validate_strings",
            }[entry],
        )
        payload = '{"value":1}' if entry == "json" else {"value": "1"}
        with pytest.raises(PydanticUserError, match="by_alias"):
            method(payload, by_alias=False, by_name=False)


@pytest.mark.parametrize("entry", ["constructor", "python", "json", "strings", "config"])
@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: PydanticUserError("developer hint", code="schema-for-unknown-type"),
        lambda: PydanticSchemaGenerationError("developer hint"),
        lambda: PydanticImportError("developer hint"),
        lambda: PydanticUndefinedAnnotation("MissingType", "developer hint"),
    ],
)
def test_developer_errors_from_validators_are_unchanged(
    entry: str, error_factory: Callable[[], Exception]
) -> None:
    expected = error_factory()

    class Config(_SafeModel):
        @model_validator(mode="before")
        @classmethod
        def reject(cls, value: Any) -> Any:
            raise expected

    action = (
        (lambda: _validate_config(Config, {}, where="m:T"))
        if entry == "config"
        else (lambda: validate(Config, entry, {}))
    )
    with pytest.raises(type(expected)) as caught:
        action()
    assert caught.value is expected


def test_alias_name_ambiguity_does_not_reveal_integer_mapping_key() -> None:
    class Config(_SafeModel):
        model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
        credentials: dict[int, int] = Field(alias="items")
        lookup: list[int] = Field(alias="credentials")

    key = 739201
    with pytest.raises(ValidationError) as caught:
        Config(credentials={key: SECRET})
    assert any(item["loc"] == ("credentials", "<…>") for item in caught.value.errors())
    assert_secret_absent(caught.value, "739201")


def test_same_name_union_uses_mapping_node_for_integer_keys() -> None:
    sequence = create_model("Duplicate", __base__=_SafeModel, value=(list[int], ...))
    mapping = create_model("Duplicate", __base__=_SafeModel, value=(dict[int, int], ...))
    model = create_model("UnionModel", __base__=_SafeModel, branch=(sequence | mapping, ...))
    key = 739201
    with pytest.raises(ValidationError) as caught:
        model(branch={"value": {key: SECRET}})
    assert any(item["loc"] == ("branch", "<…>", "value", "<…>") for item in caught.value.errors())
    assert_secret_absent(caught.value, "739201")
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize("entry", ["constructor", "python", "json", "strings"])
def test_escaping_explicit_error_keeps_inner_schema(entry: str) -> None:
    class Inner(_SafeModel):
        count: int

    class Outer(_SafeModel):
        @model_validator(mode="after")
        def check_inner(self) -> Self:
            validate(Inner, entry, {"count": SECRET})
            return self

    with pytest.raises(ValidationError) as caught:
        Outer()
    assert caught.value.errors()[0]["loc"] == ("count",)
    assert_secret_absent(caught.value, SECRET, sanitized=True)


def test_explicit_error_preserves_field_prefix_and_redacts_mapping_key() -> None:
    class Inner(_SafeModel):
        values: dict[int, int]

    class Outer(_SafeModel):
        value: int

        @field_validator("value")
        @classmethod
        def check_inner(cls, value: int) -> int:
            Inner.model_validate({"values": {739201: SECRET}})
            return value

    with pytest.raises(ValidationError) as caught:
        Outer(value=1)
    assert caught.value.errors()[0]["loc"] == ("value", "values", "<…>")
    assert_secret_absent(caught.value, "739201")
    assert_secret_absent(caught.value, SECRET, sanitized=True)


def test_copied_context_becomes_inactive_when_owner_finishes() -> None:
    copies: list[Context] = []

    class Config(_SafeModel):
        count: int

    class Outer(_SafeModel):
        @model_validator(mode="after")
        def save_context(self) -> Self:
            copies.append(copy_context())
            return self

    Outer()
    with pytest.raises(ValidationError) as caught:
        copies[0].run(Config.__pydantic_validator__.validate_python, {"count": SECRET})
    assert_secret_absent(caught.value, SECRET, sanitized=True)


def test_sanitizer_itself_is_total(monkeypatch: pytest.MonkeyPatch) -> None:
    class Config(BaseModel):
        count: int

    with pytest.raises(ValidationError) as caught:
        Config(count=SECRET)

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(SECRET)

    monkeypatch.setattr(safe, "_schema_index", crash)
    sanitized = base._sanitize_validation_error(Config, caught.value)
    assert sanitized.errors() == [{"type": "invalid", "loc": (), "msg": "invalid", "input": None}]
    assert_secret_absent(sanitized, SECRET, sanitized=True)
