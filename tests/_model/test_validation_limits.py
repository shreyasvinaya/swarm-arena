"""Bound validation details and preserve nested error ownership."""

import subprocess
import sys
import textwrap
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import BaseModel, ValidationError, create_model, field_validator
from pydantic_core import PydanticCustomError

from swarm_arena._errors import ConfigError
from swarm_arena._model import base, locations, ref, safe
from swarm_arena._model.base import FrozenModel, _SafeModel
from swarm_arena._model.plan import Plan

SECRET = "45e2f48d-adf7-5f9c-a949-55a178179b81"


@pytest.mark.parametrize("count", [20, 21, 100_000])
def test_error_detail_cap_preserves_order_and_secrecy(count: int) -> None:
    class Input(_SafeModel):
        items: list[int]

    with pytest.raises(ValidationError) as caught:
        Input(items=[SECRET] * count)
    details = caught.value.errors()
    assert len(details) == min(count, 20) + (count > 20)
    assert [item["loc"] for item in details[:20]] == [("items", i) for i in range(20)]
    if count > 20:
        assert details[-1] == {
            "type": "invalid",
            "loc": (),
            "msg": f"and {count - 20} more errors",
            "input": None,
        }
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("safe", [False, True])
def test_config_summary_preserves_error_cap_count(safe: bool) -> None:
    model = create_model(
        "Config", __base__=_SafeModel if safe else BaseModel, items=(list[int], ...)
    )
    with pytest.raises(ConfigError) as caught:
        ref._validate_config(model, {"items": [SECRET] * 25}, where="m:T")
    assert str(caught.value) == "m:T: " + "; ".join(
        [*(f"items.{index}: int_parsing" for index in range(20)), "and 5 more errors"]
    )
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize("count", [21, 25, 47])
@pytest.mark.parametrize("config_boundary", [False, True])
def test_explicit_nested_cap_counts_all_omitted_errors(count: int, config_boundary: bool) -> None:
    class Inner(_SafeModel):
        items: list[int]

    class Outer(_SafeModel):
        nested: Any
        later: list[int]

        @field_validator("nested")
        @classmethod
        def validate_nested(cls, value: Any) -> Inner:
            return Inner.model_validate(value)

    data = {"nested": {"items": [SECRET] * count}, "later": [SECRET] * 3}
    if config_boundary:
        with pytest.raises(ConfigError) as config_error:
            ref._validate_config(Outer, data, where="m:T")
        assert str(config_error.value).endswith(f"; and {count + 3 - 20} more errors")
        assert_secret_absent(config_error.value, SECRET, sanitized=True)
    else:
        with pytest.raises(ValidationError) as caught:
            Outer.model_validate(data)
        details = caught.value.errors()
        assert len(details) == 21
        assert [item["loc"] for item in details[:20]] == [
            ("nested", "items", index) for index in range(20)
        ]
        assert details[-1]["msg"] == f"and {count + 3 - 20} more errors"
        assert_secret_absent(caught.value, SECRET, sanitized=True)


def test_untrusted_cap_message_is_not_preserved_or_counted() -> None:
    class Input(_SafeModel):
        items: list[int]

        @field_validator("items", mode="before")
        @classmethod
        def forged_trailer(cls, value: Any) -> Any:
            raise PydanticCustomError("invalid", "and 937 more errors")

    with pytest.raises(ValidationError) as caught:
        Input(items=[SECRET])
    assert caught.value.errors() == [
        {"type": "invalid", "loc": ("items",), "msg": "invalid", "input": None}
    ]
    assert_secret_absent(caught.value, SECRET, sanitized=True)


@pytest.mark.parametrize("boundary", ["safe", "config"])
def test_fallback_construction_has_no_raw_exception_context(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    model = create_model(
        "Input", __base__=_SafeModel if boundary == "safe" else BaseModel, count=(int, ...)
    )

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise RecursionError()

    module = safe if boundary == "safe" else ref
    monkeypatch.setattr(module, "_sanitize_validation_error", crash)
    monkeypatch.setattr(module, "_minimal_validation_error", crash)
    action = (
        (lambda: model(**{"count": SECRET}))
        if boundary == "safe"
        else (lambda: ref._validate_config(model, {"count": SECRET}, where="m:T"))
    )
    with pytest.raises(RecursionError) as caught:
        action()
    assert caught.value.__context__ is None
    assert_secret_absent(caught.value, SECRET)


def test_plain_constructor_uses_core_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    class Input(FrozenModel):
        count: int

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("plain construction allocated another boundary")

    monkeypatch.setattr(base, "_validation_wrapper", unexpected)
    assert Input(count=1).count == 1


def test_nested_plan_errors_reuse_schema_index(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    walks = 0
    original = locations._schema_definitions
    original_location = locations._safe_location

    def spy(schema: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original(schema)

    def location(*args: Any, **kwargs: Any) -> tuple[str | int, ...]:
        nonlocal walks
        walks += 1
        return original_location(*args, **kwargs)

    Plan.model_rebuild(force=True)
    monkeypatch.setattr(locations, "_schema_definitions", spy)
    monkeypatch.setattr(safe, "_safe_location", location)
    data = {
        "goal": {"id": "g", "namespace": "test"},
        "teams": [
            {
                "name": "team",
                "rationale": "test",
                "agents": [
                    {
                        "name": f"agent-{i}",
                        "rationale": "test",
                        "manager": "m:T",
                        "tool": "m:T",
                        "replicates": SECRET,
                    }
                    for i in range(1000)
                ],
            }
        ],
        "judges": [],
    }
    for _ in range(2):
        with pytest.raises(ValidationError):
            Plan.model_validate(data)
    assert walks >= 40
    assert calls == 1
    Plan.model_rebuild(force=True)
    with pytest.raises(ValidationError):
        Plan.model_validate(data)
    assert calls == 2


def test_stack_exhaustion_keeps_every_context_value_free() -> None:
    # Isolate recursion-limit changes from pytest and inspect even suppressed contexts.
    script = textwrap.dedent("""
        import sys
        from pydantic import BaseModel, TypeAdapter, ValidationError
        from swarm_arena._model.base import FrozenModel
        from swarm_arena._model.ref import _validate_config

        class Safe(FrozenModel):
            value: int
        class Plain(BaseModel):
            value: int
        adapter = TypeAdapter(Safe)

        def action(boundary: str) -> object:
            if boundary == 'config':
                return _validate_config(Plain, {'value': 'ZQXSECRET'}, where='m:T')
            return adapter.validate_python({'value': 'ZQXSECRET'})

        def descend(n: int, boundary: str) -> object:
            if n:
                return descend(n-1, boundary)
            return action(boundary)

        sys.setrecursionlimit(120)
        for boundary in ('safe', 'config'):
            count=0
            for depth in range(80, 122):
                try:
                    descend(depth, boundary)
                except Exception as error:
                    count+=1
                    current=error
                    seen=set()
                    while current is not None and id(current) not in seen:
                        seen.add(id(current))
                        if isinstance(current, ValidationError):
                            assert 'ZQXSECRET' not in repr(current.errors()), (boundary, depth)
                        current=current.__context__
            print(boundary, count, 'depths safe')
    """)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["safe 42 depths safe", "config 42 depths safe"]
