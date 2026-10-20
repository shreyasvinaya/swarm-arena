"""Pin the shared sentinel assertion's visible exception surfaces."""

import traceback
from typing import Any
from uuid import uuid4

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import BaseModel, ConfigDict, ValidationError

from swarm_arena import Goal


@pytest.mark.parametrize("surface", ["message", "notes", "cause", "validation"])
def test_secret_assertion_detects_visible_leaks(surface: str) -> None:
    secret = str(uuid4())
    error: BaseException = ValueError("invalid")
    if surface == "message":
        error = ValueError(secret)
    elif surface == "notes":
        error.add_note(secret)
    elif surface == "cause":
        error.__cause__ = ValueError(secret)
    else:

        class Inputs(BaseModel):
            model_config = ConfigDict(hide_input_in_errors=True)
            count: int

        with pytest.raises(ValidationError) as caught:
            Inputs.model_validate({"count": secret})
        error = caught.value
    with pytest.raises(AssertionError):
        assert_secret_absent(error, secret)


def test_secret_assertion_accepts_sanitized_public_validation() -> None:
    secret = str(uuid4())
    with pytest.raises(ValidationError) as caught:
        Goal.model_validate({"id": "example", "namespace": "demo", "description": {"data": secret}})
    assert_secret_absent(caught.value, secret, sanitized=True)


def test_secret_assertion_accepts_suppressed_context() -> None:
    secret = str(uuid4())
    try:
        try:
            raise ValueError(secret)
        except ValueError:
            raise RuntimeError("invalid") from None
    except RuntimeError as error:
        assert_secret_absent(error, secret, sanitized=True)


SECRET = "secrecy-surface-sentinel"


class _QuietError(ValueError):
    def __str__(self) -> str:
        return "invalid"

    def __repr__(self) -> str:
        return "QuietError()"


def test_catches_str_leak() -> None:
    class StrLeak(_QuietError):
        def __str__(self) -> str:
            return SECRET

    with pytest.raises(AssertionError):
        assert_secret_absent(StrLeak(), SECRET)


def test_catches_repr_leak() -> None:
    class ReprLeak(_QuietError):
        def __repr__(self) -> str:
            return SECRET

    with pytest.raises(AssertionError):
        assert_secret_absent(ReprLeak(), SECRET)


def test_catches_args_leak() -> None:
    with pytest.raises(AssertionError):
        assert_secret_absent(_QuietError(SECRET), SECRET)


def test_catches_notes_leak() -> None:
    error = _QuietError()
    error.add_note(SECRET)
    with pytest.raises(AssertionError):
        assert_secret_absent(error, SECRET)


def test_catches_cause_leak() -> None:
    error = ValueError("invalid")
    error.__cause__ = _QuietError(SECRET)
    with pytest.raises(AssertionError):
        assert_secret_absent(error, SECRET)


def test_catches_unsuppressed_context_leak() -> None:
    error = ValueError("invalid")
    error.__context__ = _QuietError(SECRET)
    with pytest.raises(AssertionError):
        assert_secret_absent(error, SECRET)


@pytest.mark.parametrize("surface", ["errors", "json"])
def test_catches_pydantic_details_leak(surface: str, monkeypatch: pytest.MonkeyPatch) -> None:
    error = ValidationError.from_exception_data(
        "Invalid", [{"type": "int_parsing", "loc": ("value",), "input": SECRET}], hide_input=True
    )
    # Isolate each serialized surface: hide_input already keeps the text surfaces clean.
    if surface == "errors":
        monkeypatch.setattr(ValidationError, "json", lambda self: "[]")
    else:
        monkeypatch.setattr(ValidationError, "errors", lambda self: [])
    with pytest.raises(AssertionError):
        assert_secret_absent(error, SECRET)


def test_catches_exception_only_leak() -> None:
    error = type(SECRET, (_QuietError,), {})()
    assert SECRET not in str(error) + repr(error) + repr(error.args)
    assert SECRET in "".join(traceback.format_exception_only(error))
    with pytest.raises(AssertionError):
        assert_secret_absent(error, SECRET)


def test_catches_nested_exception_group_leak() -> None:
    class GroupedLeak(_QuietError):
        def __str__(self) -> str:
            return SECRET

    error = ExceptionGroup("invalid", [ExceptionGroup("nested", [GroupedLeak()])])
    with pytest.raises(AssertionError):
        assert_secret_absent(error, SECRET)


def test_ignores_frame_source_containing_secret() -> None:
    def reject(value: str) -> None:
        raise ValueError("invalid")

    with pytest.raises(ValueError, match="invalid") as caught:
        reject("secrecy-surface-sentinel")
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("sanitized", [False, True])
def test_allows_suppressed_context(sanitized: bool) -> None:
    error = ValueError("invalid")
    error.__context__ = ValueError(SECRET)
    error.__suppress_context__ = True
    assert_secret_absent(error, SECRET, sanitized=sanitized)


def test_cause_takes_precedence_over_context() -> None:
    error = ValueError("invalid")
    error.__context__ = ValueError(SECRET)
    error.__cause__ = ValueError("cause")
    error.__suppress_context__ = False
    assert_secret_absent(error, SECRET)


def test_checks_falsey_explicit_cause() -> None:
    class FalseyError(_QuietError):
        def __bool__(self) -> bool:
            return False

    error = ValueError("invalid")
    error.__cause__ = FalseyError(SECRET)
    with pytest.raises(AssertionError):
        assert_secret_absent(error, SECRET)


@pytest.mark.parametrize("link", ["__cause__", "__context__"])
def test_checks_each_chain_node_and_stops_cycles(link: str) -> None:
    error = ValueError("invalid")
    chained = _QuietError()
    setattr(error, link, chained)
    setattr(chained, link, error)
    assert_secret_absent(error, SECRET)
    chained.args = (SECRET,)
    with pytest.raises(AssertionError):
        assert_secret_absent(error, SECRET)


@pytest.mark.parametrize("details", [{"input": SECRET}, {"input": None, "ctx": {"key": "safe"}}])
def test_sanitized_requires_value_free_details(
    details: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ValidationError.from_exception_data("Invalid", [])
    monkeypatch.setattr(ValidationError, "errors", lambda self: [details])
    with pytest.raises(AssertionError):
        assert_secret_absent(error, "absent", sanitized=True)


def test_sanitized_requires_suppressed_original_exception() -> None:
    error = ValueError("invalid")
    error.__context__ = ValueError("safe")
    with pytest.raises(AssertionError):
        assert_secret_absent(error, SECRET, sanitized=True)
