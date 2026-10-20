"""Pin the shared sentinel assertion's visible exception surfaces."""

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
