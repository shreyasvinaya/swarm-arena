"""Detach extension-owned models inside an abandonable preparation worker."""

from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

_M = TypeVar("_M", bound=BaseModel)


def _plain(value: Any) -> Any:
    """Copy Python payloads to built-in values without lossy JSON serializers."""
    if value is None or type(value) is bool:
        return value
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, bytes):
        return memoryview(value).tobytes()
    if isinstance(value, int):
        return int.__int__(value)
    if isinstance(value, float):
        return float.__float__(value)
    if isinstance(value, datetime):
        return datetime.fromisoformat(value.isoformat())
    if isinstance(value, dict):
        return {_plain(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_plain(item) for item in value)
    raise TypeError("invalid returned model payload")


def _detach_model(model: type[_M], value: object) -> _M:
    """Detach and validate user models in a supervised daemon, applying core rules.

    Dumping, field access and container iteration may all execute extension
    code. Only a core-owned model class may be passed as model. Python-mode
    payloads retain datetime values and reject malformed values before
    any stored-format serializer can coerce or truncate them.
    """
    if not isinstance(value, model):
        raise TypeError("invalid returned model")
    return model.model_validate(_plain(value.model_dump(mode="python", warnings=False)))
