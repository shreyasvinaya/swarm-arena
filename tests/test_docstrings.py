import enum
import importlib
import inspect
import pydoc
import types

import numpydoc.validate
import pytest
from test_public_api import PUBLIC

IGNORED = {"SA01", "ES01"}


def _function(value: object) -> types.FunctionType | None:
    """Return the function behind a class-body routine or property, else None."""
    if isinstance(value, classmethod | staticmethod):
        value = value.__func__
    elif isinstance(value, property):
        value = value.fget
    return value if inspect.isfunction(value) else None


def _members(cls: type, dotted: str) -> list[str]:
    """Return the public routines, properties, and nested classes defined in the class body."""
    prefix = cls.__qualname__ + "."
    found: list[str] = []
    for attr, value in vars(cls).items():
        if attr.startswith("_"):
            continue
        if isinstance(value, type):
            if value.__qualname__.startswith(prefix):
                found.append(f"{dotted}.{attr}")
                found.extend(_members(value, f"{dotted}.{attr}"))
            continue
        func = _function(value)
        if func is not None and func.__qualname__.startswith(prefix):
            found.append(f"{dotted}.{attr}")
    return found


def public_objects() -> list[str]:
    names: list[str] = []
    for module in sorted(PUBLIC):
        mod = importlib.import_module(module)
        for name in mod.__all__:
            dotted = f"{module}.{name}"
            names.append(dotted)
            obj = getattr(mod, name)
            if isinstance(obj, type):
                names.extend(_members(obj, dotted))
    return names


def _is_enum(name: str) -> bool:
    obj = pydoc.locate(name)
    return isinstance(obj, type) and issubclass(obj, enum.Enum)


@pytest.mark.parametrize("name", public_objects())
def test_numpydoc(name: str) -> None:
    errors = [
        (code, msg)
        for code, msg in numpydoc.validate.validate(name)["errors"]
        if code not in IGNORED and not (code == "PR01" and _is_enum(name))
    ]
    assert errors == []  # numpydoc reads Enum's constructor as parameters, hence the PR01 skip
