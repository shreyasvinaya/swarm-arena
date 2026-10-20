import enum
import importlib
import inspect
import pydoc
import types

import numpydoc.validate
import pytest
from pydantic import BaseModel
from test_public_api import PUBLIC

from swarm_arena.board import Board, BoardView

IGNORED = {"SA01", "ES01"}

BOARD_READS = {
    "best",
    "learnings",
    "search_learnings",
    "get_learning",
    "get_candidate",
    "scores",
    "lineage",
    "leaderboard",
}


@pytest.mark.parametrize("cls", [Board, BoardView])
def test_inherited_board_methods_are_checked(cls: type) -> None:
    names = set(public_objects())
    assert {f"swarm_arena.board.{cls.__name__}.{name}" for name in BOARD_READS} <= names


@pytest.mark.parametrize("cls", [Board, BoardView])
def test_board_docstrings_use_public_names(cls: type) -> None:
    for name in [None, *BOARD_READS, *[n for n in vars(cls) if not n.startswith("_")]]:
        obj = cls if name is None else getattr(cls, name)
        if not (name is None or inspect.isfunction(obj)):
            continue
        assert "swarm_arena._" not in (inspect.getdoc(obj) or ""), name


def _function(value: object) -> types.FunctionType | None:
    """Return the function behind a class-body routine or property, else None."""
    if isinstance(value, classmethod | staticmethod):
        value = value.__func__
    elif isinstance(value, property):
        value = value.fget
    return value if inspect.isfunction(value) else None


def _members(cls: type, dotted: str) -> list[str]:
    """Include inherited package contracts without traversing Pydantic internals."""
    found: list[str] = []
    seen: set[str] = set()
    for owner in cls.__mro__:
        if owner is not cls and (
            not owner.__module__.startswith("swarm_arena") or issubclass(owner, BaseModel)
        ):
            continue
        prefix = owner.__qualname__ + "."
        for attr, value in vars(owner).items():
            if attr.startswith("_") or attr in seen:
                continue
            seen.add(attr)
            if isinstance(value, type):
                if owner is cls and value.__qualname__.startswith(prefix):
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
