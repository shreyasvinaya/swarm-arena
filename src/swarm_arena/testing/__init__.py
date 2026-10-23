"""Conformance kit for connector authors: check tools, checks, judges, and managers offline."""

import importlib as _importlib
from typing import Any as _Any

from swarm_arena._facade import publish as _publish
from swarm_arena.testing._artifacts import read_logs, read_output_file
from swarm_arena.testing._candidates import make_candidate
from swarm_arena.testing._conformance import check_check, check_judge, check_manager, check_tool

__all__: list[str] = [
    "FakeModel",
    "FakeTurn",
    "check_check",
    "check_judge",
    "check_manager",
    "check_tool",
    "make_candidate",
    "read_logs",
    "read_output_file",
]

_LAZY = ("FakeModel", "FakeTurn")


def __getattr__(name: str) -> _Any:
    if name not in _LAZY:
        raise AttributeError(name)
    _importlib.import_module("swarm_arena.llm")
    obj = getattr(_importlib.import_module("swarm_arena.testing._fake_model"), name)
    globals()[name] = obj
    _publish(__name__, [name])
    return obj


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


_publish(__name__, [n for n in __all__ if n not in _LAZY])
