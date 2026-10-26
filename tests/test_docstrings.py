import doctest
import enum
import importlib.metadata
import inspect
import pydoc
import types

import numpydoc.validate
import pytest
from _pytest.mark.structures import ParameterSet
from _surface import LAZY_DEPS, PUBLIC_ALL, deps_present, import_public, resolve
from pydantic import BaseModel

from swarm_arena.board import Board, BoardView

IGNORED = frozenset({"SA01", "ES01"})
DOCTEST_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

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
    names = {p.values[0] for p in public_objects()}
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
                if owner is cls and value.__qualname__ == f"{prefix}{attr}":
                    found.append(f"{dotted}.{attr}")
                    found.extend(_members(value, f"{dotted}.{attr}"))
                continue
            func = _function(value)
            if func is not None and func.__qualname__.startswith(prefix):
                found.append(f"{dotted}.{attr}")
    return found


def public_objects() -> list[ParameterSet]:
    """Collect public contracts, retaining explicit cases for unavailable extras."""
    params: list[ParameterSet] = []
    for module, names in PUBLIC_ALL.items():
        if not deps_present(module):
            extra = module.rsplit(".", 1)[-1]
            params.append(
                pytest.param(module, marks=pytest.mark.skip(reason=f"needs the {extra} extra"))
            )
            continue
        mod = import_public(module)
        for name in names:
            dotted = f"{module}.{name}"
            if not deps_present(module, name):
                params.append(
                    pytest.param(dotted, marks=pytest.mark.skip(reason="needs the llm extra"))
                )
                continue
            params.append(pytest.param(dotted))
            obj = getattr(mod, name)
            if isinstance(obj, type):
                params.extend(pytest.param(member) for member in _members(obj, dotted))
    return params


def test_public_objects_cover_nested_classes() -> None:
    names = {p.values[0] for p in public_objects()}
    assert {
        "swarm_arena.ScriptTool.Config",
        "swarm_arena.ScriptedManager.Request",
        "swarm_arena.executor.Job.Stage",
    } <= names


def test_public_objects_exclude_injected_methods_and_config_aliases() -> None:
    names = {p.values[0] for p in public_objects()}
    assert "swarm_arena.Plan.model_post_init" not in names
    assert "swarm_arena.tool.Tool.Config" not in names
    assert len(names) == len(public_objects())


def test_members_recurse_into_owned_nested_classes() -> None:
    class Outer:
        Config = BaseModel
        injected = public_objects

        class Nested:
            class Leaf:
                def method(self) -> None:
                    pass

            @classmethod
            def build(cls) -> None:
                pass

            @staticmethod
            def helper() -> None:
                pass

            @property
            def value(self) -> int:
                return 1

        Alias = Nested

    assert set(_members(Outer, "example.Outer")) == {
        "example.Outer.Nested",
        "example.Outer.Nested.Leaf",
        "example.Outer.Nested.Leaf.method",
        "example.Outer.Nested.build",
        "example.Outer.Nested.helper",
        "example.Outer.Nested.value",
    }


def test_public_objects_preserve_installed_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    error = ModuleNotFoundError("broken installed package", name="agents.internal")

    def broken_import(module: str) -> types.ModuleType:
        raise error

    monkeypatch.setattr(__name__ + ".PUBLIC_ALL", {"swarm_arena.llm": ["LLMJudge"]})
    monkeypatch.setattr(__name__ + ".deps_present", lambda module, name=None: True)
    monkeypatch.setattr(__name__ + ".import_public", broken_import)
    with pytest.raises(ModuleNotFoundError) as caught:
        public_objects()
    assert caught.value is error


@pytest.mark.parametrize("missing", ["openai-agents", "litellm", "ray"])
def test_public_objects_keep_missing_extra_cases(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    def version(distribution: str) -> str:
        if distribution == missing:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return "1.0"

    # Model each partially installed extra without importing a missing facade/name.
    original_import = import_public

    def guarded_import(module: str) -> types.ModuleType:
        assert deps_present(module), module
        return original_import(module)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(__name__ + ".import_public", guarded_import)
    # Do not require unrelated optional packages in a core-only environment.
    modules = (
        {"swarm_arena.ray": PUBLIC_ALL["swarm_arena.ray"]}
        if missing == "ray"
        else {
            "swarm_arena.llm": PUBLIC_ALL["swarm_arena.llm"],
            "swarm_arena.testing": ["FakeModel", "FakeTurn"],
        }
    )
    monkeypatch.setattr(__name__ + ".PUBLIC_ALL", modules)
    params = public_objects()
    expected = (
        {"swarm_arena.ray"}
        if missing == "ray"
        else {"swarm_arena.llm", "swarm_arena.testing.FakeModel", "swarm_arena.testing.FakeTurn"}
    )
    assert {p.values[0] for p in params} == expected
    extra = "ray" if missing == "ray" else "llm"
    for param in params:
        assert [(mark.name, mark.kwargs) for mark in param.marks] == [
            ("skip", {"reason": f"needs the {extra} extra"})
        ]


def _is_enum(name: str) -> bool:
    obj = pydoc.locate(name)
    return isinstance(obj, type) and issubclass(obj, enum.Enum)


@pytest.mark.parametrize("dotted", public_objects())
def test_numpydoc_valid(dotted: str) -> None:
    errors = [
        (code, msg)
        for code, msg in numpydoc.validate.validate(dotted)["errors"]
        if code not in IGNORED and not (code == "PR01" and _is_enum(dotted))
    ]
    assert errors == []  # numpydoc reads Enum's constructor as parameters, hence the PR01 skip


@pytest.mark.parametrize("dotted", public_objects())
def test_examples_are_doctests(dotted: str) -> None:
    module = max((m for m in PUBLIC_ALL if dotted.startswith(f"{m}.")), key=len)
    found = doctest.DocTestFinder(recurse=False).find(
        resolve(dotted), dotted, globs=dict(vars(import_public(module)))
    )
    assert any(t.examples for t in found), f"{dotted}: the Examples section has no >>> line"
    if (module, dotted[len(module) + 1 :].partition(".")[0]) in LAZY_DEPS:
        runner = doctest.DocTestRunner(optionflags=DOCTEST_FLAGS)
        for test in found:
            runner.run(test)
        assert runner.failures == 0, f"{dotted}: an Examples line failed (see the captured stdout)"


@pytest.mark.parametrize("module", PUBLIC_ALL)
def test_facade_docstring(module: str) -> None:
    doc = (import_public(module).__doc__ or "").strip()
    assert doc != ""
    assert doc.splitlines()[0].endswith(".")
