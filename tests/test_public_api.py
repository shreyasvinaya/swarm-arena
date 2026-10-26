import doctest
import importlib
import importlib.metadata
import importlib.util
import inspect
import pkgutil
import subprocess
import sys
import types

import pytest
from _fakes.dependencies import ray_dependency
from _surface import (
    DISTRIBUTIONS,
    ENTRY_MODULES,
    PUBLIC_ALL,
    PUBLIC_SUBMODULES,
    deps_present,
    import_public,
    public_object,
    resolve,
)
from pydantic import BaseModel

import swarm_arena
from swarm_arena._facade import publish


@pytest.mark.parametrize("module", PUBLIC_ALL)
def test_all_matches_spec(module: str) -> None:
    assert import_public(module).__all__ == PUBLIC_ALL[module]


@pytest.mark.parametrize("module", PUBLIC_ALL)
def test_public_names_report_facade_module(module: str) -> None:
    import_public(module)
    objs = [(n, public_object(module, n)) for n in PUBLIC_ALL[module] if deps_present(module, n)]
    assert [n for n, o in objs if o.__module__ != module] == []
    assert [
        n
        for n, o in objs
        if isinstance(o, type) and issubclass(o, BaseModel) and not o.__pydantic_complete__
    ] == []


def test_each_public_name_has_one_home() -> None:
    names = [n for ns in PUBLIC_ALL.values() for n in ns]
    assert sorted(n for n in set(names) if names.count(n) > 1) == []


@pytest.mark.parametrize("module", PUBLIC_ALL)
def test_facades_hold_only_imports_and_all(module: str) -> None:
    mod = import_public(module)
    allowed = set(mod.__all__) | (
        PUBLIC_SUBMODULES | ENTRY_MODULES if module == "swarm_arena" else set()
    )
    assert {n for n in vars(mod) if not n.startswith("_")} - allowed == set()


def test_no_other_public_modules() -> None:
    top = {m.name for m in pkgutil.iter_modules(swarm_arena.__path__) if not m.name.startswith("_")}
    assert top == PUBLIC_SUBMODULES | ENTRY_MODULES
    for pkg in ("swarm_arena.llm", "swarm_arena.ray", "swarm_arena.testing"):
        spec = importlib.util.find_spec(pkg)
        assert spec is not None
        assert spec.submodule_search_locations is not None
        subs = {m.name for m in pkgutil.iter_modules(spec.submodule_search_locations)}
        assert {s for s in subs if not s.startswith("_")} == set(), pkg


@pytest.mark.parametrize("module", ["swarm_arena", "swarm_arena.llm", "swarm_arena.testing"])
def test_facade_has_no_extra_public_names(module: str) -> None:
    facade = import_public(module)
    children = {
        name
        for name, value in vars(facade).items()
        if isinstance(value, types.ModuleType) and value.__name__.startswith(f"{module}.")
    }
    for names in (vars(facade), dir(facade)):
        assert {name for name in names if not name.startswith("_")} <= set(
            facade.__all__
        ) | children


def test_version_is_package_metadata() -> None:
    assert swarm_arena.__version__ == importlib.metadata.version("swarm-arena")
    assert "__version__" not in swarm_arena.__all__


def test_doctest_finder_collects_method_examples() -> None:
    assert "swarm_arena.Ref.of" in {t.name for t in doctest.DocTestFinder().find(swarm_arena)}


def test_publish_lets_doctest_find_method_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    class Demo:
        def m(self) -> int:
            """
            Return one.

            >>> Demo().m()
            1
            """
            return 1

        @classmethod
        def c(cls) -> int:
            """
            Return two.

            >>> Demo.c()
            2
            """
            return 2

        @staticmethod
        def s() -> int:
            """
            Return three.

            >>> Demo.s()
            3
            """
            return 3

        @property
        def p(self) -> int:
            """
            Four.

            >>> Demo().p
            4
            """
            return 4

    module = types.ModuleType("demo_facade")
    vars(module)["Demo"] = Demo
    monkeypatch.setitem(sys.modules, "demo_facade", module)
    publish("demo_facade", ["Demo"])
    found = {t.name for t in doctest.DocTestFinder().find(sys.modules["demo_facade"])}
    assert {
        "demo_facade.Demo.m",
        "demo_facade.Demo.c",
        "demo_facade.Demo.s",
        "demo_facade.Demo.p",
    } <= found


CORE = sorted(
    set(PUBLIC_ALL) - {"swarm_arena", "swarm_arena.testing", "swarm_arena.llm", "swarm_arena.ray"}
)


@pytest.mark.parametrize("pydantic_model", [False, True])
def test_publish_invalidates_original_class_source_location(
    pydantic_model: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Plain:
        class Nested:
            pass

    class Model(BaseModel):
        class Nested(BaseModel):
            pass

    demo = Model if pydantic_model else Plain
    # Python 3.13 adds this metadata; also exercise invalidation on older versions.
    for cls in (demo, demo.Nested):
        type.__setattr__(cls, "__firstlineno__", 1)
    module = types.ModuleType("source_location_facade")
    module.__file__ = __file__
    vars(module)["Demo"] = demo
    monkeypatch.setitem(sys.modules, module.__name__, module)
    publish(module.__name__, ["Demo"])
    for cls in (demo, demo.Nested):
        assert cls.__module__ == module.__name__
        assert "__firstlineno__" not in vars(cls)
        try:
            source, _ = inspect.getsourcelines(cls)
        except OSError:
            pass  # Retargeted classes need not have inspectable source.
        else:
            assert source[0].lstrip().startswith(f"class {cls.__name__}")


def test_root_import_publishes_core_facades() -> None:  # __module__ never depends on import order
    code = f"import sys, swarm_arena; print(sorted(set({CORE!r}) - set(sys.modules)))"
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True
    )
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize("missing", ["agents", "litellm"])
def test_llm_facade_names_the_extra_when_missing(missing: str) -> None:
    code = f"import sys; sys.modules[{missing!r}] = None; import swarm_arena.llm"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", timeout=30
    )
    assert proc.returncode != 0
    assert (
        'uv add "swarm-arena[llm] @ git+https://github.com/shreyasvinaya/swarm-arena"'
        in proc.stderr
    )
    assert "ModuleNotFoundError" in proc.stderr


def test_llm_facade_publishes_its_names() -> None:
    llm = import_public("swarm_arena.llm")
    assert all(getattr(llm, name).__module__ == "swarm_arena.llm" for name in llm.__all__)


@pytest.mark.parametrize(("dist", "module"), [("openai-agents", "agents"), ("litellm", "litellm")])
def test_llm_facade_checks_distribution_metadata(dist: str, module: str) -> None:
    code = f"""
import importlib.metadata

def version(name):
    if name == {dist!r}:
        raise importlib.metadata.PackageNotFoundError(name)
    return "0"

importlib.metadata.version = version
try:
    import swarm_arena.llm
except ModuleNotFoundError as error:
    assert error.name == {module!r}
    assert (
        'uv add "swarm-arena[llm] @ git+https://github.com/shreyasvinaya/swarm-arena"'
        in str(error)
    )
else:
    raise AssertionError("missing extra imported")
"""
    subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True, timeout=30
    )


@pytest.mark.parametrize(
    ("error_type", "name"),
    [("ModuleNotFoundError", "transitive_dependency"), ("ImportError", "agents")],
)
def test_llm_facade_preserves_other_import_errors(error_type: str, name: str) -> None:
    import_public("swarm_arena.llm")
    code = f"""
import builtins
original = builtins.__import__
error = {error_type}("upstream failure", name={name!r})

def importing(module, *args, **kwargs):
    if module == "agents":
        raise error
    return original(module, *args, **kwargs)

builtins.__import__ = importing
try:
    import swarm_arena.llm
except ImportError as caught:
    assert caught is error
else:
    raise AssertionError("broken upstream imported")
"""
    subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True, timeout=30
    )


def test_ray_all() -> None:
    module = ray_dependency()
    assert module.__all__ == ["RayExecutor"]


def test_ray_facade_publishes_methods() -> None:
    module = ray_dependency()
    cls = module.RayExecutor
    assert {
        cls.__module__,
        cls.Config.__module__,
        cls.submit.__module__,
        cls.confirm_dead.__module__,
    } == {"swarm_arena.ray"}


@pytest.mark.parametrize("missing", [None, "openai-agents", "litellm", "ray"])
@pytest.mark.parametrize(
    ("module", "name", "required"),
    [
        ("swarm_arena", None, ()),
        ("swarm_arena.testing", None, ()),
        ("swarm_arena.testing", "read_logs", ()),
        ("swarm_arena.testing", "FakeModel", ("openai-agents", "litellm")),
        ("swarm_arena.testing", "FakeTurn", ("openai-agents", "litellm")),
        ("swarm_arena.llm", None, ("openai-agents", "litellm")),
        ("swarm_arena.ray", None, ("ray",)),
    ],
)
def test_surface_dependencies_use_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
    missing: str | None,
    module: str,
    name: str | None,
    required: tuple[str, ...],
) -> None:
    checked: list[str] = []

    def version(distribution: str) -> str:
        checked.append(distribution)
        if distribution == missing:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return "0"

    def no_probe(*args: object, **kwargs: object) -> None:
        pytest.fail("dependency detection must not import or locate packages")

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib, "import_module", no_probe)
    monkeypatch.setattr(importlib.util, "find_spec", no_probe)
    assert deps_present(module, name) == (missing not in required)
    assert set(checked) <= set(required)
    if missing not in required:
        assert checked == list(required)


@pytest.mark.parametrize("missing", DISTRIBUTIONS.values())
def test_surface_import_skips_missing_extra(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    def version(distribution: str) -> str:
        if distribution == missing:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return "0"

    monkeypatch.setattr(importlib.metadata, "version", version)
    extra = "ray" if missing == "ray" else "llm"
    with pytest.raises(pytest.skip.Exception, match=f"needs the {extra} extra"):
        import_public(f"swarm_arena.{extra}")


@pytest.mark.parametrize("missing", ["openai-agents", "litellm"])
@pytest.mark.parametrize("name", ["FakeModel", "FakeTurn"])
def test_surface_lazy_object_skips_either_missing_llm_dependency(
    monkeypatch: pytest.MonkeyPatch, missing: str, name: str
) -> None:
    def version(distribution: str) -> str:
        if distribution == missing:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return "0"

    monkeypatch.setattr(importlib.metadata, "version", version)
    with pytest.raises(pytest.skip.Exception, match="needs the llm extra"):
        public_object("swarm_arena.testing", name)
    assert public_object("swarm_arena", "Goal") is swarm_arena.Goal


@pytest.mark.parametrize(
    "error",
    [
        ImportError("broken import"),
        ModuleNotFoundError("broken vendor module", name="agents.models"),
    ],
)
@pytest.mark.parametrize("loader", [import_public, resolve])
def test_surface_helpers_preserve_installed_package_import_failures(
    monkeypatch: pytest.MonkeyPatch, error: ImportError, loader: types.FunctionType
) -> None:
    def version(distribution: str) -> str:
        return "0"

    def importing(module: str) -> types.ModuleType:
        raise error

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib, "import_module", importing)
    with pytest.raises(type(error)) as caught:
        loader("swarm_arena.llm")
    assert caught.value is error


@pytest.mark.parametrize(
    ("dotted", "expected"),
    [
        ("swarm_arena", swarm_arena),
        ("swarm_arena.tool", swarm_arena.tool),
        ("swarm_arena.Goal", swarm_arena.Goal),
        ("swarm_arena.Ref.of", swarm_arena.Ref.of),
        ("swarm_arena.ScriptTool.Config", swarm_arena.ScriptTool.Config),
    ],
)
def test_surface_resolve_modules_and_nested_members(dotted: str, expected: object) -> None:
    assert resolve(dotted) == expected


def test_surface_resolve_reports_missing_module_and_attribute() -> None:
    with pytest.raises(ModuleNotFoundError, match="nonexistent_surface_module"):
        resolve("nonexistent_surface_module.Thing")
    with pytest.raises(AttributeError, match="missing_surface_member"):
        resolve("swarm_arena.Goal.missing_surface_member")


@pytest.mark.parametrize("missing", ["openai-agents", "litellm", "ray"])
def test_docstring_collector_does_not_import_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    from test_docstrings import public_objects

    original_version = importlib.metadata.version

    def version(distribution: str) -> str:
        if distribution == missing:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return original_version(distribution)

    original = importlib.import_module
    extra = "ray" if missing == "ray" else "llm"

    def importing(module: str) -> types.ModuleType:
        assert module != f"swarm_arena.{extra}"
        return original(module)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(importlib, "import_module", importing)
    names = [str(param.values[0]) for param in public_objects() if not param.marks]
    assert not any(name.startswith(f"swarm_arena.{extra}.") for name in names)
    if extra == "llm":
        assert not any(name.startswith("swarm_arena.testing.Fake") for name in names)
    assert "swarm_arena.testing.read_logs" in names
