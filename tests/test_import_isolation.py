import importlib
import inspect
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _surface import LAZY_DEPS, PUBLIC_ALL, public_object

_RAY_HINT = 'uv add "swarm-arena[ray]"'

# The module list of test_public_api.PUBLIC, repeated so this test stands alone.
PUBLIC = (
    "swarm_arena",
    "swarm_arena.tool",
    "swarm_arena.scoring",
    "swarm_arena.manager",
    "swarm_arena.planning",
    "swarm_arena.board",
    "swarm_arena.executor",
    "swarm_arena.config",
    "swarm_arena.protocol",
    "swarm_arena.testing",
)
OPTIONAL = ("agents", "litellm", "ray", "s3fs", "langfuse", "a2a")


def test_core_import_does_not_load_gateway_transport() -> None:
    code = """
import sys
import swarm_arena
assert 'swarm_arena._gateway.litellm_proxy' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True, timeout=30
    )


@pytest.mark.parametrize("missing", ["openai-agents", "litellm"])
@pytest.mark.parametrize(
    "entry", ["import", "FakeModel", "FakeTurn", "ref", "validate", "snapshot"]
)
def test_missing_llm_extra_keeps_hint(missing: str, entry: str) -> None:
    code = f"""
import importlib, importlib.metadata, sys
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0, 'tests')
from _fakes.run import plan
from _fakes.secrecy import assert_secret_absent
from swarm_arena import Ref, RunConfig, validate_plan
from swarm_arena._planning.validate import admit_revision
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.local import LocalStore
original = importlib.metadata.version
def version(name):
    if name == {missing!r}:
        raise importlib.metadata.PackageNotFoundError('missing-extra-secret-sentinel')
    return original(name)
importlib.metadata.version = version
try:
    entry = {entry!r}
    ref = Ref(path='swarm_arena.llm:AgentsSDKManager', config={{'model': 'm'}})
    if entry == 'import':
        importlib.import_module('swarm_arena.llm')
    elif entry in ('FakeModel', 'FakeTurn'):
        getattr(importlib.import_module('swarm_arena.testing'), entry)
    elif entry == 'ref':
        ref.load()
    else:
        draft = plan(rounds=1)
        draft.teams[0].agents[0] = draft.teams[0].agents[0].model_copy(update={{'manager': ref}})
        if entry == 'validate':
            validate_plan(draft, RunConfig())
        else:
            with TemporaryDirectory() as root:
                admit_revision(draft, RunConfig(), ArtifactStore(LocalStore(Path(root))))
except (ModuleNotFoundError, ValueError) as error:
    assert_secret_absent(error, 'missing-extra-secret-sentinel', sanitized=True)
    print(str(error))
else:
    raise AssertionError('missing extra was accepted')
assert 'agents' not in sys.modules
assert 'litellm' not in sys.modules
"""
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True, timeout=60
    )
    assert "swarm_arena.llm needs the 'llm' extra:" in done.stdout
    assert "swarm-arena[llm]" in done.stdout
    assert 'uv add "swarm-arena[llm]"' in done.stdout
    assert "missing-extra-secret-sentinel" not in done.stdout + done.stderr


def test_core_imports_load_no_optional_dependency() -> None:
    code = (
        "import importlib, sys\n"
        f"for m in {sorted(PUBLIC)!r}: importlib.import_module(m)\n"
        f"print(sorted({{n.split('.')[0] for n in sys.modules}} & set({OPTIONAL!r})))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True
    )
    assert done.stdout.strip() == "[]"


def test_testing_facade_does_not_load_agents() -> None:
    code = """
import sys
import swarm_arena.testing as testing
sys.path.insert(0, 'tests')
import _fakes.fake_models
assert 'agents' not in sys.modules
assert 'litellm' not in sys.modules
assert {'FakeModel', 'FakeTurn'} <= set(testing.__all__) <= set(dir(testing))
assert 'agents' not in sys.modules
assert not hasattr(testing, 'unknown_testing_name')
"""
    subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True, timeout=30
    )


def test_testing_lazy_exports_publish_and_cache() -> None:
    code = """
import importlib.metadata
try:
    importlib.metadata.version('openai-agents')
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(0)
import swarm_arena.testing as testing
assert 'FakeModel' not in vars(testing)
model = testing.FakeModel
assert vars(testing)['FakeModel'] is model
assert testing.FakeModel is model
assert model.__module__ == model.get_response.__module__ == 'swarm_arena.testing'
assert 'FakeTurn' not in vars(testing)
assert testing.FakeTurn.__module__ == 'swarm_arena.testing'
assert dir(testing) == sorted(set(dir(testing)))
"""
    subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=True, timeout=30
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
BLOCK_RAY = "import sys; sys.modules['ray'] = None\n"
FAKE_RAY_DIST = """
import importlib.metadata
_original_version = importlib.metadata.version
def version(name):
    return '2.59.0' if name == 'ray' else _original_version(name)
importlib.metadata.version = version
"""
HIDE_RAY_DIST = (
    FAKE_RAY_DIST
    + """
def version(name):
    if name == 'ray':
        raise importlib.metadata.PackageNotFoundError('ray-secret-sentinel')
    return _original_version(name)
importlib.metadata.version = version
"""
)
BOOM_FINDER = """
import sys
class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ray':
            raise ImportError('boom inside ray')
sys.meta_path.insert(0, Finder())
"""
MISSING_SUB_FINDER = BOOM_FINDER.replace(
    "ImportError('boom inside ray')",
    """ModuleNotFoundError("No module named 'ray._private'", name='ray._private')""",
)


def test_ray_facade_error_without_extra() -> None:
    result = subprocess.run(
        [sys.executable, "-c", FAKE_RAY_DIST + BLOCK_RAY + "import swarm_arena.ray"],
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode != 0
    assert _RAY_HINT in result.stderr


def test_ray_facade_error_with_tests_dir_on_path() -> None:
    code = HIDE_RAY_DIST + "import sys; sys.path.insert(0, 'tests'); import swarm_arena.ray"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert result.returncode != 0
    assert _RAY_HINT in result.stderr
    assert "ModuleNotFoundError" in result.stderr
    assert "ray-secret-sentinel" not in result.stderr


def test_ray_facade_keeps_other_import_errors() -> None:
    result = subprocess.run(
        [sys.executable, "-c", FAKE_RAY_DIST + BOOM_FINDER + "import swarm_arena.ray"],
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode != 0
    assert "boom inside ray" in result.stderr
    assert "pip install" not in result.stderr


def test_ray_facade_keeps_missing_vendor_submodule() -> None:
    result = subprocess.run(
        [sys.executable, "-c", FAKE_RAY_DIST + MISSING_SUB_FINDER + "import swarm_arena.ray"],
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode != 0
    assert "No module named 'ray._private'" in result.stderr
    assert "swarm-arena[" not in result.stderr


def test_ray_facade_refuses_namespace_package_with_distribution() -> None:
    code = (
        FAKE_RAY_DIST
        + """
import importlib.machinery, sys
class NamespaceFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ray':
            spec = importlib.machinery.ModuleSpec('ray', loader=None, is_package=True)
            spec.submodule_search_locations = ['tests/ray']
            return spec
sys.meta_path.insert(0, NamespaceFinder())
sys.path.insert(0, 'tests')
import swarm_arena.ray
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
        timeout=30,
    )
    assert result.returncode != 0
    assert _RAY_HINT in result.stderr


class _Hide:
    """Hide a module prefix while retaining distribution-metadata discovery."""

    def __init__(self, finder: Any, dep: str) -> None:
        self.finder = finder
        self.dep = dep

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name == self.dep or name.startswith(self.dep + "."):
            return None
        find_spec = getattr(self.finder, "find_spec", None)
        return None if find_spec is None else find_spec(name, path, target)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.finder, attr)


def _hide(prefix: str) -> None:
    sys.meta_path[:] = [_Hide(finder, prefix) for finder in sys.meta_path]


HIDE_SOURCE = (
    "import sys\nfrom typing import Any\n"
    + inspect.getsource(_Hide)
    + "\n"
    + inspect.getsource(_hide)
)


@pytest.mark.parametrize(
    ("module", "dep", "extra"),
    [
        ("swarm_arena.llm", "agents", "llm"),
        ("swarm_arena.llm", "litellm", "llm"),
        ("swarm_arena.ray", "ray", "ray"),
    ],
)
def test_extra_facade_names_its_extra(module: str, dep: str, extra: str) -> None:
    # A fresh interpreter exercises the guard even if a prior test loaded it.
    code = (
        HIDE_SOURCE
        + f"""
_hide({dep!r})
import importlib
try:
    importlib.import_module({module!r})
except ModuleNotFoundError as error:
    assert 'swarm-arena[{extra}]' in str(error), str(error)
else:
    raise AssertionError('missing extra was accepted')
"""
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", timeout=60
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stderr == ""


@pytest.mark.parametrize(
    "name", [name for module, name in LAZY_DEPS if module == "swarm_arena.testing"]
)
@pytest.mark.parametrize("dep", LAZY_DEPS[("swarm_arena.testing", "FakeModel")])
def test_fake_model_names_the_llm_extra(name: str, dep: str) -> None:
    code = (
        HIDE_SOURCE
        + f"""
_hide({dep!r})
import swarm_arena.testing as testing
assert {name!r} not in vars(testing)
try:
    getattr(testing, {name!r})
except ModuleNotFoundError as error:
    assert 'swarm-arena[llm]' in str(error), str(error)
else:
    raise AssertionError('missing extra was accepted')
assert {name!r} not in vars(testing)
"""
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", timeout=60
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stderr == ""


@pytest.mark.parametrize(
    ("module", "sub"),
    [("swarm_arena.llm", "agents.models"), ("swarm_arena.ray", "ray._private")],
)
def test_extra_facade_keeps_missing_vendor_submodule(module: str, sub: str) -> None:
    public_object(module, PUBLIC_ALL[module][0])
    code = HIDE_SOURCE + f"_hide({sub!r})\nimport {module}\n"
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, encoding="utf-8", timeout=60
    )
    assert done.returncode != 0
    assert f"No module named '{sub}'" in done.stderr
    assert "swarm-arena[" not in done.stderr


def test_llm_facade_keeps_missing_vendor_name() -> None:
    public_object("swarm_arena.llm", PUBLIC_ALL["swarm_arena.llm"][0])
    code = """
import agents
del agents.Model
import swarm_arena.llm
"""
    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
    )
    assert done.returncode != 0
    assert "ImportError: cannot import name 'Model' from 'agents'" in done.stderr
    assert "swarm-arena[" not in done.stderr


@pytest.mark.parametrize("module", ["swarm_arena.llm", "swarm_arena.ray"])
def test_extra_facade_keeps_other_import_errors(
    module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = PUBLIC_ALL[module][0]
    obj = public_object(module, name)
    homes = [
        home
        for key, home in list(sys.modules.items())
        if key.startswith("swarm_arena.") and key != module and vars(home).get(name) is obj
    ]
    assert homes != []
    for home in homes:
        monkeypatch.delitem(vars(home), name)
    monkeypatch.delitem(sys.modules, module)
    with pytest.raises(ImportError, match=re.escape(f"cannot import name '{name}'")) as err:
        importlib.import_module(module)
    assert "swarm-arena[" not in str(err.value)


def test_s3_missing_extra_names_release_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._errors import ConditionalWritesUnsupported
    from swarm_arena._store.s3 import S3Store

    original = builtins.__import__

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "s3fs":
            raise ImportError("s3-secret-sentinel")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ConditionalWritesUnsupported) as caught:
        S3Store("s3://bucket/prefix")
    assert 'uv add "swarm-arena[s3]"' in str(caught.value)
    assert "https://github.com/shreyasvinaya/swarm-arena#install" in str(caught.value)
    assert_secret_absent(caught.value, "s3-secret-sentinel")


@pytest.mark.parametrize("parameter", ["IfMatch", "IfNoneMatch"])
def test_s3_old_sdk_hint_requires_upgrade_without_put(
    monkeypatch: pytest.MonkeyPatch, parameter: str
) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._errors import ConditionalWritesUnsupported
    from swarm_arena._store.s3 import S3Store

    store = object.__new__(S3Store)
    monkeypatch.setattr(store, "_key", lambda key: key)
    monkeypatch.setattr(store, "_conditional_params", lambda: frozenset({"Bucket", "Key", "Body"}))

    def no_put(*args: Any, **kwargs: Any) -> str:
        pytest.fail("an unsupported SDK must not issue PutObject")

    monkeypatch.setattr(store, "_put_object", no_put)
    with pytest.raises(ConditionalWritesUnsupported) as caught:
        store._write("s3-key-secret", b"s3-data-secret", parameter, "s3-value-secret")
    assert parameter in str(caught.value)
    assert 'uv add --upgrade "swarm-arena[s3]"' in str(caught.value)
    for secret in ("s3-key-secret", "s3-data-secret", "s3-value-secret"):
        assert_secret_absent(caught.value, secret)
