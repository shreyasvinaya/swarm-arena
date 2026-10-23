import subprocess
import sys

import pytest

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
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert "swarm_arena.llm needs the 'llm' extra:" in done.stdout
    assert "swarm-arena[llm]" in done.stdout
    assert (
        'uv add "swarm-arena[llm] @ git+https://github.com/shreyasvinaya/swarm-arena"'
        in done.stdout
    )
    assert "missing-extra-secret-sentinel" not in done.stdout + done.stderr


def test_core_imports_load_no_optional_dependency() -> None:
    code = (
        "import importlib, sys\n"
        f"for m in {sorted(PUBLIC)!r}: importlib.import_module(m)\n"
        f"print(sorted({{n.split('.')[0] for n in sys.modules}} & set({OPTIONAL!r})))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
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
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=30
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
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=30
    )
