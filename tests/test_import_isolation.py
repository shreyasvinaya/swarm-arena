import subprocess
import sys

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
