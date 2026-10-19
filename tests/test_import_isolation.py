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
