import os
import subprocess
import sys


def test_cost_map_pinned_at_import() -> None:
    env = dict(os.environ)
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import swarm_arena.llm, os; print(os.environ['LITELLM_LOCAL_MODEL_COST_MAP'])",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert proc.stdout.splitlines()[0] == "True"


def test_cost_map_repinned_when_litellm_imported_first() -> None:
    env = dict(os.environ)
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    env["LITELLM_MODEL_COST_MAP_URL"] = "http://127.0.0.1:9/"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import litellm; litellm.model_cost['zz-test'] = {}; "
            "import swarm_arena.llm; print('zz-test' in litellm.model_cost)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert proc.stdout.strip() == "False"
