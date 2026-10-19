"""Doctest collection for ``src/``: skip the modules whose optional extra is not installed."""

import importlib.metadata


def _missing(dist: str) -> bool:
    """Return True when the distribution ``dist`` is not installed."""
    try:
        importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return True
    return False


# By distribution, never find_spec: pythonpath = ["tests"] makes tests/ray/ a namespace package.
collect_ignore: list[str] = []
if _missing("openai-agents"):
    collect_ignore += ["src/swarm_arena/llm", "src/swarm_arena/testing/_fake_model.py"]
if _missing("ray"):
    collect_ignore.append("src/swarm_arena/ray")
if _missing("s3fs"):
    collect_ignore.append("src/swarm_arena/_store/s3.py")
