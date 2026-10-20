"""Expose an import side effect for pre-import catalog authorization regressions."""

import os
from pathlib import Path

if marker := os.environ.get("SWARM_ARENA_TEST_CANARY"):
    Path(marker).write_text("imported", encoding="utf-8")


class Tool:
    """Stand in for an executable component that authorization must not import."""
