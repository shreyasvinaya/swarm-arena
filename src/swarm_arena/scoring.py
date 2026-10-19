"""Checks, judges, and scores."""

from swarm_arena._facade import publish as _publish
from swarm_arena._model.results import CheckResult
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._scoring.judge import Judge, JudgeContext
from swarm_arena._scoring.lexicographic import Lexicographic  # Task 21

__all__: list[str] = ["Check", "CheckContext", "CheckResult", "Judge", "JudgeContext"]

# --- Task 21 ---
__all__ += ["Lexicographic"]

_publish(__name__, __all__)
