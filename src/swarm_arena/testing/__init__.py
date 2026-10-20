"""Conformance kit for connector authors: check tools, checks, judges, and managers offline."""

from swarm_arena._facade import publish as _publish
from swarm_arena.testing._artifacts import read_logs, read_output_file
from swarm_arena.testing._candidates import make_candidate
from swarm_arena.testing._conformance import check_check, check_judge, check_manager, check_tool

__all__: list[str] = [
    "check_check",
    "check_judge",
    "check_manager",
    "check_tool",
    "make_candidate",
    "read_logs",
    "read_output_file",
]

_publish(__name__, __all__)
