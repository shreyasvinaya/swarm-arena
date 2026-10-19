"""Conformance kit for connector authors: check tools, checks, and judges offline."""

from swarm_arena._facade import publish as _publish
from swarm_arena.testing._candidates import make_candidate
from swarm_arena.testing._conformance import check_check, check_judge, check_tool

__all__: list[str] = ["check_check", "check_judge", "check_tool", "make_candidate"]

_publish(__name__, __all__)
