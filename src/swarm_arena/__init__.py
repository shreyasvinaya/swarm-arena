"""
Run a swarm of agents on hard goals.

A goal fans out to teams of agents, weak branches are pruned, and the good results converge
on answers you can prove.
"""

from swarm_arena._facade import publish as _publish
from swarm_arena._model.goal import Goal
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import (
    AgentSpec,
    ConvergeSpec,
    PartSpec,
    Plan,
    Resources,
    RoundSpec,
    TeamSpec,
)
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Score
from swarm_arena._model.run_config import RunConfig
from swarm_arena._planning.policies import Chain, KeepTopTeams, Patience, SuccessiveHalving
from swarm_arena._scoring.program_judge import ProgramJudge
from swarm_arena._scoring.script_check import ScriptCheck
from swarm_arena._scoring.script_judge import ScriptJudge
from swarm_arena._tooling.function_tool import FunctionTool
from swarm_arena._tooling.script_tool import ScriptTool
from swarm_arena._version import __version__ as __version__

__all__: list[str] = [
    "AgentSpec",
    "Chain",
    "ConvergeSpec",
    "FunctionTool",
    "Goal",
    "KeepTopTeams",
    "Limits",
    "PartSpec",
    "Patience",
    "Plan",
    "ProgramJudge",
    "Ref",
    "Resources",
    "RoundSpec",
    "RunConfig",
    "Score",
    "ScriptCheck",
    "ScriptJudge",
    "ScriptTool",
    "SuccessiveHalving",
    "TeamSpec",
]

_publish(__name__, __all__)

# Import the core facades after _publish, so every published __module__ is set once swarm_arena is
# imported, whatever the import order (doctest collection imports private modules first).
from swarm_arena import board as board
from swarm_arena import config as config
from swarm_arena import executor as executor
from swarm_arena import manager as manager
from swarm_arena import planning as planning
from swarm_arena import protocol as protocol
from swarm_arena import scoring as scoring
from swarm_arena import tool as tool
