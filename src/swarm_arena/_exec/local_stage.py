"""Private local preparation aliases for the shared executor machinery."""

from swarm_arena._exec.job_steps import _Preparation as _Preparation
from swarm_arena._exec.job_steps import _Staged as _Staged
from swarm_arena._exec.job_steps import prepare_job

_stage = prepare_job
