import pytest
from _fakes.tools import GOAL, RecordingExecutor

from swarm_arena._exec.local import LocalExecutor
from swarm_arena._scoring.check import CheckContext
from swarm_arena._scoring.judge import JudgeContext


@pytest.fixture
def judge_ctx(artifact_root: str) -> JudgeContext:
    return JudgeContext(goal=GOAL, executor=RecordingExecutor(), artifact_root=artifact_root)


@pytest.fixture
def ctx(local_executor: LocalExecutor, artifact_root: str) -> CheckContext:
    return CheckContext(goal=GOAL, executor=local_executor, artifact_root=artifact_root)


@pytest.fixture
def jctx(local_executor: LocalExecutor, artifact_root: str) -> JudgeContext:
    return JudgeContext(goal=GOAL, executor=local_executor, artifact_root=artifact_root)
