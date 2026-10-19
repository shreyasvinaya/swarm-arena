from typing import cast

import pytest
from _fakes.tools import GOAL, RecordingExecutor

from swarm_arena._exec.executor import Executor
from swarm_arena._scoring.check import CheckContext
from swarm_arena._scoring.judge import JudgeContext


@pytest.fixture
def judge_ctx(artifact_root: str) -> JudgeContext:
    return JudgeContext(goal=GOAL, executor=RecordingExecutor(), artifact_root=artifact_root)


@pytest.fixture
def ctx(request: pytest.FixtureRequest, artifact_root: str) -> CheckContext:
    executor = cast(Executor, request.getfixturevalue("local_executor"))
    return CheckContext(goal=GOAL, executor=executor, artifact_root=artifact_root)


@pytest.fixture
def jctx(request: pytest.FixtureRequest, artifact_root: str) -> JudgeContext:
    executor = cast(Executor, request.getfixturevalue("local_executor"))
    return JudgeContext(goal=GOAL, executor=executor, artifact_root=artifact_root)
