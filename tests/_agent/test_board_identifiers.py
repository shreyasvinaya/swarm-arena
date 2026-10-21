"""The empty-query listing path never compares extension string objects."""

from _fakes.engine import CtxEnv
from _fakes.pr1_r5_ids import hostile


async def test_empty_learning_query_is_normalized(ctx_env: CtxEnv) -> None:
    ctx = ctx_env.ctx()
    assert ctx.search_learnings(hostile("")) == []
