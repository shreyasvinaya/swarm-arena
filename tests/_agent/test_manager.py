from _fakes.engine import CtxEnv
from _fakes.tools import NoopManager

from swarm_arena import Ref
from swarm_arena._agent.context import AgentContext
from swarm_arena.manager import LimitReached, Manager, RoundReport


class _M(Manager):
    async def run_round(self, ctx: AgentContext) -> RoundReport:
        return RoundReport(summary="ok")


def test_manager_defaults() -> None:
    m = _M()
    assert m.models() == ()
    assert m.preflight() is None
    noop = Ref(path="_fakes.tools:NoopManager", config={"models": ["opus"]}).build(Manager)
    assert isinstance(noop, Manager)
    assert noop.models() == ("opus",)


def test_limit_reached() -> None:
    err = LimitReached("budget", "goal budget used up")
    assert isinstance(err, RuntimeError)
    assert err.limit == "budget"
    assert str(err) == "goal budget used up"


async def test_round_reports(ctx_env: CtxEnv) -> None:
    assert await _M().run_round(ctx_env.ctx()) == RoundReport(summary="ok")
    assert await NoopManager().run_round(ctx_env.ctx()) == RoundReport()
    assert RoundReport().summary == ""
