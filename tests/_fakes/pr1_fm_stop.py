"""Request-only hooks that block until the test releases their worker."""

from threading import Thread, current_thread

from _fakes.pr1_fm import ENTERED, EXITED, RELEASE, PaidTool

from swarm_arena.tool import Estimate, ToolRequest

WORKERS: list[Thread] = []


def _block() -> None:
    WORKERS.append(current_thread())
    ENTERED.set()
    try:
        assert RELEASE.wait(300), "test did not release hook"
    finally:
        EXITED.set()


class BlockingEstimate(PaidTool):
    def estimate(self, request: ToolRequest) -> Estimate:
        if request.inputs.get("n") == 4:
            _block()
        return super().estimate(request)


class BlockingProviders(PaidTool):
    def estimate(self, request: ToolRequest) -> Estimate:
        self._requested = request.inputs.get("n") == 4
        return super().estimate(request)

    def providers(self) -> frozenset[str]:
        if getattr(self, "_requested", False):
            _block()
        return frozenset()
