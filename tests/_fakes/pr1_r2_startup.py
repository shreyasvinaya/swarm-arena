"""Importable startup failures for CLI boundary probes."""

from _fakes.pr1_r2 import SECRET, failure

from swarm_arena import FunctionTool
from swarm_arena.tool import Estimate, ToolRequest


class StartupTool(FunctionTool):
    class Config(FunctionTool.Config):  # ty: ignore[invalid-attribute-override]
        kind: str = "group"

    def estimate(self, request: ToolRequest) -> Estimate | None:
        assert isinstance(self.config, self.Config)
        if self.config.kind == "ordinary":
            raise ValueError(SECRET)
        raise failure(self.config.kind)
