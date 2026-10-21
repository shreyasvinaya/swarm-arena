"""Public type-only diagnostics retain the originating extension exception type."""

import json
from pathlib import Path

import anyio
import pytest
from _fakes import pr1_fm as hooks

from swarm_arena import Swarm
from swarm_arena.tool import Estimate
from tests._engine.test_pr1_fm_outcomes import config, events, plan


@pytest.mark.parametrize("where", ["manager", "check", "command", "collect", "estimate"])
@pytest.mark.parametrize(
    ("kind", "expected"), [("cancel", "CancelledError"), ("group", "BaseExceptionGroup")]
)
async def test_public_diagnostics_keep_original_type(
    tmp_path: Path, where: str, kind: str, expected: str
) -> None:
    hooks.PEER_READY = None
    cfg = config(tmp_path)
    cfg = cfg.model_copy(
        update={
            "limits": cfg.limits.model_copy(
                update={"default_run_reservation": Estimate(tokens=7, usd=0.000007)}
            )
        }
    )
    await Swarm(cfg).run(plan(where, kind), run_id="r")
    journal = events(tmp_path)
    text = json.dumps(journal)
    for event in journal:
        if event["type"] != "tool_end":
            continue
        logs = event["payload"]["candidate"]["output"].get("logs_ref")
        if logs:
            text += await anyio.Path(tmp_path / "runs" / "r" / (logs + ".err")).read_text()
    assert hooks.MARK not in text
    assert "_CoreRuntimeError" not in text
    assert "_CoreValueError" not in text
    assert expected in text
