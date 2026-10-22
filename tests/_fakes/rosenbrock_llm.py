"""Isolated example and fake-response fixtures with core-only import safety."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from _fakes.fake_models import registered_model

from swarm_arena import Plan, RunConfig
from swarm_arena.tool import Usage

if TYPE_CHECKING:
    from swarm_arena.testing import FakeModel, FakeTurn

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/rosenbrock"


def _micros(usd: float) -> int:
    units = Decimal(str(usd)) * 1_000_000
    assert units == units.to_integral_value()
    return int(units)


def _judge_turn(n: int, usage: Usage | None) -> FakeTurn:
    from swarm_arena.testing import FakeTurn

    return FakeTurn(
        text=json.dumps(
            {
                "scores": [
                    {"index": i, "value": float(i), "verdict": "ok", "notes": ""} for i in range(n)
                ]
            }
        ),
        usage=Usage(input_tokens=100, output_tokens=20) if usage is None else usage,
    )


@contextmanager
def _registered_judge(alias: str, *, usage: Usage | None = None) -> Generator[FakeModel]:
    from swarm_arena.testing import FakeModel

    with registered_model([], name=alias) as fake, pytest.MonkeyPatch.context() as patch:

        async def respond(*args: Any, **kwargs: Any) -> Any:
            # Candidate IDs identify the actual batch independently of prompt wrappers.
            ids = set(re.findall(r'"id"\s*:\s*"(c-[^"]+)"', args[1]))
            assert ids
            delegate = FakeModel([_judge_turn(len(ids), usage)])
            response = await delegate.get_response(*args, **kwargs)
            fake.calls.extend(delegate.calls)
            return response

        patch.setattr(fake, "get_response", respond)
        yield fake


def _manager_turns(
    random: bool = False, *, agent_number: int = 0, round_number: int = 0
) -> list[FakeTurn]:
    from swarm_arena.testing import FakeTurn

    inputs = (
        {"n": 2000, "radius": 2.0, "seed": 7 + 10 * round_number + agent_number}
        if random
        else {
            "lr": 0.001,
            "steps": 1000 + 100 * round_number,
            "x0": -1.5 + 0.25 * agent_number,
            "y0": 2.0,
        }
    )
    return [
        FakeTurn(tool_calls=(("run_tool", {"inputs": inputs}),)),
        FakeTurn(tool_calls=(("finish_round", {"summary": ""}),)),
    ]


@contextmanager
def _registered_manager(alias: str, *, random: bool = False) -> Generator[FakeModel]:
    from swarm_arena.testing import FakeModel

    delegates: dict[tuple[str, ...], FakeModel] = {}
    with registered_model([], name=alias) as fake, pytest.MonkeyPatch.context() as patch:

        async def respond(*args: Any, **kwargs: Any) -> Any:
            metadata = args[2].metadata
            key = tuple(
                metadata[field] for field in ("run_id", "goal_id", "team", "agent", "round")
            )
            if key not in delegates:
                delegates[key] = FakeModel(
                    _manager_turns(
                        random,
                        agent_number=int(metadata["agent"].rsplit("-", 1)[1]),
                        round_number=int(metadata["round"]),
                    )
                )
            delegate = delegates[key]
            response = await delegate.get_response(*args, **kwargs)
            fake.calls.append(delegate.calls[-1])
            return response

        patch.setattr(fake, "get_response", respond)
        yield fake


def _example(root: Path) -> tuple[Plan, RunConfig]:
    source = root / "source"
    shutil.copytree(EXAMPLE, source)
    plan = Plan.load(source / "plan_llm.toml")
    config = RunConfig.load(source / "run_llm.toml")
    return plan, config.model_copy(
        update={
            "run_root": str(root / "runs"),
            "arena_root": str(root / "arena"),
            "artifact_root": str(root / "artifacts"),
            "settle_timeout_s": 2,
            "executor": config.executor.model_copy(
                update={"config": {"workspace_root": str(root / "ws")}}
            ),
        }
    )


def large_output(size: int) -> dict[str, Any]:
    """
    Return a configurable candidate payload without making model calls.

    Parameters
    ----------
    size : int
        Number of characters included in the candidate data.

    Returns
    -------
    dict[str, Any]
        Numeric objective and deliberately large, finite JSON data.

    Examples
    --------
    >>> large_output(3)
    {'y': 0.0, 'payload': 'xxx'}
    """
    return {"y": 0.0, "payload": "x" * size}
