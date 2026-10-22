"""Run the shipped model-driven optimizer offline through the public API."""

import json
import math
import os
import shutil
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest

pytest.importorskip("agents")

from _fakes.fake_models import registered_model
from _fakes.run import all_events, events, run_dir
from agents import ModelResponse

from swarm_arena import Plan, RunConfig, Swarm
from swarm_arena._journal.replay import RunState
from swarm_arena.config import ModelCatalog, ModelSpec
from swarm_arena.llm import LLMJudge
from swarm_arena.testing import FakeModel, FakeTurn

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/rosenbrock"


def _micros(usd: float) -> int:
    units = Decimal(str(usd)) * 1_000_000
    assert units == units.to_integral_value()
    return int(units)


def _judge_turn(n: int) -> FakeTurn:
    return FakeTurn(
        text=json.dumps(
            {
                "scores": [
                    {"index": i, "value": float(i), "verdict": "ok", "notes": ""} for i in range(n)
                ]
            }
        )
    )


@contextmanager
def _registered_judge(alias: str) -> Generator[FakeModel]:
    # Each response gets a fresh script matching the actual batch membership.
    with registered_model([], name=alias) as fake, pytest.MonkeyPatch.context() as patch:

        async def respond(*args: Any, **kwargs: Any) -> ModelResponse:
            rows = json.loads(args[1].split("\n", 1)[1].rsplit("\n", 1)[0])
            delegate = FakeModel([_judge_turn(len(rows))])
            response = await delegate.get_response(*args, **kwargs)
            fake.calls.extend(delegate.calls)
            return response

        patch.setattr(fake, "get_response", respond)
        yield fake


def _manager_turns(random: bool = False) -> list[FakeTurn]:
    inputs = (
        {"n": 2000, "radius": 2.0, "seed": 7}
        if random
        else {"lr": 0.001, "steps": 1000, "x0": -1.5, "y0": 2.0}
    )
    return [
        FakeTurn(tool_calls=(("run_tool", {"inputs": inputs}),)),
        FakeTurn(tool_calls=(("finish_round", {"summary": ""}),)),
    ]


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


async def test_rosenbrock_with_llm_managers_and_judge_panel(tmp_path: Path) -> None:
    plan, config = _example(tmp_path)
    config = config.model_copy(
        update={
            "models": ModelCatalog(
                {
                    alias: ModelSpec(
                        model=f"fake/{alias}",
                        roles=spec.roles,
                        prices=ModelSpec.Prices(input=1.0, output=2.0),
                    )
                    for alias, spec in config.models.root.items()
                }
            ),
            "limits": config.limits.model_copy(update={"max_usd": 5.0}),
        }
    )
    with ExitStack() as stack:
        fakes = [
            stack.enter_context(registered_model(turns, name=alias))
            for alias, turns in (
                ("opus", _manager_turns()),
                ("sonnet", _manager_turns(True)),
            )
        ]
        fakes.extend(stack.enter_context(_registered_judge(alias)) for alias in ("gpt", "haiku"))
        swarm = Swarm(config)
        result = await swarm.run(plan)
        assert swarm.run_id == run_dir(config).name
        state = RunState.from_events(all_events(config))
        assert result[plan.goal.id]
        assert {c.role for c in state.llm_calls.values()} == {"manager", "judge"}
        assert all(f.calls for f in fakes)
        assert sum(len(f.calls) for f in fakes) == len(state.llm_calls)
        settled = [
            r for r in state.reservations.values() if r.reservation["kind"] in {"tranche", "call"}
        ]
        assert all(r.state == "settled" for r in settled)
        assert sum(_micros(r.actual["usd"]) for r in settled if r.actual is not None) == sum(
            _micros(c.cost_usd) for c in state.llm_calls.values()
        )
        produced = [c for c in state.candidates.values() if c.origin == "agent"]
        assert produced
        assert all(c.requested_by_call in state.llm_calls for c in produced)
        assert {s.panel_hash for s in state.scores.values() if s.judge == "llm"} == {
            LLMJudge("r", ["gpt", "haiku"]).panel_hash()
        }
        assert all(s.llm_call_ids for s in state.scores.values() if s.judge == "llm")
        assert not events(config, "reserve_exhausted")
        assert not events(config, "judge_error")

        # Completed public resumes must not repeat a paid manager or judge call.
        before = all_events(config)
        assert swarm.run_id is not None
        for _ in range(2):
            resumed = await Swarm(config).resume(swarm.run_id, force=True)
            assert [c.id for c in resumed[plan.goal.id]] == [c.id for c in result[plan.goal.id]]
        replayed = RunState.from_events(all_events(config))
        assert replayed.llm_calls == state.llm_calls
        assert replayed.reservations == state.reservations
        assert all(e.type == "takeover" for e in all_events(config)[len(before) :])
        assert sum(len(f.calls) for f in fakes) == len(state.llm_calls)

        from swarm_arena._board.trace import trace
        from swarm_arena.board import ArenaBoard

        board = ArenaBoard(arena_root=config.arena_root, cache_dir=str(tmp_path / "trace-cache"))
        board.open("", "", [plan.goal.namespace])
        try:
            board.frontier()
            candidate = result[plan.goal.id][0]
            rendered = trace(board, candidate.id)
            assert any(f"llm {candidate.requested_by_call} manager " in line for line in rendered)
            assert any(line.startswith("score    f ") for line in rendered)
        finally:
            board.close()


@pytest.mark.parametrize("aggregate", ["mean", "median", "min", "max"])
async def test_concurrent_recursive_panels_keep_independent_accounting(
    aggregate: Literal["mean", "median", "min", "max"],
) -> None:
    from swarm_arena import Goal
    from swarm_arena.scoring import Lexicographic
    from swarm_arena.testing import check_judge, make_candidate
    from swarm_arena.tool import ToolOutput

    entered, release, second_done = anyio.Event(), anyio.Event(), anyio.Event()
    runs: set[str] = set()
    results: dict[str, list[Any]] = {}
    first_run: str | None = None
    payload = json.dumps({"scores": [{"index": 0, "value": 1.6e308, "verdict": "ok", "notes": ""}]})

    with registered_model([], name="j") as fake, pytest.MonkeyPatch.context() as patch:

        async def respond(*args: Any, **kwargs: Any) -> ModelResponse:
            nonlocal first_run
            run_id = args[2].metadata["run_id"]
            runs.add(run_id)
            if first_run is None:
                first_run = run_id
                entered.set()
            if run_id == first_run:
                await release.wait()
            delegate = FakeModel([FakeTurn(text=payload)])
            response = await delegate.get_response(*args, **kwargs)
            fake.calls.extend(delegate.calls)
            return response

        patch.setattr(fake, "get_response", respond)
        config = RunConfig(
            settle_timeout_s=2,
            models=ModelCatalog(
                {
                    "j": ModelSpec(
                        model="fake/j",
                        roles=["judge"],
                        max_tokens=1000,
                        prices=ModelSpec.Prices(input=1, output=2),
                    )
                }
            ),
        )
        panel = LLMJudge(
            LLMJudge.Config(
                rubric="r",
                models=["j", "j"],
                aggregate=aggregate,
                scale_max=1.7e308,
            )
        )
        judge = Lexicographic([panel, Lexicographic([panel])])
        candidate = make_candidate(ToolOutput(status="ok"), id="c", goal_id="g")

        async def check(label: str) -> None:
            try:
                # A caller's handled exception must not suppress kit accounting.
                raise ValueError("unrelated")
            except ValueError:
                results[label] = await check_judge(
                    judge, Goal(id="g", namespace="n"), [candidate], config=config
                )
            finally:
                if label == "second":
                    second_done.set()

        with anyio.fail_after(30):
            async with anyio.create_task_group() as group:
                group.start_soon(check, "first")
                await entered.wait()
                try:
                    group.start_soon(check, "second")
                    await second_done.wait()
                    assert len(runs) == 2
                    assert "first" not in results
                finally:
                    release.set()
        assert set(results) == {"first", "second"}
        for scores in results.values():
            assert len(scores) == 1
            assert scores[0].key == (1.6e308, 1.6e308)
            assert all(math.isfinite(value) for value in scores[0].key)
            assert len(set(scores[0].llm_call_ids)) == 4
        assert len(fake.calls) == 8


async def test_late_manager_callback_cannot_change_completed_run(tmp_path: Path) -> None:
    from _fakes.llm_components import LateFakeModel
    from _fakes.llm_run import record_kwargs

    from swarm_arena._journal.llm_record import record_call
    from swarm_arena.llm import _catalog

    plan, config = _example(tmp_path)
    team = plan.teams[0].model_copy(update={"agents": [plan.teams[0].agents[0]]})
    plan = plan.model_copy(update={"teams": [team], "judges": [plan.judges[0]]})
    config = config.model_copy(
        update={
            "models": ModelCatalog(
                {
                    "opus": ModelSpec(
                        model="fake/opus",
                        roles=["manager"],
                        prices=ModelSpec.Prices(input=1, output=2),
                    )
                }
            ),
            "limits": config.limits.model_copy(update={"max_rounds": 1, "max_usd": 5.0}),
        }
    )
    fake = LateFakeModel(_manager_turns(), delay_s=None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_catalog, "_FAKE_MODELS", {**_catalog._FAKE_MODELS, "opus": fake})
        try:
            assert (await Swarm(config).run(plan))[plan.goal.id]
            before = all_events(config)
            state = RunState.from_events(before)
            assert len(state.llm_calls) == len(fake.calls) == 2
            assert all((c.error or "").startswith("estimated:") for c in state.llm_calls.values())
            charges = [
                r.actual
                for r in state.reservations.values()
                if r.reservation["kind"] in {"tranche", "call"}
            ]
            assert sum(_micros(a["usd"]) for a in charges if a is not None) == sum(
                _micros(c.cost_usd) for c in state.llm_calls.values()
            )

            def objects() -> list[Path]:
                return sorted(Path(config.artifact_root).rglob("*"))

            uploaded = await anyio.to_thread.run_sync(objects)
            for call in fake.calls:
                assert await record_call(**record_kwargs(call["model_settings"].metadata)) is None
            assert all_events(config) == before
            assert await anyio.to_thread.run_sync(objects) == uploaded
        finally:
            await fake.aclose()


@pytest.mark.llm_live
async def test_rosenbrock_live(tmp_path: Path) -> None:
    if not all(os.environ.get(key) for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
        pytest.skip("requires ANTHROPIC_API_KEY and OPENAI_API_KEY")
    plan, config = _example(tmp_path)
    assert (await Swarm(config).run(plan))[plan.goal.id]
