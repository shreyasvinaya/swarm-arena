"""Exercise final ranking and a model-managed converge team on a capped run."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

pytest.importorskip("agents")

from _fakes.fake_models import registered_model
from _fakes.rosenbrock_llm import _example, _manager_turns, _micros, _registered_judge
from _fakes.run import all_events, events, run_dir

from swarm_arena import ConvergeSpec, Goal, Limits, Plan, Ref, RunConfig, Score, Swarm
from swarm_arena._budget.amount import Amount
from swarm_arena._journal.replay import RunState
from swarm_arena.board import Candidate
from swarm_arena.config import ModelCatalog, ModelSpec
from swarm_arena.llm import LLMJudge
from swarm_arena.scoring import JudgeContext
from swarm_arena.tool import Usage


class _FinalJudge(LLMJudge):
    """Use an exact free objective at barriers and the real LLM for final ranking."""

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        """
        Defer paid scoring until the final public engine operation.

        Parameters
        ----------
        goal : Goal
            Goal being judged.
        candidates : sequence of Candidate
            Published data-only outputs.
        ctx : JudgeContext
            Engine-owned phase and execution context.

        Returns
        -------
        list of Score
            Free barrier scores or the real LLM judge's final scores.

        Examples
        --------
        >>> _FinalJudge("r", ["j"]).models()
        ('j',)
        """
        if ctx.final:
            return await super().judge(goal, candidates, ctx)
        return [Score.maximize(1).model_copy(update={"candidate_id": c.id}) for c in candidates]


async def test_llm_ranking_judge_and_llm_converge_manager_fit_the_final_slice(
    tmp_path: Path,
) -> None:
    plan, config = _example(tmp_path)
    judge = Ref.of(LLMJudge("r", ["j"])).model_copy(update={"name": "llm"})
    # Keep the scripted optimizer plan and give only convergence an LLM manager.

    plan = Plan.load(plan.base_dir / "plan.toml")
    plan = plan.model_copy(update={"goal": plan.goal.model_copy(update={"id": "g"})})
    team = plan.teams[0].model_copy(
        update={
            "name": "join",
            "agents": [
                plan.teams[0]
                .agents[0]
                .model_copy(
                    update={
                        "manager": Ref(
                            path="swarm_arena.llm:AgentsSDKManager", config={"model": "m"}
                        )
                    }
                )
            ],
        }
    )
    plan = plan.model_copy(
        update={
            "judges": [judge],
            "ranking": "llm",
            "converge": ConvergeSpec(mode="team", team=team),
        }
    )
    config = config.model_copy(
        update={
            "models": ModelCatalog(
                {
                    alias: ModelSpec(
                        model=f"fake/{alias}",
                        roles=[role],
                        max_tokens=1000,
                        prices=ModelSpec.Prices(input=1, output=2),
                    )
                    for alias, role in (("j", "judge"), ("m", "manager"))
                }
            ),
            "limits": config.limits.model_copy(update={"max_usd": 5.0}),
        }
    )
    with (
        _registered_judge("j") as judge_fake,
        registered_model(_manager_turns(), name="m") as manager_fake,
    ):
        result = await Swarm(config).run(plan)
        assert result[plan.goal.id]
        solution = json.loads(
            (run_dir(config) / "goals" / plan.goal.id / "solution.json").read_text()
        )
        assert solution["status"] == "ok"
        assert not events(config, "reserve_exhausted")
        assert not events(config, "judge_error")
        assert judge_fake.calls
        assert len(manager_fake.calls) == 2
        final_scores = [e.payload["score"] for e in events(config, "score") if e.round == 0]
        expected = [
            s["candidate_id"] for s in sorted(final_scores, key=lambda s: s["value"], reverse=True)
        ]
        assert (
            events(config, "final_order")[-1].payload["candidate_ids"]
            == expected[: plan.converge.top_n]
        )
        state = RunState.from_events(all_events(config))
        assert all(c.status == "ok" for c in state.llm_calls.values())
        assert any(c.team == "join" for c in state.candidates.values())
        start = events(config, "converge_started")[0].seq
        ids = {
            e.payload["reservation"]["id"]
            for e in events(config, "reserve")
            if e.seq > start and e.payload["reservation"]["goal_id"] == plan.goal.id
        }
        spent = sum((Amount.model_validate(state.reservations[id].actual) for id in ids), Amount())
        final_slice = Amount.model_validate(
            events(config, "run_started")[0].payload["final_slices"][plan.goal.id]
        )
        # Independent output-only oracle: two attempts * 20 manager turns * 1,000
        # tokens, plus max(25,000 tranche, 2 * five final candidates * 1,000).
        # At input/output prices 1/2, the terms hold $0.08 and $0.25 respectively.
        assert final_slice == Amount(tokens=65_000, usd=0.33)
        llm_reservations = {
            id: state.reservations[id].reservation
            for id in ids
            if state.reservations[id].reservation["kind"] in {"tranche", "call"}
        }
        final_call_ids = {id for score in final_scores for id in score["llm_call_ids"]}
        phase_reservations: dict[str, set[str]] = {
            "manager": set(),
            "barrier": set(),
            "ranking": set(),
        }
        for call in state.llm_calls.values():
            assert call.reservation_id is not None
            owned = {
                id
                for id in llm_reservations
                if call.reservation_id == id or call.reservation_id.startswith(f"{id}/call-")
            }
            if not owned:
                continue
            assert len(owned) == 1
            assert call.role in {"manager", "judge"}
            phase = (
                "manager"
                if call.role == "manager"
                else "ranking"
                if call.id in final_call_ids
                else "barrier"
            )
            phase_reservations[phase].update(owned)
        for phase, holder in (
            ("manager", "g/:final"),
            ("barrier", "g/:judges"),
            ("ranking", "g/:final"),
        ):
            assert phase_reservations[phase]
            assert {llm_reservations[id]["holder"] for id in phase_reservations[phase]} == {holder}
        assert set().union(*phase_reservations.values()) == set(llm_reservations)
        assert spent.fits_in(final_slice, cap_tokens=True, cap_usd=True)
        assert sum(_micros(c.cost_usd) for c in state.llm_calls.values()) > 0


@pytest.mark.parametrize("size", [10, 100_000])
async def test_final_input_size_controls_predispatch_refusal(tmp_path: Path, size: int) -> None:
    from _fakes.engine import with_executor
    from _fakes.run import plan as simple_plan

    plan = simple_plan(
        rounds=1,
        tool_ref=Ref(
            path="swarm_arena:FunctionTool",
            config={"function": "_fakes.rosenbrock_llm:large_output"},
        ),
    )
    plan.teams[0].agents[0] = (
        plan.teams[0]
        .agents[0]
        .model_copy(
            update={
                "manager": Ref(
                    path="swarm_arena:ScriptedManager",
                    config={"requests": [{"inputs": {"size": size}}]},
                ),
            }
        )
    )
    plan = plan.model_copy(
        update={
            "judges": [Ref.of(_FinalJudge("r", ["j"])).model_copy(update={"name": "llm"})],
            "ranking": "llm",
        }
    )
    config = with_executor(
        RunConfig(
            run_root=str(tmp_path / "runs"),
            arena_root=str(tmp_path / "arena"),
            artifact_root=str(tmp_path / "artifacts"),
            settle_timeout_s=2,
            limits=Limits(
                max_tokens=10_000,
                llm_tranche_tokens=1000,
                max_usd=5,
                max_rounds=1,
                default_run_reservation={"tokens": 0, "usd": 0.0},
            ),
            models=ModelCatalog(
                {
                    "j": ModelSpec(
                        model="fake/j",
                        roles=["judge"],
                        max_tokens=100,
                        prices=ModelSpec.Prices(input=1, output=2),
                    )
                }
            ),
        ),
        workspace_root=str(tmp_path / "ws"),
    )
    with _registered_judge("j", usage=Usage(input_tokens=40, output_tokens=20)) as fake:
        result = await Swarm(config).run(plan)
        if size == 10:
            assert result["g"]
            assert len(fake.calls) == 1
            [call] = events(config, "llm_call")
            assert call.payload["call"]["status"] == "ok"
            assert call.payload["call"]["error"] is None
            assert not events(config, "no_result")
            solution = json.loads((run_dir(config) / "goals/g/solution.json").read_text())
            assert solution["status"] == "ok"
        else:
            assert result == {"g": []}
            assert not fake.calls
            assert events(config, "no_result")[-1].payload["reason"] == "reserve_exhausted"
            assert not events(config, "llm_call")
    state = RunState.from_events(all_events(config))
    assert all(r.state == "settled" for r in state.reservations.values())
    spent = sum((Amount.model_validate(r.actual) for r in state.reservations.values()), Amount())
    assert spent.fits_in(Amount(tokens=10_000, usd=5), cap_tokens=True, cap_usd=True)
