"""Rubric judging, anchors, panels, funding and accounting."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock

import pytest
from _fakes.llm_run import open_llm_run
from agents import ModelResponse

from swarm_arena import Goal, RunConfig
from swarm_arena._budget.llm_scope import llm_scope
from swarm_arena.config import CachePolicy, ModelCatalog, ModelSpec
from swarm_arena.llm import LLMJudge
from swarm_arena.scoring import JudgeContext
from swarm_arena.testing import FakeModel, FakeTurn, make_candidate
from swarm_arena.tool import RunStatus, ToolOutput


def test_models_lists_each_panel_member() -> None:
    assert LLMJudge("r", ["a", "b", "a"]).models() == ("a", "b", "a")


def test_identity_hashes() -> None:
    assert LLMJudge("r", ["a"]).panel_hash() == ""
    assert LLMJudge("r", ["a", "b"]).panel_hash() == LLMJudge("r", ["b", "a"]).panel_hash()
    assert LLMJudge("r", ["a"]).rubric_hash() != LLMJudge("r", ["b"]).rubric_hash()


def test_anchor_count_default_is_two() -> None:
    assert LLMJudge("r", ["a"]).anchor_count() == 2


def test_anchors_must_be_fewer_than_batch_size() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMJudge.Config(rubric="r", models=["a"], anchors=8, batch_size=8)


def candidates(n: int) -> list:
    return [
        make_candidate(ToolOutput(status=RunStatus.OK), id=f"c{i}", goal_id="g") for i in range(n)
    ]


def turn(n: int, value: float = 7.5) -> FakeTurn:
    return FakeTurn(
        text=json.dumps(
            {
                "scores": [
                    {"index": i, "value": value, "verdict": "ok", "notes": "rubric"}
                    for i in range(n)
                ]
            }
        )
    )


def configuration(aliases: tuple[str, ...] = ("a",), *, cache_dir: Path | None = None) -> RunConfig:
    return RunConfig(
        models=ModelCatalog(
            {
                alias: ModelSpec(
                    model=f"fake/{alias}",
                    roles=["judge"],
                    max_tokens=1000,
                    prices=ModelSpec.Prices(input=1, output=2),
                )
                for alias in aliases
            }
        ),
        cache=None if cache_dir is None else CachePolicy(backend="disk", disk_dir=str(cache_dir)),
        settle_timeout_s=2,
    )


@pytest.mark.parametrize(
    ("n", "anchors", "final", "sizes"),
    [
        (2, False, False, [2]),
        (9, False, False, [8, 1]),
        (12, False, True, [12]),
        (20, True, False, [8, 8, 6]),
    ],
)
async def test_batches_anchors_cache_and_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n: int,
    anchors: bool,
    final: bool,
    sizes: list[int],
) -> None:
    from swarm_arena.llm._judge import JUDGE_SCHEMA

    fake = FakeModel([])

    async def respond(*args: Any, **kwargs: Any) -> ModelResponse:
        rows = json.loads(args[1].split("\n", 1)[1].rsplit("\n", 1)[0])
        ids = [row["candidate"]["id"] for row in rows]
        value = 7.5 + (1 if "c8" in ids and anchors else 2 if "c15" in ids and anchors else 0)
        delegate = FakeModel([turn(len(rows), value)])
        response = await delegate.get_response(*args, **kwargs)
        fake.calls.extend(delegate.calls)
        return response

    monkeypatch.setattr(fake, "get_response", respond)
    fake.register("a")
    goal = Goal(id="g", namespace="n")
    batch = candidates(n)
    ctx = JudgeContext(
        goal=goal,
        executor=Mock(),
        artifact_root=str(tmp_path),
        anchors=frozenset({"c0"}) if anchors else frozenset(),
        final=final,
    )
    async with open_llm_run(tmp_path, config=configuration(cache_dir=tmp_path / "cache")) as run:
        scope = run.scope("judge")
        with llm_scope(scope):
            scores = await LLMJudge("r", ["a"]).judge(goal, batch, ctx)
        assert len(scores) == n
        assert scores[0].value == 7.5
        assert scores[0].key == (7.5,)
        assert {i for score in scores for i in score.llm_call_ids} == {c.id for c in run.calls()}
        assert scope.tranche.pending() == 0
        assert run.charged(scope).usd == pytest.approx(sum(c.cost_usd for c in run.calls()))
    assert len(fake.calls) == len(sizes)
    for call in fake.calls:
        assert call["model_settings"].temperature == 0
        assert call["output_schema"].json_schema() == JUDGE_SCHEMA
        assert call["model_settings"].extra_args["cache"]["no-cache"] is (anchors or final)
        if anchors:
            assert "c0" in json.dumps(call["input"])
    if anchors:
        assert scores[0].notes.endswith("anchor values in later batches: [8.5, 9.5]")
        assert len(scores[0].llm_call_ids) == len(sizes)


@pytest.mark.parametrize(
    ("agg", "want"), [("mean", 8.0), ("median", 8.0), ("min", 6.0), ("max", 10.0)]
)
async def test_panel_aggregate(tmp_path: Path, agg: str, want: float) -> None:
    from swarm_arena.testing import check_judge

    for alias, value in zip(("a", "b", "c"), (6.0, 8.0, 10.0), strict=True):
        FakeModel([turn(2, value)]).register(alias)
    scores = await check_judge(
        LLMJudge("r", ["a", "b", "c"], agg),
        Goal(id="g", namespace="n"),
        candidates(2),
        config=configuration(("a", "b", "c")),
    )
    assert [s.value for s in scores] == [want, want]
    assert all(len(s.llm_call_ids) == 3 for s in scores)


@pytest.mark.parametrize(
    "payload",
    [
        [{"index": 0, "value": 11.0, "verdict": "ok", "notes": ""}],
        [],
        [{"index": True, "value": 1.0, "verdict": "ok", "notes": ""}],
        [{"index": 0, "value": True, "verdict": "ok", "notes": ""}],
    ],
)
async def test_out_of_scale_or_missing_value_omits_the_candidate(
    tmp_path: Path, payload: list
) -> None:
    fake = FakeModel([FakeTurn(text=json.dumps({"scores": payload}))])
    fake.register("a")
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration(cache_dir=tmp_path / "cache")) as run:
        with llm_scope(run.scope("judge")):
            scores = await LLMJudge("r", ["a"]).judge(
                goal,
                candidates(1),
                JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
            )
    assert scores == []


def test_explicit_survivor_panel_survives_core_stamping() -> None:
    from swarm_arena import Score
    from swarm_arena._model.ids import IdGen
    from swarm_arena._scoring.judge import stamp_scores

    judge = LLMJudge("r", ["a", "b"])
    raw = Score(key=(1.0,), candidate_id="c", panel_hash="survivor")
    assert (
        stamp_scores([raw], judge=judge, judge_name="j", ids=IdGen("r", 0), batch_id="b")[
            0
        ].panel_hash
        == "survivor"
    )


def test_llm_judge_holds_nothing_at_the_barrier() -> None:
    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.estimate import judge_amount, make_price_of

    config = configuration()
    assert (
        judge_amount(LLMJudge("r", ["a"]), 8, config, make_price_of(config.models), llm=False)
        == Amount()
    )


def test_spec_forms_build() -> None:
    from swarm_arena import Ref
    from swarm_arena.scoring import Judge

    judge = LLMJudge("rubric", ["opus", "gpt", "flash"], "median")
    assert Ref.of(judge).build(Judge).models() == judge.models()


@pytest.mark.parametrize("delay_s", [0.05, None])
async def test_judge_drains_before_it_returns(delay_s: float | None, tmp_path: Path) -> None:
    from _fakes.llm_components import LateFakeModel

    fake = LateFakeModel([turn(1)], delay_s=delay_s)
    fake.register("a")
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        scope = run.scope("judge")
        try:
            with llm_scope(scope):
                scores = await LLMJudge("r", ["a"]).judge(
                    goal,
                    candidates(1),
                    JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
                )
            ids = {i for s in scores for i in s.llm_call_ids}
            by_id = {c.id: c for c in run.calls()}
            assert ids <= set(scope.call_ids)
            assert ids <= set(by_id)
            assert delay_s is not None or all(
                (by_id[i].error or "").startswith("estimated:") for i in ids
            )
            assert run.charged(scope).usd == pytest.approx(sum(c.cost_usd for c in run.calls()))
        finally:
            await fake.aclose()


@pytest.mark.parametrize("too_large", [False, True])
async def test_two_stage_final_judge_fits_its_final_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, too_large: bool
) -> None:
    from _budget.test_estimate import _plan

    from swarm_arena import Limits
    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.estimate import final_slice_estimate, make_price_of
    from swarm_arena._budget.ledger import Ledger
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena.llm import BudgetExceeded, _structured

    judge = LLMJudge("r", ["a"])
    config = configuration(("a", "fmt"))
    config.models.root["fmt"] = ModelSpec(
        model="fake/fmt",
        roles=["judge"],
        max_tokens=100000,
        prices=ModelSpec.Prices(input=0.5, output=1),
    )
    slice = final_slice_estimate(
        _plan(ranking_judge=judge, top_n=1), config, {}, make_price_of(config.models)
    )
    config = config.model_copy(
        update={"limits": Limits(max_tokens=10 * slice.tokens, max_usd=10 * slice.usd)}
    )
    first = FakeModel([FakeTurn(text="rubric reasoning")])
    formatter = FakeModel([turn(1)])
    first.register("a")
    formatter.register("fmt")
    monkeypatch.setattr(_structured, "supports", lambda spec, feature: spec.model == "fake/fmt")
    goal = Goal(id="g", namespace="n")
    batch = candidates(1)
    if too_large:
        batch[0] = batch[0].model_copy(
            update={
                "request": batch[0].request.model_copy(
                    update={"inputs": {"text": "private-input" * 10000}}
                )
            }
        )
    async with open_llm_run(tmp_path, config=config) as run:
        ledger = Ledger(
            config.limits,
            reserve_fraction=0.15,
            goal_weights=None,
            journal=run.journal,
            ids=run.ids,
        )
        run.ledger = ledger
        await ledger.hold_final_slices({"g": slice})
        ledger.open_goal("g")
        available = ledger.remaining("g/:judges")
        reservation = await ledger.reserve("g/:judges", available, kind="judge", goal_id="g")
        await ledger.settle(reservation.id, available)
        assert ledger.remaining("g/:judges") == Amount()
        assert ledger.remaining("g/:final") == slice
        scope = run.scope("judge")
        scope = replace(
            scope,
            tranche=Tranche(
                ledger,
                "g/:final",
                "g",
                size=Amount(
                    tokens=config.limits.llm_tranche_tokens, usd=config.limits.llm_tranche_usd
                ),
            ),
        )
        with llm_scope(scope):
            if too_large:
                with pytest.raises(BudgetExceeded):
                    await judge.judge(
                        goal,
                        batch,
                        JudgeContext(
                            goal=goal, executor=Mock(), artifact_root=str(tmp_path), final=True
                        ),
                    )
                assert first.calls == []
                assert formatter.calls == []
            else:
                (score,) = await judge.judge(
                    goal,
                    batch,
                    JudgeContext(
                        goal=goal, executor=Mock(), artifact_root=str(tmp_path), final=True
                    ),
                )
                assert score.value == 7.5
                assert formatter.calls[0]["model_settings"].max_tokens == 1000
        assert scope.tranche.pending() == 0
        spent = run.charged(scope)
        assert spent.tokens <= slice.tokens
        assert spent.usd <= slice.usd
        assert spent.fits_in(slice, cap_tokens=True, cap_usd=True)


@pytest.mark.parametrize("nested", [False, True])
async def test_survivor_identity_is_global_and_kept_through_stamping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    from swarm_arena import Ref
    from swarm_arena._model.canonical import canonical_json, sha256_hex
    from swarm_arena._scoring.runner import judge_round
    from swarm_arena.llm import _judge
    from swarm_arena.scoring import Lexicographic

    async def scripted(**kwargs: object) -> tuple[dict, list[str]]:
        alias = kwargs["alias"]
        first = '"c0"' in str(kwargs["user"])
        if (alias == "a" and first) or (alias == "b" and not first):
            raise ValueError("private-member-failure")
        n = 8 if first else 1
        return json.loads(turn(n, 6.0 if alias == "c" else 10.0).text), [f"{alias}-{first}"]

    monkeypatch.setattr(_judge, "structured_call", scripted)
    judge = LLMJudge("r", ["a", "b", "c"])
    expected = sha256_hex(canonical_json(["c"]))
    if nested:
        judge = Lexicographic([Lexicographic([judge])])
        expected = sha256_hex(canonical_json([sha256_hex(canonical_json([expected]))]))
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration(("a", "b", "c"))) as run:
        with llm_scope(run.scope("judge")):
            outcome = await judge_round(
                "g",
                [(Ref.of(judge), judge)],
                candidates(9),
                [],
                lambda _: JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
            )
    assert len(outcome.scores) == 9
    assert {s.value for s in outcome.scores} == {6.0}
    assert {s.panel_hash for s in outcome.scores} == {expected}


async def test_failed_member_calls_remain_linked(tmp_path: Path) -> None:
    FakeModel([turn(1)]).register("a")
    FakeModel([FakeTurn(text="private malformed output")]).register("b")
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration(("a", "b"))) as run:
        scope = run.scope("judge")
        with llm_scope(scope):
            (score,) = await LLMJudge("r", ["a", "b"]).judge(
                goal,
                candidates(1),
                JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
            )
        assert set(score.llm_call_ids) == {c.id for c in run.calls()}
        assert len(score.llm_call_ids) == 2


async def test_repeated_panel_votes_are_kept(tmp_path: Path) -> None:
    from swarm_arena.testing import check_judge

    FakeModel([turn(1, 6), turn(1, 6)]).register("a")
    FakeModel([turn(1, 9)]).register("b")
    (score,) = await check_judge(
        LLMJudge("r", ["a", "b", "a"]),
        Goal(id="g", namespace="n"),
        candidates(1),
        config=configuration(("a", "b")),
    )
    assert score.value == 7
    assert len(score.llm_call_ids) == 3


@pytest.mark.parametrize("kind", ["terminal", "fenced", "user_cancel", "nested_terminal"])
async def test_member_outcomes_use_shared_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import asyncio

    from swarm_arena._errors import Fenced, _core_error, _stop_cleanup_error
    from swarm_arena.llm import _judge

    error = {
        "terminal": _stop_cleanup_error(),
        "fenced": _core_error(Fenced, "run ownership lost"),
        "user_cancel": asyncio.CancelledError("private"),
        "nested_terminal": BaseExceptionGroup("private", [_stop_cleanup_error()]),
    }[kind]

    async def fail(**kwargs: object) -> tuple[dict, list[str]]:
        raise error

    monkeypatch.setattr(_judge, "structured_call", fail)
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        with llm_scope(run.scope("judge")):
            ctx = JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path))
            if kind == "user_cancel":
                assert await LLMJudge("r", ["a"]).judge(goal, candidates(1), ctx) == []
            else:
                with pytest.raises(Fenced if kind == "fenced" else RuntimeError):
                    await LLMJudge("r", ["a"]).judge(goal, candidates(1), ctx)


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_caller_cancellation_drains_without_waiting_for_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    import asyncio

    import anyio
    from _fakes.llm_components import LateFakeModel

    from swarm_arena._errors import _stop_cleanup_error
    from swarm_arena.llm import _judge

    finish = _judge._finish_accounting

    async def cleanup(*args: Any, **kwargs: Any) -> None:
        await finish(*args, **kwargs)
        if cleanup_fails:
            raise _stop_cleanup_error()

    monkeypatch.setattr(_judge, "_finish_accounting", cleanup)
    release = anyio.Event()
    fake = LateFakeModel([turn(1)], delay_s=0, release=release)
    fake.register("a")
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        scope = run.scope("judge")
        try:
            with llm_scope(scope):
                task = asyncio.create_task(
                    LLMJudge("r", ["a"]).judge(
                        goal,
                        candidates(1),
                        JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
                    )
                )
                with anyio.fail_after(10):
                    await fake.ready.wait()
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
            assert scope.tranche.pending() == 0
            assert len(run.calls()) == 1
            assert (run.calls()[0].error or "").startswith("estimated:")
            release.set()
            await fake.flush()
            assert len(run.calls()) == 1
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await fake.aclose()


async def test_formatter_input_refusal_preserves_first_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena import Limits
    from swarm_arena.llm import BudgetExceeded, _structured

    first = FakeModel([FakeTurn(text="draft" * 10000)])
    formatter = FakeModel([turn(1)])
    first.register("a")
    formatter.register("fmt")
    monkeypatch.setattr(_structured, "supports", lambda spec, feature: spec.model == "fake/fmt")
    config = configuration(("a", "fmt")).model_copy(update={"limits": Limits(max_tokens=20000)})
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=config) as run:
        scope = run.scope("judge")
        with llm_scope(scope), pytest.raises(BudgetExceeded):
            await LLMJudge("r", ["a"]).judge(
                goal,
                candidates(1),
                JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path), final=True),
            )
        assert len(first.calls) == 1
        assert formatter.calls == []
        assert scope.tranche.pending() == 0
        assert len(run.calls()) == 1
        assert run.charged(scope).tokens == 120


async def test_all_members_failed_omits_all_candidates(tmp_path: Path) -> None:
    FakeModel([FakeTurn(text="bad")]).register("a")
    FakeModel([FakeTurn(text="bad")]).register("b")
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration(("a", "b"))) as run:
        scope = run.scope("judge")
        with llm_scope(scope):
            assert (
                await LLMJudge("r", ["a", "b"]).judge(
                    goal,
                    candidates(1),
                    JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
                )
                == []
            )
        assert len(run.calls()) == 2
        assert scope.tranche.pending() == 0


async def test_all_batch_members_start_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio

    from swarm_arena.llm import _judge

    entered: list[str] = []
    all_entered = anyio.Event()

    async def response(**kwargs: Any) -> tuple[dict[str, Any], list[str]]:
        rows = json.loads(kwargs["user"].split("\n", 1)[1].rsplit("\n", 1)[0])
        entered.append(kwargs["alias"])
        if len(entered) == 4:
            all_entered.set()
        await all_entered.wait()
        return json.loads(turn(len(rows)).text or ""), []

    monkeypatch.setattr(_judge, "structured_call", response)
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration(("a", "b"))) as run:
        with llm_scope(run.scope("judge")), anyio.fail_after(10):
            scores = await LLMJudge("r", ["a", "b"]).judge(
                goal,
                candidates(9),
                JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
            )
        assert len(scores) == 9
        assert sorted(entered) == ["a", "a", "b", "b"]


async def test_anchor_in_only_batch_has_empty_later_values(tmp_path: Path) -> None:
    FakeModel([turn(1)]).register("a")
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration()) as run:
        with llm_scope(run.scope("judge")):
            (score,) = await LLMJudge("r", ["a"]).judge(
                goal,
                candidates(1),
                JudgeContext(
                    goal=goal,
                    executor=Mock(),
                    artifact_root=str(tmp_path),
                    anchors=frozenset({"c0"}),
                ),
            )
        assert score.notes.endswith("anchor values in later batches: []")
        assert len(score.llm_call_ids) == 1


@pytest.mark.parametrize("aggregate", ["mean", "median", "min", "max"])
@pytest.mark.parametrize(
    ("votes", "lower", "upper"),
    [
        ([1e308, 1e308], 0.0, 1.5e308),
        ([sys.float_info.max, sys.float_info.max], 0.0, sys.float_info.max),
        ([-sys.float_info.max, -sys.float_info.max], -sys.float_info.max, 2.0),
        ([-sys.float_info.max, sys.float_info.max], -sys.float_info.max, sys.float_info.max),
        ([5e-324, 5e-324], 5e-324, 2.0),
        ([1e308, 1.2e308, 1.4e308, 1.6e308], 0.0, sys.float_info.max),
    ],
)
async def test_aggregate_finite_extreme_votes(
    aggregate: Literal["mean", "median", "min", "max"],
    votes: list[float],
    lower: float,
    upper: float,
) -> None:
    from fractions import Fraction

    from swarm_arena.testing import check_judge

    aliases = tuple(f"member{i}" for i in range(len(votes)))
    for alias, value in zip(aliases, votes, strict=True):
        payload = {
            "scores": [
                {"index": i, "value": v, "verdict": "ok", "notes": ""}
                for i, v in enumerate((value, 2.0))
            ]
        }
        FakeModel([FakeTurn(text=json.dumps(payload))]).register(alias)
    ordered = sorted(votes)
    middle = ordered[(len(ordered) - 1) // 2 : len(ordered) // 2 + 1]
    expected = {
        "mean": float(sum(map(Fraction, votes)) / len(votes)),
        "median": float(sum(map(Fraction, middle)) / len(middle)),
        "min": min(votes),
        "max": max(votes),
    }[aggregate]
    scores = await check_judge(
        LLMJudge(
            LLMJudge.Config(
                rubric="r",
                models=list(aliases),
                aggregate=aggregate,
                scale_min=lower,
                scale_max=upper,
            )
        ),
        Goal(id="g", namespace="n"),
        candidates(2),
        config=configuration(aliases),
    )
    assert [score.candidate_id for score in scores] == ["c0", "c1"]
    assert [score.value for score in scores] == [expected, 2.0]
    for score in scores:
        assert score.value is not None
        assert math.isfinite(score.value)
        assert lower <= score.value <= upper
    assert all(len(score.llm_call_ids) == len(votes) for score in scores)


@pytest.mark.parametrize("nested", [False, True])
async def test_multiple_llm_composite_members(nested: bool) -> None:
    from swarm_arena.scoring import Lexicographic
    from swarm_arena.testing import check_judge

    FakeModel([turn(2, 6)]).register("a")
    FakeModel([turn(2, 8)]).register("b")
    second = LLMJudge("secondary", ["b"])
    judge = Lexicographic(
        [LLMJudge("primary", ["a"]), Lexicographic([second]) if nested else second]
    )
    scores = await check_judge(
        judge, Goal(id="g", namespace="n"), candidates(2), config=configuration(("a", "b"))
    )
    assert len(scores) == 2
    assert all(score.key == (6.0, 8.0) for score in scores)
    assert all(len(score.llm_call_ids) == 2 for score in scores)


async def test_swarm_with_multiple_llm_ranking_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.run import all_events, plan

    from swarm_arena import Limits, Ref, Swarm
    from swarm_arena.scoring import Lexicographic

    fake = FakeModel([])

    async def respond(*args: Any, **kwargs: Any) -> ModelResponse:
        rows = json.loads(args[1].split("\n", 1)[1].rsplit("\n", 1)[0])
        return await FakeModel([turn(len(rows))]).get_response(*args, **kwargs)

    monkeypatch.setattr(fake, "get_response", respond)
    fake.register("a")
    ref = Ref.of(Lexicographic([LLMJudge("primary", ["a"]), LLMJudge("secondary", ["a"])]))
    ref = ref.model_copy(update={"name": "composite"})
    planned = plan(rounds=1, agents=2).model_copy(update={"judges": [ref], "ranking": "composite"})
    config = configuration().model_copy(
        update={
            "limits": Limits(max_tokens=1000000),
            "run_root": str(tmp_path / "runs"),
            "arena_root": str(tmp_path / "arena"),
            "artifact_root": str(tmp_path / "artifacts"),
        }
    )
    result = await Swarm(config).run(planned)
    assert result["g"]
    events = all_events(config)
    assert not [event for event in events if event.type == "judge_error"]
    calls = {event.payload["call"]["id"] for event in events if event.type == "llm_call"}
    assert calls
    for event in events:
        if event.type == "score":
            assert set(event.payload["score"]["llm_call_ids"]) <= calls


async def test_limit_stops_sibling_dispatch_and_preserves_inflight_charge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio

    from swarm_arena import Limits
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena.llm import BudgetExceeded, _judge, _structured

    draft_started, refused = anyio.Event(), anyio.Event()
    original_reserve = Tranche.reserve_call

    async def reserve(self: Tranche, estimate: Any) -> str:
        if estimate.tokens > 1000000:
            await draft_started.wait()
        return await original_reserve(self, estimate)

    monkeypatch.setattr(Tranche, "reserve_call", reserve)
    original_structured = _judge.structured_call
    refusals: list[BaseException] = []

    async def structured(**kwargs: Any) -> tuple[dict[str, Any], list[str]]:
        try:
            return await original_structured(**kwargs)
        except BudgetExceeded as error:
            refusals.append(error)
            refused.set()
            raise

    monkeypatch.setattr(_judge, "structured_call", structured)
    monkeypatch.setattr(_structured, "supports", lambda spec, feature: spec.model != "fake/draft")
    draft = FakeModel([FakeTurn(text="reasoning")])
    original_response = draft.get_response

    async def held(*args: Any, **kwargs: Any) -> ModelResponse:
        draft_started.set()
        await refused.wait()
        return await original_response(*args, **kwargs)

    monkeypatch.setattr(draft, "get_response", held)
    draft.register("draft")
    formatter = FakeModel([turn(1)])
    formatter.register("fmt")
    huge = FakeModel([])
    huge.register("huge")
    config = configuration(("draft", "fmt", "huge"))
    config.models.root["huge"] = config.models.root["huge"].model_copy(
        update={"max_tokens": 5000000}
    )
    config.models.root["fmt"] = config.models.root["fmt"].model_copy(
        update={
            "prices": ModelSpec.Prices(input=0.5, output=1),
        }
    )
    config = config.model_copy(update={"limits": Limits(max_tokens=1000000)})
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=config) as run:
        scope = run.scope("judge")
        with llm_scope(scope), anyio.fail_after(10), pytest.raises(BudgetExceeded) as caught:
            await LLMJudge("r", ["draft", "huge"]).judge(
                goal,
                candidates(1),
                JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
            )
        assert caught.value is refusals[0]
        assert len(draft.calls) == 1
        assert not formatter.calls
        assert not huge.calls
        assert scope.tranche.pending() == 0
        assert [call.model_alias for call in run.calls()] == ["draft"]
        assert run.calls()[0].error is None
        assert run.charged(scope).usd == pytest.approx(sum(call.cost_usd for call in run.calls()))


@pytest.mark.parametrize("bad", ["missing", "out_of_scale"])
@pytest.mark.parametrize("bad_first", [False, True])
async def test_anchor_keeps_only_its_first_batch_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str, bad_first: bool
) -> None:
    fake = FakeModel([])

    async def respond(*args: Any, **kwargs: Any) -> ModelResponse:
        rows = json.loads(args[1].split("\n", 1)[1].rsplit("\n", 1)[0])
        ids = [row["candidate"]["id"] for row in rows]
        invalid = "c1" in ids if bad_first else "c3" in ids
        payload = []
        for row in rows:
            anchor = row["candidate"]["id"] == "c0"
            if anchor and invalid and bad == "missing":
                continue
            value = 11 if anchor and invalid else (9 if anchor else 5)
            payload.append(
                {"index": row["index"], "value": value, "verdict": "ok", "notes": "rubric"}
            )
        return await FakeModel([FakeTurn(text=json.dumps({"scores": payload}))]).get_response(
            *args, **kwargs
        )

    monkeypatch.setattr(fake, "get_response", respond)
    fake.register("a")
    goal = Goal(id="g", namespace="n")
    judge = LLMJudge(LLMJudge.Config(rubric="r", models=["a"], batch_size=3, anchors=1))
    async with open_llm_run(tmp_path, config=configuration()) as run:
        scope = run.scope("judge")
        with llm_scope(scope):
            scores = await judge.judge(
                goal,
                candidates(6),
                JudgeContext(
                    goal=goal,
                    executor=Mock(),
                    artifact_root=str(tmp_path),
                    anchors=frozenset({"c0"}),
                ),
            )
        by_id = {score.candidate_id: score for score in scores}
        assert len(scores) == (5 if bad_first else 6)
        if bad_first:
            assert "c0" not in by_id
        else:
            assert by_id["c0"].value == 9
            assert "invalid" in by_id["c0"].notes
            assert "9.0" in by_id["c0"].notes
            assert set(by_id["c0"].llm_call_ids) == {call.id for call in run.calls()}
