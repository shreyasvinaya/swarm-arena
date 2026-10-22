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


async def test_empty_draft_member_is_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena.llm import _structured
    from swarm_arena.testing import check_judge

    empty = FakeModel([FakeTurn(text="")])
    empty.register("a")
    FakeModel([turn(1, 6)]).register("b")
    formatter = FakeModel([turn(1, 10)])
    formatter.register("fmt")
    config = configuration(("a", "b", "fmt"))
    config.models.root["fmt"] = config.models.root["fmt"].model_copy(
        update={
            "prices": ModelSpec.Prices(input=0.5, output=1),
        }
    )
    monkeypatch.setattr(_structured, "supports", lambda spec, feature: spec.model != "fake/a")
    scores = await check_judge(
        LLMJudge("r", ["a", "b"]), Goal(id="g", namespace="n"), candidates(1), config=config
    )
    assert len(scores) == 1
    assert scores[0].value == 6
    assert len(scores[0].llm_call_ids) == 2
    assert formatter.calls == []


@pytest.mark.parametrize(
    "aliases", [("a", "b", "c"), ("b",), ("b", "b"), ("a", "b", "b"), ("a", "b", "a")]
)
async def test_provider_panel_drop_reports_after_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aliases: tuple[str, ...]
) -> None:
    import litellm

    from swarm_arena._engine.stops import provider_stops
    from swarm_arena._model.canonical import canonical_json, sha256_hex
    from swarm_arena._model.ref import Ref
    from swarm_arena._scoring.runner import judge_round

    for alias in set(aliases):
        fake = FakeModel([turn(2), turn(2)])
        if alias == "b":

            async def fail(*args: Any, **kwargs: Any) -> ModelResponse:
                raise litellm.ServiceUnavailableError(
                    message="PRIVATE", llm_provider="untrusted", model="wrong"
                )

            monkeypatch.setattr(fake, "get_response", fail)
        fake.register(alias)
    judge = LLMJudge("r", aliases)
    ctx = JudgeContext(
        goal=Goal(id="g", namespace="n"), executor=Mock(), artifact_root=str(tmp_path)
    )
    async with open_llm_run(tmp_path, config=configuration(tuple(set(aliases)))) as run:
        scope = run.scope("judge")
        service = provider_stops(scope.journal)
        cancelled: list[str] = []
        for address in ("g/a/one", "other/b/two"):
            service.register(address, frozenset({"fake"}), lambda a=address: cancelled.append(a))
        service.register(
            "healthy/c/three", frozenset({"healthy"}), lambda: cancelled.append("healthy")
        )
        with llm_scope(scope):
            outcome = await judge_round(
                "g", [(Ref.of(judge), judge)], candidates(2), [], lambda _: ctx
            )
        assert scope.tranche.pending() == 0
        assert provider_stops(scope.journal).generations() == {"fake": 1}
        assert set(cancelled) == {"g/a/one", "other/b/two"}
        service.register("late/d/four", frozenset({"fake"}), lambda: cancelled.append("late"))
        assert cancelled[-1] == "late"
        assert run.charged(scope).usd == pytest.approx(sum(c.cost_usd for c in run.calls()))
        assert not [
            e for e in run.events("provider_failed") if e.payload.get("disposition") == "panel_drop"
        ]
        if set(aliases) == {"b"}:
            assert outcome.panel_drops == []
            assert outcome.provider_failed
            assert not outcome.errors
            assert not outcome.scores
        else:
            remaining = sorted(a for a in aliases if a != "b")
            assert {s.panel_hash for s in outcome.scores} == {
                sha256_hex(canonical_json(list(remaining)))
            }
            assert not outcome.errors

            assert [drop.dropped_member for drop in outcome.panel_drops] == ["b"] * aliases.count(
                "b"
            )
            assert {drop.panel_hash for drop in outcome.panel_drops} == {
                sha256_hex(canonical_json(list(remaining)))
            }
            assert {drop.provider for drop in outcome.panel_drops} == {"fake"}


async def test_recursive_provider_survivors_across_batches_near_scale_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._errors import ProviderFailed, _core_error
    from swarm_arena._model.canonical import canonical_json, sha256_hex
    from swarm_arena.scoring import Lexicographic
    from swarm_arena.testing import check_judge

    for alias in ("a", "b", "c"):
        fake = FakeModel([])

        async def respond(*args: Any, _alias: str = alias, **kwargs: Any) -> ModelResponse:
            rows = json.loads(args[1].split("\n", 1)[1].rsplit("\n", 1)[0])
            ids = [row["candidate"]["id"] for row in rows]
            if (_alias == "a" and "c0" in ids) or (_alias == "b" and "c8" in ids):
                raise _core_error(ProviderFailed, "fake")
            return await FakeModel([turn(len(rows), 1e308)]).get_response(*args, **kwargs)

        monkeypatch.setattr(fake, "get_response", respond)
        fake.register(alias)
    panel = LLMJudge(
        LLMJudge.Config(
            rubric="r", models=["a", "b", "c", "c"], aggregate="median", scale_max=1e308, anchors=0
        )
    )
    scores = await check_judge(
        Lexicographic([Lexicographic([panel])]),
        Goal(id="g", namespace="n"),
        candidates(10),
        config=configuration(("a", "b", "c")),
    )
    identity = sha256_hex(canonical_json(["c", "c"]))
    for _ in range(2):
        identity = sha256_hex(canonical_json([identity]))
    assert len(scores) == 10
    assert {s.panel_hash for s in scores} == {identity}
    assert all(s.key == (1e308,) and s.llm_call_ids for s in scores)


async def partial_panel_crash_driver(root: str) -> None:
    """Leave a completed partial panel batch in a public-run journal."""
    import os

    from _engine.test_goal import Y_JUDGE, _plan
    from _fakes.engine import with_executor

    from swarm_arena import Limits, Ref, Swarm
    from swarm_arena._errors import ProviderFailed, _core_error
    from swarm_arena._exec.threads import run_sync_daemon
    from swarm_arena._journal.writer import JournalWriter

    path = Path(root)
    config = with_executor(
        configuration(("a", "b", "c")).model_copy(
            update={
                "run_root": str(path / "runs"),
                "arena_root": str(path / "arena"),
                "artifact_root": str(path / "artifacts"),
                "limits": Limits(max_tokens=10_000_000, max_usd=1000, max_agents_per_fingerprint=8),
            }
        ),
        workspace_root=str(path / "ws"),
    )
    counts: dict[str, int] = {}

    async def respond(*args: Any, alias: str, **kwargs: Any) -> ModelResponse:
        counts[alias] = counts.get(alias, 0) + 1
        count = counts[alias]

        def record() -> None:
            with (path / "calls").open("a") as output:
                output.write(f"{alias}:{count}\n")

        await run_sync_daemon(record)
        rows = json.loads(args[1].split("\n", 1)[1].rsplit("\n", 1)[0])
        if (alias == "a" and count == 1) or (alias == "b" and count == 2):
            raise _core_error(ProviderFailed, "fake")
        return await FakeModel([turn(len(rows))]).get_response(*args, **kwargs)

    from functools import partial

    with pytest.MonkeyPatch.context() as patch:
        for alias in ("a", "b", "c"):
            fake = FakeModel([])
            patch.setattr(fake, "get_response", partial(respond, alias=alias))
            fake.register(alias)
        original = JournalWriter._publish

        async def publish(self: JournalWriter) -> None:
            hit = any(
                e.type == "provider_failed" and e.payload.get("disposition") == "panel_drop"
                for e in self._queue
            )
            await original(self)
            if hit:
                os._exit(91)

        patch.setattr(JournalWriter, "_publish", publish)
        panel = Ref.of(
            LLMJudge(
                LLMJudge.Config(rubric="r", models=["a", "b", "c", "c"], batch_size=2, anchors=0)
            )
        ).model_copy(update={"name": "panel"})
        await Swarm(config).run(
            _plan(rounds=1, teams=2, agents_per_team=3, judges=[Y_JUDGE, panel]), run_id="panel"
        )


async def test_partial_panel_public_resume_makes_no_new_calls(tmp_path: Path) -> None:
    import os

    import anyio
    from _fakes.run import all_events

    from swarm_arena import Swarm
    from swarm_arena._model.canonical import canonical_json, sha256_hex

    repo = Path(__file__).parents[2]
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(repo / "tests"), str(repo / "src"), str(repo)]),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
    )
    script = (
        "import anyio,sys; from tests.llm.test_judge import partial_panel_crash_driver; "
        "anyio.run(partial_panel_crash_driver,sys.argv[1])"
    )
    with anyio.fail_after(60):
        process = await anyio.run_process(
            ["uv", "run", "--project", str(repo), "python", "-c", script, str(tmp_path)],
            cwd=tmp_path,
            env=environment,
            check=False,
        )
    assert process.returncode == 91, process.stderr.decode()
    config = RunConfig.load(tmp_path / "runs" / "panel" / "run.toml")
    calls = (tmp_path / "calls").read_text()
    recorded = all_events(config)
    scores = [
        e.payload["score"]
        for e in recorded
        if e.type == "score" and e.payload["score"]["judge"] == "panel"
    ]
    drops = [e for e in recorded if e.payload.get("disposition") == "panel_drop"]
    assert [e.payload["dropped_member"] for e in drops] == ["a", "b"]
    assert {e.payload["panel_hash"] for e in drops} == {sha256_hex(canonical_json(["c", "c"]))}
    assert all("complete" not in e.payload for e in drops)
    assert len(scores) == 6
    assert {s["panel_hash"] for s in scores} == {sha256_hex(canonical_json(["c", "c"]))}
    spies = [FakeModel([]) for _ in range(3)]
    for fake, alias in zip(spies, ("a", "b", "c"), strict=True):
        fake.register(alias)
    with anyio.fail_after(60):
        result = await Swarm(config).resume("panel", force=True)
        again = await Swarm(config).resume("panel", force=True)
    assert result["g"] == again["g"]
    assert (tmp_path / "calls").read_text() == calls
    assert all(not fake.calls for fake in spies)


async def test_llm_ranking_final_runs_with_zero_tool_reservation(tmp_path: Path) -> None:
    from _engine.test_goal import _plan
    from _fakes.engine import with_executor
    from _fakes.run import all_events

    from swarm_arena import Limits, Ref, Swarm
    from swarm_arena.tool import Estimate

    fake = FakeModel([turn(2), turn(2)])
    fake.register("a")
    config = with_executor(
        configuration().model_copy(
            update={
                "run_root": str(tmp_path / "runs"),
                "arena_root": str(tmp_path / "arena"),
                "artifact_root": str(tmp_path / "artifacts"),
                "limits": Limits(
                    max_tokens=2_000_000, max_usd=100, default_run_reservation=Estimate()
                ),
            }
        ),
        workspace_root=str(tmp_path / "ws"),
    )
    judge = Ref.of(LLMJudge("r", ["a"])).model_copy(update={"name": "y"})
    result = await Swarm(config).run(_plan(rounds=1, judges=[judge]), run_id="zero")
    assert result["g"]
    final_scores = [
        e.payload["score"] for e in all_events(config) if e.type == "score" and e.round == 0
    ]
    assert len(final_scores) == 2
    assert all(s["llm_call_ids"] for s in final_scores)
    solution = json.loads(
        (tmp_path / "runs" / "zero" / "goals" / "g" / "solution.json").read_text()
    )
    assert solution["status"] == "ok"


@pytest.mark.parametrize("aliases", [("b",), ("a", "b")])
async def test_panel_accounting_failure_prevents_provider_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aliases: tuple[str, ...]
) -> None:
    from swarm_arena._engine.stops import provider_stops
    from swarm_arena._errors import ProviderFailed, StorageError, _core_error, _is_core_signal
    from swarm_arena.llm import _judge

    async def fail_provider(*args: Any, **kwargs: Any) -> ModelResponse:
        raise _core_error(ProviderFailed, "fake")

    async def fail_accounting(*args: Any, **kwargs: Any) -> None:
        raise _core_error(StorageError, "LLM settlement failed: OSError")

    for alias in aliases:
        fake = FakeModel([turn(1)])
        if alias == "b":
            monkeypatch.setattr(fake, "get_response", fail_provider)
        fake.register(alias)
    monkeypatch.setattr(_judge, "_finish_accounting", fail_accounting)
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=configuration(aliases)) as run:
        scope = run.scope("judge")
        ctx = JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path))
        with (
            llm_scope(scope),
            pytest.raises(StorageError, match="LLM settlement failed: OSError") as caught,
        ):
            await LLMJudge("r", aliases).judge(goal, candidates(1), ctx)
        assert _is_core_signal(caught.value)
        await scope.journal.commit()
        assert provider_stops(scope.journal).generations() == {}
        assert not run.events("provider_failed")


async def test_provider_failure_is_reported_before_sibling_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio
    import litellm

    from swarm_arena import Limits
    from swarm_arena._budget.amount import Amount
    from swarm_arena._budget.tranche import Tranche
    from swarm_arena._engine.stops import provider_stops
    from swarm_arena.llm import BudgetExceeded

    failed = anyio.Event()
    reserve = Tranche.reserve_call

    async def ordered_reserve(self: Tranche, estimate: Amount) -> str:
        if estimate.tokens > 1_000_000:
            await failed.wait()
        return await reserve(self, estimate)

    monkeypatch.setattr(Tranche, "reserve_call", ordered_reserve)

    class Down(FakeModel):
        async def get_response(self, *args: Any, **kwargs: Any) -> Any:
            failed.set()
            raise litellm.ServiceUnavailableError(
                message="PRIVATE", llm_provider="fake", model="fake/down"
            )

    Down([]).register("down")
    huge = FakeModel([])
    huge.register("huge")
    config = configuration(("down", "huge"))
    config.models.root["huge"] = config.models.root["huge"].model_copy(
        update={"max_tokens": 5_000_000}
    )
    config = config.model_copy(update={"limits": Limits(max_tokens=1_000_000)})
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=config) as run:
        scope = run.scope("judge")
        cancelled: list[bool] = []
        provider_stops(scope.journal).register(
            "other/t/a", frozenset({"fake"}), lambda: cancelled.append(True)
        )
        with llm_scope(scope), pytest.raises(BudgetExceeded):
            await LLMJudge("r", ["down", "huge"]).judge(
                goal,
                candidates(1),
                JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
            )
        assert provider_stops(scope.journal).generations() == {"fake": 1}
        assert cancelled == [True]
        assert huge.calls == []
        assert scope.tranche.pending() == 0
        assert not [
            e for e in run.events("provider_failed") if e.payload.get("variant") == "receipt"
        ]
