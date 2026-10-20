from collections.abc import Callable, Sequence
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import GOAL, FixedJudge, PanelJudge, cand, rosen_score, small_b
from pydantic import ValidationError

from swarm_arena._errors import ConfigError
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.goal import Goal
from swarm_arena._model.ids import IdGen
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate, Score
from swarm_arena._scoring.judge import JudgeContext, stamp_scores
from swarm_arena._scoring.lexicographic import Lexicographic
from swarm_arena._scoring.program_judge import ProgramJudge

SECRET = "6dca447c8d41f358f1c5dcc3d2e651cb"


async def test_key_is_member_keys_in_order(judge_ctx: JudgeContext) -> None:
    lex = Lexicographic(
        [FixedJudge(FixedJudge.Config(value=1.0)), FixedJudge(FixedJudge.Config(value=2.0))]
    )
    [s] = await lex.judge(GOAL, [cand(id="c1")], judge_ctx)
    assert (s.key, s.value, s.candidate_id) == ((1.0, 2.0), 1.0, "c1")


async def test_second_member_breaks_ties(judge_ctx: JudgeContext) -> None:
    lex = Lexicographic([ProgramJudge(rosen_score), ProgramJudge(small_b)])
    a, b = cand(id="a", data={"f": 1.0, "b": 3.0}), cand(id="b", data={"f": 1.0, "b": 0.5})
    scores = {s.candidate_id: s.key for s in await lex.judge(GOAL, [a, b], judge_ctx)}
    assert scores == {"a": (-1.0, -3.0), "b": (-1.0, -0.5)}
    assert max(scores, key=scores.__getitem__) == "b"


async def test_member_error_leaves_candidate_unjudged(judge_ctx: JudgeContext) -> None:
    lex = Lexicographic([ProgramJudge(rosen_score), ProgramJudge("_fakes.tools:raising_score")])
    assert await lex.judge(GOAL, [cand(id="c1", data={"f": 1.0})], judge_ctx) == []
    assert judge_ctx.errors["c1"].startswith("RuntimeError")


def test_hashes_and_forms() -> None:
    assert Lexicographic.name == "lexicographic"
    one = Lexicographic([FixedJudge(FixedJudge.Config(value=1.0))])
    assert (
        one.rubric_hash() != Lexicographic([FixedJudge(FixedJudge.Config(value=2.0))]).rubric_hash()
    )
    assert one.panel_hash() == ""
    ref = Ref.of(FixedJudge(FixedJudge.Config(value=1.0)))
    assert (
        Lexicographic([ref]).config
        == one.config
        == Lexicographic(Lexicographic.Config(judges=[ref])).config
    )
    renamed = ref.model_copy(update={"name": "other"})
    assert Lexicographic([renamed]).rubric_hash() != one.rubric_hash()
    assert one.rubric_hash() == sha256_hex(
        canonical_json([["fixed", FixedJudge(FixedJudge.Config(value=1.0)).rubric_hash()]])
    )


def test_init_imports_nothing() -> None:
    lex = Lexicographic([Ref(path="no_such_mod:J")])
    with pytest.raises(ValueError, match="cannot import no_such_mod:J"):
        lex.preflight()


async def test_composite_panel_hash_follows_member_scores(judge_ctx: JudgeContext) -> None:
    lex = Lexicographic(
        [
            PanelJudge(PanelJudge.Config(panel="p", per_candidate={"c2": "dropped"})),
            FixedJudge(FixedJudge.Config(value=1.0)),
        ]
    )
    a, b = stamp_scores(
        await lex.judge(GOAL, [cand(id="c1"), cand(id="c2")], judge_ctx),
        judge=lex,
        judge_name="lex",
        ids=IdGen("r", 0),
        batch_id="b",
    )
    assert a.panel_hash == lex.panel_hash() == sha256_hex(canonical_json(["p", ""]))
    assert b.panel_hash == sha256_hex(canonical_json(["dropped", ""])) != lex.panel_hash()


def test_member_requirements_propagate() -> None:
    lex = Lexicographic(
        [
            PanelJudge(PanelJudge.Config(anchors=2, providers=["openai"], models=["gpt"])),
            FixedJudge(FixedJudge.Config(value=1.0)),
        ]
    )
    assert (lex.anchor_count(), lex.providers(), lex.models()) == (
        2,
        frozenset({"openai"}),
        ("gpt",),
    )
    assert Lexicographic([FixedJudge(FixedJudge.Config(value=1.0))]).anchor_count() == 0
    combined = Lexicographic(
        [
            lex,
            PanelJudge(
                PanelJudge.Config(anchors=3, providers=["other", "openai"], models=["gpt", "next"])
            ),
        ]
    )
    assert combined.anchor_count() == 3
    assert combined.providers() == frozenset({"openai", "other"})
    assert combined.models() == ("gpt", "next")


@pytest.mark.parametrize(
    "update",
    [
        {"key": (float("nan"),)},
        {"key": ()},
        {"key": (SECRET,)},
        {"key": None},
        {"notes": {SECRET: SECRET}},
        {"llm_call_ids": [object()]},
        {"value": SECRET},
    ],
)
async def test_bad_member_score_is_candidate_error(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch, update: dict[str, Any]
) -> None:
    async def score(
        self: FixedJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            Score.maximize(1).model_copy(
                update={"candidate_id": c.id, **(update if c.id == "bad" else {})}
            )
            for c in candidates
        ]

    monkeypatch.setattr(FixedJudge, "judge", score)
    lex = Lexicographic([FixedJudge(FixedJudge.Config(value=1)), PanelJudge(PanelJudge.Config())])
    scores = await lex.judge(GOAL, [cand(id="bad"), cand(id="good")], judge_ctx)
    assert [s.candidate_id for s in scores] == ["good"]
    if "key" in update:
        assert judge_ctx.errors["bad"] == "member fixed gave a non-finite key"
    else:
        assert "cannot combine member scores" in judge_ctx.errors["bad"]
    assert SECRET not in repr(judge_ctx.errors)


async def test_metadata_and_missing_member(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    async def score(
        self: FixedJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        calls.append([c.id for c in candidates])
        return [
            Score(
                candidate_id=c.id,
                key=(self.config.value, 3),
                value=self.config.value,
                verdict=None if self.config.value == 1 else "ok",
                notes="" if self.config.value == 1 else "second",
                llm_call_ids=[str(self.config.value)],
            )
            for c in reversed(candidates)
            if c.id != "missing" or self.config.value == 2
        ]

    monkeypatch.setattr(FixedJudge, "judge", score)
    lex = Lexicographic(
        [FixedJudge(FixedJudge.Config(value=1)), FixedJudge(FixedJudge.Config(value=2))]
    )
    [s] = await lex.judge(GOAL, [cand(id="good"), cand(id="missing")], judge_ctx)
    assert s.key == (1, 3, 2, 3)
    assert s.value == 1
    assert s.verdict == "ok"
    assert s.notes == "second"
    assert s.llm_call_ids == ["1.0", "2.0"]
    assert s.panel_hash == ""
    assert judge_ctx.errors == {"missing": "member fixed gave no score"}
    assert calls == [["good", "missing"], ["good", "missing"]]


def test_members_built_once_and_preflight_called(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[FixedJudge] = []
    monkeypatch.setattr(FixedJudge, "preflight", lambda self: seen.append(self))
    lex = Lexicographic([FixedJudge(FixedJudge.Config(value=1))])
    lex.preflight()
    lex.anchor_count()
    lex.rubric_hash()
    lex.providers()
    lex.models()
    lex.panel_hash()
    lex.preflight()
    assert len(seen) == 2
    assert seen[0] is seen[1]


@pytest.mark.parametrize(
    "make",
    [
        lambda: Lexicographic.Config(judges=[{"path": "example:J", "config": {}, "oops": SECRET}]),
        lambda: Lexicographic.Config.model_validate_json('{"judges": ' + SECRET),
        lambda: Lexicographic.Config.model_validate({"judges": SECRET}),
    ],
)
def test_config_errors_are_sanitized(make: Callable[[], object]) -> None:
    with pytest.raises(ValidationError) as caught:
        make()
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ConfigError])
def test_member_build_errors_are_sanitized(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    def fail(self: FixedJudge, config: object) -> None:
        raise error_type(SECRET)

    monkeypatch.setattr(FixedJudge, "__init__", fail)
    lex = Lexicographic([Ref(path="_fakes.tools:FixedJudge", config={"value": 1})])
    with pytest.raises((ValueError, TypeError), match="judge") as caught:
        lex.preflight()
    assert_secret_absent(caught.value, SECRET)


def test_rubric_errors_are_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: FixedJudge) -> str:
        raise ValueError(SECRET)

    monkeypatch.setattr(FixedJudge, "rubric_hash", fail)
    lex = Lexicographic([FixedJudge(FixedJudge.Config(value=1))])
    with pytest.raises(ValueError, match="cannot encode lexicographic rubric") as caught:
        lex.rubric_hash()
    assert_secret_absent(caught.value, SECRET)


async def test_mixed_member_key_lengths_fail_closed_for_whole_batch(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def score(
        self: FixedJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            Score(candidate_id=c.id, key=(1,) if c.id == "short" else (1, 2))
            for c in candidates
            if c.id != "missing"
        ]

    monkeypatch.setattr(FixedJudge, "judge", score)
    lex = Lexicographic([FixedJudge(FixedJudge.Config(value=1)), PanelJudge(PanelJudge.Config())])
    candidates = [cand(id="short"), cand(id="long"), cand(id="missing")]
    assert await lex.judge(GOAL, candidates, judge_ctx) == []
    assert judge_ctx.errors == {
        c.id: "member fixed gave keys of different lengths" for c in candidates
    }


@pytest.mark.parametrize("direct", [False, True])
def test_empty_members_rejected_at_config_time(direct: bool) -> None:
    make = (lambda: Lexicographic.Config(judges=[])) if direct else (lambda: Lexicographic([]))
    with pytest.raises((ConfigError, ValidationError), match="judges") as caught:
        make()
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize("method", ["preflight", "rubric_hash"])
def test_member_import_diagnostics_keep_path(method: str) -> None:
    lex = Lexicographic([Ref(path="no_such_mod:J")])
    with pytest.raises(ValueError, match="cannot import no_such_mod:J"):
        getattr(lex, method)()


@pytest.mark.parametrize("method", ["preflight", "rubric_hash"])
def test_member_config_diagnostics_keep_safe_detail(method: str) -> None:
    lex = Lexicographic([Ref(path="_fakes.tools:FixedJudge", config={"value": SECRET})])
    with pytest.raises(ConfigError, match=r"_fakes\.tools:FixedJudge") as caught:
        getattr(lex, method)()
    assert str(caught.value) == "_fakes.tools:FixedJudge: value: float_parsing"
    assert str(caught.value).count("_fakes.tools:FixedJudge") == 1  # the path once
    assert_secret_absent(caught.value, SECRET)


@pytest.mark.parametrize(
    "item",
    [None, {"candidate_id": "bad"}, Score.maximize(1).model_copy(update={"candidate_id": []})],
)
async def test_non_score_member_item_is_candidate_error(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch, item: Any
) -> None:
    async def score(
        self: FixedJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [item, Score(candidate_id="good", key=(1,))]

    monkeypatch.setattr(FixedJudge, "judge", score)
    lex = Lexicographic([FixedJudge(FixedJudge.Config(value=1)), PanelJudge(PanelJudge.Config())])
    scores = await lex.judge(GOAL, [cand(id="bad"), cand(id="good")], judge_ctx)
    assert [s.candidate_id for s in scores] == ["good"]
    assert judge_ctx.errors == {"bad": "member fixed gave an invalid score"}


async def test_member_exception_propagates(
    judge_ctx: JudgeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = RuntimeError("provider unavailable")

    async def fail(
        self: FixedJudge, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        raise error

    monkeypatch.setattr(FixedJudge, "judge", fail)
    lex = Lexicographic([FixedJudge(FixedJudge.Config(value=1))])
    with pytest.raises(RuntimeError, match="provider unavailable") as caught:
        await lex.judge(GOAL, [cand()], judge_ctx)
    assert caught.value is error
