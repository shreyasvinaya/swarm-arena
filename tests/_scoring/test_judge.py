import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import GOAL, FixedJudge, RecordingExecutor, cand
from pydantic import BaseModel, ConfigDict, field_serializer

from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.ids import IdGen, run_tag
from swarm_arena._model.results import Score
from swarm_arena._scoring.judge import Judge, JudgeContext, is_finite_score, stamp_scores


def test_judge_defaults() -> None:
    j = FixedJudge(FixedJudge.Config(value=1.0))
    assert j.rubric_hash() == sha256_hex(
        canonical_json({"path": "_fakes.tools:FixedJudge", "config": {"value": 1.0}})
    )
    assert (j.panel_hash(), j.anchor_count(), j.providers(), j.models()) == ("", 0, frozenset(), ())
    assert j.preflight() is None
    with pytest.raises(TypeError, match="abstract"):
        Judge()  # ty: ignore[call-non-callable]


async def test_fixed_judge_scores_each_candidate() -> None:
    j = FixedJudge(FixedJudge.Config(value=1.0))
    ctx = JudgeContext(goal=GOAL, executor=RecordingExecutor(), artifact_root="/a")
    assert [
        (s.candidate_id, s.key) for s in await j.judge(GOAL, [cand(id="c1"), cand(id="c2")], ctx)
    ] == [("c1", (1.0,)), ("c2", (1.0,))]


def test_judge_context() -> None:
    ctx = JudgeContext(
        goal=GOAL,
        executor=RecordingExecutor(),
        artifact_root="/a",
        fidelity="f",
        batch_id="b",
        anchors=frozenset({"c0"}),
        final=True,
        read_cache=False,
    )
    ctx.judge_error("c1", "first")
    ctx.judge_error("c1", "second")
    assert ctx.errors == {"c1": "first"}
    assert (ctx.fidelity, ctx.final, ctx.read_cache) == (
        "f",
        True,
        False,
    )
    assert (ctx.batch_id, ctx.anchors) == ("b", frozenset({"c0"}))
    other = JudgeContext(goal=GOAL, executor=RecordingExecutor(), artifact_root="/a")
    assert other.errors == {}
    assert (other.fidelity, other.batch_id, other.anchors, other.final, other.read_cache) == (
        "",
        "",
        frozenset(),
        False,
        True,
    )


def test_stamp_scores() -> None:
    j = FixedJudge(FixedJudge.Config(value=1.0))
    raw = [
        Score.maximize(1.0).model_copy(update={"candidate_id": "c1"}),
        Score(key=(2.0,), candidate_id="c2", panel_hash="p", batch_id="own"),
    ]
    a, b = stamp_scores(raw, judge=j, judge_name="rel-error", ids=IdGen("r", 1), batch_id="batch-1")
    tag = run_tag("r")
    assert (a.id, a.judge, a.rubric_hash, a.panel_hash, a.batch_id, a.candidate_id) == (
        f"s-{tag}-e1-0001",
        "rel-error",
        j.rubric_hash(),
        "",
        "batch-1",
        "c1",
    )
    assert (b.id, b.panel_hash, b.batch_id) == (f"s-{tag}-e1-0002", "p", "own")
    assert raw[0].id == raw[0].judge == raw[0].batch_id == ""
    assert a.key == raw[0].key
    assert a.value == raw[0].value


@pytest.mark.parametrize("panel", ["", "fallback"])
@pytest.mark.parametrize("panels", [[], ["own", "owned"], ["own", "", "", "last"]])
def test_stamp_hashes_once_per_batch(panel: str, panels: list[str]) -> None:
    class CountingJudge(FixedJudge):
        rubric_calls = 0
        panel_calls = 0

        def rubric_hash(self) -> str:
            self.rubric_calls += 1
            return f"rubric-{self.rubric_calls}"

        def panel_hash(self) -> str:
            self.panel_calls += 1
            return panel

    judge = CountingJudge(CountingJudge.Config(value=1.0))
    raw = [Score(key=(1.0,), panel_hash=value) for value in panels]
    ids = IdGen("r", 1)
    for call in range(1, 3):
        scores = stamp_scores(raw, judge=judge, judge_name="j", ids=ids, batch_id="b")
        assert judge.rubric_calls == call
        assert judge.panel_calls == (call if "" in panels else 0)
        assert [s.rubric_hash for s in scores] == [f"rubric-{call}"] * len(raw)
        assert [s.panel_hash for s in scores] == [value or panel for value in panels]


def test_stamp_overwrites_core_fields_and_uses_panel_default() -> None:
    class Panel(FixedJudge):
        def panel_hash(self) -> str:
            return "panel"

    j = Panel(Panel.Config(value=2.0))
    raw = Score(
        key=(2.0,),
        candidate_id="c",
        id="old",
        judge="old",
        rubric_hash="old",
        verdict="ok",
        notes="notes",
        llm_call_ids=["call"],
    )
    ids = IdGen("r", 3)
    ctx = JudgeContext(
        goal=GOAL,
        executor=RecordingExecutor(),
        artifact_root="/a",
        ids=ids,
        run_id="r",
        round=2,
        part="p",
        secrets_file="/s.env",
        tool_timeout_s=60.0,
    )
    [score] = stamp_scores([raw], judge=j, judge_name="j", ids=ctx.ids, batch_id="b")
    assert (score.id, score.judge, score.rubric_hash, score.panel_hash) == (
        f"s-{run_tag('r')}-e3-0001",
        "j",
        j.rubric_hash(),
        "panel",
    )
    assert (score.verdict, score.notes, score.llm_call_ids) == ("ok", "notes", ["call"])
    assert ctx.ids is ids
    assert (
        ctx.run_id,
        ctx.round,
        ctx.part,
        ctx.secrets_file,
        ctx.tool_timeout_s,
    ) == ("r", 2, "p", "/s.env", 60.0)
    assert stamp_scores([], judge=j, judge_name="j", ids=ids, batch_id="b") == []


def test_is_finite_score() -> None:
    assert is_finite_score(Score.maximize(1.0))
    assert not is_finite_score(Score.model_construct(key=(float("nan"),)))
    assert not is_finite_score(Score.model_construct(key=()))


@pytest.mark.parametrize("key", [(float("inf"),), (1.0, -float("inf")), (1.0, float("nan"))])
def test_is_finite_score_model_copy(key: tuple[float, ...]) -> None:
    assert not is_finite_score(Score.maximize(1.0).model_copy(update={"key": key}))


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(0.25, id="scalar"),
        pytest.param(("bad",), id="str-element"),
        pytest.param((None,), id="none-element"),
        pytest.param((10**400,), id="overflow"),
        pytest.param([1.0], id="list-key"),
        pytest.param((True,), id="bool-element"),
        pytest.param(None, id="none-key"),
        pytest.param((), id="empty"),
        pytest.param((float("inf"),), id="inf"),
        pytest.param((float("nan"),), id="nan"),
    ],
)
@pytest.mark.parametrize("construction", ["construct", "copy"])
def test_is_finite_score_rejects_malformed_key(key: object, construction: str) -> None:
    score = (
        Score.model_construct(key=key)
        if construction == "construct"
        else Score.maximize(1.0).model_copy(update={"key": key})
    )
    assert is_finite_score(score) is False


def test_is_finite_score_rejects_missing_key() -> None:
    assert is_finite_score(Score.model_construct()) is False


@pytest.mark.parametrize("key", [(0,), (-1, 2.5), (1.0,)])
def test_is_finite_score_accepts_finite_numeric_tuple(key: tuple[int | float, ...]) -> None:
    assert is_finite_score(Score.model_construct(key=key)) is True


def test_rubric_hash_sanitizes_serialization_errors() -> None:
    secret = "1748c57f137692a66d528db35a73990b_RUBRIC"

    class BrokenConfig(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        value: str

        @field_serializer("value")
        def serialize_value(self, value: str) -> str:
            raise ValueError(value)

    j = FixedJudge(FixedJudge.Config(value=1.0))
    j._config = BrokenConfig(value=secret)
    with pytest.raises(ValueError, match="cannot encode judge rubric") as error:
        j.rubric_hash()
    assert_secret_absent(error.value, secret)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
