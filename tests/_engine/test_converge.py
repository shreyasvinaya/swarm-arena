from swarm_arena._engine.converge import select_finalists
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.results import Candidate, Score
from swarm_arena._model.tooling import ToolOutput, ToolRequest


def _c(id: str, *, data: dict[str, int]) -> Candidate:
    return Candidate(
        id=id,
        run_id="r",
        goal_id="g",
        team="t",
        agent="a",
        round=1,
        request=ToolRequest(),
        output=ToolOutput(status="ok", data=data),
    )


def _s(value: float) -> Score:
    return Score.maximize(value)


def _fp(candidate: Candidate) -> str:
    return sha256_hex(canonical_json(candidate.output.data))


def test_select_top_n_distinct_by_fingerprint() -> None:
    rk = [
        (_c("c-1", data={"x": 1}), _s(9)),
        (_c("c-2", data={"x": 1}), _s(8)),
        (_c("c-3", data={"x": 2}), _s(7)),
    ]
    assert [c.id for c in select_finalists(rk, top_n=2, distinct=True, fingerprint=_fp)] == [
        "c-1",
        "c-3",
    ]
    assert [c.id for c in select_finalists(rk, top_n=2, distinct=False, fingerprint=_fp)] == [
        "c-1",
        "c-2",
    ]
    assert len(select_finalists(rk, top_n=9, distinct=True, fingerprint=_fp)) == 2
    assert select_finalists(rk, top_n=0, distinct=True, fingerprint=_fp) == []
    assert select_finalists([], top_n=2, distinct=True, fingerprint=_fp) == []


def test_non_distinct_selection_does_not_fingerprint() -> None:
    def unexpected(candidate: Candidate) -> str:
        raise AssertionError("unnecessary fingerprint")

    candidate = _c("c-1", data={})
    assert select_finalists(
        [(candidate, _s(1))], top_n=2, distinct=False, fingerprint=unexpected
    ) == [candidate]


def test_unknown_fingerprints_are_always_distinct() -> None:
    ranking = [(_c("c-1", data={}), _s(2)), (_c("c-2", data={}), _s(1))]
    assert len(select_finalists(ranking, top_n=2, distinct=True, fingerprint=lambda c: None)) == 2
