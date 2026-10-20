import hashlib
import json

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena._model.board_types import Frontier, LineageGraph, Row
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import ToolOutput, ToolRequest


def test_frontier_digest_is_stable() -> None:
    m = Frontier.Mark(run_id="r", run_root="/x", epoch=1, seq=4, segment="journal/e1-4.jsonl")
    frontier = Frontier(marks=[m])
    assert frontier.digest() == Frontier(marks=[m]).digest() != Frontier().digest()
    raw = frontier.model_dump(mode="json")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert frontier.digest() == hashlib.sha256(canonical).hexdigest()
    assert Frontier.model_validate_json(frontier.model_dump_json()) == frontier
    assert frontier.model_copy(update={"reviews": ["review-1"]}).digest() != frontier.digest()
    assert frontier.model_copy(update={"curation": ["curation-1"]}).digest() != frontier.digest()
    assert Frontier.model_validate(dict(reversed(list(raw.items())))).digest() == frontier.digest()


def test_lineage_parents_and_ancestors() -> None:
    nodes = {
        id: Candidate(
            id=id,
            run_id="r",
            goal_id="g",
            team="t",
            agent="a",
            round=1,
            request=ToolRequest(),
            output=ToolOutput(status="ok"),
        )
        for id in ("a", "b", "c")
    }
    g = LineageGraph(root="c", nodes=nodes, edges=[("c", "b"), ("b", "a"), ("c", "a")])
    assert g.parents("c") == ["b", "a"]
    assert g.ancestors("c") == ["b", "a"]
    assert g.parents("a") == []
    assert g.ancestors("a") == []
    assert g.parents("missing") == []
    assert g.ancestors("missing") == []


def test_lineage_traversal_handles_cycles_and_deep_graphs() -> None:
    graph = LineageGraph(root="c", nodes={}, edges=[("c", "b"), ("c", "b"), ("b", "c"), ("b", "a")])
    assert graph.ancestors("c") == ["b", "a"]
    deep = LineageGraph(root="0", nodes={}, edges=[(str(i), str(i + 1)) for i in range(1200)])
    assert deep.ancestors("0") == [str(i) for i in range(1, 1201)]


def test_row_preserves_ranking_provenance() -> None:
    row = Row(
        rank=1,
        candidate_id="c",
        run_id="r",
        goal_id="g",
        part=None,
        team="t",
        agent="a",
        fidelity="full",
        key=(2.0, 1.0),
        value=None,
        judge="program",
        rubric_hash="rh",
        panel_hash="ph",
        built_on=["parent"],
        partial=True,
    )
    assert row.key == (2.0, 1.0)
    assert row.built_on == ["parent"]
    assert row.partial
    assert row.judge == "program"
    assert row.rubric_hash == "rh"
    assert row.panel_hash == "ph"
    field_name = "rank"
    with pytest.raises(ValidationError):
        setattr(row, field_name, 2)


def test_frontier_validation_is_secret_safe() -> None:
    secret = "SENTINEL-private-frontier-value"
    with pytest.raises(ValidationError) as exc:
        Frontier.model_validate({"marks": [{"epoch": secret}]})
    assert_secret_absent(exc.value, secret, sanitized=True)
    with pytest.raises(ValidationError) as exc:
        Frontier.model_validate_json('{"marks": ' + secret)
    assert_secret_absent(exc.value, secret, sanitized=True)
