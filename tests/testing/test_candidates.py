from pathlib import Path

import pytest
from _fakes.tools import PyTool

from swarm_arena import Goal, Ref, ScriptCheck, ScriptJudge, ScriptTool
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.ids import run_tag
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._tooling.staging import parent_ref, tool_identity
from swarm_arena.executor import LocalExecutor
from swarm_arena.testing import check_check, check_judge, check_tool, make_candidate
from swarm_arena.testing._conformance import _artifact_root
from swarm_arena.tool import FileSet, RunStatus, ToolOutput, ToolRequest


def test_defaults_and_unique_ids() -> None:
    a = make_candidate(ToolOutput(status=RunStatus.OK))
    b = make_candidate(ToolOutput(status=RunStatus.OK))
    assert (a.run_id, a.goal_id, a.team, a.agent, a.round, a.origin, a.tool_identity) == (
        "local",
        "check",
        "team",
        "agent",
        0,
        "agent",
        "",
    )
    assert a.id != b.id
    assert a.id.startswith(f"c-{run_tag('local')}-e0-")
    assert make_candidate(ToolOutput(status=RunStatus.OK), team="x").team == "x"


def test_overlay_from_edits_is_stored() -> None:
    c = make_candidate(ToolOutput(status=RunStatus.OK), request=ToolRequest(edits={"p/a.txt": "2"}))
    assert c.overlay == {"p/a.txt": sha256_hex(b"2")}
    assert ArtifactStore(open_store(_artifact_root())).get_bytes(sha256_hex(b"2")) == b"2"


def test_explicit_overlay_and_id_win() -> None:
    c = make_candidate(
        ToolOutput(status=RunStatus.OK),
        request=ToolRequest(edits={"p/a.txt": "2"}),
        overlay={},
        id="c-mine",
    )
    assert (c.id, c.overlay) == ("c-mine", {})


def test_make_candidate_refuses_stored_reference_edit() -> None:  # edit_hashes refuses it
    stored = ToolRequest(edits={"p/a.txt": "sha256:" + "a" * 64})
    with pytest.raises(ValueError, match="stored-edit reference"):
        make_candidate(ToolOutput(status=RunStatus.OK), request=stored)


def test_tool_sets_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    tool = PyTool(PyTool.Config(code="pass"))
    expected = tool_identity(Ref.of(tool), None)
    ok = ToolOutput(status=RunStatus.OK)
    assert make_candidate(ok, tool=tool).tool_identity == expected
    assert make_candidate(ok, tool=Ref.of(tool)).tool_identity == expected


async def test_overlay_chain_matches_tool_check_and_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_executor: LocalExecutor
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "problem").mkdir()
    for name in ("a", "b"):
        (tmp_path / f"problem/{name}.txt").write_text("original")
    (tmp_path / "run.py").write_text(
        "import json, pathlib\n"
        "data = {n: pathlib.Path(f'problem/{n}.txt').read_text() for n in ('a', 'b')}\n"
        "pathlib.Path('swarm/result.json').write_text(json.dumps({'data': data}))\n"
    )
    goal = Goal(id="chain", namespace="testing", files=FileSet(globs=["problem/*.txt"]))
    tool = ScriptTool(command=["python", "run.py"], files=["run.py"], editable=["problem/*.txt"])
    parent = make_candidate(
        ToolOutput(status=RunStatus.OK),
        tool=tool,
        goal=goal,
        request=ToolRequest(edits={"problem/a.txt": "first"}),
    )
    child = make_candidate(
        ToolOutput(status=RunStatus.OK),
        tool=tool,
        goal=goal,
        parents=[parent],
        request=ToolRequest(edits={"problem/b.txt": "second"}),
    )
    grandchild = make_candidate(
        ToolOutput(status=RunStatus.OK),
        tool=tool,
        goal=goal,
        parents=[child],
        request=ToolRequest(edits={"problem/b.txt": "third"}),
    )
    assert child.overlay == {
        "problem/a.txt": sha256_hex(b"first"),
        "problem/b.txt": sha256_hex(b"second"),
    }
    assert grandchild.overlay == {
        "problem/a.txt": sha256_hex(b"first"),
        "problem/b.txt": sha256_hex(b"third"),
    }
    assert child.parent_ids == [parent.id]
    assert grandchild.request.parents[0].candidate_id == child.id
    assert child.goal_id == grandchild.goal_id == goal.id
    out = await check_tool(tool, {}, goal=goal, parents=[grandchild], executor=local_executor)
    assert out.data == {"a": "first", "b": "third"}
    for candidate, expected in ((child, "second"), (grandchild, "third")):
        expression = (
            "Path('candidate/files/problem/a.txt').read_text() == 'first' and "
            f"Path('candidate/files/problem/b.txt').read_text() == '{expected}'"
        )
        code = (
            "from pathlib import Path; import json; "
            f"passed = {expression}; "
            "Path('swarm/result.json').write_text(json.dumps({'passed': passed}))"
        )
        result = await check_check(
            ScriptCheck(command=["python", "-c", code]),
            candidate,
            goal=goal,
            executor=local_executor,
        )
        assert result.passed
        code = code.replace("{'passed': passed}", "{'value': int(passed)}")
        [score] = await check_judge(
            ScriptJudge(command=["python", "-c", code]), goal, [candidate], executor=local_executor
        )
        assert score.key == (1.0,)


@pytest.mark.parametrize("refusal", ["identity", "editable", "protected", "layout", "missing-tool"])
def test_candidate_uses_core_base_overlay_refusal(
    refusal: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tool = ScriptTool(command=["python", "run.py"], editable=["problem/*.txt"])
    path = {"editable": "elsewhere/a.txt", "protected": "run.py", "layout": "Problem/a.txt"}.get(
        refusal, "problem/a.txt"
    )
    parent = make_candidate(
        ToolOutput(status=RunStatus.OK), overlay={path: sha256_hex(b"base")}, tool=tool
    )
    if refusal == "identity":
        parent = parent.model_copy(update={"tool_identity": "another-tool"})
    goal = Goal(
        id="check", namespace="testing", files=FileSet(manifest={"problem/b.txt": "1" * 64})
    )
    with pytest.warns(UserWarning, match="not overlaid"):
        child = make_candidate(
            ToolOutput(status=RunStatus.OK),
            tool=None if refusal == "missing-tool" else tool,
            goal=goal,
            parents=[parent],
            request=ToolRequest(edits={"problem/new.txt": "new"}),
        )
    assert child.overlay == {"problem/new.txt": sha256_hex(b"new")}
    assert child.parent_ids == [parent.id]


def test_request_parent_refs_chain_and_only_base_contributes() -> None:
    tool = ScriptTool(command=["python", "run.py"], editable=["problem/*.txt"])
    base = make_candidate(
        ToolOutput(status=RunStatus.OK),
        tool=tool,
        request=ToolRequest(edits={"problem/a.txt": "base", "problem/c.txt": "kept"}),
    )
    other = make_candidate(
        ToolOutput(status=RunStatus.OK),
        tool=tool,
        request=ToolRequest(edits={"problem/b.txt": "other"}),
    )
    request = ToolRequest(
        parents=[parent_ref(base), parent_ref(other)], edits={"problem/a.txt": "new"}
    )
    child = make_candidate(ToolOutput(status=RunStatus.OK), request=request, tool=tool)
    assert child.overlay == {
        "problem/a.txt": sha256_hex(b"new"),
        "problem/c.txt": sha256_hex(b"kept"),
    }
    assert child.parent_ids == [base.id, other.id]


def test_explicit_parents_must_match_request_parents() -> None:
    a = make_candidate(ToolOutput(status=RunStatus.OK))
    b = make_candidate(ToolOutput(status=RunStatus.OK))
    with pytest.raises(ValueError, match="parents"):
        make_candidate(
            ToolOutput(status=RunStatus.OK),
            parents=[a],
            request=ToolRequest(parents=[parent_ref(b)]),
        )
