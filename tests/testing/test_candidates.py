from pathlib import Path

import pytest
from _fakes.tools import PyTool

from swarm_arena import Ref
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.ids import run_tag
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store
from swarm_arena._tooling.staging import tool_identity
from swarm_arena.testing import make_candidate
from swarm_arena.testing._conformance import _artifact_root
from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest


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
