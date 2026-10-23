"""Generated files retain catalog authority and standalone trace provenance."""

import json
from pathlib import Path

import pytest
from _fakes.plan_cli import call, executable_files
from _fakes.planning_llm import planner_output

from swarm_arena import Plan, Ref
from swarm_arena._journal.reader import read_committed
from swarm_arena._store.local import LocalStore


def test_llm_plan_file_keeps_origin_and_run_rejects_a_non_catalog_ref(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("agents")
    from swarm_arena.testing import FakeModel, FakeTurn

    files = executable_files(tmp_path)
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    canary = tmp_path / "imported"
    monkeypatch.setenv("SWARM_ARENA_TEST_CANARY", str(canary))
    assert call("plan", "--llm", "--model", "planner", files.run, files.goal) == 0
    out = files.output()
    assert 'origin = "llm"' in out.read_text()
    draft = Plan.load(out)
    agent = draft.teams[0].agents[0]
    draft.teams[0].agents[0] = agent.model_copy(
        update={"tool": Ref(path="_fakes.import_canary:Tool")}
    )
    out.write_text(draft.dumps())
    before = set(Path(files.config.run_root).iterdir())
    assert call("run", "--yes", files.run, out) == 1
    assert "catalog" in capsys.readouterr().err
    assert set(Path(files.config.run_root).iterdir()) == before
    assert not canary.exists()


def test_run_of_an_llm_plan_file_links_its_planning_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("agents")
    from swarm_arena.testing import FakeModel, FakeTurn

    files = executable_files(tmp_path)
    FakeModel([FakeTurn(text=json.dumps(planner_output()))]).register("planner")
    assert call("plan", "--llm", "--model", "planner", files.run, files.goal) == 0
    session_id = Plan.load(files.output()).planning_session
    assert session_id is not None
    assert session_id in capsys.readouterr().out
    session = read_committed(LocalStore(Path(files.config.run_root) / session_id))
    assert session is not None
    planner_call_id = next(e.payload["call"]["id"] for e in session[1] if e.type == "llm_call")
    assert call("run", "--yes", files.run, files.output()) == 0
    run_dir = next(p for p in Path(files.config.run_root).iterdir() if p.name != session_id)
    read = read_committed(LocalStore(run_dir))
    assert read is not None
    r0 = next(e for e in read[1] if e.type == "plan_revision")
    assert r0.payload["planning_session"] == session_id
    candidate_id = next(e.payload["candidate"]["id"] for e in read[1] if e.type == "tool_end")
    capsys.readouterr()
    assert call("trace", candidate_id, "--run", run_dir.name, "--config", files.run) == 0
    assert planner_call_id in capsys.readouterr().out
