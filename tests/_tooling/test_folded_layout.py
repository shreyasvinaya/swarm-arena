"""Regressions for portable producer and sink layouts.

Each producer-side check (edits, base overlays, outputs, snapshots) must agree
with the materialize sink, so a bad path fails the run that made it.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from traceback import format_exception
from typing import Any, Never

import pytest
from _fakes.tools import cand

from swarm_arena import ScriptTool
from swarm_arena._errors import NotFound, SoftError
from swarm_arena._exec.executor import Job
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.goal import FileSet, Goal
from swarm_arena._model.tooling import CompletedRun, FileRef, ParentRef, RunStatus, ToolOutput
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.files import snapshot_fileset
from swarm_arena._tooling.materialize import collect_output_files, materialize
from swarm_arena._tooling.result_file import tool_output_from_result
from swarm_arena._tooling.staging import (
    effective_files,
    parent_ref,
    plan_script_stage,
    plan_tool_stage,
    put_edits,
    validate_edits,
)

H = "1" * 64
SECRET = "round-three-secret-contents"
EDIT_CASES = [
    ({"p/sub/a.txt"}, {"p/SUB/b.txt": SECRET}),
    (set(), {"p/sub/a.txt": SECRET, "p/SUB/b.txt": SECRET}),
    ({"p/caf\u00e9/a.txt"}, {"p/cafe\u0301/b.txt": SECRET}),
    ({"p/sub/a.txt"}, {"p/SUB": SECRET}),
    ({"problem/run.py"}, {"Problem/notes.txt": SECRET}),
    (set(), {"q/a.txt": SECRET, "Q/b.txt": SECRET}),
    ({"caf\u00e9/a.txt"}, {"cafe\u0301/b.txt": SECRET}),
    ({"p/a.txt"}, {"P/b.txt": SECRET}),
    (set(), {"P/x.txt": SECRET, "p/y.txt": SECRET}),
    ({"p/run.py"}, {"P": SECRET}),
    ({"p/a.txt"}, {"p": SECRET}),
    ({"p"}, {"p/a.txt": SECRET}),
]


@pytest.mark.parametrize(("root", "edits"), EDIT_CASES)
@pytest.mark.parametrize("operation", ["validate", "plan"])
def test_edit_layout_soft_refusal(root: set[str], edits: dict[str, str], operation: str) -> None:
    def invoke() -> None:
        if operation == "validate":
            validate_edits(
                edits, root=root, editable=["**/*"], protected=set(), max_edit_bytes=1000
            )
        else:
            plan_tool_stage(
                goal_files=dict.fromkeys(root, H),
                tool_files={},
                editable=["**/*"],
                protected=set(),
                private_paths=set(),
                parents=[],
                receiving_identity="T",
                edits=edits,
                request_json=None,
            )

    with pytest.raises(SoftError, match="folded path collision") as exc:
        invoke()
    for path in root | edits.keys():
        assert path in str(exc.value)
    assert SECRET not in "".join(format_exception(exc.value))


@pytest.mark.parametrize("variant", ["p/RUN.py", "P/RUN.py", "p/cafe\u0301.py"])
def test_tool_content_wins_at_the_goal_spelling(
    tmp_path: Path, artifacts: ArtifactStore, variant: str
) -> None:
    original = "p/caf\u00e9.py" if "cafe" in variant else "p/run.py"
    goal, tool = artifacts.put_bytes(b"GOAL"), artifacts.put_bytes(b"TOOL")
    stage, _, _ = plan_tool_stage(
        goal_files={original: goal},
        tool_files={variant: tool},
        editable=[],
        protected=set(),
        private_paths=set(),
        parents=[],
        receiving_identity="T",
        edits={},
        request_json=None,
    )
    # One spelling rule: the tool's content at the goal's spelling, as later stages see it.
    assert stage.root == {original: tool}
    materialize(stage, tmp_path / "ws", artifacts)
    assert [path.name for path in (tmp_path / "ws/p").iterdir()] == [original.split("/")[1]]
    assert (tmp_path / "ws" / original).read_bytes() == b"TOOL"


@pytest.mark.parametrize("variant", ["P/A.txt", "p/A.txt", "p/cafe\u0301.txt"])
@pytest.mark.parametrize("target", ["parent", "candidate"])
def test_output_content_wins_at_the_overlay_spelling(
    tmp_path: Path, artifacts: ArtifactStore, variant: str, target: str
) -> None:
    original = "p/caf\u00e9.txt" if "cafe" in variant else "p/a.txt"
    base, output = artifacts.put_bytes(b"BASE"), artifacts.put_bytes(b"OUTPUT")
    file = FileRef(name="a", path=variant, sha256=output, size=6)
    if target == "parent":
        stage = Job.Stage(
            parents=[
                ParentRef(
                    candidate_id="c1",
                    run_id="r",
                    tool_identity="T",
                    overlay={original: base},
                    files=[file],
                )
            ]
        )
        dest = f"parents/c1/{original}"
    else:
        stage = plan_script_stage(
            own_files={},
            effective_files={original: base},
            candidate=cand(output=ToolOutput(status=RunStatus.OK, files=[file])),
        )
        assert stage.candidate is not None
        assert stage.candidate.files == {original: output}
        dest = f"candidate/files/{original}"
    materialize(stage, tmp_path / "ws", artifacts)
    assert (tmp_path / "ws" / dest).read_bytes() == b"OUTPUT"
    names = [path.name for path in (tmp_path / "ws" / dest).parent.iterdir()]
    assert names == [dest.rsplit("/", 1)[1]]


@pytest.mark.parametrize("variant", ["DATA.json", "Data.json"])
@pytest.mark.parametrize("source", ["overlay", "output"])
def test_generated_parent_data_wins_folded_alias(
    tmp_path: Path, artifacts: ArtifactStore, variant: str, source: str
) -> None:
    digest = artifacts.put_bytes(SECRET.encode())
    parent = ParentRef(
        candidate_id="c1",
        run_id="r",
        tool_identity="T",
        data={"f": 1},
        overlay={variant: digest} if source == "overlay" else {},
        files=[FileRef(name=variant, path=variant, sha256=digest, size=len(SECRET))]
        if source == "output"
        else [],
    )
    materialize(Job.Stage(parents=[parent]), tmp_path / "ws", artifacts)
    assert (tmp_path / "ws/parents/c1/data.json").read_bytes() == b'{"f":1}'
    assert [p.name for p in (tmp_path / "ws/parents/c1").iterdir()] == ["data.json"]


@pytest.mark.parametrize(
    ("first", "second"),
    [("out/a.txt", "OUT/b.txt"), ("caf\u00e9/a.txt", "cafe\u0301/b.txt")],
)
def test_output_ancestor_conflict_before_upload(
    tmp_path: Path, artifacts: ArtifactStore, first: str, second: str
) -> None:
    for rel in [first, second]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SECRET)
    with pytest.raises(ValueError, match="output path") as exc:
        collect_output_files(tmp_path, [first, second], artifacts, stage=Job.Stage())
    assert first in str(exc.value)
    assert second in str(exc.value)
    assert SECRET not in "".join(format_exception(exc.value))
    with pytest.raises(NotFound):
        artifacts.get_bytes(sha256_hex(SECRET.encode()))


@pytest.mark.parametrize("globs", [["problem/*", "problem/RUN.py"], ["problem/RUN.py"]])
def test_snapshot_keeps_directory_listing_spelling(
    tmp_path: Path, artifacts: ArtifactStore, globs: list[str]
) -> None:
    (tmp_path / "problem").mkdir()
    (tmp_path / "problem/run.py").write_text(SECRET)
    if not (tmp_path / "problem/RUN.py").exists():
        pytest.skip("native case-insensitive filesystem repro")
    if not list(tmp_path.glob("problem/RUN.py")):
        # Python 3.12 globs literal segments case-sensitively, so the alias
        # matches nothing and the snapshot fails closed instead.
        pytest.skip("this Python's glob does not resolve a literal alias")
    snapshot = snapshot_fileset(FileSet(globs=globs), tmp_path, artifacts)
    assert snapshot.manifest == {"problem/run.py": sha256_hex(SECRET.encode())}


@pytest.mark.parametrize(
    ("root", "declared"),
    [
        ("problem/run.py", "Problem/out.txt"),
        ("p/run.py", "P/RUN.py"),
        ("p/a.txt", "P/A.txt"),
        ("caf\u00e9/a.txt", "cafe\u0301/b.txt"),
        ("p/a.txt", "P"),
        ("p", "p/a.txt"),
    ],
)
def test_collection_checks_original_stage_before_reading_outputs(
    tmp_path: Path, artifacts: ArtifactStore, root: str, declared: str
) -> None:
    # No output exists: collision validation must precede reads and uploads.
    stage = Job.Stage(root={root: H})
    with pytest.raises(ValueError, match="conflicting output path") as exc:
        collect_output_files(tmp_path, [declared], artifacts, stage=stage)
    assert root in str(exc.value)
    assert declared in str(exc.value)


def test_collection_allows_exact_staged_replacement(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    stage = Job.Stage(root={"out.txt": H})
    (tmp_path / "out.txt").write_text(SECRET)
    [ref] = collect_output_files(tmp_path, ["out.txt"], artifacts, stage=stage)
    assert ref.path == "out.txt"
    assert artifacts.get_bytes(ref.sha256) == SECRET.encode()


@pytest.mark.parametrize("kind", ["parent", "candidate"])
def test_collection_checks_full_stage_layout(
    tmp_path: Path, artifacts: ArtifactStore, kind: str
) -> None:
    # Invalid bypassed models prove that the collector includes the read-only
    # trees as well as root. Ordinary stages have already passed the same sink.
    bad = {"p/a.txt": H, "P/b.txt": H}
    stage = Job.Stage(
        parents=[ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay=bad)]
        if kind == "parent"
        else [],
        candidate=Job.CandidateFiles(request={}, output={}, files=bad)
        if kind == "candidate"
        else None,
    )
    with pytest.raises(ValueError, match=r"conflicting.*path") as exc:
        collect_output_files(tmp_path, [], artifacts, stage=stage)
    assert "p/a.txt" in str(exc.value)
    assert "P/b.txt" in str(exc.value)


# --- Canonical caseless keys (G1) ---

SCRIPT = "problem/\N{LATIN SMALL LETTER S WITH CARON}olver.py"
SCRIPT_ALIAS = "problem/\N{LATIN SMALL LETTER LONG S}\N{COMBINING CARON}olver.py"
NOTE = "notes/\N{LATIN SMALL LETTER S WITH ACUTE}.txt"
NOTE_ALIAS = "notes/\N{LATIN SMALL LETTER LONG S}\N{COMBINING ACUTE ACCENT}.txt"


def exactly(message: str) -> str:
    return f"^{re.escape(message)}$"


def _tool_stage(**kw: Any) -> tuple[Job.Stage, dict[str, str], list[str]]:
    base: dict[str, Any] = {
        "goal_files": {},
        "tool_files": {},
        "editable": ["**/*"],
        "protected": set(),
        "private_paths": set(),
        "parents": [],
        "receiving_identity": "T",
        "edits": {},
        "request_json": None,
    }
    return plan_tool_stage(**{**base, **kw})


def test_canonical_caseless_alias_cannot_replace_protected_script(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    tool = ScriptTool(command=["python", SCRIPT], editable=["problem/*"])
    goal = {SCRIPT: artifacts.put_bytes(b"HUMAN")}
    protected = tool._protected(set(goal))
    edits = {SCRIPT_ALIAS: SECRET}
    with pytest.raises(SoftError) as exc:
        validate_edits(
            edits, root=set(goal), editable=tool.editable(), protected=protected, max_edit_bytes=100
        )
    assert str(exc.value) == f"{SCRIPT_ALIAS}: protected"
    with pytest.raises(RuntimeError, match="edit path not editable here"):
        _tool_stage(goal_files=goal, editable=tool.editable(), protected=protected, edits=edits)
    stage = Job.Stage.model_construct(root={**goal, SCRIPT_ALIAS: artifacts.put_bytes(b"AGENT")})
    with pytest.raises(ValueError, match="conflicting workspace path"):
        materialize(stage, tmp_path / "ws", artifacts)
    assert not (tmp_path / "ws").exists()


def test_canonical_caseless_edits_collide() -> None:
    edits = {NOTE: SECRET, NOTE_ALIAS: SECRET}
    with pytest.raises(SoftError, match="folded path collision") as exc:
        validate_edits(edits, editable=["**/*"], protected=set(), max_edit_bytes=100)
    assert NOTE in str(exc.value)
    assert NOTE_ALIAS in str(exc.value)
    assert SECRET not in "".join(format_exception(exc.value))
    with pytest.raises(SoftError, match="folded path collision"):
        _tool_stage(edits=edits)


# --- One spelling rule from the producing root to later stages (G2) ---


def test_tool_spelling_cannot_pass_the_producing_run(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    goal_files = {"p/run.py": artifacts.put_bytes(b"GOAL")}
    tool_files = {"P/RUN.py": artifacts.put_bytes(b"TOOL")}
    goal = Goal(id="g", namespace="n", files=FileSet(globs=["p/*"], manifest=goal_files))
    common: dict[str, Any] = {
        "goal_files": goal_files,
        "tool_files": tool_files,
        "protected": {"P/RUN.py"},
    }
    produced, _, _ = _tool_stage(**common)
    assert produced.root == {"p/run.py": tool_files["P/RUN.py"]}

    alias = {"P/notes.txt": SECRET}
    with pytest.raises(SoftError, match=r"^P/notes\.txt: folded path collision with p/run\.py$"):
        validate_edits(
            alias, root=set(produced.root), editable=["**/*"], protected=set(), max_edit_bytes=100
        )
    with pytest.raises(SoftError, match=r"^P/notes\.txt: folded path collision with p/run\.py$"):
        _tool_stage(**common, edits=alias)
    with pytest.raises(ValueError, match="conflicting output path") as exc:
        collect_output_files(tmp_path, ["P/out.txt"], artifacts, stage=produced)
    assert "p/run.py" in str(exc.value)

    # The shared spelling passes the producing run, and every later stage accepts it.
    edits = {"p/notes.txt": "agent text"}
    validate_edits(
        edits, root=set(produced.root), editable=["**/*"], protected=set(), max_edit_bytes=100
    )
    put_edits(edits, artifacts)
    stage, overlay, _ = _tool_stage(**common, edits=edits)
    ws = tmp_path / "producing"
    materialize(stage, ws, artifacts)
    (ws / "p/out.txt").write_bytes(b"OUT")
    files = collect_output_files(ws, ["p/out.txt"], artifacts, stage=stage)
    candidate = cand(
        tool_identity="T", overlay=overlay, output=ToolOutput(status=RunStatus.OK, files=files)
    )
    check = plan_script_stage(
        own_files={}, candidate=candidate, effective_files=effective_files(goal, candidate)
    )
    materialize(check, tmp_path / "check", artifacts)
    assert (tmp_path / "check/candidate/files/p/notes.txt").read_bytes() == b"agent text"
    later, _, warnings = _tool_stage(**common, parents=[parent_ref(candidate)])
    assert warnings == []
    materialize(later, tmp_path / "later", artifacts)
    assert (tmp_path / "later/parents/c1/p/out.txt").read_bytes() == b"OUT"


# --- data.json is reserved at the top of a parent tree (G3) ---


@pytest.mark.parametrize("path", ["data.json/x", "Data.json/x", "DATA.json/notes.txt"])
def test_parent_data_directory_edit_is_soft_refused(path: str) -> None:
    with pytest.raises(SoftError) as exc:
        validate_edits({path: SECRET}, editable=["**/*"], protected=set(), max_edit_bytes=100)
    assert str(exc.value) == f"{path}: folded path collision with parent data.json"
    assert SECRET not in "".join(format_exception(exc.value))
    with pytest.raises(SoftError) as exc:
        _tool_stage(edits={path: SECRET})
    assert str(exc.value) == f"{path}: folded path collision with parent data.json"


def test_parent_data_file_edit_is_skipped_not_refused(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    edits = {"Data.json": "agent text", "p/data.json/x": "agent text"}
    validate_edits(edits, editable=["**/*"], protected=set(), max_edit_bytes=100)
    put_edits(edits, artifacts)
    _, overlay, _ = _tool_stage(edits=edits)
    parent = parent_ref(cand(overlay=overlay, data={"f": 1}))
    materialize(Job.Stage(parents=[parent]), tmp_path / "ws", artifacts)
    assert (tmp_path / "ws/parents/c1/data.json").read_bytes() == b'{"f":1}'
    assert (tmp_path / "ws/parents/c1/p/data.json/x").read_bytes() == b"agent text"


@pytest.mark.parametrize("path", ["data.json/out.txt", "DATA.json/notes.txt"])
def test_parent_data_directory_output_fails_producing_run(
    tmp_path: Path, artifacts: ArtifactStore, path: str
) -> None:
    (tmp_path / path).parent.mkdir(parents=True)
    (tmp_path / path).write_text(SECRET)
    with pytest.raises(ValueError, match=r"with parent data\.json") as exc:
        collect_output_files(tmp_path, [path], artifacts, stage=Job.Stage())
    assert str(exc.value) == f"conflicting output path with parent data.json: {path}"
    with pytest.raises(NotFound):
        artifacts.get_bytes(sha256_hex(SECRET.encode()))

    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/result.json").write_text(json.dumps({"data": {}, "files": [path]}))
    (tmp_path / "swarm/stdout.log").touch()
    (tmp_path / "swarm/stderr.log").touch()
    when = datetime(2026, 10, 20, tzinfo=UTC)
    completed = CompletedRun(
        returncode=0,
        status=RunStatus.OK,
        stdout_path=tmp_path / "swarm/stdout.log",
        stderr_path=tmp_path / "swarm/stderr.log",
        started_at=when,
        ended_at=when,
    )
    assert tool_output_from_result(tmp_path, completed).status == RunStatus.ERROR
    assert path in completed.stderr_path.read_text()


# --- A base overlay that does not fit the root is a soft warning (G4) ---


@pytest.mark.parametrize(
    "overlay",
    [
        {"problem/notes.txt": H},
        {"Problem/RUN.PY/x": H},
        {"problem": H},
        {"PROBLEM/RUN.py": H},
        {"Data.json/x": H},
    ],
)
def test_base_overlay_layout_conflict_is_a_soft_warning(overlay: dict[str, str]) -> None:
    root = {"Problem/run.py": "2" * 64}
    parent = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay=overlay)
    stage, effective, warnings = _tool_stage(goal_files=root, parents=[parent])
    assert (stage.root, effective) == (root, {})
    assert warnings == [
        "base parent c1 not overlaid: paths conflict with the workspace layout; "
        f"paths: {', '.join(sorted(overlay))}"
    ]


# --- Merge and layout errors keep the tree context (G7) ---


def test_merge_errors_name_the_parent_or_tree(tmp_path: Path, artifacts: ArtifactStore) -> None:
    good = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay={"p/a.txt": H})
    bad = good.model_copy(update={"candidate_id": "c2", "overlay": {"p/a.txt": H, "P/b.txt": H}})
    parents = exactly("conflicting workspace path: parents/c2/p/a.txt and parents/c2/P/b.txt")
    with pytest.raises(ValueError, match=parents):
        materialize(Job.Stage(parents=[good, bad]), tmp_path / "ws", artifacts)
    with pytest.raises(ValueError, match=parents):
        collect_output_files(tmp_path, [], artifacts, stage=Job.Stage(parents=[good, bad]))

    files = exactly(
        "conflicting workspace path: candidate/files/p/a.txt and candidate/files/P/b.txt"
    )
    with pytest.raises(ValueError, match=files):
        plan_script_stage(
            own_files={}, candidate=cand(), effective_files={"p/a.txt": H, "P/b.txt": H}
        )
    goal = Goal(id="g", namespace="n", files=FileSet(globs=["p/*"], manifest={"p/a.txt": H}))
    with pytest.raises(ValueError, match=files):
        effective_files(goal, cand(overlay={"P/b.txt": H}))


# --- The generated layout is computed once (G8) ---


def test_collection_reuses_the_stage_layout_without_serializing(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import swarm_arena._tooling.materialize as materialize_module

    def fail(*args: Any, **kwargs: Any) -> Never:
        raise AssertionError("collection serialized metadata")

    stage = Job.Stage(
        parents=[ParentRef(candidate_id="c1", run_id="r", tool_identity="T", data={"f": 1})],
        candidate=Job.CandidateFiles(request={}, output={}, files={}),
        request_json={"params": {}},
    )
    materialize(stage, tmp_path, artifacts)
    monkeypatch.setattr(materialize_module, "_json_bytes", fail)
    (tmp_path / "out.txt").write_text("x")
    [ref] = collect_output_files(tmp_path, ["out.txt"], artifacts, stage=stage)
    assert ref.path == "out.txt"
