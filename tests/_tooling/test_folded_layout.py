"""Round-three regressions for portable producer and sink layouts."""

from pathlib import Path
from traceback import format_exception

import pytest
from _fakes.tools import cand

from swarm_arena._errors import NotFound, SoftError
from swarm_arena._exec.executor import Job
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.goal import FileSet
from swarm_arena._model.tooling import FileRef, ParentRef, RunStatus, ToolOutput
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.files import snapshot_fileset
from swarm_arena._tooling.materialize import collect_output_files, materialize
from swarm_arena._tooling.staging import plan_script_stage, plan_tool_stage, validate_edits

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
def test_tool_wins_folded_goal_path(tmp_path: Path, artifacts: ArtifactStore, variant: str) -> None:
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
    assert stage.root == {variant: tool}
    materialize(stage, tmp_path / "ws", artifacts)
    assert (tmp_path / "ws" / variant).read_bytes() == b"TOOL"


@pytest.mark.parametrize("variant", ["P/A.txt", "p/A.txt", "p/cafe\u0301.txt"])
@pytest.mark.parametrize("target", ["parent", "candidate"])
def test_outputs_win_folded_overlay_path(
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
        dest = f"parents/c1/{variant}"
    else:
        stage = plan_script_stage(
            own_files={},
            effective_files={original: base},
            candidate=cand(output=ToolOutput(status=RunStatus.OK, files=[file])),
        )
        assert stage.candidate is not None
        assert stage.candidate.files == {variant: output}
        dest = f"candidate/files/{variant}"
    materialize(stage, tmp_path / "ws", artifacts)
    assert (tmp_path / "ws" / dest).read_bytes() == b"OUTPUT"


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
