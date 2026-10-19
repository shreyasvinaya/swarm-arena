import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Never

import pytest

from swarm_arena._errors import NotFound
from swarm_arena._exec.executor import Job
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.tooling import FileRef, ParentRef
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.materialize import collect_output_files, materialize


# tests/_tooling/test_materialize.py
def test_materialize_tool_stage(tmp_path: Path, artifacts: ArtifactStore) -> None:
    ha, hb, hc = artifacts.put_bytes(b"A"), artifacts.put_bytes(b"B"), artifacts.put_bytes(b"C")
    out = FileRef(name="o.txt", path="out/o.txt", sha256=ha, size=1)
    clash = FileRef(name="b.txt", path="p/b.txt", sha256=hc, size=1)  # also an overlay path
    parent = ParentRef(
        candidate_id="c1",
        run_id="r",
        tool_identity="T",
        data={"f": 1},
        overlay={"p/a.txt": hb, "p/b.txt": hb},
        files=[out, clash],
    )
    materialize(
        Job.Stage(root={"p/a.txt": ha}, parents=[parent], request_json={"params": {}}),
        tmp_path / "ws",
        artifacts,
    )
    ws = tmp_path / "ws"
    assert (ws / "p/a.txt").read_bytes() == b"A"
    assert (ws / "parents/c1/p/a.txt").read_bytes() == b"B"
    assert (
        ws / "parents/c1/out/o.txt"
    ).read_bytes() == b"A"  # the parent's output files are staged
    assert (ws / "parents/c1/p/b.txt").read_bytes() == b"C"  # an output file wins over the overlay
    assert (ws / "parents/c1/data.json").read_bytes() == b'{"f":1}'
    for rel in (
        "parents/c1/p/a.txt",
        "parents/c1/p/b.txt",
        "parents/c1/out/o.txt",
        "parents/c1/data.json",
    ):
        assert (ws / rel).stat().st_mode & 0o222 == 0
    assert (ws / "p/a.txt").stat().st_mode & 0o200
    assert json.loads((ws / "swarm/request.json").read_text()) == {"params": {}}


def test_materialize_script_stage(tmp_path: Path, artifacts: ArtifactStore) -> None:
    stage = Job.Stage(
        candidate=Job.CandidateFiles(request={"params": {}}, output={"status": "ok"}, files={})
    )
    ws = tmp_path / "ws"
    materialize(stage, ws, artifacts)
    assert json.loads((ws / "candidate/request.json").read_text()) == {"params": {}}
    assert json.loads((ws / "candidate/output.json").read_text()) == {"status": "ok"}
    assert (ws / "candidate/files").is_dir()
    assert list((ws / "candidate/files").iterdir()) == []
    assert (ws / "swarm").is_dir()


PARENT = ParentRef(candidate_id="c1", run_id="r", tool_identity="T")


@pytest.mark.parametrize("path", ["SWARM/result.json", "Parents/x", "Candidate/x", "\u017fwarm/x"])
def test_folded_reserved_root_rejected_before_write(
    tmp_path: Path, artifacts: ArtifactStore, path: str
) -> None:
    h = artifacts.put_bytes(b"planted")
    stage = Job.Stage.model_construct(root={"valid.txt": h, path: h})
    with pytest.raises(ValueError, match="reserved"):
        materialize(stage, tmp_path / "ws", artifacts)
    assert not (tmp_path / "ws").exists()


def test_folded_parent_ids_rejected_before_write(tmp_path: Path, artifacts: ArtifactStore) -> None:
    parents = [PARENT.model_copy(update={"candidate_id": name}) for name in ("c1", "C1")]
    with pytest.raises(ValueError, match="duplicate parent id"):
        materialize(Job.Stage(parents=parents), tmp_path / "ws", artifacts)
    assert not (tmp_path / "ws").exists()


@pytest.mark.parametrize("location", ["root", "parent", "candidate"])
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("p/run.py", "p/RUN.py"),
        ("caf\u00e9.txt", "cafe\u0301.txt"),
        ("p", "P/x"),
        ("p/a", "P/b"),
        ("caf\u00e9/a", "cafe\u0301/b"),
    ],
)
def test_folded_destination_collisions_fail_before_write(
    tmp_path: Path, artifacts: ArtifactStore, first: str, second: str, location: str
) -> None:
    h = artifacts.put_bytes(b"content")
    files = {first: h, second: h}
    stage = Job.Stage(root=files)
    if location == "parent":
        # Check collisions across overlay and output sources as well as within root/candidate.
        stage = Job.Stage(
            parents=[
                PARENT.model_copy(
                    update={
                        "overlay": {first: h},
                        "files": [FileRef(name="output", path=second, sha256=h, size=7)],
                    }
                )
            ]
        )
    elif location == "candidate":
        stage = Job.Stage(candidate=Job.CandidateFiles(request={}, output={}, files=files))
    with pytest.raises(ValueError, match="conflicting workspace path"):
        materialize(stage, tmp_path / "ws", artifacts)
    assert not (tmp_path / "ws").exists()


@pytest.mark.parametrize("path", ["DATA.json", "DATA.json/x"])
def test_folded_parent_metadata_collision_fails_before_write(
    tmp_path: Path, artifacts: ArtifactStore, path: str
) -> None:
    parent = PARENT.model_copy(update={"overlay": {path: artifacts.put_bytes(b"planted")}})
    with pytest.raises(ValueError, match="conflicting workspace path"):
        materialize(Job.Stage(parents=[parent]), tmp_path / "ws", artifacts)
    assert not (tmp_path / "ws").exists()


def test_distinct_destination_names_still_materialize(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    stage = Job.Stage(
        root={"p/a.txt": artifacts.put_bytes(b"a"), "p/b.txt": artifacts.put_bytes(b"b")}
    )
    materialize(stage, tmp_path / "ws", artifacts)
    assert (tmp_path / "ws/p/a.txt").read_bytes() == b"a"
    assert (tmp_path / "ws/p/b.txt").read_bytes() == b"b"


def test_materialize_reads_artifacts_without_scratch_fetch(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    digest = artifacts.put_bytes(b"content")

    def refuse_fetch(sha256: str, dest: Path, *, read_only: bool = True) -> Never:
        pytest.fail("materialize must read verified bytes directly")

    monkeypatch.setattr(artifacts, "fetch", refuse_fetch)
    materialize(Job.Stage(root={"a.txt": digest}), tmp_path / "ws", artifacts)
    assert (tmp_path / "ws/a.txt").read_bytes() == b"content"


@pytest.mark.parametrize(
    "make",
    [  # model_copy skips validators, so the sink checks again
        lambda h: Job.Stage(parents=[PARENT.model_copy(update={"candidate_id": "../../outside"})]),
        lambda h: Job.Stage(parents=[PARENT.model_copy(update={"overlay": {"/abs/x": h}})]),
        lambda h: Job.Stage(parents=[PARENT.model_copy(update={"overlay": {"../x": h}})]),
        lambda h: Job.Stage().model_copy(update={"root": {"../x": h}}),
        lambda h: Job.Stage(
            candidate=Job.CandidateFiles(request={}, output={}, files={}).model_copy(
                update={"files": {"../x": h}}
            )
        ),
    ],
)
def test_materialize_rechecks_paths_at_the_sink(
    tmp_path: Path, artifacts: ArtifactStore, make: Callable[[str], Job.Stage]
) -> None:
    ha = artifacts.put_bytes(b"A")
    with pytest.raises(ValueError, match=r"path|parent id"):
        materialize(make(ha), tmp_path / "ws", artifacts)
    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize(
    ("declared", "match"),
    [
        ("../x", "path"),
        ("swarm/result.json", "reserved"),
        ("missing.txt", "missing.txt"),
        ("link.txt", "symlink"),
        ("fifo.txt", "regular file"),
    ],
)
def test_collect_output_files_rejects(
    tmp_path: Path, artifacts: ArtifactStore, declared: str, match: str
) -> None:
    (tmp_path / "link.txt").symlink_to(tmp_path / "target")
    (tmp_path / "target").write_text("t")
    os.mkfifo(tmp_path / "fifo.txt")
    with pytest.raises(ValueError, match=match):
        collect_output_files(tmp_path, [declared], artifacts)


def test_collect_output_files_uploads(tmp_path: Path, artifacts: ArtifactStore) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out/p.txt").write_bytes(b"p")
    [ref] = collect_output_files(tmp_path, ["out/p.txt"], artifacts)
    assert (ref.name, ref.path, ref.size) == ("p.txt", "out/p.txt", 1)
    assert artifacts.get_bytes(ref.sha256) == b"p"


@pytest.mark.parametrize(
    "bad", ["../x", "/abs", "./p/a.txt", "p//a.txt", "swarm/x", "parents/x", "candidate/x", "p/\0x"]
)
@pytest.mark.parametrize("location", ["root", "overlay", "output", "candidate"])
def test_all_materialized_paths_checked_before_any_write(
    tmp_path: Path, artifacts: ArtifactStore, bad: str, location: str
) -> None:
    h = artifacts.put_bytes(b"A")
    stage = Job.Stage(root={"valid.txt": h})
    if location == "root":
        stage = stage.model_copy(update={"root": {"valid.txt": h, bad: h}})
    elif location == "candidate":
        stage = stage.model_copy(
            update={
                "candidate": Job.CandidateFiles.model_construct(
                    request={}, output={}, files={bad: h}
                )
            }
        )
    else:
        fields = (
            {"overlay": {bad: h}}
            if location == "overlay"
            else {"files": [FileRef.model_construct(name="bad", path=bad, sha256=h, size=1)]}
        )
        stage = stage.model_copy(update={"parents": [PARENT.model_copy(update=fields)]})
    ws = tmp_path / "ws"
    with pytest.raises(ValueError, match="path"):
        materialize(stage, ws, artifacts)
    assert not ws.exists()


@pytest.mark.parametrize("parent_id", ["", ".", "..", "a/b", "a\\b", "/x", "x\0", "../../outside"])
def test_parent_ids_rechecked_before_any_write(
    tmp_path: Path, artifacts: ArtifactStore, parent_id: str
) -> None:
    parent = PARENT.model_copy(update={"candidate_id": parent_id})
    ws = tmp_path / "ws"
    with pytest.raises(ValueError, match="parent id"):
        materialize(Job.Stage(parents=[parent]), ws, artifacts)
    assert not ws.exists()


@pytest.mark.parametrize(
    "rel", ["p", "p/a.txt", "swarm", "parents", "parents/c1", "candidate", "candidate/files"]
)
def test_materialize_rejects_existing_symlink_components(
    tmp_path: Path, artifacts: ArtifactStore, rel: str
) -> None:
    h = artifacts.put_bytes(b"A")
    outside = tmp_path / "outside"
    outside.mkdir()
    ws = tmp_path / "ws"
    link = ws / rel
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    stage = Job.Stage(
        root={"p/a.txt": h},
        parents=[PARENT],
        request_json={},
        candidate=Job.CandidateFiles(request={}, output={}, files={"a.txt": h}),
    )
    with pytest.raises(ValueError, match="symlink"):
        materialize(stage, ws, artifacts)
    assert list(outside.iterdir()) == []
    assert not (ws / "p/a.txt").is_file()


def test_materialize_rejects_workspace_symlink(tmp_path: Path, artifacts: ArtifactStore) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    ws = tmp_path / "ws"
    ws.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        materialize(Job.Stage(), ws, artifacts)
    assert list(outside.iterdir()) == []


def test_materialize_rechecks_symlinks_after_preflight(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = artifacts.put_bytes(b"new")
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    get_bytes = artifacts.get_bytes

    def swap(digest: str) -> bytes:
        data = get_bytes(digest)
        (ws / "p").symlink_to(outside)
        return data

    monkeypatch.setattr(artifacts, "get_bytes", swap)
    with pytest.raises(ValueError, match="workspace path"):
        materialize(Job.Stage(root={"p/a.txt": h}), ws, artifacts)
    assert list(outside.iterdir()) == []


def test_materialize_does_not_truncate_existing_hardlink(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"old")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").hardlink_to(outside)
    materialize(Job.Stage(root={"a.txt": artifacts.put_bytes(b"new")}), ws, artifacts)
    assert outside.read_bytes() == b"old"
    assert (ws / "a.txt").read_bytes() == b"new"


def test_parent_data_json_wins_without_fetching_shadowed_artifact(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    parent = PARENT.model_copy(update={"overlay": {"data.json": "0" * 64}, "data": {"x": 1}})
    materialize(Job.Stage(parents=[parent]), tmp_path / "ws", artifacts)
    assert (tmp_path / "ws/parents/c1/data.json").read_bytes() == b'{"x":1}'


@pytest.mark.parametrize("files", [{"a": "h", "a/b": "h"}, {"data.json/x": "h"}])
def test_layout_conflicts_fail_before_writes(
    tmp_path: Path, artifacts: ArtifactStore, files: dict[str, str]
) -> None:
    parent = PARENT.model_copy(update={"overlay": files})
    ws = tmp_path / "ws"
    with pytest.raises(ValueError, match="conflicting workspace path"):
        materialize(Job.Stage(parents=[parent]), ws, artifacts)
    assert not ws.exists()


def test_candidate_files_are_read_only(tmp_path: Path, artifacts: ArtifactStore) -> None:
    h = artifacts.put_bytes(b"candidate")
    stage = Job.Stage(candidate=Job.CandidateFiles(request={}, output={}, files={"p/a": h}))
    materialize(stage, tmp_path / "ws", artifacts)
    path = tmp_path / "ws/candidate/files/p/a"
    assert path.read_bytes() == b"candidate"
    assert path.stat().st_mode & 0o222 == 0


def test_collection_rejects_symlinked_parent(tmp_path: Path, artifacts: ArtifactStore) -> None:
    (tmp_path / "target").mkdir()
    (tmp_path / "target/a").write_text("secret-content")
    (tmp_path / "p").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="symlink not allowed: p/a"):
        collect_output_files(tmp_path, ["p/a"], artifacts)


def test_collection_normalizes_and_refuses_collisions(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    (tmp_path / "a.txt").write_bytes(b"A")
    with pytest.raises(ValueError, match="duplicate output path"):
        collect_output_files(tmp_path, ["a.txt", "./a.txt"], artifacts)
    [ref] = collect_output_files(tmp_path, ["./a.txt"], artifacts)
    assert ref.path == "a.txt"


def test_metadata_errors_do_not_disclose_values(tmp_path: Path, artifacts: ArtifactStore) -> None:
    import traceback

    secret = "secret-sentinel-0123456789"

    class BadDict(dict[str, object]):
        def items(self) -> Never:
            raise ValueError(secret)

    parent = PARENT.model_copy(update={"data": BadDict(x=1)})
    with pytest.raises(ValueError, match="cannot serialize workspace path") as exc:
        materialize(Job.Stage(parents=[parent]), tmp_path / "ws", artifacts)
    assert secret not in "".join(traceback.format_exception(exc.value))
    assert not (tmp_path / "ws").exists()


def test_collection_storage_errors_suppress_secret_exception_chains(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traceback import format_exception

    secret = "storage-secret-sentinel"
    (tmp_path / "a.txt").write_text(secret)

    def fail(data: bytes) -> Never:
        raise ValueError(secret)

    monkeypatch.setattr(artifacts, "put_bytes", fail)
    with pytest.raises(ValueError, match=r"cannot store output path: a\.txt") as exc:
        collect_output_files(tmp_path, ["a.txt"], artifacts)
    assert secret not in "".join(format_exception(exc.value))


def test_materialize_write_errors_suppress_secret_exception_chains(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from traceback import format_exception
    from typing import Any

    secret = "write-secret-sentinel"
    digest = artifacts.put_bytes(secret.encode())

    def fail(*args: Any, **kwargs: Any) -> Never:
        raise OSError(secret)

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(ValueError, match=r"cannot write workspace path: a\.txt") as exc:
        materialize(Job.Stage(root={"a.txt": digest}), tmp_path / "ws", artifacts)
    assert secret not in "".join(format_exception(exc.value))
    assert list((tmp_path / "ws").iterdir()) == [tmp_path / "ws/swarm"]


@pytest.mark.parametrize(
    ("first", "second"),
    [("out.txt", "OUT.txt"), ("caf\u00e9.txt", "cafe\u0301.txt")],
)
def test_collection_refuses_folded_output_duplicates_before_upload(
    tmp_path: Path, artifacts: ArtifactStore, first: str, second: str
) -> None:
    content = b"secret-output-sentinel"
    (tmp_path / first).write_bytes(content)
    (tmp_path / second).write_bytes(content)
    with pytest.raises(ValueError, match="duplicate output path") as exc:
        collect_output_files(tmp_path, [first, second], artifacts)
    assert first in str(exc.value)
    assert second in str(exc.value)
    assert content.decode() not in str(exc.value)
    with pytest.raises(NotFound):
        artifacts.get_bytes(sha256_hex(content))
