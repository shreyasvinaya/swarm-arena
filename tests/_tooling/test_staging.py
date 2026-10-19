from typing import Any

import pytest
from _fakes.tools import PyTool, cand

from swarm_arena._errors import NotFound, SoftError
from swarm_arena._exec.executor import Job
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.catalog import CatalogEntry
from swarm_arena._model.goal import FileSet, Goal
from swarm_arena._model.ref import Ref
from swarm_arena._model.tooling import FileRef, ParentRef, RunStatus, ToolOutput, ToolRequest, Usage
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.result_file import request_json
from swarm_arena._tooling.staging import (
    edit_hashes,
    effective_files,
    fixed_config,
    parent_ref,
    plan_script_stage,
    plan_tool_stage,
    put_edits,
    tool_identity,
    validate_edits,
)

# tests/_tooling/test_staging.py
H1, H2, H3 = "1" * 64, "2" * 64, "3" * 64


def test_tool_identity_ignores_spelling() -> None:  # Review Focus 5
    toml_form = Ref(path="_fakes.tools:PyTool", config={"code": "x"})
    python_form = Ref.of(PyTool(PyTool.Config(code="x", grace_s=10.0)))
    assert tool_identity(toml_form, None) == tool_identity(python_form, None)
    assert tool_identity(
        Ref(path="_fakes.tools:PyTool", config={"code": "y"}), None
    ) != tool_identity(toml_form, None)


def test_tool_identity_drops_settable_fields() -> None:
    entry = CatalogEntry(
        name="py",
        kind="tool",
        ref="_fakes.tools:PyTool",
        description="d",
        settable={"timeout_s": {"type": "number"}},
    )
    a = Ref(path="_fakes.tools:PyTool", config={"code": "x", "timeout_s": 5})
    b = Ref(path="_fakes.tools:PyTool", config={"code": "x", "timeout_s": 50})
    assert tool_identity(a, entry) == tool_identity(b, entry)
    assert tool_identity(a, None) != tool_identity(b, None)
    assert "timeout_s" not in fixed_config(a, entry)


def stage(**kw: Any) -> tuple[Job.Stage, dict[str, str], list[str]]:
    base: dict[str, Any] = {
        "goal_files": {"p/run.py": H1, "p/a.txt": H2},
        "tool_files": {},
        "editable": ["p/*.txt"],
        "protected": {"p/run.py"},
        "private_paths": set(),
        "parents": [],
        "receiving_identity": "T",
        "edits": {},
        "request_json": None,
    }
    return plan_tool_stage(**{**base, **kw})


def test_order_goal_then_same_identity_overlay_then_edits() -> None:
    parent = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay={"p/a.txt": H3})
    st, overlay, warnings = stage(parents=[parent], edits={"p/b.txt": "new"})
    hb = sha256_hex(b"new")
    assert st.root == {"p/run.py": H1, "p/a.txt": H3, "p/b.txt": hb}
    assert overlay == {
        "p/a.txt": H3,
        "p/b.txt": hb,
    }
    assert warnings == []
    assert st.parents == [parent]


def test_other_tool_identity_is_not_overlaid() -> None:
    st, overlay, warnings = stage(
        parents=[
            ParentRef(candidate_id="c1", run_id="r", tool_identity="U", overlay={"p/a.txt": H3})
        ]
    )
    assert st.root["p/a.txt"] == H2
    assert overlay == {}
    assert warnings == ["base parent c1 not overlaid: made by another tool; paths: p/a.txt"]


def test_overlay_is_all_or_nothing() -> None:
    parent = ParentRef(
        candidate_id="c1", run_id="r", tool_identity="T", overlay={"p/a.txt": H3, "p/run.py": H3}
    )
    st, _, warnings = stage(parents=[parent])
    assert st.root == {"p/run.py": H1, "p/a.txt": H2}
    assert warnings == ["base parent c1 not overlaid: paths not editable here; paths: p/run.py"]


def test_empty_identity_and_later_parents_never_overlay() -> None:
    p0 = ParentRef(candidate_id="c0", run_id="r", tool_identity="", overlay={"p/a.txt": H3})
    p1 = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay={"p/a.txt": H3})
    assert stage(receiving_identity="", parents=[p0])[0].root["p/a.txt"] == H2
    st, _, _ = stage(parents=[ParentRef(candidate_id="c2", run_id="r", tool_identity="T"), p1])
    assert st.root["p/a.txt"] == H2
    assert len(st.parents) == 2


def test_private_path_in_root_is_internal_error() -> None:
    with pytest.raises(RuntimeError, match=r"judge/hidden/x\.csv"):
        stage(goal_files={"judge/hidden/x.csv": H1}, private_paths={"judge/hidden/x.csv"})


def test_validate_edits_lists_every_reason() -> None:
    ref = "sha256:" + "a" * 64  # 71 bytes, under the limit, so its own reason shows
    with pytest.raises(SoftError) as exc:
        validate_edits(
            {
                "p/run.py": "x",
                "q.txt": "y",
                "p/big.txt": "z" * 101,
                "swarm/x": "w",
                "p/ref.txt": ref,
            },
            editable=["p/*", "swarm/*"],
            protected={"p/run.py"},
            max_edit_bytes=100,
        )
    msg = str(exc.value)
    for part in (
        "p/run.py: protected",
        "q.txt: not editable",
        "p/big.txt: 101 bytes > max_edit_bytes 100",
        "swarm/x: uses the reserved name 'swarm'",
        "p/ref.txt: content is a stored-edit reference",
    ):
        assert part in msg


def test_validate_edits_rejects_alias_spellings() -> None:  # never normalized onto a protected path
    with pytest.raises(SoftError) as exc:
        validate_edits(
            {"./p/run.py": "x", "p//run.py": "y"},
            editable=["**/*"],
            protected={"p/run.py"},
            max_edit_bytes=10,
        )
    msg = str(exc.value)
    assert "./p/run.py: not in normal form (use 'p/run.py')" in msg
    assert "p//run.py: not in normal form (use 'p/run.py')" in msg


def test_live_edit_helpers_refuse_stored_reference(artifacts: ArtifactStore) -> None:
    ref = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match=r"p/a\.txt: content is a stored-edit reference"):
        edit_hashes({"p/a.txt": ref})
    with pytest.raises(ValueError, match=r"p/a\.txt: content is a stored-edit reference"):
        put_edits({"p/a.txt": ref, "p/b.txt": "ok"}, artifacts)
    with pytest.raises(NotFound):
        artifacts.get_bytes(sha256_hex(b"ok"))  # every value is checked before any store
    assert edit_hashes({"p/a.txt": "sha256:hello"}) == {"p/a.txt": sha256_hex(b"sha256:hello")}


def test_plan_script_stage_and_helpers() -> None:
    out_file = FileRef(name="o.txt", path="out/o.txt", sha256=H3, size=1)
    c = cand(
        request=ToolRequest(inputs={"lr": 0.1}),
        parent_ids=["c0"],
        overlay={"p/a.txt": H2},
        output=ToolOutput(status=RunStatus.OK, data={"f": 1.0}, files=[out_file]),
    )
    goal = Goal(
        id="g",
        namespace="n",
        files=FileSet(globs=["p/*"], manifest={"p/a.txt": H1, "p/run.py": H1}),
    )
    assert effective_files(goal, c) == {"p/a.txt": H2, "p/run.py": H1}
    st = plan_script_stage(
        own_files={"judge/score.py": H1}, candidate=c, effective_files=effective_files(goal, c)
    )
    assert st.root == {"judge/score.py": H1}
    assert st.request_json is None
    assert st.candidate is not None
    assert st.candidate.request == {**request_json(c.request), "parents": ["c0"]}
    assert st.candidate.output == {
        "schema_version": 1,
        "status": "ok",
        "partial": False,
        "data": {"f": 1.0},
        "usage": Usage().model_dump(mode="json"),
        "files": ["out/o.txt"],
    }
    assert st.candidate.files == {"p/a.txt": H2, "p/run.py": H1, "out/o.txt": H3}
    assert parent_ref(c) == ParentRef(
        candidate_id=c.id,
        run_id=c.run_id,
        tool_identity=c.tool_identity,
        data={"f": 1.0},
        overlay={"p/a.txt": H2},
        files=[out_file],
    )


@pytest.mark.parametrize("protected", [{"p/run.py"}, {"./p/run.py"}, {"p//run.py"}])
def test_protected_command_aliases_are_normalized(protected: set[str]) -> None:
    with pytest.raises(SoftError, match=r"p/run\.py: protected"):
        validate_edits({"p/run.py": "x"}, editable=["p/*"], protected=protected, max_edit_bytes=10)
    with pytest.raises(RuntimeError, match=r"p/run\.py"):
        stage(editable=["p/*"], protected=protected, edits={"p/run.py": "x"})
    parent = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay={"p/run.py": H3})
    st, overlay, warnings = stage(editable=["p/*"], protected=protected, parents=[parent])
    assert st.root["p/run.py"] == H1
    assert overlay == {}
    assert warnings == ["base parent c1 not overlaid: paths not editable here; paths: p/run.py"]


@pytest.mark.parametrize("path", ["p/run.py", "judge/private.txt", "other.txt"])
def test_stage_rechecks_edit_permissions(path: str) -> None:
    with pytest.raises(RuntimeError, match=path):
        stage(private_paths={"judge/private.txt"}, edits={path: "secret-content"})


def test_private_aliases_are_normalized() -> None:
    with pytest.raises(RuntimeError, match=r"judge/private\.txt"):
        stage(goal_files={"judge/private.txt": H1}, private_paths={"./judge//private.txt"})


def test_overlay_refusal_lists_sorted_protected_and_private_paths() -> None:
    parent = ParentRef(
        candidate_id="c1",
        run_id="r",
        tool_identity="T",
        overlay={"p/z.txt": H1, "p/a.txt": H2, "p/run.py": H3},
    )
    st, overlay, warnings = stage(parents=[parent], private_paths={"p/z.txt"})
    assert st.root["p/a.txt"] == H2
    assert not overlay
    assert warnings == [
        "base parent c1 not overlaid: paths not editable here; paths: p/run.py, p/z.txt"
    ]
    assert stage(receiving_identity="", parents=[parent])[2] == [
        "base parent c1 not overlaid: made by another tool; paths: p/a.txt, p/run.py, p/z.txt"
    ]


@pytest.mark.parametrize("path", ["./p/a.txt", "p//a.txt", "../outside", "/absolute", "swarm/x"])
def test_overlay_paths_are_checked_even_for_later_or_other_tool_parents(path: str) -> None:
    bad = ParentRef(candidate_id="bad", run_id="r", tool_identity="U").model_copy(
        update={"overlay": {path: H1}}
    )
    first = ParentRef(candidate_id="c1", run_id="r", tool_identity="T")
    with pytest.raises(ValueError, match="path"):
        stage(parents=[first, bad])


def test_alias_collisions_refused_before_hashing_or_storing(artifacts: ArtifactStore) -> None:
    edits = {"p/a.txt": "secret-content", "./p/a.txt": "other-content"}
    with pytest.raises(SoftError, match="not in normal form"):
        validate_edits(edits, editable=["p/*"], protected=set(), max_edit_bytes=100)
    with pytest.raises(ValueError, match="not in normal form"):
        stage(edits=edits)
    with pytest.raises(ValueError, match="not in normal form"):
        edit_hashes(edits)
    with pytest.raises(ValueError, match="not in normal form"):
        put_edits(edits, artifacts)
    with pytest.raises(NotFound):
        artifacts.get_bytes(sha256_hex(b"secret-content"))


def test_edit_validation_uses_utf8_bytes_and_does_not_echo_contents() -> None:
    secret = "secret-sentinel-0123456789"
    with pytest.raises(SoftError) as exc:
        validate_edits(
            {"p/a.txt": "éé", "p/b.txt": secret},
            editable=["p/*"],
            protected=set(),
            max_edit_bytes=3,
        )
    assert "p/a.txt: 4 bytes > max_edit_bytes 3" in str(exc.value)
    assert secret not in str(exc.value)


def test_fixed_config_sanitizes_validation_errors() -> None:
    import traceback

    secret = "secret-sentinel-0123456789"
    ref = Ref(path="_fakes.tools:PyTool", config={"code": "x", "timeout_s": secret})
    with pytest.raises(ValueError, match="timeout_s") as exc:
        fixed_config(ref, None)
    assert secret not in "".join(traceback.format_exception(exc.value))
    assert "timeout_s" in str(exc.value)


def test_goal_tool_overlay_edit_precedence_and_output_only_parent() -> None:
    output = FileRef(name="a", path="p/a.txt", sha256=H3, size=1)
    parent = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", files=[output])
    st, overlay, warnings = stage(
        tool_files={"tool.py": H3}, parents=[parent], edits={"p/a.txt": "edit"}
    )
    assert st.root == {"p/run.py": H1, "p/a.txt": sha256_hex(b"edit"), "tool.py": H3}
    assert overlay == {"p/a.txt": sha256_hex(b"edit")}
    assert warnings == []


def test_script_stage_output_wins_and_keeps_effective_and_own_files_separate() -> None:
    c = cand(
        output=ToolOutput(
            status=RunStatus.OK, files=[FileRef(name="a", path="p/a.txt", sha256=H3, size=1)]
        )
    )
    st = plan_script_stage(
        own_files={"judge/score.py": H1}, candidate=c, effective_files={"p/a.txt": H2}
    )
    assert st.candidate is not None
    assert st.candidate.files == {"p/a.txt": H3}
    assert st.root == {"judge/score.py": H1}
