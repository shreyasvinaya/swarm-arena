import os
import sys
import types
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import FilesWidget, Widget
from pydantic import (
    AliasChoices,
    AliasPath,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_serializer,
)
from pydantic_core import PydanticCustomError

from swarm_arena._errors import ConfigError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Plan
from swarm_arena._model.ref import Configurable, Ref, RefField, _ConfigInputModel
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.files import snapshot_fileset, snapshot_plan, snapshot_ref


def _write(base: Path, rel: str, data: bytes = b"contents") -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_snapshot_fileset(tmp_path: Path, artifacts: ArtifactStore) -> None:
    for rel in ["problem/a.phyk", "problem/b.phyk", "problem/run.py", "problem/sub/c.phyk"]:
        _write(tmp_path, rel, rel.encode())
    fs = snapshot_fileset(FileSet(globs=["problem/*.phyk", "problem/run.py"]), tmp_path, artifacts)
    assert list(fs.manifest) == ["problem/a.phyk", "problem/b.phyk", "problem/run.py"]
    assert fs.globs == ["problem/*.phyk", "problem/run.py"]
    assert (
        artifacts.get_bytes(fs.manifest["problem/run.py"])
        == (tmp_path / "problem/run.py").read_bytes()
    )


def test_glob_matching_nothing_raises(tmp_path: Path, artifacts: ArtifactStore) -> None:
    _write(tmp_path, "problem/a.phyk")
    with pytest.raises(ValueError, match=r"glob 'problem/\*\.phky' matched no file"):
        snapshot_fileset(FileSet(globs=["problem/*.phky"]), tmp_path, artifacts)


@pytest.mark.parametrize("kind", ["file", "directory", "dangling"])
def test_symlinks_rejected(tmp_path: Path, artifacts: ArtifactStore, kind: str) -> None:
    base = tmp_path / "source"
    (base / "problem").mkdir(parents=True)
    target = _write(tmp_path, "outside/file.txt", b"secret-file-contents")
    if kind == "directory":
        (base / "problem/linkdir").symlink_to(target.parent, target_is_directory=True)
        glob, match = "problem/linkdir/*", "symlink.*problem/linkdir"
    else:
        if kind == "dangling":
            target.unlink()
        (base / "problem/link.txt").symlink_to(target)
        glob, match = "problem/*", "symlink.*problem/link.txt"
    with pytest.raises(ValueError, match=match):
        snapshot_fileset(FileSet(globs=[glob]), base, artifacts)


@pytest.mark.parametrize("reserved", ["swarm", "parents", "candidate"])
def test_reserved_names_rejected(tmp_path: Path, artifacts: ArtifactStore, reserved: str) -> None:
    _write(tmp_path, f"{reserved}/x.txt")
    with pytest.raises(ValueError, match="reserved"):
        snapshot_fileset(FileSet(globs=[f"{reserved}/*"]), tmp_path, artifacts)


def test_reserved_basenames_allowed(tmp_path: Path, artifacts: ArtifactStore) -> None:
    for name in ["swarm", "parents", "candidate"]:
        _write(tmp_path, f"problem/{name}", name.encode())
    fs = snapshot_fileset(FileSet(globs=["problem/*"]), tmp_path, artifacts)
    assert list(fs.manifest) == ["problem/candidate", "problem/parents", "problem/swarm"]
    assert artifacts.get_bytes(fs.manifest["problem/candidate"]) == b"candidate"


def test_already_snapshotted_reads_no_disk(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, "problem/a.txt")
    fs = snapshot_fileset(FileSet(globs=["problem/*"]), tmp_path, artifacts)
    path.unlink()

    from swarm_arena._tooling import files as files_module

    def forbid_glob(base: Path, pattern: str, listings: object) -> Iterator[str]:
        raise AssertionError("snapshotted sets must not expand globs")

    monkeypatch.setattr(files_module, "_glob_files", forbid_glob)
    assert snapshot_fileset(fs, tmp_path, artifacts) is fs
    empty = FileSet()
    assert snapshot_fileset(empty, tmp_path, artifacts) is empty


def test_snapshot_glob_order_overlap_and_regular_files(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    for rel in ["data/z.txt", "data/a.txt", "data/.hidden.txt", "data/sub/b.txt"]:
        _write(tmp_path, rel)
    os.mkfifo(tmp_path / "data/pipe.txt")
    fs = snapshot_fileset(
        FileSet(globs=["data/z.txt", "data/**/*.txt", "data/*"]), tmp_path, artifacts
    )
    assert list(fs.manifest) == ["data/.hidden.txt", "data/a.txt", "data/sub/b.txt", "data/z.txt"]


def test_each_glob_must_match_a_regular_file(tmp_path: Path, artifacts: ArtifactStore) -> None:
    _write(tmp_path, "data/ok.txt")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="matched no file"):
        snapshot_fileset(FileSet(globs=["data/*", "empty"]), tmp_path, artifacts)


def test_reserved_directory_is_not_a_selected_file(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    _write(tmp_path, "input.txt")
    _write(tmp_path, "swarm/internal.txt")
    fs = snapshot_fileset(FileSet(globs=["*"]), tmp_path, artifacts)
    assert list(fs.manifest) == ["input.txt"]


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("data/*.txt", ["data/a.txt"]),
        ("data/**/*.txt", ["data/a.txt", "data/sub/b.txt"]),
        ("data/**/**", ["data/a.txt", "data/sub/b.txt"]),
        ("data/[a-z].t?t", ["data/a.txt"]),
        ("data/*/b.txt", ["data/sub/b.txt"]),
    ],
)
def test_snapshot_globs_use_shared_dialect(
    tmp_path: Path, artifacts: ArtifactStore, pattern: str, expected: list[str]
) -> None:
    for rel in ["data/a.txt", "data/sub/b.txt"]:
        _write(tmp_path, rel)
    fs = snapshot_fileset(FileSet(globs=[pattern]), tmp_path, artifacts)
    assert list(fs.manifest) == expected


def test_snapshot_fileset_sanitizes_io_errors(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from swarm_arena._tooling import files as files_module

    def fail_listing(directory: Path) -> dict[str, list[str]]:
        raise OSError("fa89d1a4ebc7ed804e52ad2b3cdc0c7b")

    monkeypatch.setattr(files_module, "_listing", fail_listing)
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError, match="cannot snapshot glob") as exc:
        snapshot_fileset(FileSet(globs=["data/*"]), tmp_path, artifacts)
    assert_secret_absent(exc.value, "fa89d1a4ebc7ed804e52ad2b3cdc0c7b", sanitized=True)


def test_snapshot_ref_nested(tmp_path: Path, artifacts: ArtifactStore) -> None:
    _write(tmp_path, "data/a.csv")
    _write(tmp_path, "more/b.txt")
    ref = Ref(
        path="_fakes.tools:FilesWidget",
        config={
            "files": ["data/*.csv"],
            "label": "x",
            "members": [{"path": "_fakes.tools:FilesWidget", "config": {"files": ["more/*.txt"]}}],
        },
    )
    before = deepcopy(ref)
    out = snapshot_ref(ref, tmp_path, artifacts)
    assert list(out.config["files"]["manifest"]) == ["data/a.csv"]
    assert out.config["label"] == "x"
    assert list(out.config["members"][0]["config"]["files"]["manifest"]) == ["more/b.txt"]
    assert snapshot_ref(
        Ref(path="_fakes.tools:Widget", config={"size": 2}), tmp_path, artifacts
    ) == Ref(path="_fakes.tools:Widget", config={"size": 2})
    assert ref == before
    assert snapshot_ref(out, tmp_path, artifacts) == out


def test_snapshot_ref_keeps_omitted_fileset_omitted(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    out = snapshot_ref(
        Ref(path="_fakes.tools:FilesWidget", config={"label": "x"}), tmp_path, artifacts
    )
    assert out == Ref(path="_fakes.tools:FilesWidget", config={"label": "x"})
    assert "files" not in out.config


@pytest.mark.parametrize("kind", ["files", "ref", "refs"])
@pytest.mark.parametrize(
    ("alias", "raw_path", "settings", "unused"),
    [
        ("sources", ("sources",), {}, {}),
        ("sources", ("files",), {"populate_by_name": True}, {}),
        ("sources", ("files",), {"validate_by_alias": False}, {"sources": ["unused"]}),
        (
            "sources",
            ("sources",),
            {"populate_by_name": True},
            {"files": ["unused"]},
        ),
        (AliasChoices("sources", "legacy"), ("legacy",), {}, {}),
        (
            AliasChoices("sources", "legacy"),
            ("sources",),
            {},
            {"legacy": ["unused"]},
        ),
        (AliasPath("payload", "sources"), ("payload", "sources"), {}, {}),
        (AliasPath("payload", -1), ("payload", -1), {}, {}),
        (
            AliasChoices(AliasPath("payload", 1), "legacy"),
            ("legacy",),
            {},
            {"payload": []},
        ),
    ],
)
def test_snapshot_ref_validation_aliases(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    alias: str | AliasChoices | AliasPath,
    raw_path: tuple[str | int, ...],
    settings: dict[str, Any],
    unused: dict[str, Any],
) -> None:
    source = _write(tmp_path, "problem/run.py")
    payload: Any = ["problem/run.py"]
    annotation: Any = FileSet
    if kind != "files":
        payload = {"path": "_fakes.tools:FilesWidget", "config": {"files": payload}}
        annotation = RefField
        if kind == "refs":
            payload = [payload]
            annotation = list[RefField]
    schema = create_model(
        "Config",
        __base__=Widget.Config,
        __config__=ConfigDict(extra="ignore", **settings),
        files=(annotation, Field(validation_alias=alias)),
    )
    monkeypatch.setattr(Widget, "Config", schema)
    raw: dict[str, Any] = dict(unused)
    if len(raw_path) == 1:
        raw[str(raw_path[0])] = payload
    elif isinstance(raw_path[1], int):
        raw[str(raw_path[0])] = [payload]
    else:
        raw[str(raw_path[0])] = {raw_path[1]: payload, "other": 7}
    raw["size"] = "2"
    ref = Ref(path="_fakes.tools:Widget", config=raw)
    before = ref.model_copy(deep=True)
    frozen = snapshot_ref(ref, tmp_path, artifacts)
    assert ref == before
    assert frozen.config.keys() == ref.config.keys()
    assert frozen.config["size"] == "2"
    for key, value in unused.items():
        assert frozen.config[key] == value
    result: Any = frozen.config
    for part in raw_path:
        result = result[part]
    if kind != "files":
        result = (result[0] if kind == "refs" else result)["config"]["files"]
    assert result["manifest"] == {"problem/run.py": sha256_hex(source.read_bytes())}
    source.unlink()
    reloaded = Ref.model_validate_json(frozen.model_dump_json())
    assert snapshot_ref(reloaded, tmp_path, artifacts) == frozen


def test_snapshot_ref_preserves_raw_scalars(tmp_path: Path, artifacts: ArtifactStore) -> None:
    ref = Ref(path="_fakes.tools:Widget", config={"size": "2"}, name="custom")
    assert snapshot_ref(ref, tmp_path, artifacts) == ref


def test_snapshot_nested_lexicographic_ref_reloads(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    ref = Ref(
        path="swarm_arena.scoring:Lexicographic",
        config={"judges": [{"path": "_fakes.tools:FixedJudge", "config": {"value": 1.0}}]},
    )
    snap = snapshot_ref(ref, tmp_path, artifacts)
    assert Ref.model_validate_json(snap.model_dump_json()) == snap
    assert "name" not in snap.config["judges"][0]
    assert type(snap.config["judges"][0]["config"]["value"]) is float


def test_snapshot_ref_revalidates_serialized_config(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NullableFileSet(FileSet):
        @model_serializer
        def serialize(self) -> dict[str, Any]:
            return {"manifest": None}

    config = create_model(
        "Config", __base__=FilesWidget.Config, files=(NullableFileSet, NullableFileSet())
    )
    monkeypatch.setattr(FilesWidget, "Config", config)
    secret = "96adb125d6424fbe835d80730f10b94c"
    ref = Ref(path="_fakes.tools:FilesWidget", config={"files": {}, "label": secret})
    with pytest.raises(ValidationError, match="TOML data cannot contain null") as caught:
        snapshot_ref(ref, tmp_path, artifacts)
    assert_secret_absent(caught.value, secret, sanitized=True)


@pytest.mark.parametrize("wire_format", ["json", "toml"])
def test_snapshot_nested_plan_round_trip_after_source_removal(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    wire_format: str,
) -> None:
    class Chain(Configurable):
        Config = create_model("Config", __base__=_ConfigInputModel, policies=(list[RefField], ...))

    module = types.ModuleType("snapshot_components")
    monkeypatch.setattr(module, "Chain", Chain, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = _write(tmp_path, "data/input.txt", b"frozen input")
    plan = Plan.loads(
        """
teams = []
judges = [{path = "swarm_arena.scoring:Lexicographic", config = {judges = [
    {path = "swarm_arena.scoring:Lexicographic", config = {judges = [
        {path = "_fakes.tools:FilesWidget", config = {files = ["data/*"], label = "raw"}}
    ]}}
]}}]
policy = {path = "snapshot_components:Chain", config = {policies = [
    {path = "snapshot_components:Chain", config = {policies = [
        {path = "_fakes.tools:FilesWidget", name = "named", config = {files = ["data/*"]}}
    ]}}
]}}
[goal]
id = "g"
namespace = "n"
""",
        base_dir=tmp_path,
    )
    original = deepcopy(plan)
    snap = snapshot_plan(plan, artifacts)
    assert plan == original
    judge = snap.judges[0].config["judges"][0]
    policy = snap.policy.config["policies"][0]
    for nested, field in [(judge, "judges"), (policy, "policies")]:
        assert "name" not in nested
        manifest = nested["config"][field][0]["config"]["files"]["manifest"]
        assert manifest == {"data/input.txt": sha256_hex(b"frozen input")}
        assert artifacts.get_bytes(manifest["data/input.txt"]) == b"frozen input"
    assert policy["config"]["policies"][0]["name"] == "named"
    source.unlink()
    source.parent.rmdir()
    if wire_format == "json":
        restored = Plan.model_validate_json(snap.model_dump_json())
    else:
        restored = Plan.loads(snap.dumps(), base_dir=tmp_path)
    assert restored.model_dump(mode="json") == snap.model_dump(mode="json")
    assert snapshot_plan(snap, artifacts) == snap
    assert snapshot_plan(restored, artifacts) == restored


@pytest.mark.parametrize("raw_size", [True, 2, 2.0, "2"])
def test_snapshot_ref_single_ref_and_omitted_defaults(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch, raw_size: Any
) -> None:
    _write(tmp_path, "data/a.csv")
    config = create_model(
        "Config",
        __base__=FilesWidget.Config,
        member=(RefField, Ref(path="no_such_mod:OmittedDefault")),
        files=(FileSet, FileSet()),
        size=(int, 1),
    )
    monkeypatch.setattr(FilesWidget, "Config", config)
    empty = Ref(path="_fakes.tools:FilesWidget")
    assert snapshot_ref(empty, tmp_path, artifacts) == empty
    ref = Ref(
        path=empty.path,
        config={"member": {"path": empty.path, "config": {"files": ["data/*"], "size": raw_size}}},
    )
    out = snapshot_ref(ref, tmp_path, artifacts)
    assert Ref.model_validate_json(out.model_dump_json()) == out
    assert "name" not in out.config["member"]
    assert out.config["member"]["config"]["size"] == raw_size
    assert type(out.config["member"]["config"]["size"]) is type(raw_size)
    assert list(out.config["member"]["config"]["files"]["manifest"]) == ["data/a.csv"]
    assert "files" not in out.config


@pytest.mark.parametrize("path", ["builtins:str", "json:dumps"])
def test_snapshot_ref_requires_configurable_class(
    tmp_path: Path, artifacts: ArtifactStore, path: str
) -> None:
    with pytest.raises(TypeError, match=r"class|Configurable"):
        snapshot_ref(Ref(path=path), tmp_path, artifacts)


@pytest.mark.parametrize(
    "config",
    [
        {"label": {"fa89d1a4ebc7ed804e52ad2b3cdc0c7b": "fa89d1a4ebc7ed804e52ad2b3cdc0c7b"}},
        {"fa89d1a4ebc7ed804e52ad2b3cdc0c7b": "fa89d1a4ebc7ed804e52ad2b3cdc0c7b"},
        {"files": {"manifest": {"fa89d1a4ebc7ed804e52ad2b3cdc0c7b": "invalid"}}},
        {
            "members": [
                {
                    "path": "_fakes.tools:Widget",
                    "config": {"size": "fa89d1a4ebc7ed804e52ad2b3cdc0c7b"},
                }
            ]
        },
    ],
)
def test_snapshot_ref_sanitizes_validation_errors(
    tmp_path: Path, artifacts: ArtifactStore, config: dict[str, Any]
) -> None:
    with pytest.raises(ConfigError) as exc:
        snapshot_ref(Ref(path="_fakes.tools:FilesWidget", config=config), tmp_path, artifacts)
    assert_secret_absent(exc.value, "fa89d1a4ebc7ed804e52ad2b3cdc0c7b", sanitized=True)


def test_snapshot_ref_sanitizes_validator_message(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Config(Widget.Config):
        @field_validator("size", mode="before")
        @classmethod
        def reject(cls, value: object) -> object:
            raise ValueError(f"invalid secret: {value}")

    monkeypatch.setattr(Widget, "Config", Config)
    # Python 3.13 tracebacks quote the whole multi-line call, so the sentinel
    # must not be a literal inside it.
    secret = "fa89d1a4ebc7ed804e52ad2b3cdc0c7b"
    with pytest.raises(ConfigError) as exc:
        snapshot_ref(Ref(path="_fakes.tools:Widget", config={"size": secret}), tmp_path, artifacts)
    assert_secret_absent(exc.value, secret)


def test_snapshot_ref_sanitizes_import_errors(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_load(self: Ref) -> type:
        raise ValueError("fa89d1a4ebc7ed804e52ad2b3cdc0c7b")

    monkeypatch.setattr("swarm_arena._model.ref.importlib.import_module", fail_load)
    with pytest.raises(ValueError, match=r"cannot import sentinel_import:Widget") as exc:
        snapshot_ref(Ref(path="sentinel_import:Widget"), tmp_path, artifacts)
    assert_secret_absent(exc.value, "fa89d1a4ebc7ed804e52ad2b3cdc0c7b")


PLAN_TOML = """
policy = { path = "_fakes.tools:Widget" }
checks = [{ path = "_fakes.tools:FilesWidget", config = { files = ["check/*"] } }]
judges = [{ path = "_fakes.tools:FilesWidget", config = { files = ["judge/*"] } }]
[goal]
id = "g"
namespace = "n"
files = ["problem/*"]
[[parts]]
name = "part"
checks = [{ path = "_fakes.tools:FilesWidget", config = { files = ["part-check/*"] } }]
judges = [{ path = "_fakes.tools:FilesWidget", config = { files = ["part-judge/*"] } }]
[parts.goal]
id = "p"
namespace = "n"
files = ["part/*"]
[[parts]]
name = "inherit"
[parts.goal]
id = "inherit"
namespace = "n"
[[parts]]
name = "empty"
checks = []
judges = []
[parts.goal]
id = "empty"
namespace = "n"
[[teams]]
name = "t"
rationale = "Explore"
[[teams.agents]]
name = "a"
rationale = "Search"
tool = { path = "_fakes.tools:FilesWidget", config = { files = ["tool/*"] } }
manager = { path = "_fakes.tools:FilesWidget", config = { files = ["manager/*"] } }
[converge]
mode = "team"
[converge.team]
name = "join"
rationale = "Combine"
[[converge.team.agents]]
name = "join"
rationale = "Combine"
tool = { path = "_fakes.tools:FilesWidget", config = { files = ["join-tool/*"] } }
manager = { path = "_fakes.tools:FilesWidget", config = { files = ["join-manager/*"] } }
"""


def test_snapshot_plan(tmp_path: Path, artifacts: ArtifactStore) -> None:
    for directory in [
        "problem",
        "part",
        "tool",
        "manager",
        "check",
        "judge",
        "part-check",
        "part-judge",
        "join-tool",
        "join-manager",
        "policy",
    ]:
        _write(tmp_path, f"{directory}/file.txt", directory.encode())
    (tmp_path / "plan.toml").write_text(PLAN_TOML)
    original = Plan.load(tmp_path / "plan.toml")
    before = deepcopy(original)
    plan = snapshot_plan(original, artifacts)
    assert plan.goal.key().files_sha256
    assert plan.parts[0].goal.files.snapshotted
    assert plan.teams[0].agents[0].tool.config["files"]["manifest"]
    assert plan.judges[0].config["files"]["manifest"]
    assert plan.base_dir == tmp_path
    assert snapshot_plan(plan, artifacts) == plan
    assert original == before
    assert plan.parts[1].checks is None
    assert plan.parts[1].judges is None
    assert plan.parts[2].checks == []
    assert plan.parts[2].judges == []
    assert plan.converge.team is not None
    assert plan.parts[0].checks is not None
    assert plan.parts[0].judges is not None
    refs = [
        (plan.teams[0].agents[0].tool, "tool"),
        (plan.teams[0].agents[0].manager, "manager"),
        (plan.checks[0], "check"),
        (plan.judges[0], "judge"),
        (plan.parts[0].checks[0], "part-check"),
        (plan.parts[0].judges[0], "part-judge"),
        (plan.converge.team.agents[0].tool, "join-tool"),
        (plan.converge.team.agents[0].manager, "join-manager"),
    ]
    for ref, directory in refs:
        manifest = ref.config["files"]["manifest"]
        assert list(manifest) == [f"{directory}/file.txt"]
        assert artifacts.get_bytes(manifest[f"{directory}/file.txt"]) == directory.encode()
    policy_plan = original.model_copy(
        update={"policy": Ref(path="_fakes.tools:FilesWidget", config={"files": ["policy/*"]})}
    )
    assert list(snapshot_plan(policy_plan, artifacts).policy.config["files"]["manifest"]) == [
        "policy/file.txt"
    ]
    for path in tmp_path.glob("*/file.txt"):
        path.unlink()
    assert snapshot_plan(plan, artifacts) == plan


def test_snapshot_plan_unimportable_ref(tmp_path: Path, artifacts: ArtifactStore) -> None:
    plan = Plan.loads(
        'teams = []\njudges = [{path = "no_such_mod:J"}]\n'
        'policy = {path = "_fakes.tools:Widget"}\n[goal]\nid = "g"\nnamespace = "n"',
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="cannot import no_such_mod:J"):
        snapshot_plan(plan, artifacts)


class _ValidatorFailure(Exception):
    pass


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TypeError("982fa6f15da245419f1f479d3e60a8cd"), "invalid"),
        (ValueError("982fa6f15da245419f1f479d3e60a8cd"), "size: value_error"),
        (AssertionError("982fa6f15da245419f1f479d3e60a8cd"), "size: assertion_error"),
        (_ValidatorFailure("982fa6f15da245419f1f479d3e60a8cd"), "invalid"),
        (
            PydanticCustomError(
                "982fa6f15da245419f1f479d3e60a8cd",
                "bad input: {value}",
                {"value": "982fa6f15da245419f1f479d3e60a8cd"},
            ),
            "size: invalid",
        ),
        (
            PydanticCustomError("unregistered_type", "982fa6f15da245419f1f479d3e60a8cd"),
            "size: invalid",
        ),
        (None, "size: int_parsing"),
    ],
)
def test_snapshot_ref_config_validation_failures_are_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts: ArtifactStore,
    error: Exception | None,
    expected: str,
) -> None:
    class Config(Widget.Config):
        @field_validator("size", mode="before")
        @classmethod
        def reject_size(cls, value: object) -> object:
            if error is not None:
                raise error
            return value

    monkeypatch.setattr(Widget, "Config", Config)
    secret = "982fa6f15da245419f1f479d3e60a8cd"
    ref = Ref(path="_fakes.tools:Widget", config={"size": secret})
    with pytest.raises(ConfigError) as exc:
        snapshot_ref(ref, tmp_path, artifacts)

    assert_secret_absent(exc.value, secret, sanitized=True)
    assert str(exc.value) == f"_fakes.tools:Widget: {expected}"


@pytest.mark.parametrize(
    ("upper_listed", "same_inode"),
    [
        (False, True),  # APFS: RUN.py is a lookup alias of the one listed name
        (True, False),  # case-sensitive volume: two distinct files
        (True, True),  # case-sensitive volume: two listed hard links to one file
    ],
)
def test_snapshot_folded_alias_identity(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    upper_listed: bool,
    same_inode: bool,
) -> None:
    """Exercise the filesystem identity cases independently of the host volume."""
    import stat

    import swarm_arena._tooling.files as files_module

    lower, upper = tmp_path / "run.py", tmp_path / "RUN.py"
    original_inspect = files_module._inspect_path
    original_listing = files_module._listing

    def inspect(base: Path, rel: str) -> tuple[str | None, os.stat_result | None]:
        if base / rel in (lower, upper):
            inode = 1 if base / rel == lower or same_inode else 2
            return None, os.stat_result((stat.S_IFREG | 0o644, inode, 1, 1, 0, 0, 1, 0, 0, 0))
        return original_inspect(base, rel)

    def listing(directory: Path) -> dict[str, list[str]]:
        if directory == tmp_path:
            return {"run.py": ["run.py", "RUN.py"] if upper_listed else ["run.py"]}
        return original_listing(directory)

    def identity(path: Path) -> tuple[int, int]:
        _, info = inspect(path.parent, path.name)
        assert info is not None
        return info.st_dev, info.st_ino

    def read(base: Path, rel: str, *, max_bytes: int) -> bytes:
        return b"x"

    monkeypatch.setattr(files_module, "_inspect_path", inspect)
    monkeypatch.setattr(files_module, "_listing", listing)
    monkeypatch.setattr(files_module, "_identity", identity)
    monkeypatch.setattr(files_module, "read_regular_file", read)
    if not upper_listed:
        snapshot = snapshot_fileset(FileSet(globs=["*"]), tmp_path, artifacts)
        assert snapshot.manifest == {"run.py": sha256_hex(b"x")}
        alias = snapshot_fileset(FileSet(globs=["RUN.py"]), tmp_path, artifacts)
        assert alias.manifest == snapshot.manifest
    else:
        with pytest.raises(ValueError, match=r"conflicting.*path") as exc:
            snapshot_fileset(FileSet(globs=["*"]), tmp_path, artifacts)
        assert "run.py" in str(exc.value)
        assert "RUN.py" in str(exc.value)


def test_snapshot_refuses_listed_case_variant_hard_links(
    tmp_path: Path, artifacts: ArtifactStore
) -> None:
    _write(tmp_path, "p/run.py")
    if (tmp_path / "p/RUN.py").exists():
        pytest.skip("needs a case-sensitive filesystem")
    os.link(tmp_path / "p/run.py", tmp_path / "p/RUN.py")
    with pytest.raises(ValueError, match="conflicting snapshot paths") as exc:
        snapshot_fileset(FileSet(globs=["p/*"]), tmp_path, artifacts)
    assert str(exc.value) == "conflicting snapshot paths: p/RUN.py and p/run.py"


def test_snapshot_lists_each_directory_once(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import swarm_arena._tooling.files as files_module

    for index in range(40):
        _write(tmp_path, f"data/f{index:02}.txt", b"%d" % index)
    _write(tmp_path, "data/sub/g.txt")
    listed: list[Path] = []
    original_listing = files_module._listing

    def listing(directory: Path) -> dict[str, list[str]]:
        listed.append(directory)
        return original_listing(directory)

    monkeypatch.setattr(files_module, "_listing", listing)
    snapshot = snapshot_fileset(FileSet(globs=["data/*", "data/**/*.txt"]), tmp_path, artifacts)
    assert len(snapshot.manifest) == 41
    assert sorted(listed) == [tmp_path, tmp_path / "data", tmp_path / "data/sub"]


def test_snapshot_refuses_a_file_over_its_cap_unread(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    bytes_read: list[int],
    sparse: Any,
) -> None:
    from swarm_arena._tooling import files as files_module

    assert files_module.MAX_SNAPSHOT_FILE_BYTES == 2**30  # 1 GiB per goal or tool file
    monkeypatch.setattr(files_module, "MAX_SNAPSHOT_FILE_BYTES", 1024)
    sparse(tmp_path / "data/fits.bin", 1024)
    sparse(tmp_path / "data/big.bin", 2 * 1024)
    bytes_read.clear()
    with pytest.raises(ValueError, match=r"^data/big\.bin crosses the size cap$"):
        snapshot_fileset(FileSet(globs=["data/*"]), tmp_path, artifacts)
    assert sum(bytes_read) <= 1024  # data/fits.bin at most; the big file is refused by its size
    snapshot = snapshot_fileset(FileSet(globs=["data/fits.bin"]), tmp_path, artifacts)
    assert snapshot.manifest == {"data/fits.bin": sha256_hex(bytes(1024))}


def test_unmatched_glob_names_its_base(tmp_path: Path, artifacts: ArtifactStore) -> None:
    with pytest.raises(ValueError, match="matched no file under") as caught:
        snapshot_fileset(FileSet(globs=["*.physk"]), tmp_path, artifacts)
    assert str(tmp_path) in str(caught.value)


@pytest.mark.parametrize("recursive", [False, True], ids=["literal", "recursive"])
def test_snapshot_maximum_length_path(
    tmp_path: Path, artifacts: ArtifactStore, recursive: bool
) -> None:
    relative = "/".join(["a" * 255] * 3 + ["a" * 254, "x"])
    assert len(relative.encode()) == 1024
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in relative.split("/")[:-1]:
            os.mkdir(part, dir_fd=fd)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        leaf = os.open("x", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=fd)
        try:
            os.write(leaf, b"hi")
        finally:
            os.close(leaf)
    finally:
        os.close(fd)
    snapshot = snapshot_fileset(
        FileSet(globs=["**/*" if recursive else relative]), tmp_path, artifacts
    )
    assert snapshot.manifest == {relative: sha256_hex(b"hi")}
    assert artifacts.get_bytes(snapshot.manifest[relative]) == b"hi"


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("shape", ["dict", "mapping", "lists"])
@pytest.mark.parametrize("alias_kind", ["canonical", "alias", "choices", "path"])
def test_snapshot_typed_ref_mappings_preserve_raw_values(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
    shape: str,
    alias_kind: str,
) -> None:
    from collections.abc import Mapping

    from _fakes.admission import MapJudge

    annotation = dict[str, RefField] if shape == "dict" else Mapping[str, RefField]
    if shape == "lists":
        annotation = dict[str, list[RefField]]
    alias: Any = None
    key = "children"
    if alias_kind == "alias":
        alias = key = "members"
    elif alias_kind == "choices":
        alias = AliasChoices("members", "legacy")
        key = "legacy"
    elif alias_kind == "path":
        alias = AliasPath("payload", 0)
        key = "payload"
    schema = create_model(
        "Config",
        __base__=MapJudge.Config,
        children=(annotation, Field(default_factory=dict, validation_alias=alias)),
    )
    monkeypatch.setattr(MapJudge, "Config", schema)
    source = _write(tmp_path, "judge.py", b"judge source")
    leaf = Ref(path="_fakes.admission:DefaultScriptJudge", config={"command": ["score"]})
    for _ in range(depth):
        child: Any = leaf.model_dump(exclude_none=True)
        if shape == "lists":
            child = [child]
        children: Any = {"raw/key~name": child}
        if alias_kind == "path":
            children = [children]
        leaf = Ref(path="_fakes.admission:MapJudge", config={"value": 1, key: children})
    original = leaf.model_dump(mode="json")
    frozen = snapshot_ref(leaf, tmp_path, artifacts)
    assert leaf.model_dump(mode="json") == original
    current = frozen
    for _ in range(depth):
        assert type(current.config["value"]) is int
        assert ("children" in current.config) == (alias_kind == "canonical")
        _, cfg = current._resolve_config(Configurable)
        current = cast(Any, cfg).children["raw/key~name"]
        if shape == "lists":
            current = current[0]
    files = current.config["files"]
    assert files["manifest"] == {"judge.py": sha256_hex(b"judge source")}
    assert "env" not in current.config
    assert "name" not in current.model_dump(exclude_none=True)
    source.unlink()
    restored = Ref.model_validate_json(frozen.model_dump_json())
    assert snapshot_ref(restored, tmp_path, artifacts) == frozen


def test_snapshot_materializes_only_nonempty_file_defaults(
    tmp_path: Path,
    artifacts: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _fakes.admission import DefaultScriptJudge

    schema = create_model(
        "Config",
        __base__=DefaultScriptJudge.Config,
        empty=(FileSet, FileSet()),
        label=(str, "omitted"),
    )
    monkeypatch.setattr(DefaultScriptJudge, "Config", schema)
    source = _write(tmp_path, "judge.py", b"score")
    raw = Ref(path="_fakes.admission:DefaultScriptJudge", config={"command": ["score"]})
    frozen = snapshot_ref(raw, tmp_path, artifacts)
    assert frozen.config == {
        "command": ["score"],
        "files": {"globs": ["judge.py"], "manifest": {"judge.py": sha256_hex(b"score")}},
    }
    source.unlink()
    assert snapshot_ref(frozen, tmp_path, artifacts) == frozen
    explicit_empty = raw.model_copy(update={"config": {**raw.config, "files": []}})
    assert snapshot_ref(explicit_empty, tmp_path, artifacts).config["files"]["manifest"] == {}
