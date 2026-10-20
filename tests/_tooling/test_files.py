import os
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from _fakes.tools import FilesWidget, Widget
from pydantic import create_model, field_validator
from pydantic_core import PydanticCustomError

from swarm_arena._errors import ConfigError
from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Plan
from swarm_arena._model.ref import Ref, RefField
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

    def forbid_glob(self: Path, pattern: str) -> Iterator[Path]:
        raise AssertionError("snapshotted sets must not expand globs")

    monkeypatch.setattr(Path, "glob", forbid_glob)
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


def test_path_glob_hits_use_shared_dialect(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [_write(tmp_path, rel) for rel in ["data/a.txt", "data/sub/b.txt"]]

    def broad_glob(self: Path, pattern: str) -> Iterator[Path]:
        return iter(paths)

    monkeypatch.setattr(Path, "glob", broad_glob)
    fs = snapshot_fileset(FileSet(globs=["data/*.txt"]), tmp_path, artifacts)
    assert list(fs.manifest) == ["data/a.txt"]


def test_snapshot_fileset_sanitizes_io_errors(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_glob(self: Path, pattern: str) -> Iterator[Path]:
        raise OSError("fa89d1a4ebc7ed804e52ad2b3cdc0c7b")

    monkeypatch.setattr(Path, "glob", fail_glob)
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


def test_snapshot_ref_preserves_raw_scalars(tmp_path: Path, artifacts: ArtifactStore) -> None:
    ref = Ref(path="_fakes.tools:Widget", config={"size": "2"}, name="custom")
    assert snapshot_ref(ref, tmp_path, artifacts) == ref


def test_snapshot_ref_single_ref_and_omitted_defaults(
    tmp_path: Path, artifacts: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "data/a.csv")
    config = create_model(
        "Config",
        __base__=FilesWidget.Config,
        member=(RefField, Ref(path="no_such_mod:OmittedDefault")),
        files=(FileSet, FileSet(globs=["missing-default/*"])),
    )
    monkeypatch.setattr(FilesWidget, "Config", config)
    empty = Ref(path="_fakes.tools:FilesWidget")
    assert snapshot_ref(empty, tmp_path, artifacts) == empty
    ref = Ref(
        path=empty.path, config={"member": {"path": empty.path, "config": {"files": ["data/*"]}}}
    )
    out = snapshot_ref(ref, tmp_path, artifacts)
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
        (TypeError("validator_secret_sentinel"), "invalid"),
        (ValueError("validator_secret_sentinel"), "size: value_error"),
        (AssertionError("validator_secret_sentinel"), "size: assertion_error"),
        (_ValidatorFailure("validator_secret_sentinel"), "invalid"),
        (
            PydanticCustomError(
                "validator_secret_sentinel",
                "bad input: {value}",
                {"value": "validator_secret_sentinel"},
            ),
            "size: invalid",
        ),
        (PydanticCustomError("unregistered_type", "validator_secret_sentinel"), "size: invalid"),
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
    secret = "validator_secret_sentinel"
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
    original_stat = Path.stat
    original_listing = files_module._listing

    def stat_path(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if self in (lower, upper):
            inode = 1 if self == lower or same_inode else 2
            return os.stat_result((stat.S_IFREG | 0o644, inode, 1, 1, 0, 0, 1, 0, 0, 0))
        return original_stat(self, follow_symlinks=follow_symlinks)

    def glob(self: Path, pattern: str) -> Iterator[Path]:
        assert self == tmp_path
        yield from [lower, upper]

    def listing(directory: Path) -> dict[str, list[str]]:
        if directory == tmp_path:
            return {"run.py": ["run.py", "RUN.py"] if upper_listed else ["run.py"]}
        return original_listing(directory)

    def identity(path: Path) -> tuple[int, int]:
        info = stat_path(path, follow_symlinks=False)
        return info.st_dev, info.st_ino

    def read(base: Path, rel: str, *, max_bytes: int) -> bytes:
        return b"x"

    monkeypatch.setattr(Path, "stat", stat_path)
    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(files_module, "_listing", listing)
    monkeypatch.setattr(files_module, "_identity", identity)
    monkeypatch.setattr(files_module, "read_regular_file", read)
    if not upper_listed:
        snapshot = snapshot_fileset(FileSet(globs=["*"]), tmp_path, artifacts)
        assert snapshot.manifest == {"run.py": sha256_hex(b"x")}
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
