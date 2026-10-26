"""Verify release artifacts, metadata, and the single dated release history."""

import email.parser
import hashlib
import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta, tzinfo
from email.message import Message
from pathlib import Path

import pytest
from _fakes.process_cleanup import reap_children

import swarm_arena
from swarm_arena._exec.procinfo import process_start_time

REPO = Path(__file__).resolve().parents[1]


def _checkout_snapshot(root: Path) -> dict[str, str]:
    # Runtime caches can be populated by pytest/xdist independently of the build.
    ignored = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    # Local workspaces and docs output are written by other tools while a build runs.
    local = {".superpowers", "tasks", ".claude", "docs/_build"}
    snapshot: dict[str, str] = {}
    for directory, dirs, files in os.walk(root):
        here = Path(directory).relative_to(root)
        dirs[:] = sorted(
            name for name in dirs if name not in ignored and (here / name).as_posix() not in local
        )
        for name in [*dirs, *(name for name in files if name not in ignored)]:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                snapshot[relative] = f"symlink:{os.readlink(path)}"
            elif path.is_file():
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                snapshot[relative] = f"{path.stat().st_mode}:{digest}"
            else:
                snapshot[relative] = "directory"
    return snapshot


@pytest.fixture(scope="module")
def distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to verify release artifacts"
    out = tmp_path_factory.mktemp("dist")
    assert not out.resolve().is_relative_to(REPO.resolve()), "build output is inside the checkout"
    before = _checkout_snapshot(REPO)
    # Without --wheel, uv builds the sdist, then builds the wheel from that sdist.
    process = subprocess.Popen(
        [uv, "build", "--out-dir", str(out)], cwd=REPO, start_new_session=True
    )
    children = [(process, True, None)]
    try:
        children = [(process, True, process_start_time(process.pid))]
        assert process.wait(timeout=120) == 0
    finally:
        assert not reap_children(children)
    assert _checkout_snapshot(REPO) == before, "build changed the checkout"
    [wheel_path] = out.glob("swarm_arena-*-py3-none-any.whl")
    [sdist_path] = out.glob("swarm_arena-*.tar.gz")
    return wheel_path, sdist_path


@pytest.fixture(scope="module")
def wheel(distributions: tuple[Path, Path]) -> Iterator[zipfile.ZipFile]:
    with zipfile.ZipFile(distributions[0]) as archive:
        yield archive


def metadata(wheel: zipfile.ZipFile) -> Message:
    [name] = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
    return email.parser.Parser().parsestr(wheel.read(name).decode("utf-8"))


def test_wheel_files(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    expected = {
        path.relative_to(REPO / "src").as_posix()
        for path in (REPO / "src/swarm_arena").rglob("*.py")
    } | {"swarm_arena/py.typed"}
    packaged = {
        name
        for name in names
        if name.startswith("swarm_arena/") and (name.endswith(".py") or name.endswith("/py.typed"))
    }
    assert packaged == expected
    for needed in (
        "swarm_arena/__init__.py",
        "swarm_arena/py.typed",
        "swarm_arena/_launch.py",
        "swarm_arena/run_function.py",
        "swarm_arena/llm/__init__.py",
        "swarm_arena/ray/__init__.py",
        "swarm_arena/testing/__init__.py",
    ):
        assert needed in names
    assert not [n for n in names if n.split("/")[0] in {"tests", "docs", "examples", "tasks"}]
    [license_path] = [n for n in names if n.endswith(".dist-info/licenses/LICENSE")]
    assert wheel.read(license_path) == (REPO / "LICENSE").read_bytes()


def test_wheel_metadata(wheel: zipfile.ZipFile) -> None:
    md = metadata(wheel)
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert (md["Name"], md["Version"], md["Requires-Python"]) == (
        "swarm-arena",
        project["version"],
        ">=3.11",
    )
    assert md["License-Expression"] == "MIT"
    assert md["Description-Content-Type"] == "text/markdown"
    assert md["Author"] == "Shreyas Vinaya"
    assert md["Author-email"] is None
    assert set(md.get_all("Provides-Extra", [])) == {"llm", "cache", "ray", "s3", "langfuse"}
    base = "https://github.com/shreyasvinaya/swarm-arena"
    assert dict(u.split(", ", 1) for u in md.get_all("Project-URL", [])) == {
        "Homepage": base,
        "Repository": base,
        "Documentation": "https://swarm-arena.readthedocs.io/",
        "Changelog": f"{base}/blob/main/CHANGELOG.md",
        "Issues": f"{base}/issues",
    }
    classifiers = md.get_all("Classifier", [])
    assert {
        "Operating System :: POSIX",
        "Typing :: Typed",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    } <= set(classifiers)
    assert not [c for c in classifiers if c.startswith("License ::")]


def test_core_requirements_are_exactly_three(wheel: zipfile.ZipFile) -> None:
    core: dict[str, set[str]] = {}
    for req in metadata(wheel).get_all("Requires-Dist", []):
        if "extra ==" not in req:
            match = re.match(r"([A-Za-z0-9._-]+)\s*(.*)", req.split(";")[0])
            assert match is not None, req
            name, spec = match.groups()
            core[name.lower()] = {s.strip() for s in spec.split(",") if s.strip()}
    assert core == {
        "anyio": {">=4.15", "<5"},
        "pydantic": {">=2.14.0", "<3"},
        "typing-extensions": {">=4.16.0"},
    }


def test_entry_point(wheel: zipfile.ZipFile) -> None:
    [name] = [n for n in wheel.namelist() if n.endswith(".dist-info/entry_points.txt")]
    assert "swarm-arena = swarm_arena._cli.main:main" in wheel.read(name).decode("utf-8")


def test_sdist_metadata_and_contents(
    distributions: tuple[Path, Path], wheel: zipfile.ZipFile
) -> None:
    with tarfile.open(distributions[1]) as archive:
        names = archive.getnames()
        [root] = {name.split("/")[0] for name in names}
        relative = {name.removeprefix(f"{root}/") for name in names}
        assert {
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "PKG-INFO",
            "src/swarm_arena/py.typed",
        } <= relative
        assert not {"tasks", ".superpowers", ".claude", ".venv", "dist"} & {
            name.split("/")[0] for name in relative
        }
        project_file = archive.extractfile(f"{root}/pyproject.toml")
        assert project_file is not None
        with project_file:
            assert tomllib.loads(project_file.read().decode("utf-8")) == tomllib.loads(
                (REPO / "pyproject.toml").read_text(encoding="utf-8")
            )
        info = archive.extractfile(f"{root}/PKG-INFO")
        assert info is not None
        with info:
            md = email.parser.Parser().parsestr(info.read().decode("utf-8"))
        assert md.items() == metadata(wheel).items()
        assert md.get_payload() == metadata(wheel).get_payload()
        for filename in ("LICENSE", "README.md"):
            content = archive.extractfile(f"{root}/{filename}")
            assert content is not None
            with content:
                assert content.read() == (REPO / filename).read_bytes()


def test_version_matches_everywhere() -> None:
    version = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    released = re.findall(
        r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$",
        (REPO / "CHANGELOG.md").read_text(encoding="utf-8"),
        re.M,
    )
    assert released, "No dated release heading"
    assert (
        swarm_arena.__version__
        == version
        == released[0]
        == runpy.run_path(str(REPO / "docs/conf.py"))["release"]
    )


def test_changelog_format() -> None:
    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    _assert_changelog_format(text)


def _assert_changelog_format(text: str) -> None:
    headings = re.findall(r"^## (.+)$", text, re.M)
    assert headings[0] == "[Unreleased]"
    releases: list[str] = []
    for heading in headings[1:]:
        match = re.fullmatch(r"\[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})", heading)
        assert match is not None, heading
        version, released_on = match.groups()
        assert date.fromisoformat(released_on) <= (datetime.now(UTC) + timedelta(hours=14)).date()
        releases.append(version)
    assert releases
    assert len(releases) == len(set(releases))
    base = "https://github.com/shreyasvinaya/swarm-arena"
    assert f"[Unreleased]: {base}/compare/v{releases[0]}...HEAD" in text
    assert all(f"[{v}]: {base}/releases/tag/v{v}" in text for v in releases)
    section = text.split(f"## [{releases[0]}]", 1)[1].split("\n## [", 1)[0]
    assert "### Added" in section
    assert re.search(r"^- ", section, re.M)


def test_license_is_mit() -> None:
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert "Permission is hereby granted, free of charge" in text
    assert "Shreyas Vinaya" in text


@pytest.mark.parametrize(
    ("hour", "released_on", "valid"),
    # At 05:00Z the latest calendar day anywhere is still 2026-10-26.
    [(19, "2026-10-27", True), (19, "2026-10-28", False), (5, "2026-10-27", False)],
)
def test_changelog_date_uses_latest_calendar_day(
    monkeypatch: pytest.MonkeyPatch, hour: int, released_on: str, valid: bool
) -> None:
    class FrozenDate(date):
        @classmethod
        def today(cls) -> "FrozenDate":
            return cls(2026, 10, 26)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> "FrozenDatetime":
            assert tz is UTC
            return cls(2026, 10, 26, hour, tzinfo=UTC)

    monkeypatch.setattr(sys.modules[__name__], "date", FrozenDate)
    monkeypatch.setattr(sys.modules[__name__], "datetime", FrozenDatetime)
    base = "https://github.com/shreyasvinaya/swarm-arena"
    text = (
        f"## [Unreleased]\n\n## [0.1.0] - {released_on}\n\n### Added\n- Initial release.\n"
        f"[Unreleased]: {base}/compare/v0.1.0...HEAD\n"
        f"[0.1.0]: {base}/releases/tag/v0.1.0\n"
    )
    if valid:
        _assert_changelog_format(text)
    else:
        with pytest.raises(AssertionError):
            _assert_changelog_format(text)
