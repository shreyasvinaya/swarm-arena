"""Shared executable documentation contracts, independent of page ownership."""

import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from _fakes.process_cleanup import reap_children
from _surface import DISTRIBUTIONS, EXTRA_DEPS, deps_present

from swarm_arena._exec.procinfo import group_has_live_member, process_start_time

REPO: Path = Path(__file__).resolve().parents[1]
PAGES: list[tuple[str, str]] = [
    ("Getting started", "getting_started"),
    ("Concepts", "concepts"),
    ("Use your own files and script", "own_files"),
    ("Writing a connector", "connector"),
    ("Planning modes", "planning"),
    ("Scoring and review", "scoring"),
    ("Limits and safety", "limits"),
    ("Running on Ray", "ray"),
    ("The arena", "arena"),
    ("Versioning policy", "versioning"),
    ("Roadmap", "roadmap"),
    ("API reference", "api"),
]
EXTRA_IMPORTS: dict[str, str] = {
    "llm": "agents",
    "cache": "litellm",
    "ray": "ray",
    "s3": "s3fs",
    "langfuse": "langfuse",
}
# Cache and tracing examples also import the LLM facade, which needs both vendors.
_EXTRA_DEPS = {
    "llm": EXTRA_DEPS["swarm_arena.llm"],
    "cache": EXTRA_DEPS["swarm_arena.llm"],
    "ray": EXTRA_DEPS["swarm_arena.ray"],
    "s3": ("s3fs",),
    "langfuse": (*EXTRA_DEPS["swarm_arena.llm"], "langfuse"),
}
STUB_TEXT = re.compile(r"written in M\d|under construction|M5 task", re.I)
_MARKER = re.compile(r"^<!-- docs-test: (skip|requires|file|run)(?: (.*\S))? -->$")
_FENCE = re.compile(r"^(`{3,})([^`]*)$")


@dataclass(frozen=True)
class Block:
    """A fenced example with its execution policy and source location."""

    lang: str
    code: str
    kind: Literal["none", "skip", "requires", "file", "run"]
    arg: str
    line: int


def parse_blocks(text: str) -> list[Block]:
    """
    Parse column-zero Markdown fences and validate adjacent execution markers.

    Parameters
    ----------
    text : str
        Markdown source.

    Returns
    -------
    list of Block
        Non-directive blocks in document order.

    Raises
    ------
    ValueError
        A docs-test marker is malformed or attached to an unsuitable fence.

    Examples
    --------
    >>> parse_blocks('```python\\nprint(1)\\n```')[0].line
    1
    """
    lines = text.splitlines(keepends=True)
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip("\r\n")
        kind: Literal["none", "skip", "requires", "file", "run"] = "none"
        arg = ""
        if "docs-test:" in line:
            marker_line = index + 1

            def invalid(marker_line: int = marker_line) -> ValueError:
                return ValueError(f"docs-test invalid marker at line {marker_line}")

            marker = _MARKER.fullmatch(line)
            if marker is None:
                raise invalid()
            raw_kind, arg = marker.group(1), marker.group(2) or ""
            if raw_kind == "skip":
                kind = "skip"
                if not arg:
                    raise invalid()
            elif raw_kind == "requires":
                kind = "requires"
                if arg not in EXTRA_IMPORTS:
                    raise invalid()
            elif raw_kind == "file":
                kind = "file"
                if not arg or Path(arg).is_absolute() or ".." in Path(arg).parts:
                    raise invalid()
            else:
                kind = "run"
                if arg:
                    raise invalid()
            index += 1
            if index >= len(lines):
                raise invalid()
            line = lines[index].rstrip("\r\n")
            fence = _FENCE.fullmatch(line)
            lang = fence.group(2).strip().split()[0] if fence and fence.group(2).strip() else ""
            if fence is None or lang.startswith("{"):
                raise invalid()
            if (kind in {"skip", "requires"} and lang != "python") or (
                kind == "run" and lang != "bash"
            ):
                raise invalid()
        fence = _FENCE.fullmatch(line)
        if fence is None:
            index += 1
            continue
        opening = index
        info = fence.group(2).strip().split()
        lang = info[0] if info else ""
        closing = re.compile(rf"^`{{{len(fence.group(1))},}}$")
        index += 1
        start = index
        while index < len(lines) and not closing.fullmatch(lines[index].rstrip("\r\n")):
            index += 1
        if not lang.startswith("{"):
            blocks.append(Block(lang, "".join(lines[start:index]), kind, arg, opening + 1))
        index += 1
    return blocks


def docs_bindir() -> Path:
    """
    Select the interpreter directory for editable or installed-wheel examples.

    Returns
    -------
    Path
        Target virtual environment's bin directory.

    Examples
    --------
    >>> docs_bindir().name
    'bin'
    """
    target = os.environ.get("SWARM_ARENA_DOCS_VENV")
    return Path(target) / "bin" if target is not None else Path(sys.executable).parent


def block_env(workdir: Path, bindir: Path) -> dict[str, str]:
    """
    Create the minimal no-key environment shared by blocks and dependency probes.

    Parameters
    ----------
    workdir : Path
        Isolated example directory.
    bindir : Path
        Target interpreter and CLI directory.

    Returns
    -------
    dict of str to str
        Allowlisted environment, with failing local proxies.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     block_env(Path(directory), docs_bindir())['PYTHONUTF8']
    '1'
    """
    home, temp = workdir / ".home", workdir / ".tmp"
    home.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(home),
        "TMPDIR": str(temp),
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "",
    }


def _execute(
    command: list[str], *, workdir: Path, env: dict[str, str], timeout_s: float
) -> tuple[int, str, str]:
    """Capture one owned session and clean its descendants on every exit path."""
    process = subprocess.Popen(
        command,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        start_new_session=True,
    )
    started = None
    try:
        started = process_start_time(process.pid)
        stdout, stderr = process.communicate(timeout=timeout_s)
        return process.returncode, stdout, stderr
    finally:
        # Identity-aware cleanup also handles descendants whose leader already exited.
        errors = reap_children([(process, True, started)])
        deadline = time.monotonic() + 10
        while group_has_live_member(process.pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        if errors or group_has_live_member(process.pid):
            raise AssertionError("docs-test owned process cleanup failed")
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def requires_available(
    extra: str, *, workdir: Path, bindir: Path, timeout_s: float = 300.0
) -> bool:
    """
    Probe all optional requirements in the target, preserving broken imports.

    Parameters
    ----------
    extra : str
        Recognized documentation extra.
    workdir : Path
        Isolated working directory.
    bindir : Path
        Target interpreter directory.
    timeout_s : float, optional
        Maximum probe execution time.

    Returns
    -------
    bool
        False only when required distribution metadata is absent.

    Raises
    ------
    AssertionError
        An installed dependency cannot be imported.

    Examples
    --------
    >>> sorted(EXTRA_IMPORTS)
    ['cache', 'langfuse', 'llm', 'ray', 's3']
    """
    deps = _EXTRA_DEPS[extra]
    distributions = tuple(DISTRIBUTIONS.get(dep, dep) for dep in deps)
    code = (
        "import importlib, importlib.metadata, importlib.util\n"
        "try:\n"
        f"    for distribution in {distributions!r}: importlib.metadata.version(distribution)\n"
        "except importlib.metadata.PackageNotFoundError:\n    raise SystemExit(77)\n"
    )
    rc, stdout, stderr = _execute(
        [str(bindir / "python"), "-c", code],
        workdir=workdir,
        env=block_env(workdir, bindir),
        timeout_s=timeout_s,
    )
    if rc == 77:
        return False
    assert rc == 0, f"docs-test metadata probe exited {rc}\n{stdout}\n{stderr}"
    code = (
        "import importlib, importlib.util\n"
        f"for module in {deps!r}:\n"
        "    if importlib.util.find_spec(module) is None: raise ModuleNotFoundError(name=module)\n"
        "    importlib.import_module(module)\n"
    )
    rc, stdout, stderr = _execute(
        [str(bindir / "python"), "-c", code],
        workdir=workdir,
        env=block_env(workdir, bindir),
        timeout_s=timeout_s,
    )
    assert rc == 0, f"docs-test dependency probe exited {rc}\n{stdout}\n{stderr}"
    return True


def block_runs(block: Block, *, workdir: Path, bindir: Path, timeout_s: float = 300.0) -> bool:
    """
    Use the runner's execution predicate for the final runnable-page audit.

    Parameters
    ----------
    block : Block
        Parsed example.
    workdir : Path
        Isolated working directory.
    bindir : Path
        Target interpreter directory.
    timeout_s : float, optional
        Maximum dependency probe execution time.

    Returns
    -------
    bool
        Whether the block executes in this target environment.

    Examples
    --------
    >>> block = parse_blocks('```bash\\ntrue\\n```')[0]
    >>> block_runs(block, workdir=Path('.'), bindir=docs_bindir())
    False
    """
    if block.kind in {"skip", "file"}:
        return False
    if block.kind == "requires":
        return requires_available(block.arg, workdir=workdir, bindir=bindir, timeout_s=timeout_s)
    return block.lang == "python" or (block.lang == "bash" and block.kind == "run")


def run_blocks(
    blocks: Sequence[Block], *, source: str, workdir: Path, bindir: Path, timeout_s: float = 300.0
) -> None:
    """
    Execute self-contained examples in document order, sharing only files.

    Parameters
    ----------
    blocks : sequence of Block
        Parsed examples.
    source : str
        Source path used for failure locations.
    workdir : Path
        Isolated working directory.
    bindir : Path
        Target interpreter directory.
    timeout_s : float, optional
        Per-block execution timeout, separate from cleanup bounds.

    Raises
    ------
    AssertionError
        An example or installed dependency fails.
    subprocess.TimeoutExpired
        An executable block exceeds its deadline; owned children are cleaned.

    Examples
    --------
    >>> run_blocks([], source='empty.md', workdir=Path('.'), bindir=docs_bindir())
    """
    for block in blocks:
        if block.kind == "file":
            path = workdir / block.arg
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(block.code, encoding="utf-8")
            continue
        if not block_runs(block, workdir=workdir, bindir=bindir, timeout_s=timeout_s):
            if block.kind == "requires":
                print(f"{source}:{block.line} not executed: missing {block.arg} extra")
            continue
        if block.lang == "python":
            path = workdir / f".docs_block_L{block.line}.py"
            path.write_text(block.code, encoding="utf-8")
            command = [str(bindir / "python"), str(path)]
        else:
            command = ["bash", "-euo", "pipefail", "-c", block.code]
        rc, stdout, stderr = _execute(
            command, workdir=workdir, env=block_env(workdir, bindir), timeout_s=timeout_s
        )
        if rc != 0:
            raise AssertionError(f"{source}:{block.line} exited {rc}\n{stdout}\n{stderr}")


def page_path(title: str) -> Path:
    """
    Find the unique active Markdown page by its first H1 title.

    Parameters
    ----------
    title : str
        Exact first heading.

    Returns
    -------
    Path
        Matching source page, excluding build and design archives.

    Examples
    --------
    >>> page_path('API reference').relative_to(REPO).as_posix()
    'docs/api/index.md'
    """
    matches = []
    for path in (REPO / "docs").rglob("*.md"):
        if {"superpowers", "_build"} & set(path.relative_to(REPO / "docs").parts):
            continue
        heading = re.search(r"^# (.+)$", path.read_text(encoding="utf-8"), re.M)
        if heading and heading.group(1) == title:
            matches.append(path)
    assert len(matches) == 1, f"Expected one docs page: {title}"
    return matches[0]


def make_workdir(tmp_path: Path, *, with_examples: bool) -> Path:
    """
    Prepare an isolated directory with optional checked-in example fixtures.

    Parameters
    ----------
    tmp_path : Path
        Test temporary directory.
    with_examples : bool
        Copy examples without caches or prior run output.

    Returns
    -------
    Path
        The work subdirectory.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     make_workdir(Path(directory), with_examples=False).name
    'work'
    """
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    if with_examples:
        shutil.copytree(
            REPO / "examples",
            workdir / "examples",
            ignore=shutil.ignore_patterns(".swarm-arena", "__pycache__"),
        )
    return workdir


def spec_picture() -> str:
    """
    Read the byte-preserved tree from the design's first fenced block.

    Returns
    -------
    str
        Tree text including its trailing newline.

    Examples
    --------
    >>> 'Solution' in spec_picture()
    True
    """
    text = (REPO / "docs/superpowers/specs/2026-10-16-swarm-arena-design.md").read_text(
        encoding="utf-8"
    )
    return parse_blocks(text)[0].code


def inventory_names(path: Path) -> set[str]:
    """
    Decode names from a Sphinx version-two inventory.

    Parameters
    ----------
    path : Path
        Compressed objects.inv file.

    Returns
    -------
    set of str
        Documented object names.

    Examples
    --------
    >>> callable(inventory_names)
    True
    """
    data = path.read_bytes().split(b"\n", 4)[4]
    return {
        line.split()[0]
        for line in zlib.decompress(data).decode("utf-8").splitlines()
        if line.strip()
    }


def toctree_titles(index: Path) -> list[str]:
    """
    Resolve a Markdown toctree's entries to their page titles.

    Parameters
    ----------
    index : Path
        Source containing a MyST toctree.

    Returns
    -------
    list of str
        First headings, in navigation order.

    Examples
    --------
    >>> toctree_titles(REPO / 'docs/index.md')[0]
    'Getting started'
    """
    text = index.read_text(encoding="utf-8")
    tree = re.search(r"^```\{toctree\}\n(.*?)^```$", text, re.M | re.S)
    assert tree is not None
    titles = []
    for line in tree.group(1).splitlines():
        entry = line.strip()
        if not entry or entry.startswith(":"):
            continue
        page = (index.parent / entry).with_suffix(".md")
        heading = re.search(r"^# (.+)$", page.read_text(encoding="utf-8"), re.M)
        assert heading is not None
        titles.append(heading.group(1))
    return titles


def required_facts(title: str, facts: Sequence[str]) -> None:
    """
    Check a page owner's exact required source substrings.

    Parameters
    ----------
    title : str
        Page H1 title.
    facts : sequence of str
        Required substrings, supplied by that page's tests.

    Examples
    --------
    >>> required_facts('API reference', ['# API reference'])
    """
    text = page_path(title).read_text(encoding="utf-8")
    assert [fact for fact in facts if fact not in text] == []


@pytest.fixture(scope="session")
def sphinx_html(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the full API once per worker; core-only lanes skip this fixture."""
    if not (deps_present("swarm_arena.llm") and deps_present("swarm_arena.ray")):
        pytest.skip("full Sphinx HTML/inventory needs the llm and ray extras")
    output = tmp_path_factory.mktemp("sphinx") / "html"
    rc, stdout, stderr = _execute(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-q",
            "-b",
            "html",
            str(REPO / "docs"),
            str(output),
        ],
        workdir=REPO,
        env=dict(os.environ),
        timeout_s=300,
    )
    assert rc == 0, stdout + stderr
    return output
