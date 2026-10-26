"""Exercise the public documentation and its offline script workflow."""

import ast
import importlib
import importlib.metadata
import inspect
import re
import runpy
import subprocess
import sys
import time
import zlib
from html.parser import HTMLParser
from pathlib import Path

import pytest
from _docs import (
    EXTRA_IMPORTS,
    PAGES,
    REPO,
    block_env,
    block_runs,
    docs_bindir,
    inventory_names,
    make_workdir,
    page_path,
    parse_blocks,
    required_facts,
    requires_available,
    run_blocks,
    spec_picture,
    sphinx_html,
    toctree_titles,
)
from _fakes.process_cleanup import owned_processes
from _surface import PUBLIC_ALL

from swarm_arena._exec.procinfo import group_has_live_member

assert sphinx_html  # Imported pytest fixtures.
assert owned_processes

ROOT = Path(__file__).resolve().parents[1]
GUIDES = sorted(
    path for path in (ROOT / "docs").glob("*.md") if path.name != "writing-a-connector.md"
)
ROADMAP = Path("docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md")
BARE_OPTIONAL_INSTALL = re.compile(
    r"\b(?:uv\s+add|(?:uv\s+)?pip(?:3)?\s+install)\s+[\"']?"
    r"swarm-arena\[(?:llm|cache|langfuse|ray)[^\]]*\](?!\s+@\s+git\+)"
)


def test_optional_install_examples_use_prerelease_uv() -> None:
    source = "git+https://github.com/shreyasvinaya/swarm-arena"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for extra in ("llm", "cache", "langfuse"):
        assert f'uv add "swarm-arena[{extra}] @ {source}"' in readme
    assert f"uv add 'swarm-arena[ray] @ {source}'" in readme
    concepts = re.sub(r"\s+", " ", (ROOT / "docs/concepts.md").read_text(encoding="utf-8"))
    for extras in ("llm", "llm,cache"):
        assert f'uv add "swarm-arena[{extras}] @ {source}"' in concepts
    stale: list[str] = []
    for directory in (ROOT / "docs", ROOT / "src"):
        for path in sorted(directory.rglob("*")):
            if path.suffix not in {".md", ".rst", ".py"}:
                continue
            if ROOT / "docs/superpowers" in path.parents:
                continue
            text = path.read_text(encoding="utf-8")
            for match in BARE_OPTIONAL_INSTALL.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                stale.append(f"{path.relative_to(ROOT)}:{line}")
    assert not stale, "Bare optional install commands: " + ", ".join(stale)


@pytest.mark.parametrize("command", ["pip install", "pip3 install", "uv pip install", "uv add"])
@pytest.mark.parametrize("quote", ["", "'", '"'])
@pytest.mark.parametrize("extras", ["ray", "llm", "llm,cache", "langfuse"])
def test_bare_optional_install_guard_matches_quoted_extras(
    command: str, quote: str, extras: str
) -> None:
    assert BARE_OPTIONAL_INSTALL.search(f"{command} {quote}swarm-arena[{extras}]{quote}")
    source = "git+https://github.com/shreyasvinaya/swarm-arena"
    assert not BARE_OPTIONAL_INSTALL.search(f'{command} "swarm-arena[{extras}] @ {source}"')


def test_ray_missing_extra_hint_uses_prerelease_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        assert name == "ray"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ModuleNotFoundError) as caught:
        runpy.run_path(str(ROOT / "src/swarm_arena/ray/_imports.py"))
    assert caught.value.name == "ray"
    assert str(caught.value) == (
        "swarm_arena.ray needs the 'ray' extra: uv add "
        '"swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena"'
    )
    assert caught.value.__cause__ is None


def test_roadmap_records_supported_llm_dependency_deviation() -> None:
    text = (ROOT / ROADMAP).read_text(encoding="utf-8")
    constraints = text.split("### R5.1 ", 1)[1].split("### R5.2 ", 1)[0]
    decisions = text.split("### R5.4 ", 1)[1]
    for section in (constraints, decisions):
        assert "openai-agents[litellm]>=0.24,<0.25" in section
        assert "litellm>=1.107.0rc2,<1.108" in section
    assert "Ruling P39" in decisions
    assert "M5" in decisions
    assert "1.107.0 final" in decisions


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("_budget.ledger", "Ledger.reserve"),
        ("_journal.views", "ViewWriter.write"),
        ("_board.cache", "ArenaCache.apply_human_objects"),
        ("_scoring.human_review", "HumanReview.export"),
        ("_scoring.human_review", "HumanReview.import_csv"),
        ("_board.curator", "Curator.curate"),
        ("_board.curator", "Curator.models"),
        ("_board.digest", "build_digest"),
        ("_board.digest", "estimate_tokens"),
        ("_board.frontier", "build_frontier"),
        ("_tooling.result_file", "read_regular_file"),
        ("_tooling.materialize", "collect_output_files"),
        ("_planning.catalog_check", "check_catalog"),
        ("_planning.catalog_check", "authorize_revision"),
    ],
)
def test_registry_admission_and_byte_budget_signatures(module: str, name: str) -> None:
    text = (ROOT / ROADMAP).read_text(encoding="utf-8")
    func = importlib.import_module(f"swarm_arena.{module}")
    for part in name.split("."):
        func = getattr(func, part)
    assert callable(func)
    declarations: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for block in re.findall(r"```python\n(.*?)```", text, re.DOTALL):
        nodes = ast.parse(block).body
        for part in name.split("."):
            selected = [node for node in nodes if getattr(node, "name", None) == part]
            if len(selected) != 1:
                break
            node = selected[0]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                declarations.append(node)
            elif isinstance(node, ast.ClassDef):
                nodes = node.body
    assert len(declarations) == 1
    declaration = declarations[0]
    arguments = declaration.args
    signature = inspect.signature(func)
    parameters = list(signature.parameters.values())
    assert [arg.arg for arg in [*arguments.args, *arguments.kwonlyargs]] == [
        param.name for param in parameters
    ]
    assert [arg.arg for arg in arguments.kwonlyargs] == [
        param.name for param in parameters if param.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        actual = signature.parameters[arg.arg].default
        assert (None if default is None else ast.unparse(default)) == (
            None if actual is inspect.Parameter.empty else repr(actual)
        )


def test_registry_describes_default_files_and_callback_ownership() -> None:
    text = (ROOT / ROADMAP).read_text(encoding="utf-8")
    snapshot = text.split("def snapshot_ref(", 1)[1].split("# _tooling/staging.py", 1)[0]
    assert "non-empty FileSet defaults under their canonical" in snapshot
    assert "Only empty FileSet defaults and other defaults stay omitted" in snapshot
    reserve = text.split("async def reserve(", 1)[1].split("async def settle(", 1)[0]
    assert "before" in reserve
    assert "durability wait" in reserve
    assert "does not authorize dispatch" in reserve
    assert "unwinds the booking exactly, writes no event" in reserve


def test_changelog_describes_one_initial_release() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.findall(r"^## .+$", text, re.MULTILINE) == ["## [0.1.0] - Unreleased"]
    assert re.findall(r"^### .+$", text, re.MULTILINE) == ["### Added"]
    assert "swarm_arena._" not in text


@pytest.mark.parametrize(
    ("relative_path", "heading"),
    [("README.md", "Quickstart"), ("docs/getting-started.md", "Install")],
)
def test_quickstarts_distinguish_release_and_source_installs(
    relative_path: str, heading: str
) -> None:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    section = text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    release, prerelease = section.split("Before the first release", 1)
    assert "For releases" in release
    assert "```sh\nuv add swarm-arena\n```" in release
    assert 'uv add "swarm-arena @ git+https://github.com/shreyasvinaya/swarm-arena"' in prerelease
    assert (
        "git clone https://github.com/shreyasvinaya/swarm-arena && cd swarm-arena && uv sync"
        in prerelease
    )


@pytest.mark.parametrize("path", GUIDES, ids=lambda path: path.name)
def test_guides_use_public_interfaces_without_stub_markers(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"swarm_arena\._|_human_only|\bMAX_[A-Z_]+\b|\bPyTool\b", text)
    assert not re.search(r"Written in M\d", text)


@pytest.mark.parametrize("name", ["concepts.md", "own-files.md"])
def test_script_guides_explain_import_shadowing(name: str) -> None:
    text = (ROOT / "docs" / name).read_text(encoding="utf-8")
    assert "import shadowing" in text.lower()
    assert "problem/*.physk" in text


def test_judge_result_guide_rejects_unknown_fields() -> None:
    text = (ROOT / "docs" / "own-files.md").read_text(encoding="utf-8")
    result = text.split("Judges likewise write a raw object:", 1)[1].split(
        "(isolation-boundary)=", 1
    )[0]
    assert "unknown fields are rejected" in result
    assert "extra fields are ignored" not in result


class RenderedText(HTMLParser):
    """Read displayed HTML text without scanning styles, URLs or element attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self.chunks.append(data)


def test_api_html_hides_model_implementation_details(sphinx_html: Path) -> None:
    output = sphinx_html
    pages = sorted((output / "api").rglob("*.html"))
    leaks: list[str] = []
    assert pages
    for path in pages:
        html = path.read_text(encoding="utf-8")
        assert not re.search(r'id="[^"]+\.(model_config|model_post_init|model_fields)"', html)
        assert "_Pydantic" not in html, path.name
        rendered = RenderedText()
        rendered.feed(html)
        text = "".join(rendered.chunks)
        assert ".. py:" not in text, path.name
        assert ".. automodule::" not in text, path.name
        if "swarm_arena._" in html or "swarm_arena._" in text:
            leaks.append(f"{path.name}: private import path")
        names = sorted(set(re.findall(r"(?<![\w])_(?!_)[A-Za-z]\w*", text)))
        if names:
            leaks.append(f"{path.name}: {names}")
        if "BEFORE_KILL_TIMEOUT_S" in text:
            leaks.append(f"{path.name}: private timeout constant")
    assert not leaks, "\n".join(leaks)
    root_api = (output / "api" / "swarm_arena.html").read_text(encoding="utf-8")
    assert 'id="swarm_arena.Goal"' in root_api
    assert 'id="swarm_arena.Plan.loads"' in root_api
    assert "Parameters" in root_api
    llm = (output / "api" / "llm.html").read_text(encoding="utf-8")
    assert 'id="swarm_arena.llm.AgentsSDKManager"' in llm
    testing = (output / "api" / "testing.html").read_text(encoding="utf-8")
    assert 'id="swarm_arena.testing.FakeModel"' in testing
    protocol = RenderedText()
    protocol.feed((output / "api" / "protocol.html").read_text(encoding="utf-8"))
    assert "data=..." in "".join(protocol.chunks).replace(" ", "")
    assert "omission, not a content value" in "".join(protocol.chunks)
    board = (output / "api" / "board.html").read_text(encoding="utf-8")
    for cls in ("Board", "BoardView"):
        for name in (
            "best",
            "learnings",
            "search_learnings",
            "get_learning",
            "get_candidate",
            "scores",
            "lineage",
            "leaderboard",
        ):
            assert f'id="swarm_arena.board.{cls}.{name}"' in board


def test_own_files_offline_example_runs(tmp_path: Path) -> None:
    text = (ROOT / "docs" / "own-files.md").read_text(encoding="utf-8")
    section = text.split("## Try it offline\n", 1)
    assert len(section) == 2, "The script guide needs a runnable offline workflow"
    example = section[1].split("\n## ", 1)[0]
    files = re.findall(r"### `([^`]+\.py)`\n.*?```python\n(.*?)```", example, re.DOTALL)
    assert {name for name, _ in files} == {
        "problem/run.py",
        "checks/check.py",
        "judge/score.py",
        "try_scripts.py",
    }
    for name, source in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "try_scripts.py"],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "ok: loss=0.25, check=True, key=(-0.25,)"
    assert done.stderr == ""


@pytest.mark.parametrize(
    "module",
    [
        "_errors.py",
        "_agent/runtime.py",
        "_budget/tranche.py",
        "_cli/run_cmds.py",
        "_journal/views.py",
        "_model/safe.py",
    ],
)
def test_registry_marks_inherited_m2_modules(module: str) -> None:
    text = (ROOT / ROADMAP).read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith(f"| `{module}` |"))
    assert "+M2" in row


def test_registry_includes_shared_human_operations() -> None:
    text = (ROOT / ROADMAP).read_text(encoding="utf-8")
    assert "| `_board/human_ops.py` |" in text
    assert "class Curator(Configurable, ABC):" in text


@pytest.mark.parametrize(
    ("event", "keys"),
    [
        ("round_started", {"round", "shares", "converge"}),
        ("decision", {"round", "decision"}),
        ("converge_started", {"reason", "select_only"}),
        ("curated", {"round", "revision", "decision_seq", "summaries", "supersedes", "errors"}),
        ("review_imported", {"reviewer", "scores", "objects"}),
    ],
)
def test_registry_documents_m2_payload_keys(event: str, keys: set[str]) -> None:
    text = (ROOT / ROADMAP).read_text(encoding="utf-8").split("### R4.14 Journal event types", 1)[1]
    row = next((line for line in text.splitlines() if line.startswith(f"| `{event}` |")), "")
    assert keys <= set(re.findall(r"`([a-z_]+)`", row))


def test_registry_documents_page_closure_position() -> None:
    text = (ROOT / ROADMAP).read_text(encoding="utf-8")
    assert "closed_seq" in text


def test_arena_guide_describes_available_cli() -> None:
    text = (ROOT / "docs/arena.md").read_text(encoding="utf-8")
    assert "CLI integration is delivered separately" not in text


def test_changelog_has_no_superseded_board_or_fix_round_details() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "an in-memory run board" not in text
    assert "Hide temporary and lock basenames" not in text
    assert "Preserve valid raw key characters" not in text
    assert text.count("Configured-board lifecycle") + text.count("Configured-board open") <= 1


def test_registry_collision_contract_uses_recorded_frontier() -> None:
    text = (ROOT / ROADMAP).read_text(encoding="utf-8")
    assert "skips an item whose id it already holds from another run" not in text
    assert "lexicographically smallest visible producer key" in text
    assert "`run:<run_id>`" in text
    assert "`object:<object_name>`" in text


def test_source_has_no_dispatch_or_fix_round_markers() -> None:
    marker = re.compile(r"\b(?:task|wave)\s+\d+|\b(?:fix|review)[ -]round\b|__all__\s*\+=", re.I)
    assert not [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*.py")
        if marker.search(path.read_text(encoding="utf-8"))
    ]


SAMPLE = """\
```python
print(1)
```
<!-- docs-test: skip needs a key -->
```python
print(2)
```
<!-- docs-test: file m.py -->
```python
X = 41
```
<!-- docs-test: run -->
```bash
echo ok
```
```{literalinclude} ../examples/x.py
```
"""


def test_parse_blocks_reads_markers() -> None:
    blocks = parse_blocks(SAMPLE)
    assert [(b.lang, b.kind, b.arg) for b in blocks] == [
        ("python", "none", ""),
        ("python", "skip", "needs a key"),
        ("python", "file", "m.py"),
        ("bash", "run", ""),
    ]
    assert [b.line for b in blocks] == [1, 5, 9, 13]
    assert blocks[0].code == "print(1)\n"


@pytest.mark.parametrize(
    "text",
    [
        "<!-- docs-test: skip -->\n```python\npass\n```",
        "<!-- docs-test: maybe -->\n```python\npass\n```",
        " <!-- docs-test: skip reason -->\n```python\npass\n```",
        "<!-- docs-test: requires gpu -->\n```python\npass\n```",
        "<!-- docs-test: requires -->\n```python\npass\n```",
        "<!-- docs-test: file -->\n```python\npass\n```",
        "<!-- docs-test: file ../x.py -->\n```python\npass\n```",
        "<!-- docs-test: file /tmp/x.py -->\n```python\npass\n```",
        "<!-- docs-test: file a/../x.py -->\n```python\npass\n```",
        "<!-- docs-test: run argument -->\n```bash\ntrue\n```",
        "<!-- docs-test: run -->\n```python\npass\n```",
        "<!-- docs-test: skip reason -->\n```bash\ntrue\n```",
        "<!-- docs-test: requires llm -->\n```bash\ntrue\n```",
        "<!-- docs-test: skip reason -->\n\n```python\npass\n```",
        "<!-- docs-test: skip reason -->",
        "<!-- docs-test: skip reason -->\n```{literalinclude} a.py\n```",
    ],
)
def test_parse_blocks_rejects_bad_markers(text: str) -> None:
    with pytest.raises(ValueError, match=r"docs-test.*line 2"):
        parse_blocks("heading\n" + text)


def test_parse_blocks_honors_long_fences_and_ignores_marker_content() -> None:
    text = "````text extra\n```\n<!-- docs-test: maybe -->\n````\n"
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].code == "```\n<!-- docs-test: maybe -->\n"
    assert parse_blocks(" ```python\npass\n ```\n") == []


def test_run_blocks_shares_files_between_blocks(tmp_path: Path) -> None:
    blocks = parse_blocks(
        "<!-- docs-test: file nested/m.py -->\n```python\nX = 41\n```\n"
        "```python\nfrom nested import m\nassert m.X + 1 == 42\n```\n"
        "<!-- docs-test: run -->\n```bash\ntest -f nested/m.py\n```\n"
    )
    run_blocks(
        blocks,
        source="t.md",
        workdir=make_workdir(tmp_path, with_examples=False),
        bindir=docs_bindir(),
    )


def test_run_blocks_reports_failing_line(tmp_path: Path) -> None:
    blocks = parse_blocks('text\n```python\nprint("out")\nraise SystemExit(3)\n```\n')
    with pytest.raises(AssertionError, match=r"t\.md:2 exited 3\nout"):
        run_blocks(
            blocks,
            source="t.md",
            workdir=make_workdir(tmp_path, with_examples=False),
            bindir=docs_bindir(),
        )


def test_run_blocks_do_not_share_python_globals(tmp_path: Path) -> None:
    blocks = parse_blocks("```python\nx = 1\n```\n```python\nassert x == 1\n```\n")
    with pytest.raises(AssertionError, match="NameError"):
        run_blocks(
            blocks,
            source="t.md",
            workdir=make_workdir(tmp_path, with_examples=False),
            bindir=docs_bindir(),
        )


def test_blocks_get_no_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "PYTHONPATH", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, "sentinel-secret")
    workdir = make_workdir(tmp_path, with_examples=False)
    env = block_env(workdir, docs_bindir())
    assert set(env) == {
        "PATH",
        "HOME",
        "TMPDIR",
        "PYTHONUTF8",
        "PYTHONDONTWRITEBYTECODE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    }
    assert Path(env["HOME"]).is_dir()
    assert Path(env["TMPDIR"]).is_dir()
    code = (
        'import os\nassert "sentinel-secret" not in repr(dict(os.environ))\n'
        'assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:9"\n'
    )
    run_blocks(
        parse_blocks(f"```python\n{code}```\n"),
        source="t.md",
        workdir=workdir,
        bindir=docs_bindir(),
    )


def test_docs_bindir_selects_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SWARM_ARENA_DOCS_VENV", raising=False)
    assert docs_bindir() == Path(sys.executable).parent
    monkeypatch.setenv("SWARM_ARENA_DOCS_VENV", str(tmp_path))
    assert docs_bindir() == tmp_path / "bin"


@pytest.fixture
def target_bin(tmp_path: Path) -> Path:
    """Create a target interpreter with no optional packages."""
    done = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(tmp_path / "venv")],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return tmp_path / "venv" / "bin"


def test_target_interpreter_executes_blocks(target_bin: Path, tmp_path: Path) -> None:
    code = f"import sys\nassert sys.prefix == {str(target_bin.parent)!r}\n"
    run_blocks(
        parse_blocks(f"```python\n{code}```"),
        source="t.md",
        workdir=make_workdir(tmp_path, with_examples=False),
        bindir=target_bin,
    )


@pytest.mark.parametrize("extra", list(EXTRA_IMPORTS))
def test_requires_probe_uses_target_environment(
    extra: str, target_bin: Path, tmp_path: Path
) -> None:
    workdir = make_workdir(tmp_path, with_examples=False)
    assert not requires_available(extra, workdir=workdir, bindir=target_bin)
    block = parse_blocks(
        f"<!-- docs-test: requires {extra} -->\n```python\nraise AssertionError\n```"
    )[0]
    assert not block_runs(block, workdir=workdir, bindir=target_bin)
    run_blocks([block], source="t.md", workdir=workdir, bindir=target_bin)


@pytest.mark.parametrize("extra", ["llm", "cache", "langfuse"])
@pytest.mark.parametrize("missing", ["agents", "litellm"])
def test_requires_llm_needs_both_distributions(
    extra: str, missing: str, target_bin: Path, tmp_path: Path
) -> None:
    site = next((target_bin.parent / "lib").glob("python*/site-packages"))
    present = {"agents": "openai_agents", "litellm": "litellm", "langfuse": "langfuse"}
    for module, distribution in present.items():
        (site / f"{module}.py").write_text("", encoding="utf-8")
        if module != missing:
            metadata = site / f"{distribution}-1.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(
                f"Name: {distribution}\nVersion: 1.0\n", encoding="utf-8"
            )
    assert not requires_available(
        extra, workdir=make_workdir(tmp_path, with_examples=False), bindir=target_bin
    )


@pytest.mark.parametrize("extra", ["llm", "cache", "langfuse"])
@pytest.mark.parametrize(
    ("broken", "message"),
    [
        ('raise ImportError("broken vendor module")\n', "ImportError: broken vendor module"),
        ("import docs_missing_vendor_submodule\n", "ModuleNotFoundError: No module named"),
        ("raise SystemExit(77)\n", "dependency probe exited 77"),
    ],
)
def test_requires_runs_available_and_preserves_broken_imports(
    extra: str, broken: str, message: str, target_bin: Path, tmp_path: Path
) -> None:
    site = next((target_bin.parent / "lib").glob("python*/site-packages"))
    for module, distribution in {
        "agents": "openai_agents",
        "litellm": "litellm",
        "langfuse": "langfuse",
    }.items():
        (site / f"{module}.py").write_text("", encoding="utf-8")
        metadata = site / f"{distribution}-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Name: {distribution}\nVersion: 1.0\n", encoding="utf-8"
        )
    workdir = make_workdir(tmp_path, with_examples=False)
    block = parse_blocks(
        f"<!-- docs-test: requires {extra} -->\n```python\n"
        'from pathlib import Path\nPath("ran").touch()\n```'
    )[0]
    assert block_runs(block, workdir=workdir, bindir=target_bin)
    run_blocks([block], source="t.md", workdir=workdir, bindir=target_bin)
    assert (workdir / "ran").exists()
    (site / "agents.py").write_text(broken, encoding="utf-8")
    with pytest.raises(AssertionError, match=message):
        run_blocks([block], source="t.md", workdir=workdir, bindir=target_bin)


def test_execution_predicate_excludes_ignored_blocks(tmp_path: Path) -> None:
    workdir = make_workdir(tmp_path, with_examples=False)
    assert [block_runs(b, workdir=workdir, bindir=docs_bindir()) for b in parse_blocks(SAMPLE)] == [
        True,
        False,
        False,
        True,
    ]
    assert not block_runs(
        parse_blocks("```bash\nfalse\n```")[0], workdir=workdir, bindir=docs_bindir()
    )


@pytest.mark.usefixtures("owned_processes")
@pytest.mark.parametrize("outcome", ["timeout", "failure", "success", "interrupt"])
def test_run_blocks_reaps_driver_and_descendant(
    outcome: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import _docs

    workdir = make_workdir(tmp_path, with_examples=False)
    # The child signals readiness before the driver can exit/fail. No elapsed-time oracle.
    code = (
        "import os, subprocess, sys, time\nfrom pathlib import Path\n"
        'Path("driver").write_text(str(os.getpid()))\n'
        'child = subprocess.Popen([sys.executable, "-c", '
        "'import os, time; from pathlib import Path; "
        'Path("child").write_text(str(os.getpid())); time.sleep(120)\'], '
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        'while not Path("child").exists(): time.sleep(0.01)\n'
    )
    code += (
        "time.sleep(120)\n"
        if outcome in {"timeout", "interrupt"}
        else f"raise SystemExit({3 if outcome == 'failure' else 0})\n"
    )
    blocks = parse_blocks(f"```python\n{code}```")
    if outcome == "interrupt":
        communicate = subprocess.Popen.communicate
        interrupted = False

        def interrupt_once(
            self: subprocess.Popen[str], input: str | None = None, timeout: float | None = None
        ) -> tuple[str, str]:
            nonlocal interrupted
            if (
                not interrupted
                and isinstance(self.args, list)
                and self.args[0] == str(docs_bindir() / "python")
            ):
                deadline = time.monotonic() + 30
                while not (workdir / "child").exists():
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                interrupted = True
                raise KeyboardInterrupt
            return communicate(self, input=input, timeout=timeout)

        monkeypatch.setattr(_docs.subprocess.Popen, "communicate", interrupt_once)
    expected = {
        "timeout": subprocess.TimeoutExpired,
        "failure": AssertionError,
        "interrupt": KeyboardInterrupt,
    }
    if outcome in expected:
        with pytest.raises(expected[outcome]):
            run_blocks(
                blocks,
                source="t.md",
                workdir=workdir,
                bindir=docs_bindir(),
                timeout_s=3 if outcome == "timeout" else 30,
            )
    else:
        run_blocks(blocks, source="t.md", workdir=workdir, bindir=docs_bindir())
    assert (workdir / "child").exists()
    pgid = int((workdir / "driver").read_text(encoding="utf-8"))
    assert not group_has_live_member(pgid)


def test_shared_lookup_helpers(tmp_path: Path) -> None:
    assert page_path("Getting started") == REPO / "docs/getting-started.md"
    with pytest.raises(AssertionError):
        page_path("No such page")
    required_facts("Getting started", ["## Install"])
    with pytest.raises(AssertionError):
        required_facts("Getting started", ["absent required phrase"])
    workdir = make_workdir(tmp_path, with_examples=True)
    assert (workdir / "examples/rosenbrock/plan.toml").is_file()
    assert not list(workdir.rglob("__pycache__"))
    inv = tmp_path / "objects.inv"
    inv.write_bytes(
        b"# Sphinx inventory version 2\n# Project: test\n# Version: 1\n# zlib\n"
        + zlib.compress(b"swarm_arena.Goal py:class 1 api.html#$ -\n")
    )
    assert inventory_names(inv) == {"swarm_arena.Goal"}


def test_index_lists_the_12_pages_in_order() -> None:
    assert toctree_titles(REPO / "docs/index.md") == [title for title, _ in PAGES]


def test_index_shows_the_picture() -> None:
    assert spec_picture() in (REPO / "docs/index.md").read_text(encoding="utf-8")


def test_conf_settings() -> None:
    import swarm_arena

    conf = runpy.run_path(str(REPO / "docs/conf.py"))
    assert {"_build", "superpowers"} <= set(conf["exclude_patterns"])
    assert {"sphinx.ext.autodoc", "numpydoc", "myst_parser"} <= set(conf["extensions"])
    assert conf["project"] == "swarm-arena"
    assert conf["author"] == "Shreyas Vinaya"
    assert conf["copyright"] == "2026, Shreyas Vinaya"
    assert conf["html_theme"] == "sphinx_rtd_theme"
    assert conf["autodoc_member_order"] == "bysource"
    assert conf["numpydoc_show_class_members"] is False
    assert conf["release"] == swarm_arena.__version__
    assert conf["version"] == ".".join(swarm_arena.__version__.split(".")[:2])


def test_api_reference_has_one_page_per_public_module() -> None:
    pages = list((REPO / "docs/api").glob("*.md"))
    assert toctree_titles(REPO / "docs/api/index.md") == list(PUBLIC_ALL)
    for module in PUBLIC_ALL:
        pattern = re.compile(
            rf"^```\{{eval-rst\}}\n\.\. automodule:: {re.escape(module)}\n +:members:.*$", re.M
        )
        assert sum(bool(pattern.search(p.read_text(encoding="utf-8"))) for p in pages) == 1, module


@pytest.mark.slow
def test_api_reference_documents_every_public_name(sphinx_html: Path) -> None:
    names = inventory_names(sphinx_html / "objects.inv")
    assert [f"{m}.{n}" for m, ns in PUBLIC_ALL.items() for n in ns if f"{m}.{n}" not in names] == []
    assert sorted(n for n in names if "._" in n) == []


@pytest.mark.parametrize(("title", "pid"), PAGES, ids=[pid for _, pid in PAGES])
def test_page_blocks_run(title: str, pid: str, tmp_path: Path) -> None:
    path = page_path(title)
    run_blocks(
        parse_blocks(path.read_text(encoding="utf-8")),
        source=str(path.relative_to(REPO)),
        workdir=make_workdir(tmp_path, with_examples=True),
        bindir=docs_bindir(),
    )
