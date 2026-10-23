"""Exercise the public documentation and its offline script workflow."""

import ast
import importlib
import importlib.metadata
import inspect
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDES = sorted(
    path for path in (ROOT / "docs").glob("*.md") if path.name != "writing-a-connector.md"
)
ROADMAP = Path("docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md")


def test_optional_install_examples_use_prerelease_uv() -> None:
    source = "git+https://github.com/shreyasvinaya/swarm-arena"
    readme = (ROOT / "README.md").read_text()
    for extra in ("llm", "cache", "langfuse"):
        assert f'uv add "swarm-arena[{extra}] @ {source}"' in readme
    concepts = re.sub(r"\s+", " ", (ROOT / "docs/concepts.md").read_text())
    for extras in ("llm", "llm,cache"):
        assert f'uv add "swarm-arena[{extras}] @ {source}"' in concepts


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
    text = (ROOT / ROADMAP).read_text()
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
    text = (ROOT / ROADMAP).read_text()
    snapshot = text.split("def snapshot_ref(", 1)[1].split("# _tooling/staging.py", 1)[0]
    assert "non-empty FileSet defaults under their canonical" in snapshot
    assert "Only empty FileSet defaults and other defaults stay omitted" in snapshot
    reserve = text.split("async def reserve(", 1)[1].split("async def settle(", 1)[0]
    assert "before" in reserve
    assert "durability wait" in reserve
    assert "does not authorize dispatch" in reserve
    assert "unwinds the booking exactly, writes no event" in reserve


def test_changelog_describes_one_initial_release() -> None:
    text = (ROOT / "CHANGELOG.md").read_text()
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
    text = (ROOT / relative_path).read_text()
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
    text = path.read_text()
    assert not re.search(r"swarm_arena\._|_human_only|\bMAX_[A-Z_]+\b|\bPyTool\b", text)
    assert not re.search(r"Written in M\d", text)


@pytest.mark.parametrize("name", ["concepts.md", "own-files.md"])
def test_script_guides_explain_import_shadowing(name: str) -> None:
    text = (ROOT / "docs" / name).read_text()
    assert "import shadowing" in text.lower()
    assert "problem/*.physk" in text


def test_judge_result_guide_rejects_unknown_fields() -> None:
    text = (ROOT / "docs" / "own-files.md").read_text()
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


def test_api_html_hides_model_implementation_details(tmp_path: Path) -> None:
    for distribution in ("openai-agents", "litellm"):
        try:
            importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("Full Sphinx API docs require the 'llm' extra: swarm-arena[llm]")
    output = tmp_path / "html"
    done = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", str(ROOT / "docs"), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    pages = sorted((output / "api").rglob("*.html"))
    leaks: list[str] = []
    assert pages
    for path in pages:
        html = path.read_text()
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
    root_api = (output / "api" / "swarm_arena.html").read_text()
    assert 'id="swarm_arena.Goal"' in root_api
    assert 'id="swarm_arena.Plan.loads"' in root_api
    assert "Parameters" in root_api
    llm = (output / "api" / "llm.html").read_text()
    assert 'id="swarm_arena.llm.AgentsSDKManager"' in llm
    testing = (output / "api" / "testing.html").read_text()
    assert 'id="swarm_arena.testing.FakeModel"' in testing
    protocol = RenderedText()
    protocol.feed((output / "api" / "protocol.html").read_text())
    assert "data=..." in "".join(protocol.chunks).replace(" ", "")
    assert "omission, not a content value" in "".join(protocol.chunks)
    board = (output / "api" / "board.html").read_text()
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
    text = (ROOT / "docs" / "own-files.md").read_text()
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
        path.write_text(source)
    done = subprocess.run(
        [sys.executable, "try_scripts.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
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
    text = (ROOT / ROADMAP).read_text()
    row = next(line for line in text.splitlines() if line.startswith(f"| `{module}` |"))
    assert "+M2" in row


def test_registry_includes_shared_human_operations() -> None:
    text = (ROOT / ROADMAP).read_text()
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
    text = (ROOT / ROADMAP).read_text().split("### R4.14 Journal event types", 1)[1]
    row = next((line for line in text.splitlines() if line.startswith(f"| `{event}` |")), "")
    assert keys <= set(re.findall(r"`([a-z_]+)`", row))


def test_registry_documents_page_closure_position() -> None:
    text = (ROOT / ROADMAP).read_text()
    assert "closed_seq" in text


def test_arena_guide_describes_available_cli() -> None:
    text = (ROOT / "docs/arena.md").read_text()
    assert "CLI integration is delivered separately" not in text


def test_changelog_has_no_superseded_board_or_fix_round_details() -> None:
    text = (ROOT / "CHANGELOG.md").read_text()
    assert "an in-memory run board" not in text
    assert "Hide temporary and lock basenames" not in text
    assert "Preserve valid raw key characters" not in text
    assert text.count("Configured-board lifecycle") + text.count("Configured-board open") <= 1


def test_registry_collision_contract_uses_recorded_frontier() -> None:
    text = (ROOT / ROADMAP).read_text()
    assert "skips an item whose id it already holds from another run" not in text
    assert "lexicographically smallest visible producer key" in text
    assert "`run:<run_id>`" in text
    assert "`object:<object_name>`" in text


def test_source_has_no_dispatch_or_fix_round_markers() -> None:
    marker = re.compile(r"\b(?:task|wave)\s+\d+|\b(?:fix|review)[ -]round\b|__all__\s*\+=", re.I)
    assert not [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*.py")
        if marker.search(path.read_text())
    ]
