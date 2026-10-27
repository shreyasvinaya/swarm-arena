"""Exercise the public documentation and its offline script workflow."""

import ast
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import runpy
import shlex
import signal
import subprocess
import sys
import time
import tomllib
import zlib
from contextlib import suppress
from html.parser import HTMLParser
from pathlib import Path

import pytest
from _docs import (
    EXTRA_IMPORTS,
    PAGES,
    REPO,
    STUB_TEXT,
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
    run_command,
    spec_picture,
    sphinx_html,
    toctree_titles,
)
from _fakes.process_cleanup import owned_processes
from _surface import PUBLIC_ALL

from swarm_arena._exec.procinfo import group_has_live_member, process_start_time

assert sphinx_html  # Imported pytest fixtures.
assert owned_processes

ROOT = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
GUIDES = sorted(
    path for path in (ROOT / "docs").glob("*.md") if path.name != "writing-a-connector.md"
)
ROADMAP = Path("docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md")
INSTALL_COMMAND = re.compile(
    r"(?P<command>uv\s+add|(?:uv\s+)?pip3?\s+install|--pip)\s+"
    r"(?P<upgrade>--upgrade\s+)?(?P<quote>[\"']?)"
    r"(?P<requirement>swarm-arena(?:\[[^\]]+\])?(?:\s*@\s*[^\s\"'`]+)?)"
    r"(?P=quote)"
)
INSTALL_GUIDANCE = "https://github.com/shreyasvinaya/swarm-arena#install"
UV_INSTALL = "uv init swarm-demo\ncd swarm-demo\nuv add swarm-arena\nsource .venv/bin/activate\n"
VENV_INSTALL = "python -m venv .venv\nsource .venv/bin/activate\npip install swarm-arena\n"
FALLBACK = (
    "If 0.1.0 is not on PyPI yet, use "
    '`uv add "swarm-arena @ git+https://github.com/shreyasvinaya/swarm-arena"`'
    " (add the desired extras inside the quoted requirement)."
)


def _install_violations(text: str, *, python: bool = False) -> list[str]:
    # AST constants fold adjacent string literals, including multiline hints.
    if python:
        text = "\n".join(
            node.value
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    violations = []
    for match in INSTALL_COMMAND.finditer(text):
        requirement = match["requirement"]
        command = " ".join(match["command"].split())
        if (
            command not in {"uv add", "--pip"}
            or "@" in requirement
            or ("[" in requirement and match["quote"] != '"')
            or ("[" not in requirement and bool(match["quote"]))
        ):
            violations.append(match.group())
    # Catch old requirements even without a command in the same literal.
    if "git+https://github.com/shreyasvinaya/swarm-arena" in text:
        violations.append("GitHub requirement")
    # Only the explicitly labeled contributor setup may install a local checkout.
    local_text = (
        re.sub(
            r"(?ms)^Contributors can clone the source and set up its environment:\n\n"
            r"```sh\n.*?^```$",
            "",
            text,
        )
        if not python
        else text
    )
    for match in re.finditer(
        r"\b(?:uv\s+add|(?:uv\s+)?pip3?\s+install)\s+(?P<args>[^\n`]+)", local_text
    ):
        try:
            arguments = shlex.split(match["args"])
        except ValueError:
            violations.append("Malformed install command")
            continue
        takes_value = {"--python", "-p", "--index-url", "--extra-index-url", "--find-links", "-f"}
        arguments = iter(arguments)
        for argument in arguments:
            if argument in takes_value:
                next(arguments, None)  # an option's value is not a requirement
                continue
            requirement = argument.rstrip(";,")
            if (
                requirement in {"-e", "--editable"}
                or requirement.startswith(("--editable=", ".", "/", "~"))
                or "file://" in requirement
            ):
                violations.append(match.group())
                break
    return violations


def test_install_examples_and_hints_use_one_release_form() -> None:
    readme = README.read_text(encoding="utf-8")
    assert readme.count(FALLBACK) == 1
    for extra in ("llm", "cache", "ray", "s3", "langfuse"):
        assert f'uv add "swarm-arena[{extra}]"' in readme
    paths = [
        README,
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "docs").rglob("*.rst"),
        *(ROOT / "src").rglob("*.py"),
    ]
    stale: dict[str, list[str]] = {}
    for path in paths:
        if {"superpowers", "_build"} & set(path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8")
        if path == README:
            text = text.replace(FALLBACK, "", 1)
        if path in {README, ROOT / "docs/getting-started.md"}:
            # Only these complete, activated-venv instructions may use plain pip.
            text = text.replace(f"```sh\n{VENV_INSTALL}```", "", 1)
        violations = _install_violations(text, python=path.suffix == ".py")
        if violations:
            stale[str(path.relative_to(ROOT))] = violations
    assert not stale, stale


@pytest.mark.parametrize(
    "extras", ["", "llm", "cache", "ray", "s3", "langfuse", "llm,cache", "ray,llm"]
)
@pytest.mark.parametrize("quote", ["", "'", '"'])
@pytest.mark.parametrize("command", ["uv add", "pip install", "pip3 install", "uv pip install"])
def test_install_guard_rejects_mixed_forms(command: str, quote: str, extras: str) -> None:
    requirement = "swarm-arena" + (f"[{extras}]" if extras else "")
    source = "git+https://github.com/shreyasvinaya/swarm-arena"
    assert _install_violations(f"{command} {quote}{requirement} @ {source}{quote}")
    assert _install_violations(f"{command} {quote}{requirement}@{source}{quote}")
    assert _install_violations(f"{command} {quote}{source}{quote}")
    violations = _install_violations(f"{command} {quote}{requirement}{quote}")
    valid = command == "uv add" and quote == ('"' if extras else "")
    assert bool(violations) is not valid


@pytest.mark.parametrize(
    "command",
    [
        "uv add --editable /absolute/path/to/agent-swarm",
        'uv add --editable "/absolute/path/to/agent-swarm"',
        "uv add -e ../swarm-arena",
        "uv add /absolute/path/to/agent-swarm",
        'uv add "../swarm-arena"',
        "uv add ./agent-swarm",
        "uv add swarm-arena@file:///absolute/path/to/agent-swarm",
        "uv add --editable .",
        "uv pip install -e .",
        "uv add -e .[ray]",
        'uv add --editable ".[ray]"',
        'uv pip install -e ".[dev]"',
        "uv add --editable ~/src/arena",
        "uv add --editable /home/me/code/swarm-arena-main",
        "uv add ~/src/arena",
        "uv add .[ray]",
    ],
)
def test_install_guard_rejects_local_package_installs(command: str) -> None:
    assert _install_violations(command)
    assert _install_violations(f"uv add swarm-arena\n{command}")


def test_install_guard_allows_option_values_and_a_built_wheel() -> None:
    command = 'uv pip install --python "$VENV/bin/python" dist/swarm_arena-0.1.0-py3-none-any.whl'
    assert _install_violations(command) == []


def test_install_guard_limits_local_exception_to_contributor_setup() -> None:
    setup = (
        "Contributors can clone the source and set up its environment:\n\n"
        "```sh\nuv add --editable /absolute/path/to/agent-swarm\n```"
    )
    assert not _install_violations(setup)
    assert _install_violations(setup + "\nuv add --editable /absolute/path/to/agent-swarm")
    assert _install_violations(setup.replace("Contributors", "Users"))


@pytest.mark.parametrize("upgrade", ["", "--upgrade "])
def test_install_guard_accepts_release_and_upgrade_forms(upgrade: str) -> None:
    assert not _install_violations(f'uv add {upgrade}"swarm-arena[s3]"')
    assert not _install_violations('--pip "swarm-arena[ray]"')


def test_install_guard_folds_multiline_python_literals() -> None:
    source = "git+https://github.com/shreyasvinaya/swarm-arena"
    stale = f"HINT = ('uv add ' \n    '\"swarm-arena[llm,cache] @ {source}\"')"
    assert _install_violations(stale, python=True)
    assert not _install_violations(stale.replace(f" @ {source}", ""), python=True)
    assert _install_violations('uv add\n"swarm-arena[ray] @ ' + source + '"')
    assert _install_violations('uv add "swarm-arena[llm]"; pip install swarm-arena')


def test_ray_missing_extra_hint_uses_release_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        assert name == "ray"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ModuleNotFoundError) as caught:
        runpy.run_path(str(ROOT / "src/swarm_arena/ray/_imports.py"))
    assert caught.value.name == "ray"
    assert str(caught.value) == (
        "swarm_arena.ray needs the 'ray' extra: uv add "
        '"swarm-arena[ray]"; see https://github.com/shreyasvinaya/swarm-arena#install'
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
    _assert_changelog_overview(text)


def _assert_changelog_overview(text: str) -> None:
    headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
    assert headings[0] == "[Unreleased]"
    assert re.fullmatch(r"\[\d+\.\d+\.\d+\] - \d{4}-\d{2}-\d{2}", headings[1])
    newest = text.split(f"## {headings[1]}", 1)[1].split("\n## ", 1)[0]
    assert "### Added" in newest
    assert "swarm_arena._" not in text


def test_changelog_overview_accepts_later_releases() -> None:
    _assert_changelog_overview(
        "## [Unreleased]\n\n## [0.1.1] - 2026-10-28\n\n### Added\n- New release.\n"
        "\n## [0.1.0] - 2026-10-27\n\n### Added\n- Initial release.\n"
    )


@pytest.mark.parametrize(
    ("relative_path", "heading"),
    [("README.md", "Install"), ("docs/getting-started.md", "Install")],
)
def test_quickstarts_distinguish_release_and_source_installs(
    relative_path: str, heading: str
) -> None:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    section = text.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    assert f"```sh\n{UV_INSTALL}```" in section
    assert f"```sh\n{VENV_INSTALL}```" in section
    commands = "\n".join(block.code for block in parse_blocks(text))
    steps = [
        UV_INSTALL,
        VENV_INSTALL,
        "mkdir hello-world\ncd hello-world",
        "swarm-arena run --yes run.toml plan.toml",
    ]
    positions = [commands.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "same activated shell" in text
    if relative_path == "README.md":
        assert FALLBACK in section
        assert "name-reservation placeholder" in section
    else:
        assert INSTALL_GUIDANCE in section
        assert "until 0.1.0 is on PyPI" in section
    assert (
        "git clone https://github.com/shreyasvinaya/swarm-arena && cd swarm-arena && uv sync"
        in section
    )


def test_readme_quickstart(tmp_path: Path) -> None:
    work = make_workdir(tmp_path, with_examples=False)
    assert list(work.iterdir()) == []
    text = README.read_text(encoding="utf-8")
    assert re.findall(r"^### (.+)$", text, re.M) == [
        "Run hello-world",
        "View results",
        "Share a review",
    ]
    blocks = parse_blocks(text)
    assert [block for block in blocks if block.kind in {"requires", "skip"}] == []
    assert [(block.lang, block.arg) for block in blocks if block.kind == "file"] == [
        ("toml", "run.toml"),
        ("toml", "plan.toml"),
    ]
    commands = "\n".join(
        block.code for block in blocks if block_runs(block, workdir=work, bindir=docs_bindir())
    )
    expected = [
        "swarm-arena run --yes run.toml plan.toml",
        "swarm-arena leaderboard hello",
        "swarm-arena review export",
        "swarm-arena review import",
    ]
    positions = [commands.index(command) for command in expected]
    assert positions == sorted(positions)
    viewer = [block for block in blocks if "swarm-arena board serve hello" in block.code]
    assert len(viewer) == 1
    assert viewer[0].lang == "sh"
    assert viewer[0].kind == "none"
    assert "Ctrl-C" in text
    assert "not executed" in text
    run_blocks(blocks, source="README.md", workdir=work, bindir=docs_bindir())
    solutions = list(work.glob(".swarm-arena/runs/*/goals/*/solution.json"))
    assert len(solutions) == 1
    assert (solutions[0].parent / "rounds/round-01.md").is_file()
    solution = json.loads(solutions[0].read_text(encoding="utf-8"))
    assert solution["status"] == "ok"
    assert solution["source_head"]
    assert len(solution["finalists"]) == 1
    candidate = solution["finalists"][0]["candidate"]
    assert candidate["output"]["status"] == "ok"
    assert candidate["output"]["data"] == {"message": "hello world"}
    assert (work / "hello-run-id.txt").read_text(encoding="utf-8").strip() == solution["run_id"]
    assert candidate["id"] in (work / "hello-leaderboard.txt").read_text(encoding="utf-8")
    assert "imported 1 scores as human:demo" in (work / "hello-import.txt").read_text(
        encoding="utf-8"
    )
    leaderboard = (work / "hello-human-leaderboard.txt").read_text(encoding="utf-8")
    assert "human:demo" in leaderboard
    row = next(line for line in leaderboard.splitlines() if candidate["id"] in line)
    assert row.split("\t")[4] == "1"


def test_readme_has_picture() -> None:
    assert spec_picture() in README.read_text(encoding="utf-8")


def test_readme_has_no_stub_text() -> None:
    assert STUB_TEXT.search(README.read_text(encoding="utf-8")) is None


def test_readme_sections() -> None:
    assert re.findall(r"^## (.+)$", README.read_text(encoding="utf-8"), re.M) == [
        "Install",
        "Quickstart (offline)",
        "Documentation",
        "Status and versioning",
        "Contributing",
        "License",
    ]


def test_readme_links_are_absolute() -> None:
    targets = re.findall(r"\]\(([^)\s]+)\)", README.read_text(encoding="utf-8"))
    assert [target for target in targets if not target.startswith(("https://", "#"))] == []


def test_readme_names_every_extra() -> None:
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert [
        extra
        for extra in extras
        if f"swarm-arena[{extra}]" not in README.read_text(encoding="utf-8")
    ] == []


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
    stdout, stderr = run_command(
        [str(docs_bindir() / "python"), "try_scripts.py"],
        source="docs/own-files.md",
        line=next(
            block.line
            for block in parse_blocks(text)
            if block.kind == "file" and block.arg == "try_scripts.py"
        ),
        workdir=tmp_path,
        bindir=docs_bindir(),
        timeout_s=60,
    )
    assert stdout.strip() == OWN_FILES_EXPECTED
    assert stderr == ""


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


@pytest.mark.parametrize(
    "code",
    ['echo "$DOCS_UNSET_VARIABLE"\n', "source ./missing-file\n", "exit 3\n"],
    ids=["unbound-variable", "missing-source", "explicit-exit"],
)
def test_bash_block_failure_keeps_its_status(tmp_path: Path, code: str) -> None:
    # The exit handshake's trap must not replace the block's own exit status.
    blocks = parse_blocks(f"<!-- docs-test: run -->\n```bash\n{code}```\n")
    with pytest.raises(AssertionError, match=r"t\.md:2 exited [1-9]"):
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


def test_shell_blocks_source_target_activate(
    target_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", os.defpath)
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel-secret")
    activate = target_bin / "activate"
    with activate.open("a", encoding="utf-8") as stream:
        stream.write("\nexport DOCS_ACTIVATE_FILE_WAS_SOURCED=yes\n")
    code = (
        'test "${DOCS_ACTIVATE_FILE_WAS_SOURCED:-}" = yes\n'
        f'test "${{VIRTUAL_ENV:-}}" = {shlex.quote(str(target_bin.parent))}\n'
        f'test "$(command -v python)" = {shlex.quote(str(target_bin / "python"))}\n'
        'test -z "${OPENAI_API_KEY:-}"\n'
        "python -c "
        + shlex.quote(f"import sys; assert sys.prefix == {str(target_bin.parent)!r}")
        + "\n"
    )
    blocks = parse_blocks(f"<!-- docs-test: run -->\n```bash\n{code}```\n")
    workdir = make_workdir(tmp_path, with_examples=False)
    run_blocks(blocks, source="activation.md", workdir=workdir, bindir=target_bin)
    activate.unlink()
    with pytest.raises(AssertionError, match=r"activation\.md:2 exited"):
        run_blocks(blocks, source="activation.md", workdir=workdir, bindir=target_bin)


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


def _wait_ready(
    workdir: Path, names: tuple[str, ...], process: subprocess.Popen[str] | None = None
) -> dict[int, str]:
    """Wait for test-owned process identities independently of the behavior deadline."""
    deadline = time.monotonic() + 60
    while not all((workdir / name).exists() for name in names):
        assert process is None or process.poll() is None, (
            "driver exited before descendants were ready"
        )
        assert time.monotonic() < deadline, "held descendants did not become ready"
        time.sleep(0.01)
    identities = {}
    for name in names:
        pid = int((workdir / name).read_text(encoding="utf-8"))
        started = process_start_time(pid)
        identities[pid] = started if started is not None else "<exited>"
    return identities


def _assert_recorded_pids_gone(identities: dict[int, str]) -> None:
    """Use each pid's identity, independently of the driver's process-group oracle."""
    assert {
        pid: started for pid, started in identities.items() if process_start_time(pid) == started
    } == {}


def _emergency_reap_recorded(identities: dict[int, str]) -> None:
    """Clean deliberately failing RED/mutation probes without hiding their assertions."""
    for pid, started in reversed(identities.items()):
        if process_start_time(pid) == started:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 60
    while any(process_start_time(pid) == started for pid, started in identities.items()):
        assert time.monotonic() < deadline, "emergency pid cleanup did not finish"
        time.sleep(0.01)


def _held_tree(workdir: Path, *, executor: bool, detached: bool) -> str:
    """Build a held three-generation tree, optionally using the real LocalExecutor."""
    grandchild = (
        "import os, time; from pathlib import Path; "
        + ("os.setsid(); " if detached else "")
        + f"Path({str(workdir / 'grandchild')!r}).write_text(str(os.getpid())); time.sleep(300)"
    )
    child = (
        "import os, subprocess, sys, time\nfrom pathlib import Path\n"
        f"Path({str(workdir / 'child')!r}).write_text(str(os.getpid()))\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "time.sleep(300)\n"
    )
    if executor:
        worker = (
            "import anyio, sys\nfrom swarm_arena import ScriptTool\n"
            "from swarm_arena.testing import check_tool\n"
            f"anyio.run(check_tool, ScriptTool(command=[sys.executable, '-c', {child!r}]), {{}})\n"
        )
    else:
        worker = child
    return (
        "import os, subprocess, sys, time\nfrom pathlib import Path\n"
        "assert os.getpgid(0) == os.getpid(), 'driver must own its group'\n"
        f"Path({str(workdir / 'driver')!r}).write_text(str(os.getpid()))\n"
        f"subprocess.Popen([sys.executable, '-c', {worker!r}], "
        f"start_new_session={detached!r}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"while not Path({str(workdir / 'grandchild')!r}).exists(): time.sleep(0.01)\n"
        "print('held stdout', flush=True)\nprint('held stderr', file=sys.stderr, flush=True)\n"
    )


@pytest.mark.usefixtures("owned_processes")
@pytest.mark.parametrize("topology", ["same_group", "sessions", "executor"])
@pytest.mark.parametrize("outcome", ["timeout", "failure", "success", "interrupt"])
def test_run_blocks_reaps_driver_and_descendant(
    outcome: str, topology: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = make_workdir(tmp_path, with_examples=False)
    code = _held_tree(workdir, executor=topology == "executor", detached=topology != "same_group")
    # Failed/successful leaders can exit before the test thread returns from communicate.
    code += (
        "time.sleep(300)\n"
        if outcome in {"timeout", "interrupt"}
        else f"raise SystemExit({3 if outcome == 'failure' else 0})\n"
    )
    communicate = subprocess.Popen.communicate
    identities: dict[int, str] = {}
    injected = False

    def after_readiness(
        self: subprocess.Popen[str], input: str | None = None, timeout: float | None = None
    ) -> tuple[str, str]:
        nonlocal injected
        if (
            not injected
            and isinstance(self.args, list)
            and self.args[0] == str(docs_bindir() / "python")
        ):
            injected = True
            identities.update(_wait_ready(workdir, ("driver", "child", "grandchild"), self))
            if outcome == "interrupt":
                raise KeyboardInterrupt
            if outcome == "timeout":
                # Real communicate drains output and expires only after readiness, not cold startup.
                assert timeout == 60
                return communicate(self, input=input, timeout=0.05)
        return communicate(self, input=input, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "communicate", after_readiness)
    try:
        if outcome in {"timeout", "failure", "interrupt"}:
            with pytest.raises(
                KeyboardInterrupt if outcome == "interrupt" else AssertionError,
                match=None
                if outcome == "interrupt"
                else ("tree.md:1 timed out" if outcome == "timeout" else "tree.md:1 exited 3"),
            ):
                run_blocks(
                    parse_blocks(f"```python\n{code}```"),
                    source="tree.md",
                    workdir=workdir,
                    bindir=docs_bindir(),
                    timeout_s=60,
                )
        else:
            run_blocks(
                parse_blocks(f"```python\n{code}```"),
                source="tree.md",
                workdir=workdir,
                bindir=docs_bindir(),
                timeout_s=60,
            )
        assert len(identities) == 3
        _assert_recorded_pids_gone(identities)
        assert not group_has_live_member(int((workdir / "driver").read_text(encoding="utf-8")))
    finally:
        _emergency_reap_recorded(identities)


@pytest.mark.parametrize("lang", ["python", "bash"])
def test_timeout_reports_source_line_and_captured_output(
    lang: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdir = make_workdir(tmp_path, with_examples=False)
    code = (
        "from pathlib import Path; import time; print('timeout stdout', flush=True); import sys; "
        "print('timeout stderr', file=sys.stderr, flush=True); "
        "Path('timeout-ready').touch(); time.sleep(300)"
    )
    if lang == "bash":
        code = "echo 'timeout stdout'; echo 'timeout stderr' >&2; touch timeout-ready; sleep 300"
    communicate = subprocess.Popen.communicate
    forced = False

    def expire(
        self: subprocess.Popen[str], input: str | None = None, timeout: float | None = None
    ) -> tuple[str, str]:
        nonlocal forced
        if (
            not forced
            and isinstance(self.args, list)
            and (self.args[0] == str(docs_bindir() / "python") or self.args[0] == "bash")
        ):
            forced = True
            deadline = time.monotonic() + 60
            while not (workdir / "timeout-ready").exists():
                assert self.poll() is None, "driver exited before output readiness"
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert timeout == 60
            return communicate(self, input=input, timeout=0.05)
        return communicate(self, input=input, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "communicate", expire)
    marker = "<!-- docs-test: run -->\n" if lang == "bash" else ""
    line = 3 if lang == "bash" else 2
    with pytest.raises(AssertionError, match=f"timeout.md:{line} timed out") as caught:
        run_blocks(
            parse_blocks(f"heading\n{marker}```{lang}\n{code}\n```"),
            source="timeout.md",
            workdir=workdir,
            bindir=docs_bindir(),
            timeout_s=60,
        )
    assert "timeout stdout" in str(caught.value)
    assert "timeout stderr" in str(caught.value)


def test_skip_reasons_have_no_stub_text() -> None:
    paths = [REPO / "README.md", *(REPO / "docs").rglob("*.md")]
    assert [
        f"{path.relative_to(REPO)}:{block.line}"
        for path in paths
        if not {"superpowers", "_build"} & set(path.relative_to(REPO).parts)
        for block in parse_blocks(path.read_text(encoding="utf-8"))
        if block.kind == "skip" and STUB_TEXT.search(block.arg)
    ] == []


OWN_FILES_EXPECTED = "ok: loss=0.25, check=True, key=(-0.25,)"


def test_own_files_expected_output_matches_documentation() -> None:
    text = page_path("Use your own files and script").read_text(encoding="utf-8")
    example = text.split("## Try it offline\n", 1)[1].split("\n## ", 1)[0]
    assert [block.code.strip() for block in parse_blocks(example) if block.lang == "text"] == [
        OWN_FILES_EXPECTED
    ]


@pytest.mark.usefixtures("owned_processes")
def test_own_files_offline_example_timeout_owns_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.read_text
    text = original(page_path("Use your own files and script"), encoding="utf-8")
    driver = next(
        b.code for b in parse_blocks(text) if b.kind == "file" and b.arg == "try_scripts.py"
    )
    held = (
        _held_tree(tmp_path, executor=True, detached=True).replace(
            "assert os.getpgid(0) == os.getpid(), 'driver must own its group'\n", ""
        )
        + "time.sleep(300)\n"
    )
    mutated = text.replace(driver, held, 1)
    identities: dict[int, str] = {}
    communicate = subprocess.Popen.communicate
    forced = False

    def read_text(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if self == ROOT / "docs/own-files.md":
            return mutated
        return original(self, encoding=encoding, errors=errors)

    def expire(
        self: subprocess.Popen[str], input: str | None = None, timeout: float | None = None
    ) -> tuple[str, str]:
        nonlocal forced
        if (
            not forced
            and isinstance(self.args, list)
            and self.args[0] in {sys.executable, str(docs_bindir() / "python")}
        ):
            forced = True
            identities.update(_wait_ready(tmp_path, ("driver", "child", "grandchild"), self))
            return communicate(self, input=input, timeout=0.05)
        return communicate(self, input=input, timeout=timeout)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(subprocess.Popen, "communicate", expire)
    try:
        with pytest.raises(AssertionError, match=r"own-files\.md:.* timed out"):
            test_own_files_offline_example_runs(tmp_path)
        _assert_recorded_pids_gone(identities)
    finally:
        _emergency_reap_recorded(identities)


@pytest.mark.parametrize("mutation", ["route_file", "validator_file", "assertion"])
def test_connector_conformance_detects_broken_files_and_assertions(
    mutation: str, tmp_path: Path
) -> None:
    text = page_path("Writing a connector").read_text(encoding="utf-8")
    old, new = {
        "route_file": ("file my_connector/route.py", "file my_connector/routes.py"),
        "validator_file": ("file my_connector/validate.py", "file my_connector/validates.py"),
        "assertion": ('out.data == {"target": "CCO"', 'out.data == {"target": "WRONG"'),
    }[mutation]
    assert old in text
    with pytest.raises(AssertionError, match="exited"):
        run_blocks(
            parse_blocks(text.replace(old, new, 1)),
            source="connector.md",
            workdir=make_workdir(tmp_path, with_examples=False),
            bindir=docs_bindir(),
        )


def test_connector_ray_example_is_explicitly_not_credited(tmp_path: Path) -> None:
    text = page_path("Writing a connector").read_text(encoding="utf-8")
    block = next(b for b in parse_blocks(text) if "async def test_tool_on_ray" in b.code)
    assert block.kind == "skip"
    assert "cluster" in block.arg
    assert not block_runs(
        block, workdir=make_workdir(tmp_path, with_examples=False), bindir=docs_bindir()
    )


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


def test_no_stub_text() -> None:
    files = [
        path
        for path in (REPO / "docs").rglob("*.md")
        if "superpowers" not in path.parts and "_build" not in path.parts
    ]
    assert [
        str(path.relative_to(REPO))
        for path in files
        if STUB_TEXT.search(path.read_text(encoding="utf-8"))
    ] == []


@pytest.mark.parametrize(
    "title", [title for title, _ in PAGES if title not in {"Roadmap", "API reference"}]
)
def test_every_content_page_has_a_runnable_block(title: str, tmp_path: Path) -> None:
    blocks = parse_blocks(page_path(title).read_text(encoding="utf-8"))
    workdir = make_workdir(tmp_path, with_examples=False)
    assert any(block_runs(block, workdir=workdir, bindir=docs_bindir()) for block in blocks), title


def test_process_tree_snapshot_follows_parents_across_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _fakes import process_cleanup
    from _fakes.process_cleanup import ProcessIdentity

    table = {
        r.pid: r
        for r in [
            ProcessIdentity(6000, 1, 6000, "driver"),
            ProcessIdentity(6001, 6000, 6001, "child"),
            ProcessIdentity(6002, 6001, 6002, "grandchild"),
            ProcessIdentity(6003, 1, 6003, "unrelated"),
        ]
    }
    monkeypatch.setattr(process_cleanup, "process_table", lambda: table)
    assert process_cleanup.snapshot_process_tree(6000) == {
        pid: table[pid] for pid in (6000, 6001, 6002)
    }


def test_process_tree_reaper_kills_every_owned_group(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock

    from _fakes import process_cleanup
    from _fakes.process_cleanup import ProcessIdentity

    table = {pid: ProcessIdentity(pid, pid - 1, pid, str(pid)) for pid in (6000, 6001, 6002)}
    tables = iter((table, {}))
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(process_cleanup, "process_table", lambda: next(tables))
    monkeypatch.setattr(process_cleanup, "reap_children", lambda children: [])
    monkeypatch.setattr(process_cleanup.os, "getpgrp", lambda: 0)
    monkeypatch.setattr(process_cleanup.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    process_cleanup.reap_process_tree(Mock(spec=subprocess.Popen), "6000", table)
    assert set(signals) == {(pid, signal.SIGKILL) for pid in table}


def test_process_tree_reaper_verifies_pid_exit_and_refuses_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    from _fakes import process_cleanup
    from _fakes.process_cleanup import ProcessIdentity

    record = ProcessIdentity(6000, 1, 6000, "captured")
    current = {6000: record}
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(process_cleanup, "process_table", lambda: current)
    monkeypatch.setattr(process_cleanup, "reap_children", lambda children: [])
    monkeypatch.setattr(process_cleanup.os, "getpgrp", lambda: 0)
    monkeypatch.setattr(process_cleanup.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(AssertionError, match="identities remained"):
        process_cleanup.reap_process_tree(
            Mock(spec=subprocess.Popen), "captured", {6000: record}, timeout_s=0
        )
    assert signals == [(6000, signal.SIGKILL)]
    signals.clear()
    current[6000] = ProcessIdentity(6000, 1, 6000, "replacement")
    process_cleanup.reap_process_tree(
        Mock(spec=subprocess.Popen), "captured", {6000: record}, timeout_s=0
    )
    assert signals == []


def test_connector_executable_examples_do_not_require_pytest() -> None:
    blocks = parse_blocks(page_path("Writing a connector").read_text(encoding="utf-8"))
    assert [
        block.line
        for block in blocks
        if block.lang == "python" and block.kind != "skip"
        for node in ast.walk(ast.parse(block.code))
        if (isinstance(node, ast.Import) and any(n.name == "pytest" for n in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "pytest")
    ] == []


def test_execute_does_not_adopt_reused_root_after_driver_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from io import StringIO
    from unittest.mock import Mock

    import _docs
    from _fakes import process_cleanup
    from _fakes.process_cleanup import ProcessIdentity

    original = ProcessIdentity(6000, 1, 6000, "original driver")
    replacement = {
        6000: ProcessIdentity(6000, 1, 6000, "unrelated replacement"),
        6001: ProcessIdentity(6001, 6000, 6001, "unrelated child"),
    }
    table = {original.pid: original}
    process = Mock(spec=subprocess.Popen)
    process.pid = original.pid
    process.returncode = None
    process.stdout, process.stderr = StringIO(), StringIO()
    gates: list[Path] = []
    signals: list[int] = []
    snapshots: list[str] = []

    def command_gate(command: list[str], ready: Path, release: Path) -> list[str]:
        gates.extend((ready, release))
        return command

    def communicate(*, timeout: float) -> tuple[str, str]:
        assert timeout == 60
        ready, release = gates
        ready.touch()
        deadline = time.monotonic() + 60
        while not release.exists():
            assert time.monotonic() < deadline, "exit observer did not release the fake driver"
            time.sleep(0.01)
        # communicate reaps the original; its pid is now owned by an unrelated tree.
        process.returncode = 0
        table.clear()
        table.update(replacement)
        return "driver output", ""

    def snapshot(pid: int) -> dict[int, ProcessIdentity]:
        snapshots.append(table[pid].started)
        return process_cleanup.snapshot_process_tree(pid)

    def signal_group(pgid: int, sig: int) -> None:
        assert sig == signal.SIGKILL
        signals.append(pgid)
        for pid in list(table):
            if table[pid].pgid == pgid:
                del table[pid]

    process.communicate.side_effect = communicate
    monkeypatch.setattr(_docs, "_exit_gate", command_gate)
    monkeypatch.setattr(_docs.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(_docs, "process_start_time", lambda pid: original.started)
    monkeypatch.setattr(_docs, "snapshot_process_tree", snapshot)
    monkeypatch.setattr(process_cleanup, "process_table", lambda: dict(table))
    monkeypatch.setattr(process_cleanup, "reap_children", lambda children: [])
    monkeypatch.setattr(process_cleanup.os, "getpgrp", lambda: 0)
    monkeypatch.setattr(process_cleanup.os, "killpg", signal_group)

    assert _docs._execute(
        [str(docs_bindir() / "python"), "-c", "pass"],
        workdir=tmp_path,
        env=block_env(tmp_path, docs_bindir()),
        timeout_s=60,
    ) == (0, "driver output", "")
    assert signals == []
    assert snapshots == [original.started]
    assert table == replacement
