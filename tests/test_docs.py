"""Exercise the public documentation and its offline script workflow."""

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


class RenderedText(HTMLParser):
    """Read displayed HTML text without scanning styles, URLs or element attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self.chunks.append(data)


def test_api_html_hides_model_implementation_details(tmp_path: Path) -> None:
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
    protocol = RenderedText()
    protocol.feed((output / "api" / "protocol.html").read_text())
    assert "data=..." in "".join(protocol.chunks).replace(" ", "")
    assert "omission, not a content value" in "".join(protocol.chunks)


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
