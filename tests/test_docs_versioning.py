from pathlib import Path

from _docs import docs_bindir, page_path, parse_blocks, required_facts, run_blocks
from _surface import PUBLIC_ALL

REQUIRED_FACTS = [
    "SemVer",
    "breaking changes bump the minor version",
    "CHANGELOG.md",
    "one minor release",
    "DeprecationWarning",
    "typing_extensions.deprecated",
    "Release 1.0 after both connectors have run real campaigns",
    "schema_version",
    "Newer versions are refused",
    "optional methods with defaults",
    "does no I/O",
    "preflight()",
    "check_tool",
    "check_check",
    "check_judge",
    "check_manager",
    "FakeModel",
    "Extras keep vendor churn out of the core",
    "snapshot test",
    "Everything else is private",
    "upper bound is raised only after",
]


def test_required_facts() -> None:
    required_facts("Versioning policy", REQUIRED_FACTS)


def test_versioning_page_lists_public_modules() -> None:
    text = page_path("Versioning policy").read_text(encoding="utf-8")
    assert [module for module in PUBLIC_ALL if f"`{module}`" not in text] == []


def test_versioning_page_deprecation_example(tmp_path: Path) -> None:
    page = page_path("Versioning policy")
    blocks = parse_blocks(page.read_text(encoding="utf-8"))
    assert any(block.lang == "python" and block.kind == "none" for block in blocks)
    run_blocks(blocks, source=str(page), workdir=tmp_path, bindir=docs_bindir())
