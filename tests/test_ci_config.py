"""Offline checks for documentation and workflow configuration."""

import argparse
import copy
import email.parser
import hashlib
import importlib.metadata
import io
import json
import os
import re
import runpy
import shlex
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import traceback
import venv
import zipfile
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import Specifier, SpecifierSet
from packaging.utils import canonicalize_name
from yaml.constructor import ConstructorError

REPO = Path(__file__).resolve().parents[1]


def parse_req(text: str) -> tuple[str, frozenset[str], frozenset[str]]:
    """
    Parse an unconditional version requirement for a canary override.

    Parameters
    ----------
    text : str
        Package name, optional extras, and explicit version comparisons.

    Returns
    -------
    tuple[str, frozenset[str], frozenset[str]]
        Normalized name, extras, and version specifiers.

    Raises
    ------
    ValueError
        For malformed requirements, URLs, markers or implicit bounds.

    Examples
    --------
    >>> name, extras, specs = parse_req("Example_Pkg[b,a]>=1,!=2,<3")
    >>> name, sorted(extras), sorted(specs)
    ('example-pkg', ['a', 'b'], ['!=2', '<3', '>=1'])
    """
    try:
        requirement = Requirement(text)
    except InvalidRequirement:
        raise ValueError("unsupported requirement") from None
    if (
        requirement.url
        or requirement.marker
        or any(
            spec.operator not in {"<", "<=", "==", ">", ">=", "!="}
            for spec in requirement.specifier
        )
    ):
        raise ValueError("unsupported requirement")
    return (
        canonicalize_name(requirement.name),
        frozenset(canonicalize_name(extra) for extra in requirement.extras),
        frozenset(str(spec) for spec in requirement.specifier),
    )


def override_line(name: str, reqs: list[tuple[str, frozenset[str], frozenset[str]]]) -> str:
    """
    Union a package's extras and remove only upper and equality constraints.

    Parameters
    ----------
    name : str
        Package to override.
    reqs : list[tuple[str, frozenset[str], frozenset[str]]]
        Requirements returned by ``parse_req``.

    Returns
    -------
    str
        Deterministically sorted override retaining lower bounds and exclusions.

    Raises
    ------
    ValueError
        If no requirement matches the package.

    Examples
    --------
    >>> override_line("pkg", [parse_req("pkg[b]>=1,<3"), parse_req("pkg[a]!=2,==1.*")])
    'pkg[a,b]!=2,>=1'
    """
    name = canonicalize_name(name)
    matches = [req for req in reqs if req[0] == name]
    if not matches:
        raise ValueError("missing override requirement")
    extras = sorted({extra for _, extras, _ in matches for extra in extras})
    specs = sorted(
        {
            spec
            for _, _, specs in matches
            for spec in specs
            if Specifier(spec).operator not in {"<", "<=", "=="}
        }
    )
    return name + (f"[{','.join(extras)}]" if extras else "") + ",".join(specs)


def _installed_upstream_versions() -> dict[str, str]:
    """Read the two selected distributions from this interpreter."""
    return {name: importlib.metadata.version(name) for name in ("openai-agents", "litellm")}


def _record_upstream_versions(path: Path) -> None:
    """Record and print the installed versions before the canary starts."""
    text = json.dumps(_installed_upstream_versions(), sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _check_upstream_versions() -> None:
    """Enforce production bounds or the recorded, lifted canary policy."""
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    requirements = [parse_req(req) for req in extras["llm"] + extras["cache"]]
    versions = _installed_upstream_versions()
    record = os.environ.get("SWARM_ARENA_UPSTREAM_CANARY")
    if record is not None:
        assert record, "canary requires an installed-version record"
        recorded = json.loads(Path(record).read_text(encoding="utf-8"))
        assert versions == recorded, "upstream environment changed after installation"
        lines = [
            line.strip()
            for line in (REPO / "upstream-overrides.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert lines == [override_line(name, requirements) for name in versions]
        requirements = [parse_req(line) for line in lines]
    for name, _, specs in requirements:
        assert SpecifierSet(",".join(specs)).contains(versions[name]), (
            f"installed upstream version violates {name} policy"
        )


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Read a repository YAML mapping with the safe loader.

    Parameters
    ----------
    path : Path
        YAML file to read as UTF-8.

    Returns
    -------
    dict[str, Any]
        Parsed configuration mapping.

    Examples
    --------
    >>> load_yaml(REPO / ".github/workflows/ci.yml")["name"]
    'CI'
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """
    Read workflow triggers under either YAML spelling of the on key.

    Parameters
    ----------
    workflow : dict[str, Any]
        Workflow parsed by PyYAML, which treats bare ``on`` as boolean True.

    Returns
    -------
    dict[str, Any]
        Workflow event configuration.

    Examples
    --------
    >>> triggers({"on": {"push": {}}})
    {'push': {}}
    """
    return workflow["on"] if "on" in workflow else cast(dict[str | bool, Any], workflow)[True]


def steps_run(job: dict[str, Any]) -> list[str]:
    """
    Return the shell commands from a job in step order.

    Parameters
    ----------
    job : dict[str, Any]
        Workflow job containing steps.

    Returns
    -------
    list[str]
        Commands from steps with a run key.

    Examples
    --------
    >>> steps_run({"steps": [{"uses": "actions/checkout@v7"}, {"run": "uv sync"}]})
    ['uv sync']
    """
    return [step["run"] for step in job["steps"] if "run" in step]


def steps_uses(job: dict[str, Any]) -> list[str]:
    """
    Return the action references from a job in step order.

    Parameters
    ----------
    job : dict[str, Any]
        Workflow job containing steps.

    Returns
    -------
    list[str]
        Action references from steps with a uses key.

    Examples
    --------
    >>> steps_uses({"steps": [{"uses": "actions/checkout@v7"}, {"run": "uv sync"}]})
    ['actions/checkout@v7']
    """
    return [step["uses"] for step in job["steps"] if "uses" in step]


def shell_words(line: str) -> list[str]:
    """
    Split one shell line into words, with comments as bash reads them.

    Bash starts a comment only at the start of a word, so ``a#||b`` still runs
    ``b``. Control operators become separate words so callers can reject them.

    Parameters
    ----------
    line : str
        One shell line.

    Returns
    -------
    list of str
        The words before the first comment.

    Examples
    --------
    >>> shell_words("uv sync --locked # install")
    ['uv', 'sync', '--locked']
    >>> shell_words("cmd --help#||true")
    ['cmd', '--help#', '||', 'true']
    """
    quote = ""
    escaped = False
    word_start = True
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            word_start = False
        elif quote:
            if char == quote:
                quote = ""
        elif char in {"'", '"'}:
            quote = char
            word_start = False
        elif char == "#" and word_start:
            line = line[:index]
            break
        else:
            word_start = char in " \t\r\n();<>|&"
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def shell_command(run: str) -> list[str]:
    """
    Parse one simple command line, discarding shell comments.

    Only a single physical line with literal arguments is supported. Shell
    operator and expansion characters are rejected, even inside quoted
    arguments. Compound commands, redirection and substitution must be split
    into separate workflow steps before these offline checks can inspect them.
    Nothing is executed.

    Parameters
    ----------
    run : str
        Workflow step's run string.

    Returns
    -------
    list[str]
        The executable and its own arguments, with shell quotes removed.

    Raises
    ------
    ValueError
        If the command uses unsupported syntax or has unmatched quotes.

    Examples
    --------
    >>> shell_command('uv sync --locked --group="docs" # install dependencies')
    ['uv', 'sync', '--locked', '--group=docs']
    """
    if "\n" in run.strip() or "\r" in run.strip():
        raise ValueError("expected one simple shell command")
    command = shell_words(run)
    if any(set(argument) & set(";&|<>()$`{}") for argument in command):
        raise ValueError("unsupported shell syntax")
    return command


@pytest.mark.parametrize("on_key", ["on", '"on"'])
def test_load_yaml_and_triggers(tmp_path: Path, on_key: str) -> None:
    path = tmp_path / "workflow.yaml"
    path.write_text(f"name: café\n{on_key}:\n  push: {{}}\n", encoding="utf-8")
    workflow = load_yaml(path)
    assert workflow["name"] == "café"
    assert triggers(workflow) == {"push": {}}


def test_load_yaml_rejects_python_tags(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/tuple [1, 2]\n", encoding="utf-8")
    with pytest.raises(ConstructorError):
        load_yaml(path)


def test_triggers_prefers_string_key() -> None:
    workflow = load_yaml(REPO / ".github/workflows/ci.yml")
    workflow["on"] = {"workflow_dispatch": {}}
    assert triggers(workflow) == {"workflow_dispatch": {}}


def test_steps_run_and_uses() -> None:
    job = {
        "steps": [
            {"uses": "actions/checkout@v7"},
            {"name": "Install", "run": "uv sync"},
            {"uses": "astral-sh/setup-uv@v10"},
            {"run": "uv run pytest"},
        ]
    }
    assert steps_run(job) == ["uv sync", "uv run pytest"]
    assert steps_uses(job) == ["actions/checkout@v7", "astral-sh/setup-uv@v10"]
    assert steps_run({"steps": []}) == []
    assert steps_uses({"steps": []}) == []


def test_rtd_config() -> None:
    cfg = load_yaml(REPO / ".readthedocs.yaml")
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert cfg["version"] == 2
    assert cfg["build"] == {"os": "ubuntu-24.04", "tools": {"python": "3.13"}}
    [install] = cfg["python"]["install"]
    assert {k: v for k, v in install.items() if k != "extras"} == {
        "method": "uv",
        "command": "sync",
        "groups": ["docs"],
    }
    assert sorted(install["extras"]) == sorted(extras)
    assert cfg["sphinx"] == {"configuration": "docs/conf.py", "fail_on_warning": True}


def test_rtd_matches_docs_ci_job() -> None:
    job = load_yaml(REPO / ".github" / "workflows" / "ci.yml")["jobs"]["docs"]
    commands = [shell_command(run) for run in steps_run(job)]
    assert any(
        command[:2] == ["uv", "sync"]
        and {"--locked", "--all-extras"} <= set(command)
        and (
            "--group=docs" in command
            or any(command[i : i + 2] == ["--group", "docs"] for i in range(len(command) - 1))
        )
        for command in commands
    )
    assert any(
        command[:3] == ["uv", "run", "sphinx-build"] and "-W" in command for command in commands
    )
    assert "3.13" in json.dumps(job)


@pytest.mark.parametrize(
    ("sync", "sphinx", "valid"),
    [
        pytest.param(
            "uv sync --locked --all-extras --group docs",
            "uv run sphinx-build -W docs docs/_build/html",
            True,
            id="current-command",
        ),
        pytest.param(
            "uv sync --group docs --all-extras --locked",
            "uv run sphinx-build docs docs/_build/html -W",
            True,
            id="reordered-arguments",
        ),
        pytest.param(
            "uv sync --group=docs --locked --all-extras",
            "uv run sphinx-build -W docs docs/_build/html",
            True,
            id="equals-group",
        ),
        pytest.param(
            "uv sync --all-extras --group docs",
            "uv run sphinx-build -W docs docs/_build/html",
            False,
            id="unlocked",
        ),
        pytest.param(
            "uv sync --all-extras --group docs && echo --locked",
            "uv run sphinx-build -W docs docs/_build/html",
            False,
            id="lock-in-other-command",
        ),
        pytest.param(
            "uv sync --all-extras --group docs # --locked",
            "uv run sphinx-build -W docs docs/_build/html",
            False,
            id="lock-in-comment",
        ),
        pytest.param(
            "uv sync --locked --all-extras --group docs",
            "uv run sphinx-build docs docs/_build/html",
            False,
            id="sphinx-without-warning-errors",
        ),
        pytest.param(
            "uv sync --locked --all-extras --group docs",
            "uv run sphinx-build docs docs/_build/html # -W",
            False,
            id="warning-flag-in-comment",
        ),
        pytest.param(
            "uv sync --locked --all-extras --group docs",
            "uv run sphinx-build docs docs/_build/html && echo -W",
            False,
            id="warning-flag-in-other-command",
        ),
    ],
)
def test_docs_ci_command_semantics(
    monkeypatch: pytest.MonkeyPatch, sync: str, sphinx: str, valid: bool
) -> None:
    workflow = {
        "jobs": {
            "docs": {
                "steps": [
                    {"uses": "astral-sh/setup-uv@v10", "with": {"python-version": "3.13"}},
                    {"run": sync},
                    {"run": sphinx},
                ]
            }
        }
    }
    monkeypatch.setitem(globals(), "load_yaml", lambda _: workflow)
    if valid:
        test_rtd_matches_docs_ci_job()
    else:
        with pytest.raises((AssertionError, ValueError)):
            test_rtd_matches_docs_ci_job()


def test_pyyaml_is_dev_dependency() -> None:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert "pyyaml>=6" in project["dependency-groups"]["dev"]


UPSTREAM = REPO / ".github" / "workflows" / "upstream.yml"


def test_overrides_lift_only_upper_bounds() -> None:
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    reqs = [parse_req(req) for req in extras["llm"] + extras["cache"]]
    expected = [override_line(name, reqs) for name in ("openai-agents", "litellm")]
    lines = [
        line.strip()
        for line in (REPO / "upstream-overrides.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines == expected


def test_upstream_workflow() -> None:
    workflow = load_yaml(UPSTREAM)
    assert triggers(workflow) == {
        "schedule": [{"cron": "0 6 * * 1"}],
        "workflow_dispatch": None,
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"newest-llm-deps"}
    job = workflow["jobs"]["newest-llm-deps"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == 30
    assert steps_uses(job) == ["actions/checkout@v7", "astral-sh/setup-uv@v10.3.0"]
    commands = [shell_command(run) for run in steps_run(job)]
    venv, install, show, record, check = commands
    assert venv == ["uv", "venv", "--python", "3.13"]
    assert install[:3] == ["uv", "pip", "install"]
    assert len(install) == 9
    assert dict(zip(install[3::2], install[4::2], strict=True)) == {
        "-e": ".[llm,cache,langfuse]",
        "--group": "dev",
        "--override": "upstream-overrides.txt",
    }
    assert show == ["uv", "pip", "show", "openai-agents", "litellm"]
    assert record == [
        ".venv/bin/python",
        "tests/test_ci_config.py",
        "--record-upstream",
        "upstream-installed.json",
    ]
    assert check == [".venv/bin/python", "-m", "pytest", "tests/llm", "-v"]
    assert "env" not in workflow
    assert "env" not in job
    assert "permissions" not in job
    assert all("env" not in step for step in job["steps"][:-1])
    assert job["steps"][-1]["env"] == {
        "SWARM_ARENA_UPSTREAM_CANARY": "upstream-installed.json",
    }
    assert all("if" not in step and not step.get("continue-on-error") for step in job["steps"])
    assert "if" not in job
    assert not job.get("continue-on-error")


def test_upstream_never_resyncs_to_lock() -> None:
    [job] = load_yaml(UPSTREAM)["jobs"].values()
    assert not any("uv run" in run or "uv sync" in run for run in steps_run(job))
    # Production workflows must never select the test-only policy.
    for path in (REPO / ".github" / "workflows").glob("*.y*ml"):
        if path != UPSTREAM:
            assert "SWARM_ARENA_UPSTREAM_CANARY" not in path.read_text(encoding="utf-8")


def test_contributing_states_the_bound_rule() -> None:
    text = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "## Upstream bounds" in text
    for phrase in (
        "openai-agents",
        "litellm",
        "upstream-overrides.txt",
        "!=",
        "test_overrides_lift_only_upper_bounds",
        "SWARM_ARENA_UPSTREAM_CANARY",
        "upstream-installed.json",
        "cron",
        "email",
    ):
        assert phrase in text


def test_override_parser_unions_extras_and_retains_lower_bounds_and_exclusions() -> None:
    requirements = [
        parse_req("Example_Pkg[z,a]>=1.2,!=1.4,<3"),
        parse_req("example-pkg[b,z]>1.1,<=4,==2.*,!=1.5"),
        parse_req("other[ignored]>=8"),
    ]
    assert requirements[0] == (
        "example-pkg",
        frozenset({"z", "a"}),
        frozenset({">=1.2", "!=1.4", "<3"}),
    )
    assert override_line("example-pkg", requirements) == (
        "example-pkg[a,b,z]!=1.4,!=1.5,>1.1,>=1.2"
    )
    assert override_line("other", requirements) == "other[ignored]>=8"
    with pytest.raises(ValueError, match="missing override requirement"):
        override_line("absent", requirements)


@pytest.mark.parametrize(
    "requirement",
    [
        "",
        "pkg[broken",
        "pkg>=",
        "pkg>=1,,<2",
        "pkg~=1.2",
        "pkg===arbitrary",
        "pkg @ https://example.invalid/pkg.whl",
        "pkg>=1; python_version >= '3.11'",
    ],
)
def test_override_parser_rejects_unsupported_requirements(requirement: str) -> None:
    with pytest.raises(ValueError, match="unsupported requirement"):
        parse_req(requirement)


@pytest.mark.parametrize(
    ("canary", "versions", "valid", "record_in_cwd"),
    [
        (False, {"openai-agents": "0.24.0", "litellm": "1.107.0rc2"}, True, False),
        (False, {"openai-agents": "0.25.0", "litellm": "1.108.0"}, False, False),
        (False, {"openai-agents": "0.24.0", "litellm": "1.107.0rc1"}, False, False),
        (True, {"openai-agents": "0.25.0", "litellm": "1.108.0"}, True, False),
        (True, {"openai-agents": "0.24.0", "litellm": "1.107.0rc2"}, True, False),
        (True, {"openai-agents": "0.23.0", "litellm": "1.108.0"}, False, False),
        (True, {"openai-agents": "0.25.0", "litellm": "1.106.4"}, False, False),
        pytest.param(
            False,
            {"openai-agents": "0.25.0", "litellm": "1.108.0"},
            False,
            True,
            id="record-in-cwd-does-not-select-canary",
        ),
    ],
)
def test_upstream_version_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    canary: bool,
    versions: dict[str, str],
    valid: bool,
    record_in_cwd: bool,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.delenv("SWARM_ARENA_UPSTREAM_CANARY", raising=False)
    if record_in_cwd:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "upstream-installed.json").write_text(json.dumps(versions), encoding="utf-8")
    if canary:
        record = tmp_path / "installed.json"
        record.write_text(json.dumps(versions), encoding="utf-8")
        monkeypatch.setenv("SWARM_ARENA_UPSTREAM_CANARY", str(record))
    if valid:
        _check_upstream_versions()
    else:
        with pytest.raises(AssertionError, match="installed upstream version violates"):
            _check_upstream_versions()


@pytest.mark.parametrize("package", ["openai-agents", "litellm"])
def test_upstream_canary_rejects_restored_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    package: str,
) -> None:
    versions = {"openai-agents": "0.25.0", "litellm": "1.108.0"}
    record = tmp_path / "installed.json"
    record.write_text(json.dumps(versions), encoding="utf-8")
    versions[package] = {"openai-agents": "0.24.0", "litellm": "1.107.0rc2"}[package]
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setenv("SWARM_ARENA_UPSTREAM_CANARY", str(record))
    with pytest.raises(AssertionError, match="upstream environment changed after installation"):
        _check_upstream_versions()


def test_upstream_version_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    versions = {"openai-agents": "0.25.0", "litellm": "1.108.0"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    path = tmp_path / "installed.json"
    _record_upstream_versions(path)
    assert json.loads(path.read_text(encoding="utf-8")) == versions
    assert json.loads(capsys.readouterr().out) == versions


@pytest.mark.parametrize("content", ["{}", '{"litellm": "1.107.0rc2"}', '"not a mapping"'])
def test_upstream_canary_requires_complete_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    content: str,
) -> None:
    versions = {"openai-agents": "0.24.0", "litellm": "1.107.0rc2"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    record = tmp_path / "installed.json"
    record.write_text(content, encoding="utf-8")
    monkeypatch.setenv("SWARM_ARENA_UPSTREAM_CANARY", str(record))
    with pytest.raises(AssertionError, match="upstream environment changed after installation"):
        _check_upstream_versions()


@pytest.mark.parametrize("canary", [False, True])
@pytest.mark.parametrize("package", ["openai-agents", "litellm"])
def test_upstream_policy_retains_exclusions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    canary: bool,
    package: str,
) -> None:
    versions = {"openai-agents": "0.24.0", "litellm": "1.107.0rc2"}
    project = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    lower = {"openai-agents": ">=0.24", "litellm": ">=1.107.0rc2"}[package]
    project = project.replace(lower, f"{lower},!={versions[package]}")
    (tmp_path / "pyproject.toml").write_text(project, encoding="utf-8")
    overrides = (REPO / "upstream-overrides.txt").read_text(encoding="utf-8")
    overrides = overrides.replace(lower, f"!={versions[package]},{lower}")
    (tmp_path / "upstream-overrides.txt").write_text(overrides, encoding="utf-8")
    monkeypatch.setitem(globals(), "REPO", tmp_path)
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.delenv("SWARM_ARENA_UPSTREAM_CANARY", raising=False)
    if canary:
        record = tmp_path / "installed.json"
        record.write_text(json.dumps(versions), encoding="utf-8")
        monkeypatch.setenv("SWARM_ARENA_UPSTREAM_CANARY", str(record))
    with pytest.raises(AssertionError, match="installed upstream version violates"):
        _check_upstream_versions()


def test_override_parser_errors_do_not_echo_input() -> None:
    secret = "sentinel-private-requirement"
    with pytest.raises(ValueError, match="unsupported requirement") as caught:
        parse_req(f"pkg>=1; {secret}")
    error = caught.value
    assert secret not in str(error)
    assert secret not in repr(error)
    assert secret not in repr(error.args)
    assert secret not in repr(getattr(error, "__notes__", []))
    assert error.__cause__ is None
    assert secret not in "".join(traceback.format_exception(error))


@pytest.mark.parametrize("record_value", ["", "missing.json"])
def test_upstream_canary_cannot_fall_back_to_production(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_value: str,
) -> None:
    versions = {"openai-agents": "0.24.0", "litellm": "1.107.0rc2"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWARM_ARENA_UPSTREAM_CANARY", record_value)
    with pytest.raises((AssertionError, FileNotFoundError)):
        _check_upstream_versions()


@pytest.mark.parametrize(
    ("command", "valid"),
    [
        (
            "uv pip install --group dev --override upstream-overrides.txt "
            '-e ".[llm,cache,langfuse]"',
            True,
        ),
        (
            'uv pip install -e ".[llm,cache,langfuse]" --group dev '
            "# --override upstream-overrides.txt",
            False,
        ),
        (
            'uv pip install -e ".[llm,cache,langfuse]" --group dev '
            "&& echo --override upstream-overrides.txt",
            False,
        ),
        (
            'uv pip install -e ".[llm,cache,langfuse]" --group dev '
            "--constraint upstream-overrides.txt",
            False,
        ),
    ],
)
def test_upstream_install_command_semantics(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    valid: bool,
) -> None:
    workflow = load_yaml(UPSTREAM)
    workflow["jobs"]["newest-llm-deps"]["steps"][3]["run"] = command
    monkeypatch.setitem(globals(), "load_yaml", lambda _: workflow)
    if valid:
        test_upstream_workflow()
    else:
        with pytest.raises((AssertionError, ValueError)):
            test_upstream_workflow()


def test_canary_executes_all_upstream_facts_without_skips(tmp_path: Path) -> None:
    pytest.importorskip("agents")
    record = tmp_path / "installed.json"
    environment = dict(os.environ)
    environment.pop("SWARM_ARENA_UPSTREAM_CANARY", None)
    recorded = subprocess.run(
        [sys.executable, str(REPO / "tests/test_ci_config.py"), "--record-upstream", str(record)],
        cwd=REPO,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    assert recorded.returncode == 0, recorded.stdout + recorded.stderr
    assert json.loads(record.read_text(encoding="utf-8")) == _installed_upstream_versions()
    passed_counts = []
    for canary in (False, True):
        if canary:
            environment["SWARM_ARENA_UPSTREAM_CANARY"] = str(record)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-rs",
                "tests/llm/test_upstream_facts.py",
            ],
            cwd=REPO,
            env=environment,
            capture_output=True,
            encoding="utf-8",
            timeout=180,
            check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert re.search(r"\b\d+ (?:skipped|deselected)\b", output) is None, output
        [passed] = re.findall(r"\b(\d+) passed\b", output)
        assert int(passed) > 0, output
        passed_counts.append(int(passed))
    assert passed_counts[0] == passed_counts[1]


RELEASE = REPO / ".github/workflows/release.yml"


def _release_record(
    root: Path,
    dist: Path,
    tag: str,
    commit: str,
    installed_version: str,
    runtime_version: str,
    docs_version: str,
) -> dict[str, Any]:
    """Validate the candidate and return its exact commit and artifact hashes."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]
    stable = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    if re.fullmatch("v" + stable, tag) is None or tag != f"v{version}":
        raise ValueError("release tag must exactly match the stable project version")
    if project["name"] != "swarm-arena" or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("release project or commit is invalid")
    headings = re.findall(r"^## (.+)$", (root / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
    if not headings or headings[0] != "[Unreleased]":
        raise ValueError("release changelog requires a leading Unreleased heading")
    released = []
    for heading in headings[1:]:
        match = re.fullmatch(r"\[(" + stable + r")\] - ([0-9]{4}-[0-9]{2}-[0-9]{2})", heading)
        if match is None:
            raise ValueError("release changelog heading is malformed")
        release_version, day = match.groups()
        try:
            released_on = date.fromisoformat(day)
        except ValueError:
            raise ValueError("release changelog date is invalid") from None
        if (
            released_on > (datetime.now(UTC) + timedelta(hours=14)).date()
            or release_version in released
        ):
            raise ValueError("release changelog date or duplicate version is invalid")
        released.append(release_version)
    if not released or released[0] != version:
        raise ValueError("release changelog must begin with the project version")
    if (installed_version, runtime_version, docs_version) != (version, version, version):
        raise ValueError("release installed, runtime and docs versions must agree")
    wheel = dist / f"swarm_arena-{version}-py3-none-any.whl"
    sdist = dist / f"swarm_arena-{version}.tar.gz"
    if set(dist.iterdir()) != {wheel, sdist} or any(
        p.is_symlink() or not p.is_file() for p in (wheel, sdist)
    ):
        raise ValueError("release artifacts must be exactly one wheel and one sdist")
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = f"swarm_arena-{version}.dist-info/METADATA"
        if [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")] != [metadata_path]:
            raise ValueError("release wheel metadata inventory is invalid")
        wheel_metadata = archive.read(metadata_path).decode("utf-8")
    with tarfile.open(sdist) as archive:
        prefix = f"swarm_arena-{version}/"
        if [n for n in archive.getnames() if n.endswith("/PKG-INFO")] != [prefix + "PKG-INFO"]:
            raise ValueError("release sdist metadata inventory is invalid")
        info = archive.extractfile(prefix + "PKG-INFO")
        project_file = archive.extractfile(prefix + "pyproject.toml")
        if info is None or project_file is None:
            raise ValueError("release sdist metadata is missing")
        with info, project_file:
            sdist_metadata = info.read().decode("utf-8")
            sdist_project = tomllib.loads(project_file.read().decode("utf-8"))["project"]
        if (sdist_project["name"], sdist_project["version"]) != ("swarm-arena", version):
            raise ValueError("release sdist project metadata does not match")
    for content in (wheel_metadata, sdist_metadata):
        metadata = email.parser.Parser().parsestr(content)
        if metadata.get_all("Name") != ["swarm-arena"] or metadata.get_all("Version") != [version]:
            raise ValueError("release artifact metadata does not match")
    return {
        "schema_version": 1,
        "commit": commit,
        "tag": tag,
        "version": version,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel, sdist)},
    }


def _installed_release(python: Path) -> tuple[str, str]:
    """Probe the wheel interpreter from an empty directory without ambient keys."""
    code = """
import importlib.metadata
import json
import sys
from pathlib import Path
import swarm_arena
distribution = importlib.metadata.distribution("swarm-arena")
print(json.dumps({
    "installed": distribution.version,
    "runtime": swarm_arena.__version__,
    "module": str(Path(swarm_arena.__file__).resolve()),
    "metadata_root": str(Path(distribution.locate_file("")).resolve()),
    "prefix": sys.prefix,
}))
"""
    with tempfile.TemporaryDirectory(prefix="swarm-release-probe-") as empty:
        result = subprocess.run(
            [str(python.absolute()), "-I", "-c", code],
            cwd=empty,
            env={"PATH": str(python.absolute().parent), "PYTHONUTF8": "1"},
            capture_output=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
    if result.returncode:
        raise ValueError("release wheel interpreter probe failed")
    probe = json.loads(result.stdout)
    prefix = python.absolute().parent.parent.resolve()
    if Path(probe["prefix"]).resolve() != prefix or any(
        not Path(probe[key]).is_relative_to(prefix) for key in ("module", "metadata_root")
    ):
        raise ValueError("release imports must resolve inside the wheel environment")
    return probe["installed"], probe["runtime"]


def _check_release(dist: Path, tag: str, commit: str, python: Path, record: Path) -> None:
    """Run the shared release guard and persist only successful candidate evidence."""
    installed, runtime = _installed_release(python)
    docs_version = runpy.run_path(str(REPO / "docs/conf.py"))["release"]
    evidence = _release_record(REPO, dist, tag, commit, installed, runtime, docs_version)
    record.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("line", "words"),
    [
        ("uv sync --locked # install", ["uv", "sync", "--locked"]),
        ("cmd --help#||true", ["cmd", "--help#", "||", "true"]),
        ("cmd -v#;true", ["cmd", "-v#", ";", "true"]),
        ("cmd '#' || true", ["cmd", "#", "||", "true"]),
        (r"cmd \# || true", ["cmd", "#", "||", "true"]),
        ("cmd ''# || true", ["cmd", "#", "||", "true"]),
        ("cmd;# ' ignored", ["cmd", ";"]),
    ],
)
def test_shell_words_start_comments_only_at_a_word(line: str, words: list[str]) -> None:
    # Bash starts a comment only at the start of a word, so `a#||b` still runs `b`.
    assert shell_words(line) == words


def test_shell_command_rejects_operators_after_a_hash_in_a_word() -> None:
    with pytest.raises(ValueError, match="unsupported shell syntax"):
        shell_command("uv sync --locked --group docs#||true")


def test_release_triggers() -> None:
    on = triggers(load_yaml(RELEASE))
    assert on == {"push": {"tags": ["v*"]}, "pull_request": None, "workflow_dispatch": None}


def _release_shell_lines(script: str) -> list[list[str]]:
    """Tokenize continued shell lines without hiding control operators."""
    commands = []
    for line in script.replace("\\\n", "").splitlines():
        tokens = shell_words(line)
        if tokens:
            commands.append(tokens)
    return commands


def test_release_build_job_checks_the_wheel() -> None:
    wf = load_yaml(RELEASE)
    assert wf["permissions"] == {"contents": "read"}
    job = wf["jobs"]["build"]
    runs = "\n".join(steps_run(job))
    for needed in (
        "uv build --out-dir",
        'uvx twine check --strict "$RELEASE_DIR"/dist/*',
        'uv venv "$RELEASE_DIR/venv" --python 3.11',
        '"$RELEASE_DIR/venv/bin/swarm-arena" --help',
        'SWARM_ARENA_DOCS_VENV="$RELEASE_DIR/venv"',
        "tests/test_docs.py::test_readme_quickstart",
        "tests/test_ci_config.py tests/test_packaging.py",
        "mktemp -d",
    ):
        assert needed in runs
    assert steps_uses(job) == [
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "astral-sh/setup-uv@1c37ad07a6a961277cf70c0d37d6f313000f5884",
        "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9",
    ]
    commands = [shell_command(s) for s in steps_run(job) if s.startswith("uv sync ")]
    assert commands == [["uv", "sync", "--locked", "--all-groups"]]
    assert [
        shell_words(run)
        for run in steps_run(job)
        if run.startswith(("uv build ", "uvx twine ", "uv run --no-sync pytest "))
    ] == [
        [
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "-q",
            "tests/test_ci_config.py",
            "tests/test_packaging.py",
        ],
        ["uv", "build", "--out-dir", "$RELEASE_DIR/dist", "--no-create-gitignore"],
        ["uvx", "twine", "check", "--strict", "$RELEASE_DIR/dist/*"],
    ]
    [install] = [run for run in steps_run(job) if run.startswith("uv venv ")]
    assert _release_shell_lines(install) == [
        ["uv", "venv", "$RELEASE_DIR/venv", "--python", "3.11"],
        [
            "uv",
            "pip",
            "install",
            "--python",
            "$RELEASE_DIR/venv/bin/python",
            "$RELEASE_DIR/dist/*.whl",
        ],
        ["mkdir", "$RELEASE_DIR/empty"],
        ["cd", "$RELEASE_DIR/empty"],
        ["$RELEASE_DIR/venv/bin/swarm-arena", "--help"],
    ]
    [readme] = [
        s
        for s in job["steps"]
        if s.get("name") == "README quickstart from the wheel in an empty folder"
    ]
    assert _release_shell_lines(readme["run"]) == [
        [
            "SWARM_ARENA_DOCS_VENV=$RELEASE_DIR/venv",
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "tests/test_docs.py::test_readme_quickstart",
            "-v",
        ],
    ]
    upload = job["steps"][-1]
    assert upload["with"] == {
        "name": "release-dist",
        "path": "${{ env.RELEASE_DIR }}/bundle/",
        "if-no-files-found": "error",
    }
    assert not any(step.get("continue-on-error") or step.get("if") for step in job["steps"])


def test_release_checks_tag_against_version() -> None:
    steps = load_yaml(RELEASE)["jobs"]["build"]["steps"]
    [guard] = [s for s in steps if "--check-release" in s.get("run", "")]
    # The shared executable guard runs for PRs/dispatches as well as tags.
    assert "if" not in guard
    assert '"$GITHUB_REF_TYPE" = tag' in guard["run"]
    assert 'tag="$GITHUB_REF_NAME"' in guard["run"]
    assert "uv version --short" in guard["run"]
    assert '--tag "$tag"' in guard["run"]
    assert '--commit "$GITHUB_SHA"' in guard["run"]
    assert '--wheel-python "$RELEASE_DIR/venv/bin/python"' in guard["run"]
    assert guard["run"].splitlines()[:4] == [
        'tag="v$(uv version --short)"',
        'if [ "$GITHUB_REF_TYPE" = tag ]; then',
        '  tag="$GITHUB_REF_NAME"',
        "fi",
    ]
    assert _release_shell_lines("\n".join(guard["run"].splitlines()[4:])) == [
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "tests/test_ci_config.py",
            "--check-release",
            "--dist",
            "$RELEASE_DIR/dist",
            "--tag",
            "$tag",
            "--commit",
            "$GITHUB_SHA",
            "--wheel-python",
            "$RELEASE_DIR/venv/bin/python",
            "--record",
            "$RELEASE_DIR/release.json",
        ],
    ]
    assert steps.index(guard) < len(steps) - 1


def test_release_does_not_dump_secrets() -> None:
    workflow = load_yaml(RELEASE)
    assert re.search(r"github\s*\.\s*token\b", json.dumps(workflow)) is None
    for job in workflow["jobs"].values():
        for run in steps_run(job):
            for tokens in _release_shell_lines(run):
                assert not any(Path(token).name in {"env", "printenv"} for token in tokens)
                assert tokens != ["set"]
                assert tokens[:2] not in [["export", "-p"], ["declare", "-p"]]
    [bundle] = [
        step
        for step in workflow["jobs"]["build"]["steps"]
        if step.get("name") == "Bundle the checked files without rebuilding"
    ]
    assert _release_shell_lines(bundle["run"]) == [
        ["mkdir", "$RELEASE_DIR/bundle"],
        ["mv", "$RELEASE_DIR/dist", "$RELEASE_DIR/release.json", "$RELEASE_DIR/bundle/"],
        ["cat", "$RELEASE_DIR/bundle/release.json", ">>", "$GITHUB_STEP_SUMMARY"],
    ]


def test_publish_job_is_least_privilege() -> None:
    wf = load_yaml(RELEASE)
    job = wf["jobs"]["publish"]
    assert job["needs"] == "build"
    assert job["if"] == "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    assert job["environment"] == {"name": "pypi", "url": "https://pypi.org/p/swarm-arena"}
    assert job["permissions"] == {"id-token": "write"}
    # The job that can mint a PyPI token runs only actions pinned to a commit.
    assert steps_uses(job) == [
        "actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333",
        "pypa/gh-action-pypi-publish@7245906fb61fa33a13554f9d5fb856fe61758086",
    ]
    text = RELEASE.read_text(encoding="utf-8")
    assert "actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333 # v8.0.2" in text
    assert "pypa/gh-action-pypi-publish@7245906fb61fa33a13554f9d5fb856fe61758086 # v1.14.3" in text
    assert job["steps"][0]["with"] == {"name": "release-dist", "path": "release"}
    assert job["steps"][-1]["with"] == {"packages-dir": "release/dist/"}
    assert "id-token" not in json.dumps(wf["jobs"]["build"])
    assert "secrets." not in RELEASE.read_text(encoding="utf-8")
    assert not any(s.get("if") or s.get("continue-on-error") for s in job["steps"])
    assert len(job["steps"]) == 3


@pytest.mark.parametrize(
    "mutation", ["manual", "pr", "permission", "environment", "artifact", "skip"]
)
def test_release_publish_boundary_rejects_weakened_workflow(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    wf = copy.deepcopy(load_yaml(RELEASE))
    job = wf["jobs"]["publish"]
    if mutation in {"manual", "pr"}:
        job["if"] = "startsWith(github.ref, 'refs/tags/v')" if mutation == "manual" else "true"
    elif mutation == "permission":
        wf["jobs"]["build"]["permissions"] = {"id-token": "write"}
    elif mutation == "environment":
        job["environment"]["name"] = "unprotected"
    elif mutation == "artifact":
        job["steps"][0]["with"]["name"] = "other"
    else:
        job["steps"][1]["continue-on-error"] = True
    monkeypatch.setitem(globals(), "load_yaml", lambda _: wf)
    with pytest.raises(AssertionError):
        test_publish_job_is_least_privilege()


def _release_fixture(
    root: Path,
    *,
    wheel_version: str = "0.1.0",
    sdist_version: str = "0.1.0",
    sdist_project_version: str = "0.1.0",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "swarm-arena"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-10-27\n\n- Release.\n",
        encoding="utf-8",
    )
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    with zipfile.ZipFile(dist / "swarm_arena-0.1.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr(
            "swarm_arena-0.1.0.dist-info/METADATA", f"Name: swarm-arena\nVersion: {wheel_version}\n"
        )
    with tarfile.open(dist / "swarm_arena-0.1.0.tar.gz", "w:gz") as sdist:
        for name, value in {
            "PKG-INFO": f"Name: swarm-arena\nVersion: {sdist_version}\n",
            "pyproject.toml": (
                f'[project]\nname = "swarm-arena"\nversion = "{sdist_project_version}"\n'
            ),
        }.items():
            payload = value.encode("utf-8")
            info = tarfile.TarInfo(f"swarm_arena-0.1.0/{name}")
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))
    return dist


def test_release_guard_accepts_exact_versions_and_records_hashes(tmp_path: Path) -> None:
    dist = _release_fixture(tmp_path)
    record = _release_record(tmp_path, dist, "v0.1.0", "a" * 40, "0.1.0", "0.1.0", "0.1.0")
    assert record == {
        "schema_version": 1,
        "commit": "a" * 40,
        "tag": "v0.1.0",
        "version": "0.1.0",
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dist.iterdir()},
    }


@pytest.mark.parametrize(
    "tag",
    [
        "v0.2.0",
        "v0.1.1",
        "v1.1.0",
        "0.1.0",
        "v0x1y0",
        "v0.1.0rc1",
        "v0.1.0.dev0",
        "v0.1.0+local",
        "v00.1.0",
        "v0.1.0\n",
        "refs/tags/v0.1.0",
    ],
)
def test_release_guard_rejects_tag(tmp_path: Path, tag: str) -> None:
    dist = _release_fixture(tmp_path)
    with pytest.raises(ValueError, match="release tag"):
        _release_record(tmp_path, dist, tag, "a" * 40, "0.1.0", "0.1.0", "0.1.0")


@pytest.mark.parametrize(
    "heading",
    [
        "## [0.1.0] - Unreleased",
        "## [0x1y0] - 2026-10-27",
        "## [0.1.0rc1] - 2026-10-27",
        "## [0.1.0] - 2026-02-30",
        "## [0.1.0] - 2026-13-01",
        "## [0.1.0] - 2026-1-1",
        "## [0.1.0] - 2026-10-27 extra",
        "## [0.1.0] - 9999-01-01",
        "## [0.2.0] - 2026-10-27",
        "",
        "## [0.1.0] - 2026-10-27\n## [0.1.0] - 2026-10-27",
        "## [0.1.0] - 2026-10-27\n## [0x1y0] - 2026-10-27",
    ],
)
def test_release_guard_rejects_changelog(tmp_path: Path, heading: str) -> None:
    dist = _release_fixture(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n## [Unreleased]\n{heading}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changelog"):
        _release_record(tmp_path, dist, "v0.1.0", "a" * 40, "0.1.0", "0.1.0", "0.1.0")


@pytest.mark.parametrize(
    "field", ["project", "wheel", "sdist", "sdist-project", "installed", "runtime", "docs"]
)
def test_release_guard_rejects_version_mismatch(tmp_path: Path, field: str) -> None:
    dist = _release_fixture(
        tmp_path,
        wheel_version="0.2.0" if field == "wheel" else "0.1.0",
        sdist_version="0.2.0" if field == "sdist" else "0.1.0",
        sdist_project_version="0.2.0" if field == "sdist-project" else "0.1.0",
    )
    if field == "project":
        project = tmp_path / "pyproject.toml"
        project.write_text(
            project.read_text(encoding="utf-8").replace("0.1.0", "0.2.0"), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="release"):
        _release_record(
            tmp_path,
            dist,
            "v0.1.0",
            "a" * 40,
            "0.2.0" if field == "installed" else "0.1.0",
            "0.2.0" if field == "runtime" else "0.1.0",
            "0.2.0" if field == "docs" else "0.1.0",
        )


@pytest.mark.parametrize("mutation", ["extra", "missing", "name", "commit"])
def test_release_guard_rejects_artifact_set(tmp_path: Path, mutation: str) -> None:
    dist = _release_fixture(tmp_path)
    if mutation == "extra":
        (dist / "unexpected.whl").write_bytes(b"other")
    elif mutation == "missing":
        next(dist.glob("*.whl")).unlink()
    elif mutation == "name":
        wheel = next(dist.glob("*.whl"))
        wheel.rename(dist / "swarm_arena-0.2.0-py3-none-any.whl")
    with pytest.raises(ValueError, match="release"):
        _release_record(
            tmp_path,
            dist,
            "v0.1.0",
            "bad" if mutation == "commit" else "a" * 40,
            "0.1.0",
            "0.1.0",
            "0.1.0",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "bytes",
        "extra",
        "missing",
        "commit",
        "tag",
        "hash",
        "tag-matches-ref-only",
        "tag-matches-version-only",
        "version-lookalike",
        "version-leading-zero",
        "version-prerelease",
        "version-trailing-newline",
    ],
)
def test_release_downloaded_hash_guard(tmp_path: Path, mutation: str) -> None:
    bundle = tmp_path / "release"
    dist = _release_fixture(bundle)
    record = _release_record(bundle, dist, "v0.1.0", "a" * 40, "0.1.0", "0.1.0", "0.1.0")
    wheel = next(dist.glob("*.whl"))
    ref = "v0.1.0"
    if mutation == "bytes":
        wheel.write_bytes(wheel.read_bytes() + b"tampered")
    elif mutation == "extra":
        (dist / "extra.whl").write_bytes(b"extra")
    elif mutation == "missing":
        wheel.unlink()
    elif mutation in {"commit", "tag"}:
        record[mutation] = "different"
    elif mutation == "hash":
        record["sha256"][wheel.name] = "0" * 64
    elif mutation == "tag-matches-ref-only":
        record["tag"] = ref = "v0.1.1"
    elif mutation == "tag-matches-version-only":
        ref = "v0.1.1"
    elif mutation.startswith("version-"):
        version = {
            "version-lookalike": "0x1y0",
            "version-leading-zero": "00.1.0",
            "version-prerelease": "0.1.0rc1",
            "version-trailing-newline": "0.1.0\n",
        }[mutation]
        record["version"] = version
        record["tag"] = ref = "v" + version
        # Keep every other verifier condition valid to isolate its version regex.
        for path in list(dist.iterdir()):
            renamed = path.with_name(path.name.replace("0.1.0", version))
            path.rename(renamed)
            record["sha256"][renamed.name] = record["sha256"].pop(path.name)
    (bundle / "release.json").write_text(json.dumps(record), encoding="utf-8")
    [command] = steps_run(load_yaml(RELEASE)["jobs"]["publish"])
    # Execute only the embedded local verifier, never the publish action.
    assert command.startswith("python3 - <<'PY'\n")
    assert command.endswith("\nPY\n")
    code = command.removeprefix("python3 - <<'PY'\n").removesuffix("\nPY\n")
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env={"GITHUB_SHA": "a" * 40, "GITHUB_REF_NAME": ref},
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert (result.returncode == 0) == (mutation == "none"), result.stdout + result.stderr
    if mutation.startswith("version-"):
        assert "invalid release version" in result.stderr
    elif mutation.startswith("tag-matches-"):
        assert "release tag mismatch" in result.stderr


@pytest.mark.parametrize(
    ("tag", "runtime_version", "docs_version"),
    [
        ("v0.1.0", "0.1.0", "0.1.0"),
        ("v0.2.0", "0.1.0", "0.1.0"),
        ("v0.1.1", "0.1.0", "0.1.0"),
        ("v1.1.0", "0.1.0", "0.1.0"),
        ("v0x1y0", "0.1.0", "0.1.0"),
        ("v0.1.0", "0.2.0", "0.1.0"),
        ("v0.1.0", "0.1.0", "0.2.0"),
    ],
)
def test_release_guard_cli(
    tmp_path: Path,
    tag: str,
    runtime_version: str,
    docs_version: str,
) -> None:
    dist = _release_fixture(tmp_path)
    (tmp_path / "tests").mkdir()
    script = tmp_path / "tests/test_ci_config.py"
    script.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/conf.py").write_text(f'release = "{docs_version}"\n', encoding="utf-8")
    # A synthetic installed distribution tests the CLI without a resolver or network.
    target = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(target)
    [site] = target.glob("lib/python*/site-packages")
    (site / "swarm_arena").mkdir()
    (site / "swarm_arena/__init__.py").write_text(
        f'__version__ = "{runtime_version}"\n', encoding="utf-8"
    )
    info = site / "swarm_arena-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: swarm-arena\nVersion: 0.1.0\n", encoding="utf-8")
    record = tmp_path / "release.json"
    environment = dict(os.environ, GITHUB_REF_NAME=tag)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--check-release",
            "--dist",
            str(dist),
            "--tag",
            environment["GITHUB_REF_NAME"],
            "--commit",
            "a" * 40,
            "--wheel-python",
            str(target / "bin/python"),
            "--record",
            str(record),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    valid = tag == "v0.1.0" and runtime_version == docs_version == "0.1.0"
    assert (result.returncode == 0) == valid, result.stdout + result.stderr
    assert record.exists() == valid
    if record.exists():
        assert json.loads(record.read_text(encoding="utf-8"))["tag"] == tag
    elif tag != "v0.1.0":
        assert "release tag must exactly match" in result.stderr
    else:
        assert "release installed, runtime and docs versions must agree" in result.stderr


@pytest.mark.parametrize("wrong_path", [None, "prefix", "module", "metadata_root", "failure"])
def test_release_wheel_probe_requires_isolated_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_path: str | None,
) -> None:
    prefix = tmp_path / "venv"
    python = prefix / "bin/python"
    probe = {
        "installed": "0.1.0",
        "runtime": "0.1.0",
        "prefix": str(prefix),
        "module": str(prefix / "lib/site-packages/swarm_arena/__init__.py"),
        "metadata_root": str(prefix / "lib/site-packages"),
    }
    if wrong_path in probe:
        probe[wrong_path] = str(tmp_path / "checkout")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[:3] == [str(python), "-I", "-c"]
        assert list(Path(kwargs["cwd"]).iterdir()) == []
        assert kwargs["env"] == {"PATH": str(python.parent), "PYTHONUTF8": "1"}
        return subprocess.CompletedProcess(command, int(wrong_path == "failure"), json.dumps(probe))

    monkeypatch.setattr(subprocess, "run", run)
    if wrong_path is None:
        assert _installed_release(python) == ("0.1.0", "0.1.0")
    else:
        with pytest.raises(ValueError, match="release"):
            _installed_release(python)


def test_release_guard_errors_do_not_echo_values(tmp_path: Path) -> None:
    dist = _release_fixture(tmp_path)
    secret = "sentinel-private-release-input"
    with pytest.raises(ValueError, match="release tag") as caught:
        _release_record(tmp_path, dist, secret, "a" * 40, "0.1.0", "0.1.0", "0.1.0")
    error = caught.value
    assert secret not in str(error)
    assert secret not in repr(error)
    assert secret not in repr(error.args)
    assert secret not in repr(getattr(error, "__notes__", []))
    assert error.__cause__ is None
    assert secret not in "".join(traceback.format_exception(error))


def test_releasing_checklist() -> None:
    text = (REPO / "RELEASING.md").read_text(encoding="utf-8").lower()
    for phrase in (
        "trusted publisher",
        "release.yml",
        "environment `pypi`",
        "uv version",
        "CHANGELOG.md",
        "uv build",
        "twine check",
        "git tag -a v",
        "git push origin v",
        "go-ahead",
        "yank",
        "required reviewer",
        "v*",
        "exact commit",
        "headsha",
        "conclusion",
        "sha256",
        "already-prepared",
        "second 0.1.0",
        "fallback",
        "sees 0.1.0 on pypi",
        "test_readme_quickstart",
        "sdist",
        "empty",
        "--check-release",
    ):
        assert phrase.lower() in text


def _releasing_step(number: int) -> str:
    """Read one numbered checklist step, excluding every neighboring step."""
    text = (REPO / "RELEASING.md").read_text(encoding="utf-8")
    [section] = re.findall(rf"^{number}\. (.*?)(?=^\d+\. |\Z)", text, re.M | re.S)
    return section


def test_releasing_requires_exact_commit_ci() -> None:
    step = " ".join(_releasing_step(3).lower().replace("**", "").replace("`", "").split())
    for phrase in (
        "before tagging, verify required ci success for the exact commit to be tagged, "
        "including all required matrix jobs.",
        "record the run url, headsha, status: completed and conclusion: success.",
        "a successful run on an earlier sha or a pr merge sha that differs from the "
        "candidate does not qualify.",
        "if the candidate changes, obtain new matching evidence before continuing.",
        "before approving the waiting pypi environment run, verify again that the tag "
        "resolves to the owner-approved commit with successful exact-head ci.",
    ):
        assert phrase in step


def test_releasing_candidate_commands() -> None:
    [code] = re.findall(r"```sh\n(.*?)```", _releasing_step(2), re.S)
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    assert lines[:3] == [
        "RELEASE_TMP=$(mktemp -d)",
        "RELEASE_VERSION=X.Y.Z",
        "RELEASE_COMMIT=$(git rev-parse HEAD)",
    ]
    assert _release_shell_lines("\n".join(lines[3:])) == [
        ["uv", "build", "--out-dir", "$RELEASE_TMP/dist", "--no-create-gitignore"],
        ["uvx", "twine", "check", "--strict", "$RELEASE_TMP/dist/*"],
        ["uv", "venv", "$RELEASE_TMP/venv", "--python", "3.11"],
        [
            "uv",
            "pip",
            "install",
            "--python",
            "$RELEASE_TMP/venv/bin/python",
            "$RELEASE_TMP/dist/*.whl",
        ],
        ["mkdir", "$RELEASE_TMP/empty"],
        ["(", "cd", "$RELEASE_TMP/empty", "&&", "$RELEASE_TMP/venv/bin/swarm-arena", "--help", ")"],
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "tests/test_ci_config.py",
            "--check-release",
            "--dist",
            "$RELEASE_TMP/dist",
            "--tag",
            "v$RELEASE_VERSION",
            "--commit",
            "$RELEASE_COMMIT",
            "--wheel-python",
            "$RELEASE_TMP/venv/bin/python",
            "--record",
            "$RELEASE_TMP/release.json",
        ],
        [
            "SWARM_ARENA_DOCS_VENV=$RELEASE_TMP/venv",
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "tests/test_docs.py::test_readme_quickstart",
            "-v",
        ],
    ]


@pytest.mark.parametrize("suffix", [" '#' || true", r" \# || true"])
def test_release_smoke_rejects_literal_hash_masking(
    monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    wf = copy.deepcopy(load_yaml(RELEASE))
    for step in wf["jobs"]["build"]["steps"]:
        if step.get("name") == "README quickstart from the wheel in an empty folder":
            step["run"] = step["run"].rstrip() + suffix
    monkeypatch.setitem(globals(), "load_yaml", lambda _: wf)
    with pytest.raises(AssertionError):
        test_release_build_job_checks_the_wheel()


@pytest.mark.parametrize(
    ("hour", "released_on", "valid"),
    [(19, "2026-10-28", True), (19, "2026-10-29", False), (5, "2026-10-28", False)],
)
def test_release_date_uses_latest_calendar_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hour: int, released_on: str, valid: bool
) -> None:
    class FrozenDate(date):
        @classmethod
        def today(cls) -> "FrozenDate":
            return cls(2026, 10, 27)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> "FrozenDatetime":
            assert tz is UTC
            return cls(2026, 10, 27, hour, tzinfo=UTC)

    monkeypatch.setitem(globals(), "date", FrozenDate)
    monkeypatch.setitem(globals(), "datetime", FrozenDatetime)
    dist = _release_fixture(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        f"## [Unreleased]\n\n## [0.1.0] - {released_on}\n", encoding="utf-8"
    )
    if valid:
        assert _release_record(tmp_path, dist, "v0.1.0", "a" * 40, "0.1.0", "0.1.0", "0.1.0")
    else:
        with pytest.raises(ValueError, match="changelog date"):
            _release_record(tmp_path, dist, "v0.1.0", "a" * 40, "0.1.0", "0.1.0", "0.1.0")


def test_contributing_names_public_snapshot() -> None:
    text = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "`PUBLIC_ALL` in `tests/_surface.py`" in text
    assert "spec §9 order" in text
    assert "`PUBLIC`" not in text


def test_release_build_action_pin_comments() -> None:
    text = RELEASE.read_text(encoding="utf-8")
    for action, sha, tag in (
        ("actions/checkout", "3d3c42e5aac5ba805825da76410c181273ba90b1", "v7.0.1"),
        ("astral-sh/setup-uv", "1c37ad07a6a961277cf70c0d37d6f313000f5884", "v10.3.0"),
        ("actions/upload-artifact", "cf430e030ddbb5b0abf93d22962f4752f3646cd9", "v7.0.2"),
    ):
        assert f"{action}@{sha} # {tag}" in text


def test_releasing_updates_both_jobs_action_pins() -> None:
    step = " ".join(_releasing_step(7).split())
    for phrase in (
        "build job",
        "publish job",
        "actions/checkout",
        "astral-sh/setup-uv",
        "actions/upload-artifact",
        "actions/download-artifact",
        "pypa/gh-action-pypi-publish",
        "git ls-remote",
        "full commit SHAs",
        "comment",
        "release notes",
        "test_release_build_job_checks_the_wheel",
        "test_publish_job_is_least_privilege",
        "test_release_build_action_pin_comments",
    ):
        assert phrase in step


@pytest.mark.parametrize("canary", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "import",
        "runtime",
        "mark",
        "xfail",
        "setup",
        "teardown",
        "deselect",
        "live",
        "unexpected-live",
        "outside",
    ],
)
def test_canary_rejects_bypassed_checks(tmp_path: Path, kind: str, canary: bool) -> None:
    llm = tmp_path / "tests/llm"
    llm.mkdir(parents=True)
    (llm / "conftest.py").write_text(
        (REPO / "tests/llm/conftest.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = llm_live\n", encoding="utf-8")
    (tmp_path / "blocked_dependency.py").write_text(
        'raise ModuleNotFoundError("blocked module", name="blocked_dependency")\n',
        encoding="utf-8",
    )
    sources = {
        "import": 'pytest.importorskip("blocked_dependency")\ndef test_probe(): pass\n',
        "runtime": 'def test_probe(): pytest.importorskip("blocked_dependency")\n',
        "mark": '@pytest.mark.skip(reason="synthetic skip")\ndef test_probe(): pass\n',
        "xfail": '@pytest.mark.xfail(reason="synthetic xfail")\ndef test_probe(): assert False\n',
        "setup": (
            '@pytest.fixture(autouse=True)\ndef skip_setup(): pytest.skip("setup")\n'
            "def test_probe(): pass\n"
        ),
        "teardown": (
            "@pytest.fixture(autouse=True)\ndef skip_teardown():\n    yield\n"
            '    pytest.skip("teardown")\ndef test_probe(): pass\n'
        ),
        "deselect": "def test_probe(): pass\n",
        "live": "@pytest.mark.llm_live\ndef test_live_manager_round(): assert False\n",
        "unexpected-live": "@pytest.mark.llm_live\ndef test_probe(): assert False\n",
        "outside": 'def test_probe(): pytest.skip("outside llm")\n',
    }
    target = tmp_path / "tests" if kind == "outside" else llm
    probe = target / ("test_manager.py" if kind == "live" else "test_probe.py")
    probe.write_text("import pytest\n" + sources[kind], encoding="utf-8")
    # A passing test keeps ordinary import-time skips from becoming NO_TESTS_COLLECTED.
    passing = llm / "test_keep.py"
    passing.write_text("def test_keep(): pass\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("SWARM_ARENA_UPSTREAM_CANARY", None)
    if canary:
        env["SWARM_ARENA_UPSTREAM_CANARY"] = "installed.json"
    selection = ["-k", "test_keep"] if kind == "deselect" else []
    if kind in {"live", "unexpected-live"}:
        selection = ["-m", "not llm_live"]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            str(passing),
            str(probe),
            *selection,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    output = result.stdout + result.stderr
    if canary and kind not in {"live", "outside"}:
        assert result.returncode == (2 if kind == "import" else 1), output
        assert "upstream canary forbids" in output, output
    else:
        assert result.returncode == 0, output
        assert any(word in output for word in ("skipped", "xfailed", "deselected")), output


@pytest.mark.parametrize("bypass", ["blocked-import", "synthetic-skip"])
def test_canary_rejects_review_reproductions(tmp_path: Path, bypass: str) -> None:
    pytest.importorskip("agents")
    record = tmp_path / "installed.json"
    _record_upstream_versions(record)
    if bypass == "blocked-import":
        (tmp_path / "diskcache.py").write_text(
            'raise ModuleNotFoundError("blocked diskcache", name="diskcache")\n',
            encoding="utf-8",
        )
    script = """
import pytest
import sys
class SyntheticSkip:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            if item.name == 'test_cache_constructor_parameters':
                item.add_marker(pytest.mark.skip(reason='review synthetic skip'))
plugins = [SyntheticSkip()] if sys.argv[1] == 'synthetic-skip' else []
raise SystemExit(pytest.main([
    '-q', '-rs', '-p', 'no:cacheprovider',
    'tests/llm/test_upstream_facts.py', 'tests/llm/test_cache.py',
], plugins=plugins))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, bypass],
        cwd=REPO,
        env=dict(os.environ, PYTHONPATH=str(tmp_path), SWARM_ARENA_UPSTREAM_CANARY=str(record)),
        capture_output=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "upstream canary forbids" in output, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline workflow validation helpers")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record-upstream", type=Path)
    mode.add_argument("--check-release", action="store_true")
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--commit")
    parser.add_argument("--wheel-python", type=Path)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if args.record_upstream is not None:
        _record_upstream_versions(args.record_upstream)
    else:
        if any(
            value is None
            for value in (args.dist, args.tag, args.commit, args.wheel_python, args.record)
        ):
            parser.error("release checks require dist, tag, commit, wheel-python and record")
        _check_release(args.dist, args.tag, args.commit, args.wheel_python, args.record)


def test_setup_uv_refs_name_a_published_tag() -> None:
    # astral-sh/setup-uv publishes only full version tags, so `@v10` does not resolve.
    full = re.compile(r"astral-sh/setup-uv@(?:v\d+\.\d+\.\d+|[0-9a-f]{40})(?=\s|$)")
    paths = [*(REPO / ".github" / "workflows").glob("*.yml"), *(REPO / "docs").glob("*.md")]
    bad = {
        f"{path.relative_to(REPO)}:{number}"
        for path in paths
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "astral-sh/setup-uv@" in line and not full.search(line)
    }
    assert bad == set()


@pytest.mark.parametrize(
    ("line", "valid"),
    [
        ("      - uses: astral-sh/setup-uv@v10.3.0", True),
        ("- uses: astral-sh/setup-uv@1c37ad07a6a961277cf70c0d37d6f313000f5884 # v10.3.0", True),
        ("      - uses: astral-sh/setup-uv@v10", False),
        ("      - uses: astral-sh/setup-uv@v10.3.0^invalid", False),
        # YAML keeps an attached '#' in the value; a comment needs a space first.
        ("      - uses: astral-sh/setup-uv@v10.3.0#^invalid", False),
    ],
)
def test_setup_uv_ref_pattern(line: str, valid: bool) -> None:
    full = re.compile(r"astral-sh/setup-uv@(?:v\d+\.\d+\.\d+|[0-9a-f]{40})(?=\s|$)")
    assert bool(full.search(line)) is valid
