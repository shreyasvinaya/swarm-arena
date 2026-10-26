"""Offline checks for documentation and workflow configuration."""

import argparse
import importlib.metadata
import json
import os
import shlex
import tomllib
import traceback
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
    lexer = shlex.shlex(run, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    command = list(lexer)
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
    assert steps_uses(job) == ["actions/checkout@v7", "astral-sh/setup-uv@v10"]
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
    ("canary", "versions", "valid"),
    [
        (False, {"openai-agents": "0.24.0", "litellm": "1.107.0rc2"}, True),
        (False, {"openai-agents": "0.25.0", "litellm": "1.108.0"}, False),
        (False, {"openai-agents": "0.24.0", "litellm": "1.107.0rc1"}, False),
        (True, {"openai-agents": "0.25.0", "litellm": "1.108.0"}, True),
        (True, {"openai-agents": "0.24.0", "litellm": "1.107.0rc2"}, True),
        (True, {"openai-agents": "0.23.0", "litellm": "1.108.0"}, False),
        (True, {"openai-agents": "0.25.0", "litellm": "1.106.4"}, False),
    ],
)
def test_upstream_version_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    canary: bool,
    versions: dict[str, str],
    valid: bool,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    monkeypatch.delenv("SWARM_ARENA_UPSTREAM_CANARY", raising=False)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record the installed canary versions")
    parser.add_argument("--record-upstream", type=Path, required=True)
    _record_upstream_versions(parser.parse_args().record_upstream)
