"""Offline checks for documentation and workflow configuration."""

import json
import shlex
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from yaml.constructor import ConstructorError

REPO = Path(__file__).resolve().parents[1]


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
