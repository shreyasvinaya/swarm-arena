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
    commands = [shlex.split(run) for run in steps_run(job)]
    assert any(
        command[:2] == ["uv", "sync"]
        and {"--locked", "--all-extras"} <= set(command)
        and any(command[i : i + 2] == ["--group", "docs"] for i in range(len(command) - 1))
        for command in commands
    )
    assert "3.13" in json.dumps(job)
