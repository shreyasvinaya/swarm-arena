"""Probe preparation reports upstream type names without secret values."""

from pathlib import Path
from typing import NoReturn

import pytest
from _fakes.llm_run import open_llm_run
from _fakes.planning_llm import planner_config
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import _exception_type_name
from swarm_arena.config import ModelSpec
from swarm_arena.llm import _hooks


async def test_probe_preparation_preserves_upstream_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "private-probe-preparation-sentinel"

    def fail(spec: ModelSpec) -> NoReturn:
        raise ImportError(secret)

    monkeypatch.setattr(_hooks, "build_model", fail)
    async with open_llm_run(tmp_path, config=planner_config(tmp_path)) as run:
        with pytest.raises(ValueError) as caught:
            await _hooks.probe_provider("fake", run.scope("planner"))
        assert _exception_type_name(caught.value) == "ImportError"
        assert_secret_absent(caught.value, secret)
        assert run.events("llm_call") == []
