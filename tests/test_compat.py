import builtins
import importlib
import io
import os
import re
import socket
import subprocess
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, NoReturn

import pytest
from _fakes.secrecy import assert_secret_absent
from _pytest.mark.structures import ParameterSet
from _surface import PUBLIC_ALL, deps_present, public_object, resolve

from swarm_arena import (
    Chain,
    FunctionTool,
    KeepTopTeams,
    Patience,
    Plan,
    ProgramJudge,
    RunConfig,
    ScriptCheck,
    ScriptedManager,
    ScriptJudge,
    ScriptTool,
    SuccessiveHalving,
)
from swarm_arena._model.migrate import CURRENT, upgrade
from swarm_arena.board import ArenaBoard
from swarm_arena.executor import LocalExecutor
from swarm_arena.scoring import Lexicographic

ABSTRACT: dict[str, frozenset[str]] = {
    "swarm_arena.tool.Tool": frozenset({"command", "collect"}),
    "swarm_arena.scoring.Check": frozenset({"check"}),
    "swarm_arena.scoring.Judge": frozenset({"judge"}),
    "swarm_arena.manager.Manager": frozenset({"run_round"}),
    "swarm_arena.planning.Planner": frozenset({"plan"}),
    "swarm_arena.planning.Policy": frozenset({"after_round"}),
    "swarm_arena.executor.Executor": frozenset({"submit", "capacity", "aclose"}),
    "swarm_arena.board.Board": frozenset(
        {
            "put_candidate",
            "put_score",
            "put_learning",
            "put_decision",
            "frontier",
            "snapshot",
            "best",
            "learnings",
            "search_learnings",
            "get_learning",
            "get_candidate",
            "scores",
            "lineage",
            "leaderboard",
        }
    ),
    "swarm_arena.board.BoardView": frozenset(
        {
            "best",
            "learnings",
            "search_learnings",
            "get_learning",
            "get_candidate",
            "scores",
            "lineage",
            "leaderboard",
        }
    ),
    "swarm_arena.board.Curator": frozenset({"curate"}),
}


@pytest.mark.parametrize("dotted", ABSTRACT)
def test_abstract_methods_frozen(dotted: str) -> None:
    assert resolve(dotted).__abstractmethods__ == ABSTRACT[dotted]


def test_every_public_abc_is_frozen() -> None:
    found = {
        f"{module}.{name}"
        for module, names in PUBLIC_ALL.items()
        if deps_present(module)
        for name in names
        if deps_present(module, name)
        and isinstance(obj := public_object(module, name), type)
        and getattr(obj, "__abstractmethods__", frozenset())
    }
    assert found == set(ABSTRACT)


class _NoReview:
    async def review(self, plan: Plan) -> Any:
        raise AssertionError("construction must not request review")


COMPONENTS: list[ParameterSet] = [
    pytest.param(
        "swarm_arena", lambda tmp: ScriptTool(command=["python", "run.py"]), id="ScriptTool"
    ),
    pytest.param(
        "swarm_arena", lambda tmp: FunctionTool("_fakes.tools:compat_probe"), id="FunctionTool"
    ),
    pytest.param(
        "swarm_arena", lambda tmp: ScriptCheck(command=["python", "check.py"]), id="ScriptCheck"
    ),
    pytest.param(
        "swarm_arena", lambda tmp: ScriptJudge(command=["python", "score.py"]), id="ScriptJudge"
    ),
    pytest.param(
        "swarm_arena", lambda tmp: ProgramJudge("_fakes.tools:compat_judge"), id="ProgramJudge"
    ),
    pytest.param(
        "swarm_arena.scoring",
        lambda tmp: Lexicographic([ProgramJudge("_fakes.tools:compat_judge")]),
        id="Lexicographic",
    ),
    pytest.param(
        "swarm_arena",
        lambda tmp: ScriptedManager(requests=[{"inputs": {"x": 1.0}}]),
        id="ScriptedManager",
    ),
    pytest.param("swarm_arena", lambda tmp: KeepTopTeams(2), id="KeepTopTeams"),
    pytest.param(
        "swarm_arena",
        lambda tmp: SuccessiveHalving(keep=0.5, min_per_team=1),
        id="SuccessiveHalving",
    ),
    pytest.param("swarm_arena", lambda tmp: Patience(rounds=2), id="Patience"),
    pytest.param("swarm_arena", lambda tmp: Chain(KeepTopTeams(2), Patience(rounds=2)), id="Chain"),
    pytest.param("swarm_arena.executor", lambda tmp: LocalExecutor(), id="LocalExecutor"),
    pytest.param(
        "swarm_arena.board", lambda tmp: ArenaBoard(arena_root=str(tmp / "arena")), id="ArenaBoard"
    ),
    *[
        pytest.param(
            "swarm_arena.llm",
            build,
            id=name,
            marks=pytest.mark.skipif(
                not deps_present("swarm_arena.llm"), reason="needs the llm extra"
            ),
        )
        for name, build in (
            (
                "AgentsSDKManager",
                lambda tmp: resolve("swarm_arena.llm.AgentsSDKManager")(model="opus"),
            ),
            (
                "LLMJudge",
                lambda tmp: resolve("swarm_arena.llm.LLMJudge")("Score 0 to 10.", ["opus"]),
            ),
            ("LLMPlanner", lambda tmp: resolve("swarm_arena.llm.LLMPlanner")("opus")),
            ("LLMCurator", lambda tmp: resolve("swarm_arena.llm.LLMCurator")(model="flash")),
            (
                "CollaborativePlanner",
                lambda tmp: resolve("swarm_arena.llm.CollaborativePlanner")(
                    resolve("swarm_arena.llm.LLMPlanner")("opus"), _NoReview()
                ),
            ),
        )
    ],
    pytest.param(
        "swarm_arena.ray",
        lambda tmp: resolve("swarm_arena.ray.RayExecutor")(),
        id="RayExecutor",
        marks=pytest.mark.skipif(not deps_present("swarm_arena.ray"), reason="needs the ray extra"),
    ),
]


def _forbid(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("I/O in a component __init__ (§9 rule 3)")


@pytest.mark.parametrize(("module", "build"), COMPONENTS)
def test_component_init_does_no_io(
    module: str, build: Callable[[Path], object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    importlib.import_module(module)
    # Resolve the test refs before guarding construction, just as for facade imports.
    importlib.import_module("_fakes.tools")
    dirs = [tmp_path / directory for directory in ("cwd", "home", "tmp", "cache")]
    for directory in dirs:
        directory.mkdir()
    monkeypatch.chdir(dirs[0])
    for var, directory in zip(("HOME", "TMPDIR", "XDG_CACHE_HOME"), dirs[1:], strict=True):
        monkeypatch.setenv(var, str(directory))
    monkeypatch.setattr(tempfile, "tempdir", None)
    with monkeypatch.context() as guard:
        for target, attr in (
            (socket.socket, "connect"),
            (subprocess.Popen, "__init__"),
            (builtins, "open"),
            (io, "open"),
            (os, "open"),
            (os, "mkdir"),
        ):
            guard.setattr(target, attr, _forbid)
        build(tmp_path)
    assert [directory.name for directory in dirs if any(directory.iterdir())] == []
    assert not (tmp_path / "arena").exists()


@pytest.mark.parametrize("kind", sorted(CURRENT))
def test_newer_schema_version_refused(kind: str) -> None:
    newer = CURRENT[kind] + 1
    with pytest.raises(ValueError, match="schema_version") as err:
        upgrade(kind, {"schema_version": newer})
    assert str(newer) in str(err.value)


def test_plan_and_run_files_refuse_newer_versions(tmp_path: Path) -> None:
    examples = Path(__file__).resolve().parents[1] / "examples" / "rosenbrock"
    for filename in ("plan.toml", "run.toml"):
        original = (examples / filename).read_text(encoding="utf-8")
        text = "schema_version = 999\n" + re.sub(
            r"^schema_version\s*=.*\n", "", original, flags=re.MULTILINE
        )
        load: Callable[[], object]
        if filename == "plan.toml":
            load = partial(Plan.loads, text, base_dir=tmp_path)
        else:
            path = tmp_path / filename
            path.write_text(text, encoding="utf-8")
            load = partial(RunConfig.load, path)
        with pytest.raises(ValueError, match="schema_version") as err:
            load()
        assert "newer" in str(err.value)
        assert_secret_absent(err.value, "999", sanitized=True)
