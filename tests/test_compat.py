import builtins
import importlib
import importlib.machinery
import importlib.util
import os
import re
import sys
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

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


_IO_GUARD_ACTIVE = False
_IO_EVENTS = frozenset(
    {
        "open",
        "os.mkdir",
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.fork",
        "socket.connect",
        "socket.bind",
        "socket.sendto",
    }
)


def _importing(args: tuple[object, ...]) -> bool:
    # Only importlib's own file operation is exempt. A module executing under
    # bootstrap frames must not gain permission for its top-level file I/O.
    frame = sys._getframe(2)
    if (
        frame.f_code.co_filename != "<frozen importlib._bootstrap_external>"
        or frame.f_code.co_name not in {"get_data", "_write_atomic"}
    ):
        return False
    opened_arg = args[0]
    if isinstance(opened_arg, int) and frame.f_code.co_name == "_write_atomic":
        # FileIO wraps the descriptor just opened for this bytecode temp file.
        if opened_arg != frame.f_locals.get("fd"):
            return False
        opened_arg = frame.f_locals["path_tmp"]
    if not isinstance(opened_arg, (str, bytes)):
        return False
    opened = Path(os.path.abspath(os.fsdecode(opened_arg)))
    if not any(opened.is_relative_to(os.path.abspath(entry)) for entry in sys.path):
        return False
    operation = frame
    while frame is not None and frame.f_code.co_filename.startswith("<frozen importlib."):
        origin = getattr(frame.f_locals.get("self"), "path", None)
        if isinstance(origin, str):
            source = Path(os.path.abspath(origin))
            suffixes = (".py", ".pyc", ".pyi", *importlib.machinery.EXTENSION_SUFFIXES)
            if not str(source).endswith(suffixes):
                return False
            candidates = {source}
            if source.suffix in {".py", ".pyi"}:
                candidates.add(Path(importlib.util.cache_from_source(str(source))))
            if operation.f_code.co_name == "get_data":
                # Audit open flags, including os.open, must describe a read.
                flags = args[2]
                return (
                    opened in candidates
                    and isinstance(flags, int)
                    and not flags
                    & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
                )
            cached = Path(os.path.abspath(operation.f_locals["path"]))
            return (
                cached in candidates
                and cached.parent.name == "__pycache__"
                and str(opened) == operation.f_locals["path_tmp"]
            )
        frame = frame.f_back
    return False


def _forbid(event: str, args: tuple[object, ...]) -> None:
    if _IO_GUARD_ACTIVE and event in _IO_EVENTS and not (event == "open" and _importing(args)):
        raise AssertionError(f"I/O in a component __init__ (§9 rule 3): {event}")


# Audit hooks cannot be removed; this one is dormant outside the constructor call.
sys.addaudithook(_forbid)


@pytest.mark.parametrize(("module", "build"), COMPONENTS)
def test_component_init_does_no_io(
    module: str, build: Callable[[Path], object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global _IO_GUARD_ACTIVE

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
    _IO_GUARD_ACTIVE = True
    try:
        build(tmp_path)
    finally:
        _IO_GUARD_ACTIVE = False
    assert [directory.name for directory in dirs if any(directory.iterdir())] == []
    assert not (tmp_path / "arena").exists()


def test_io_guard_allows_lazy_imports_only(monkeypatch: pytest.MonkeyPatch) -> None:
    global _IO_GUARD_ACTIVE

    monkeypatch.delitem(sys.modules, "colorsys", raising=False)
    _IO_GUARD_ACTIVE = True
    try:
        importlib.import_module("colorsys")  # the import system reads a module file
        with pytest.raises(AssertionError, match="open"):
            open(__file__, "rb").close()
    finally:
        _IO_GUARD_ACTIVE = False


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


@pytest.mark.parametrize("filename", ["payload.txt", "unrelated.py", "__pycache__/unrelated.pyc"])
def test_io_guard_rejects_top_level_io_in_lazy_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    global _IO_GUARD_ACTIVE

    payload = tmp_path / filename
    payload.parent.mkdir(exist_ok=True)
    payload.write_text("payload", encoding="utf-8")
    module = tmp_path / "lazy_io_probe.py"
    module.write_text(
        "import builtins\nbuiltins._compat_probe_started = True\n"
        f"open({str(payload)!r}, 'rb').close()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(builtins, "_compat_probe_started", False, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "lazy_io_probe", raising=False)
    # Avoid testing importlib's cache-directory creation, which remains forbidden.
    (tmp_path / "__pycache__").mkdir(exist_ok=True)
    _IO_GUARD_ACTIVE = True
    try:
        with pytest.raises(AssertionError, match=r"I/O in a component.*open"):
            importlib.import_module("lazy_io_probe")
        assert vars(builtins)["_compat_probe_started"] is True
    finally:
        _IO_GUARD_ACTIVE = False
        sys.modules.pop("lazy_io_probe", None)
