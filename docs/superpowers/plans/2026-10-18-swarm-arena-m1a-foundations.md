# swarm-arena M1a Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the swarm-arena foundations: the project scaffold, every M1a data model, write-once storage for content-addressed files, the tool, check, and judge layer (`FunctionTool`, `ScriptTool`, `ScriptCheck`, `ScriptJudge`, `ProgramJudge`, `Lexicographic`), `LocalExecutor` with a launcher handshake and process-group control, and the `swarm_arena.testing` conformance kit, so that one tool run is staged, run, checked, and judged offline through the public API.

**Architecture:** Public modules are facades over private modules (`_model`, `_store`, `_tooling`, `_exec`, `_scoring`, `testing/_*`). Data types are frozen pydantic v2 models; stored formats carry `schema_version` and load through a migration registry. Files are snapshotted by sha256 into an `ArtifactStore` on a write-once `Store`. The driver plans a workspace (`Job.Stage`); the executor materializes it, starts the program through `python -I -m swarm_arena._launch` under a `ProcessSupervisor` (own process group; SIGTERM, grace, partial collect, SIGKILL), and uploads outputs and logs by hash. Checks and judges run in the driver; `ScriptCheck` and `ScriptJudge` send command jobs through the same executor.

**Tech Stack:** Python 3.11+ (POSIX), uv with `uv_build`, pydantic 2.11+, anyio 4.15 (asyncio backend), stdlib `tomllib`, `hashlib`, `resource`, `subprocess`; pytest 9 with the anyio plugin, ruff 0.17.1, ty 0.0.89, numpydoc 1.11, Sphinx 9.0 with sphinx-rtd-theme and myst-parser.

**Spec:** `docs/superpowers/specs/2026-10-16-swarm-arena-design.md` (APPROVED r8). Binding companion: `docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md` (R2 "M1a Foundations", R3 module map, R4 interface registry, R5 constraints). A signature written "as R4.x" below is the registry signature, copied exactly. API facts: `tasks/research/api-anyio-packaging.md`, `tasks/research/api-a2a.md`.

## Global Constraints

- Python `>=3.11`, POSIX only; `.python-version` is `3.11`; build backend `uv_build>=0.13.1,<0.14`; package version `0.1.0.dev0`.
- Core dependencies exactly: `anyio>=4.15,<5`, `pydantic>=2.11,<3`, `typing-extensions>=4.12`. No `[project.optional-dependencies]` table in M1a.
- Groups: `dev` = `pytest>=9,<10`, `ruff==0.17.1`, `ty==0.0.89`, `numpydoc>=1.11`; `docs` = `sphinx>=9.0,<10`, `sphinx-rtd-theme>=3.1`, `numpydoc>=1.11`, `myst-parser>=4`.
- The core never imports `agents`, `litellm`, `ray`, `s3fs`, `langfuse`, or `a2a`. No network access anywhere in M1a.
- anyio API on the asyncio backend only. Tests: `anyio_mode = "auto"`, fixture `anyio_backend` returns `"asyncio"`.
- A tool is never started with `uv run`. A leading `"python"` in `Command.argv` becomes `Command.python` or `sys.executable`.
- Public modules (`swarm_arena`, `.tool`, `.scoring`, `.manager`, `.planning`, `.board`, `.executor`, `.config`, `.protocol`, `.testing`) hold only a docstring, imports, `__all__`, and one `_publish(__name__, __all__)` call. The helper is imported as `from swarm_arena._facade import publish as _publish`, so no public name `publish` sits in a facade namespace. Every definition lives in a private module, and private modules never import from a facade. `swarm_arena/__init__.py` also imports the eight core facades after its `_publish` call (Task 1), so every published `__module__` is set once `swarm_arena` is imported. Absolute imports only.
- Models: data types subclass `FrozenModel` (`ConfigDict(frozen=True, extra="forbid")`); stored formats subclass `StoredModel` and set `schema_kind`. Every built-in nested `Config` sets `model_config = ConfigDict(extra="forbid", frozen=True)`, so a misspelled config key is an error.
- Times are aware UTC, truncated to milliseconds, serialized as `YYYY-MM-DDTHH:MM:SS.mmmZ`.
- A component `__init__` does no I/O and imports no referenced code (spec §9 rule 3). Imports of referenced code and checks go in `preflight()` or happen lazily.
- Every function and method is fully annotated, tests included. File I/O, `pathlib` calls, and subprocess calls inside `async def` go through sync helpers run with `anyio.to_thread.run_sync` (ruff `ASYNC230`, `ASYNC240`). `pytest.raises` of a built-in exception always passes `match=` (ruff `PT011`).
- Docstrings: numpydoc with an `Examples` section on every public object: each name in a public `__all__`, its public methods and properties defined in the class body, and its public nested classes. Model and dataclass docstrings document every field under `Parameters` (`schema_version` too); ABC docstrings document `config`. Function summaries start with an imperative verb. Examples import from public paths; an example that needs a subprocess or files uses `# doctest: +SKIP`.
- User-visible errors are `ValueError` or `RuntimeError` subclasses from `_errors.py`, or pydantic `ValidationError`. An error message never contains a secret value.
- Every task that adds public names adds them to its facade's `__all__`, to `PUBLIC` in `tests/test_public_api.py`, and one line to `CHANGELOG.md` under `## [Unreleased]` / `### Added`.
- Roadmap edits (R0 rule): tasks 1, 5, 12, 13, 14, 18, 19, and 20 edit `docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md` in their own commit and say why in the commit body. The roadmap already records the registry lines of Tasks 6 (R4.2 `ParentRef` and `ToolRequest.edits` comments), 11 (`snapshot_ref`), 15 (R4.13.8 and the R3.1 `_launch.py` row), 20 (`on_started`, `is_finite_score`), and 21 (R4.3 `Lexicographic` comment), so those parts are not edits.
- **Task gate** (the "run" step of every task ends with it; all must exit 0):
  ```bash
  uv run ruff check .
  uv run ruff format --check .
  uv run ty check
  uv run pytest
  uv run sphinx-build -W --keep-going -b html docs docs/_build/html
  ```
- Branch `m1a-foundations`. One commit per task: imperative subject, body says what and why. No AI or Claude co-author line and no "generated with" text in any commit or PR description.

## Review Focus

1. **An `editable` glob that also matches the tool's own script** (`editable = ["problem/*"]`, `command = ["python", "problem/run.py"]`, also spelled `./problem/run.py`): the user expects the script to stay protected. An edit to `problem/run.py` is refused, and a base parent whose overlay holds `problem/run.py` is not overlaid. Test: Task 19, `test_editable_glob_never_unprotects_script`.
2. **A file glob that matches nothing** (a typo such as `problem/*.phky`): the user expects a clear error at snapshot time that names the glob, not a script that fails later on a missing file. Test: Task 11, `test_glob_matching_nothing_raises`.
3. **A script that exits 0 without writing `swarm/result.json`, or writes invalid JSON or a misspelled key**: the user expects status `error` with the reason in the stderr log, not `ok` with empty data and not an exception. Test: Task 13, `test_exit_zero_without_valid_result_is_error`.
4. **A `ScriptCheck` script that crashes, times out, or writes no result**: the user expects the check to fail closed (`passed=False`, so the candidate is gated out), never to pass and never to raise. Test: Task 22, `test_crash_fails_closed`.
5. **The same tool written two ways** (TOML config that omits defaults, and a Python instance whose config dump includes them): the user expects one tool identity, so the base parent overlay applies. Test: Task 14, `test_tool_identity_ignores_spelling`.

---

### Task 1: Project scaffold, CI, docs skeleton, and public-surface tests

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `LICENSE`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.github/workflows/ci.yml`, `conftest.py`
- Create: `src/swarm_arena/{__init__.py,py.typed,_version.py,_facade.py,tool.py,scoring.py,manager.py,planning.py,board.py,executor.py,config.py,protocol.py}`, `src/swarm_arena/testing/__init__.py`, `src/swarm_arena/_cli/{__init__.py,main.py}`
- Create: `docs/conf.py`, `docs/index.md`, `docs/{getting-started,concepts,own-files,writing-a-connector,planning-modes,scoring-and-review,limits-and-safety,running-on-ray,arena,versioning,roadmap}.md`, `docs/api/index.md`, `docs/api/{swarm_arena,tool,scoring,manager,planning,board,executor,config,protocol,testing}.md`
- Modify: roadmap R3.1 (add the `_facade.py` row) and R5.3 (ruff `ignore` gains `RUF012`); R3.2 already detects extras by distribution
- Test: `tests/conftest.py`, `tests/_cli/test_main.py`, `tests/test_public_api.py`, `tests/test_docstrings.py`, `tests/test_import_isolation.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `swarm_arena.__version__: str`; `_facade.publish(module_name: str, names: Sequence[str]) -> None` (sets `__module__ = module_name` on every exported class and function, on public nested classes of exported classes, recursively, and on the routines in `vars(cls)` of each of those classes: `classmethod` and `staticmethod` unwrapped to `__func__`, properties through `fget`; without the method step, `doctest.DocTestFinder` skips every method example, because the private module skips the class and the facade skips the method); `_cli.main.main(argv: Sequence[str] | None = None) -> int` (R4.13.18); `tests/test_public_api.py::PUBLIC: dict[str, list[str]]` (later tasks append names); fixture `anyio_backend() -> str`.

- [ ] **Step 1: Preconditions, design-doc commit, branch**

Run: `uv --version`. Expected: `uv 0.13.1` or newer; otherwise run `uv self update` and check again.
```bash
git add docs/superpowers tasks/research
git commit -m "docs: add swarm-arena design spec, v1 roadmap, M1a plan, and research notes" \
  -m "The approved spec (r8), the binding roadmap, and the verified API notes that later milestones cite."
git checkout -b m1a-foundations
```

- [ ] **Step 2: Initialize the package and dependencies**

```bash
uv init --lib --name swarm-arena --python 3.11 --vcs none .
uv add "anyio>=4.15,<5" "pydantic>=2.11,<3" "typing-extensions>=4.12"
uv add --dev "pytest>=9,<10" "ruff==0.17.1" "ty==0.0.89" "numpydoc>=1.11"
uv add --group docs "sphinx>=9.0,<10" "sphinx-rtd-theme>=3.1" "numpydoc>=1.11" "myst-parser>=4"
```
Edit `pyproject.toml`: `version = "0.1.0.dev0"`, `description = "Run a swarm of agents on hard goals: fan out to teams, prune weak branches, converge on answers you can prove."`, `license = "MIT"`, `license-files = ["LICENSE"]`, classifiers `"Development Status :: 3 - Alpha"`, `"Operating System :: POSIX"`, `"Typing :: Typed"`, `[project.scripts] swarm-arena = "swarm_arena._cli.main:main"`, and the R5.3 `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.lint.per-file-ignores]`, `[tool.ruff.lint.flake8-tidy-imports]` (with its `banned-api` table), and `[tool.pytest.ini_options]` blocks verbatim, with these three deltas:
```toml
[tool.ruff.lint]
ignore = ["ANN401", "RUF012"]   # RUF012: ruff cannot see that FrozenModel subclasses are pydantic models

[tool.ruff.lint.per-file-ignores]   # added to R5.3's "tests/**" and "examples/**" entries
"src/swarm_arena/__init__.py" = ["E402"]   # the core facades are imported after _publish (Step 5)
"src/swarm_arena/llm/**" = ["TID251"]
"src/swarm_arena/ray/**" = ["TID251"]
"src/swarm_arena/_store/s3.py" = ["TID251"]
"src/swarm_arena/testing/_fake_model.py" = ["TID251"]
"tests/llm/**" = ["TID251"]
"tests/ray/**" = ["TID251"]
"tests/_store/test_s3.py" = ["TID251"]
"tests/testing/test_fake_model.py" = ["TID251"]

[tool.ty.environment]
python-version = "3.11"

[tool.ty.terminal]
error-on-warning = true
```
Run: `grep -n uv_build pyproject.toml`. Expected: `requires = ["uv_build>=0.13.1,<0.14"]` (fix by hand if uv wrote another range).

- [ ] **Step 3: Write the failing tests**

```python
# tests/_cli/test_main.py
def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"swarm-arena {swarm_arena.__version__}\n"

def test_no_arguments_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out.startswith("swarm-arena ")

def test_unknown_argument_is_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--nope"])
    assert exc.value.code == 2

def test_console_script_installed() -> None:
    exe = Path(sys.executable).parent / "swarm-arena"
    done = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=False)
    assert (done.returncode, done.stdout) == (0, f"swarm-arena {swarm_arena.__version__}\n")
```
```python
# tests/test_public_api.py
PUBLIC: dict[str, list[str]] = {m: [] for m in (
    "swarm_arena", "swarm_arena.tool", "swarm_arena.scoring", "swarm_arena.manager",
    "swarm_arena.planning", "swarm_arena.board", "swarm_arena.executor", "swarm_arena.config",
    "swarm_arena.protocol", "swarm_arena.testing")}

@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_all_snapshot(module: str) -> None:
    names = importlib.import_module(module).__all__
    assert sorted(names) == sorted(PUBLIC[module]) and len(set(names)) == len(names)

@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_names_report_public_module(module: str) -> None:
    mod = importlib.import_module(module)
    for name in mod.__all__:
        obj = getattr(mod, name)
        if isinstance(obj, type) or inspect.isfunction(obj):
            assert obj.__module__ == module, name
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            assert obj.__pydantic_complete__, name

def test_version_is_package_metadata() -> None:
    assert swarm_arena.__version__ == importlib.metadata.version("swarm-arena")
    assert "__version__" not in swarm_arena.__all__

def test_publish_lets_doctest_find_method_examples(monkeypatch: pytest.MonkeyPatch) -> None: ...
    # a local class Demo whose method m, classmethod c, staticmethod s, and property p each have a docstring example;
    # monkeypatch.setitem(sys.modules, "demo_facade", <a types.ModuleType holding Demo>); publish("demo_facade", ["Demo"])
    # → {t.name for t in doctest.DocTestFinder().find(sys.modules["demo_facade"])} contains demo_facade.Demo.m, .c, .s, .p

CORE = sorted(set(PUBLIC) - {"swarm_arena", "swarm_arena.testing"})

def test_root_import_publishes_core_facades() -> None:      # __module__ never depends on import order
    code = f"import sys, swarm_arena; print(sorted(set({CORE!r}) - set(sys.modules)))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
```
```python
# tests/test_docstrings.py
IGNORED = {"SA01", "ES01"}

def public_objects() -> list[str]: ...
    # dotted names: every name in each PUBLIC module's __all__; for classes, every non-underscore
    # function, classmethod, staticmethod, and property in vars(cls), and public nested classes
    # (qualname starts with the owner's qualname + "."), recursively

@pytest.mark.parametrize("name", public_objects())
def test_numpydoc(name: str) -> None:
    errors = [(code, msg) for code, msg in numpydoc.validate.validate(name)["errors"]
              if code not in IGNORED and not (code == "PR01" and _is_enum(name))]
    assert errors == []          # numpydoc reads Enum's constructor as parameters, hence the PR01 skip
```
```python
# tests/test_import_isolation.py
OPTIONAL = ("agents", "litellm", "ray", "s3fs", "langfuse", "a2a")

def test_core_imports_load_no_optional_dependency() -> None:
    code = ("import importlib, sys\n"
            f"for m in {sorted(PUBLIC)!r}: importlib.import_module(m)\n"
            f"print(sorted({{n.split('.')[0] for n in sys.modules}} & set({OPTIONAL!r})))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
```
(`test_import_isolation.py` repeats the module list of `PUBLIC` as a constant.)

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/_cli tests/test_public_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._cli'`.

- [ ] **Step 5: Implement the package skeleton**

- `src/swarm_arena/_version.py`: `__version__: str = importlib.metadata.version("swarm-arena")`.
- `src/swarm_arena/_facade.py`: `publish(module_name: str, names: Sequence[str]) -> None` (Interfaces).
- Each facade: a one-line module docstring naming its role (for example `"""Tool connector types."""`), `__all__: list[str] = []`, then `_publish(__name__, __all__)`, with the helper imported as `from swarm_arena._facade import publish as _publish` (a bare `publish` would be a public name in the facade namespace). `swarm_arena/__init__.py` also imports `__version__` (not in `__all__`), and after its `_publish` call imports the eight core facades in the re-export alias form: `from swarm_arena import board as board, config as config, executor as executor, manager as manager, planning as planning, protocol as protocol, scoring as scoring, tool as tool` (not `.testing`). So `import_path`, `Ref.of`, `Judge.rubric_hash`, `tool_identity`, and doctest collection (which imports private modules first) see the published `__module__` whatever the import order.
- `_cli/main.py`: `main(argv: Sequence[str] | None = None) -> int`. `argparse` parser `prog="swarm-arena"` with a `--version` flag (`store_true`, not argparse's version action). Prints `swarm-arena <version>` for `--version` and for no arguments, returns 0; argparse errors exit 2. Ends with `if __name__ == "__main__": raise SystemExit(main())`.
- Root `conftest.py` (module docstring): `collect_ignore: list[str]` holds `src/swarm_arena/llm` and `src/swarm_arena/testing/_fake_model.py` when `_missing("openai-agents")`, `src/swarm_arena/ray` when `_missing("ray")`, `src/swarm_arena/_store/s3.py` when `_missing("s3fs")`. `_missing(dist: str) -> bool` is True when `importlib.metadata.version(dist)` raises `PackageNotFoundError`. Do not use `importlib.util.find_spec`: `pythonpath = ["tests"]` puts `tests/` on `sys.path` before the conftests run, so without Ray `find_spec("ray")` returns a namespace-package spec for the `tests/ray/` directory (M4) and the `core` CI job collects the Ray modules.
- `tests/conftest.py`: fixture `anyio_backend() -> str` returning `"asyncio"`; `collect_ignore` for the R3.2 extra test paths by the same rule, with the same `_missing` helper (both conftest files define it); `collect_ignore_glob = ["*/data/*", "_fakes/*"]` (script fixtures and test helpers are never imported as doctest modules).
- Docs: `docs/conf.py` (module docstring; `project = "swarm-arena"`, `author = "Shreyas Vinaya"`, `copyright = "2026, Shreyas Vinaya"`, `release = importlib.metadata.version("swarm-arena")`, `extensions = ["sphinx.ext.autodoc", "numpydoc", "myst_parser"]`, `html_theme = "sphinx_rtd_theme"`, `exclude_patterns = ["_build", "superpowers"]`, `numpydoc_show_class_members = False`, `autodoc_member_order = "bysource"`). `docs/index.md`: title, the one-paragraph pitch, a `{toctree}` of the 11 pages plus `api/index`. Each stub page is `# <Title>` plus one line: Getting started, Planning modes, Scoring and review, Limits and safety "Written in M1b."; Running on Ray "Written in M4."; The arena "Written in M2."; Versioning policy, Roadmap "Written in M5."; Concepts, Use your own files and script, Writing a connector "Written in M1a." `docs/api/index.md` has a toctree of the 10 module pages; each module page is `# swarm_arena.<module>` plus an `{eval-rst}` block with `.. automodule:: swarm_arena.<module>` and `:members:`.
- `LICENSE`: MIT, `Copyright (c) 2026 Shreyas Vinaya`. `README.md`: title, pitch (§1 first paragraph), the §1 tree picture verbatim, the four numbered points of §1, "Status: under construction. v0.1 is built milestone by milestone.", license line. `CHANGELOG.md`: Keep a Changelog 1.1.0 header, SemVer line, `## [Unreleased]` / `### Added` / `- Project scaffold: package, \`swarm-arena --version\`, CI, and docs skeleton.` `CONTRIBUTING.md`: setup (`uv sync --all-extras --all-groups`), the task gate commands, TDD, numpydoc with Examples, the facade and snapshot-test rule, "never start a tool with `uv run`", and the commit rule (imperative subject, body says what and why, no AI co-author lines or "generated with" text in commits or PR descriptions). `.gitignore`: `.venv/`, `__pycache__/`, `*.py[cod]`, `dist/`, `build/`, `docs/_build/`, `.swarm-arena/`, `.pytest_cache/`, `.ruff_cache/`.
- `.github/workflows/ci.yml` (R2.0), on `push` to `main` and `pull_request`; every job uses `actions/checkout@v7` and `astral-sh/setup-uv@v10` with `python-version`:
  - `core`: matrix `["3.11", "3.12", "3.13"]`; `uv sync --all-groups`; `uv run ruff check .`; `uv run ruff format --check .`; `uv run pytest`.
  - `extras`: matrix `["3.11", "3.13"]`; `uv sync --all-extras --all-groups`; `uv run ty check`; `uv run pytest`.
  - `docs`: `"3.13"`; `uv sync --all-extras --group docs`; `uv run sphinx-build -W --keep-going -b html docs docs/_build/html`.
- Roadmap: in R3.1 add the row `` | `_facade.py` | `publish()`: sets `__module__` of exported names, their nested classes, and their methods to their facade (R0), so doctest collects method examples | M1a | `tests/test_public_api.py` | `` after `_version.py`; in R5.3 change the ruff `ignore` line to `["ANN401", "RUF012"]`. (The R3.2 `tests/conftest.py` row already says `importlib.metadata.version`, never `find_spec`: no edit.)

- [ ] **Step 6: Check that ty resolves `_fakes` imports, then run the task gate**

Run: `uv sync --all-extras --all-groups`. Then the R5.3 [verify]: write the throwaway pair `tests/_fakes/probe.py` (`VALUE: int = 1`) and `tests/test_ty_probe.py` (`from _fakes.probe import VALUE` and `def test_probe() -> None: assert VALUE == 1`), and run `uv run ty check tests/test_ty_probe.py`. Expected: exit 0. If ty reports `unresolved-import` for `_fakes.probe`, add `extra-paths = ["tests"]` under `[tool.ty.environment]` and run it again (exit 0). Delete both probe files and record which case held in the commit body. Then the task gate.
Expected: all exit 0; pytest reports the CLI, public-API, and isolation tests passed and `test_numpydoc` skipped with "got empty parameter set".

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore LICENSE README.md CHANGELOG.md CONTRIBUTING.md \
  .github conftest.py src tests docs/conf.py docs/index.md docs/*.md docs/api \
  docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "build: scaffold the swarm-arena package, CI, and docs skeleton" \
  -m "uv src layout, ruff/ty/pytest config, empty public facades with snapshot, docstring, and import-isolation tests, a version-only CLI, and Sphinx stubs for the 12 pages. Roadmap R3.1 gains _facade.py (publish helper, methods included so doctest finds their examples) and R5.3 ignores RUF012 for pydantic fields." \
  -m "ty and the tests/_fakes imports: <resolved with the default roots | needed extra-paths = [\"tests\"]> (R5.3 verify)."
```

---

### Task 2: Errors, canonical JSON, time format, migrations, base models

**Files:**
- Create: `src/swarm_arena/_errors.py`, `src/swarm_arena/_model/{__init__.py,canonical.py,migrate.py,base.py}`
- Test: `tests/test_errors.py`, `tests/_model/test_canonical.py`, `tests/_model/test_migrate.py`, `tests/_model/test_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces: R4.13.1 M1a errors (`PlanError(errors: Sequence[str])` with `errors: list[str]`, `ConfigError`, `SoftError`, `StorageError`, `NotFound`, `AlreadyExists`, `PreconditionFailed`, `WriteOutcomeUnknown`, `ConditionalWritesUnsupported`); R4.13.0 `canonical_json`, `sha256_hex`, `utc_now`, `CURRENT`, `migration`, `upgrade`; plus `format_utc(dt: datetime) -> str`, `UtcDatetime = Annotated[datetime, AfterValidator(...), PlainSerializer(format_utc, return_type=str, when_used="json")]`, `MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]`; R4.0 `FrozenModel`, `StoredModel` (adds `schema_kind: ClassVar[str] = ""`), `Origin`, `LLMRole`, `CallRole`, `CatalogKind`, `JsonValue`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_errors.py
def test_hierarchy() -> None:
    assert all(issubclass(e, ValueError) for e in (PlanError, ConfigError, SoftError))
    assert all(issubclass(e, StorageError) for e in (NotFound, AlreadyExists, PreconditionFailed,
                                                     WriteOutcomeUnknown, ConditionalWritesUnsupported))
    assert issubclass(StorageError, RuntimeError)

def test_plan_error_lists_errors() -> None:
    err = PlanError(["a", "b"])
    assert err.errors == ["a", "b"] and str(err) == "invalid plan:\n- a\n- b"
```
```python
# tests/_model/test_canonical.py
def test_canonical_json_sorted_compact() -> None:
    assert canonical_json({"b": 1, "a": [1, "é"]}) == '{"a":[1,"é"],"b":1}'.encode()

def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json({"x": float("nan")})

def test_sha256_hex() -> None:
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

def test_format_utc() -> None:
    assert format_utc(datetime(2026, 10, 18, 13, 52, 1, 123456, tzinfo=UTC)) == "2026-10-18T13:52:01.123Z"
    plus2 = timezone(timedelta(hours=2))
    assert format_utc(datetime(2026, 10, 18, 15, 52, 1, tzinfo=plus2)) == "2026-10-18T13:52:01.000Z"
    with pytest.raises(ValueError, match="aware"):
        format_utc(datetime(2026, 10, 18))

def test_utc_now_is_aware_milliseconds() -> None:
    now = utc_now()
    assert now.tzinfo is UTC and now.microsecond % 1000 == 0

def test_utc_datetime_field_roundtrip() -> None:
    class M(FrozenModel):
        t: UtcDatetime
    m = M(t=datetime(2026, 10, 18, 13, 52, 1, 123456, tzinfo=UTC))
    assert m.model_dump(mode="json") == {"t": "2026-10-18T13:52:01.123Z"}
    assert M.model_validate_json(m.model_dump_json()) == m          # stored value is already truncated
    with pytest.raises(ValidationError):
        M(t=datetime(2026, 10, 18))
```
```python
# tests/_model/test_migrate.py  (each test: monkeypatch.setattr(migrate, "_MIGRATIONS", {}))
def test_current_version_is_identity() -> None:
    assert upgrade("plan", {"schema_version": 1, "x": 1}) == {"schema_version": 1, "x": 1}

def test_missing_version_means_one() -> None:
    assert upgrade("plan", {"x": 1}) == {"schema_version": 1, "x": 1}

def test_newer_version_refused() -> None:
    with pytest.raises(ValueError, match=r"plan schema_version 2 is newer than this swarm-arena supports \(1\)"):
        upgrade("plan", {"schema_version": 2})

def test_migrations_apply_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(migrate.CURRENT, "demo", 3)
    @migration("demo", 1)
    def one(raw: dict[str, Any]) -> dict[str, Any]: return {("b" if k == "a" else k): v for k, v in raw.items()}
    @migration("demo", 2)
    def two(raw: dict[str, Any]) -> dict[str, Any]: return {**raw, "c": raw["b"] * 2}
    assert upgrade("demo", {"a": 2}) == {"schema_version": 3, "b": 2, "c": 4}

def test_unknown_kind_missing_step_and_duplicate() -> None:
    with pytest.raises(ValueError, match="unknown stored-format kind 'nope'"): upgrade("nope", {})
    # CURRENT["demo"] = 2 with no step registered → RuntimeError match "no migration for demo 1"
    # registering ("demo", 1) twice → ValueError match "already registered"
```
```python
# tests/_model/test_base.py
def test_frozen_model_rejects_extra_and_mutation() -> None: ...   # extra field → ValidationError; setattr → ValidationError
def test_stored_model_upgrades_and_refuses_newer(monkeypatch: pytest.MonkeyPatch) -> None: ...
    # subclass Demo(StoredModel) with schema_kind "demo"; CURRENT["demo"] = 2 and a 1→2 step adding "y":
    # Demo.model_validate({"x": 1}) == Demo(x=1, y=0, schema_version=2); {"schema_version": 3} → ValidationError match "newer"
def test_stored_model_without_kind_fails() -> None: ...           # schema_kind "" → ValidationError match "schema_kind"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_errors.py tests/_model -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._errors'`.

- [ ] **Step 3: Implement**

- `canonical_json(value: JsonValue) -> bytes`: `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`.
- `utc_now() -> datetime`: now in UTC, microseconds truncated to milliseconds. `format_utc` raises `ValueError("datetime must be timezone-aware")` for a naive value. The `UtcDatetime` after-validator rejects naive values and returns `astimezone(UTC)` truncated to milliseconds.
- `CURRENT = {"candidate": 1, "learning": 1, "llm_call": 1, "plan": 1, "run_config": 1, "job_started": 1, "request": 1, "candidate_request": 1, "candidate_output": 1, "tool_result": 1}`; later milestones add their kinds (R4.15). `_MIGRATIONS: dict[tuple[str, int], MigrationFn]`. `upgrade` copies `raw`, reads `schema_version` (default 1), refuses newer, applies `(kind, v)` steps until `CURRENT[kind]`, sets `schema_version`.
- `StoredModel`: a `model_validator(mode="before")` named `_upgrade` that, for dict input, raises `ValueError("<Class> has no schema_kind")` when `schema_kind` is empty and otherwise returns `upgrade(cls.schema_kind, data)`.

- [ ] **Step 4: Run the task gate**

Run: `uv run pytest tests/test_errors.py tests/_model -v`, then the task gate. Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_errors.py src/swarm_arena/_model tests/test_errors.py tests/_model
git commit -m "feat(model): add error types, canonical JSON, UTC time format, and schema migrations" \
  -m "Every stored format gets schema_version through StoredModel; older versions migrate and newer ones fail with a clear error (spec §9 rule 2)."
```

---

### Task 3: Ids, TOML I/O, JSON-schema subset compiler

**Files:**
- Create: `src/swarm_arena/_model/{ids.py,toml_io.py,schema.py}`
- Test: `tests/_model/{test_ids.py,test_toml_io.py,test_schema.py}`

**Interfaces:**
- Consumes: Task 2 `sha256_hex`, `ConfigError`.
- Produces: R4.13.5 `IdGen(run_id, epoch, counters=None)`, `.next(prefix)`, `.counters()`, `run_tag`, `human_learning_id`, `new_run_id`; R4.13.0 `read_toml(path: Path) -> dict[str, Any]`, `dumps_toml(data: Mapping[str, Any]) -> str`, `compile_object_schema(name: str, properties: Mapping[str, Mapping[str, Any]]) -> type[BaseModel]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_model/test_ids.py
def test_run_tag() -> None:
    assert run_tag("2026-10-18T18-30-00Z_ab12cd") == sha256_hex(b"2026-10-18T18-30-00Z_ab12cd")[:16]
    assert len(run_tag("local")) == 16

def test_ids_count_per_prefix_and_continue_across_epochs() -> None:
    ids = IdGen("local", 0)
    tag = run_tag("local")
    assert [ids.next("c"), ids.next("c"), ids.next("s")] == [f"c-{tag}-e0-0001", f"c-{tag}-e0-0002", f"s-{tag}-e0-0001"]
    later = IdGen("local", 2, counters={"c": 192})
    assert later.next("c") == f"c-{tag}-e2-0193" and later.counters() == {"c": 193}
    later.counters()["c"] = 0
    assert later.next("c").endswith("-0194")                      # counters() returns a copy
    assert IdGen("r", 1, counters={"x": 9999}).next("x").endswith("-e1-10000")

def test_bad_prefix() -> None:
    with pytest.raises(ValueError, match="prefix"): IdGen("r", 0).next(cast(Any, "q"))

def test_human_learning_id() -> None:
    a, b = human_learning_id(), human_learning_id()
    assert re.fullmatch(r"L-h-[0-9a-f]{16}", a) and a != b

def test_new_run_id() -> None:
    assert re.fullmatch(r"2026-10-18T18-30-00Z_[0-9a-f]{6}", new_run_id(datetime(2026, 10, 18, 18, 30, tzinfo=UTC)))
    with pytest.raises(ValueError, match="aware"): new_run_id(datetime(2026, 10, 18))
```
```python
# tests/_model/test_toml_io.py
SAMPLE = {"a": 1, "b": 'x\ny "q" \x7f', "c": [1, 2.5, True], "t": {"k": "v", "sub": {"z": 0}},
          "arr": [{"n": 1}, {"n": 2, "m": {"q": []}}], "e": {}, "u": "é", "f": 1e-05}

def test_roundtrip() -> None:
    assert tomllib.loads(dumps_toml(SAMPLE)) == SAMPLE

def test_scalars_first_and_none_dropped() -> None:
    assert dumps_toml({"t": {"x": 1}, "a": 1, "n": None}) == "a = 1\n\n[t]\nx = 1\n"

def test_quoted_keys() -> None:
    assert dumps_toml({"a b": 1, "ok_key-1": 2}) == '"a b" = 1\nok_key-1 = 2\n'

def test_deterministic() -> None:
    assert dumps_toml(SAMPLE) == dumps_toml(dict(SAMPLE))

def test_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="finite"): dumps_toml({"x": float("inf")})
    with pytest.raises(TypeError, match="bytes"): dumps_toml({"x": b"1"})

def test_read_toml_error_names_file(tmp_path: Path) -> None:
    (tmp_path / "bad.toml").write_text("a = ")
    with pytest.raises(ConfigError, match="bad.toml"): read_toml(tmp_path / "bad.toml")
```
```python
# tests/_model/test_schema.py
SPEC_PARAMS = {"lr": {"type": "number", "minimum": 1e-5, "maximum": 1.0},
               "epochs": {"type": "integer", "maximum": 400}}             # spec §4.12

def test_spec_params_example() -> None:
    M = compile_object_schema("Params", SPEC_PARAMS)               # type[BaseModel]: read values through model_dump()
    assert M(lr=1, epochs=400).model_dump() == {"lr": 1.0, "epochs": 400}
    for bad in ({"lr": 0.1, "epochs": 401}, {"lr": 0.0, "epochs": 1}, {"epochs": 1},
                {"lr": 0.1, "epochs": "5"}, {"lr": 0.1, "epochs": True}, {"lr": 0.1, "epochs": 1, "zz": 1}):
        with pytest.raises(ValidationError): M.model_validate(bad)

def test_enum_default_and_array_items() -> None:
    M = compile_object_schema("P", {"mode": {"enum": ["a", "b"], "default": "a"},
                                    "xs": {"type": "array", "items": {"type": "integer", "minimum": 0}, "default": []}})
    assert M().model_dump() == {"mode": "a", "xs": []} and M(xs=[1, 2]).model_dump()["xs"] == [1, 2]
    with pytest.raises(ValidationError): M(mode="c")
    with pytest.raises(ValidationError): M(xs=[-1])

@pytest.mark.parametrize(("prop", "match"), [
    ({"type": "string", "format": "email"}, "unsupported JSON-schema keyword 'format'"),
    ({"type": "integer", "default": "x"}, "default"),
    ({"type": "string", "minimum": 1}, "minimum"),
    ({"description": "no type"}, "type or enum"),
    ({"type": "object"}, "type 'object'")])
def test_rejected_schemas(prop: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match): compile_object_schema("P", {"p": prop})

def test_bad_property_name() -> None:
    with pytest.raises(ValueError, match="_secret"): compile_object_schema("P", {"_secret": {"type": "string"}})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_model/test_ids.py tests/_model/test_toml_io.py tests/_model/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._model.ids'`.

- [ ] **Step 3: Implement**

- Ids: `"<prefix>-<run tag>-e<epoch>-<n:04d>"`, `n` starts at 1 per prefix; prefixes `c`, `L`, `s`, `r`, `llm`, `x`, `k`; `run_tag(run_id) = sha256_hex(run_id.encode())[:16]` (64 bits, R4.0); `human_learning_id() = "L-h-" + secrets.token_hex(8)`; `new_run_id(now)` = `now.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ") + "_" + secrets.token_hex(3)`, naive → `ValueError`.
- `read_toml`: `tomllib.loads(path.read_text("utf-8"))`; a `TOMLDecodeError` becomes `ConfigError(f"{path}: {err}")`.
- `dumps_toml`: keys in insertion order; within a table, scalar and inline values first, then sub-tables `[a.b]`, then arrays of tables `[[a.b]]` (a non-empty list whose items are all dicts); other lists and empty dicts are inline (dicts inside inline arrays are inline tables); `None` values are dropped; bare keys match `[A-Za-z0-9_-]+`, others are quoted; strings are `json.dumps(s, ensure_ascii=False)` with `\x7f` replaced by `\u007f`; floats `repr(x)`, non-finite → `ValueError`; `bool` → `true`/`false`; any other type → `TypeError` naming it. A blank line separates table headers.
- `compile_object_schema`: model config `ConfigDict(extra="forbid", strict=True, frozen=True)`; `type` ∈ `string`, `integer`, `number`, `boolean`, `array` (`items` optional, compiled recursively); `enum` → `Literal[...]`; `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum` → `ge`/`le`/`gt`/`lt` (integer and number only); `minLength`/`maxLength`/`pattern` (string only); `default` (validated against the field type at compile time; a property without `default` is required; a default that fails raises `ValueError(f"{name}.{prop}: default {value!r} does not match the schema: {err}")`, because the raw pydantic error text does not say "default"); `description`. Any other keyword → `ValueError(f"{name}.{prop}: unsupported JSON-schema keyword {kw!r}")`. Property names match `[A-Za-z][A-Za-z0-9_]*`.

- [ ] **Step 4: Run the task gate**

Run: the three test files above, then the task gate. Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_model tests/_model
git commit -m "feat(model): add id generator, deterministic TOML writer, and JSON-schema subset compiler" \
  -m "Ids carry the run tag and epoch so they never repeat across runs or epochs (R4.13.5). The TOML writer and schema compiler replace tomli-w and jsonschema (R5.4 item 7)."
```

---

### Task 4: Path rules

**Files:**
- Create: `src/swarm_arena/_tooling/{__init__.py,paths.py}` (stdlib imports only, so `_model` may import it)
- Test: `tests/_tooling/test_paths.py`

**Interfaces:**
- Consumes: nothing.
- Produces: R4.13.9 `RESERVED`, `check_relpath(path: str) -> str`, `matches(path: str, globs: Sequence[str]) -> bool`; plus `check_glob(pattern: str) -> str` (non-empty, relative, no `..` segment, no NUL) and `check_key(path: str) -> str` (the rule for every path used as a dict key: `ParentRef.overlay`, `FileSet.manifest`, `Candidate.overlay`, `Job.Stage.root`, `Job.CandidateFiles.files`, edits; the R4.2 and R4.7 comments call it "normal form").

- [ ] **Step 1: Write the failing tests**

```python
def test_check_relpath_normalizes() -> None:
    assert check_relpath("./problem//run.py") == "problem/run.py"
    assert check_relpath("swarmy/x") == "swarmy/x"

@pytest.mark.parametrize("bad", ["", ".", "/abs", "a/../b", "..", "swarm/result.json", "parents/x", "candidate", "a\x00b"])
def test_check_relpath_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="path"): check_relpath(bad)

def test_check_key_requires_normal_form() -> None:
    assert check_key("p/a.txt") == "p/a.txt"
    for bad in ("./p/a.txt", "p//a.txt", "p/a.txt/"):
        with pytest.raises(ValueError, match="normal form"): check_key(bad)
    with pytest.raises(ValueError, match="reserved"): check_key("swarm/x")

@pytest.mark.parametrize(("path", "globs", "ok"), [
    ("problem/a.phyk", ["problem/*.phyk"], True), ("problem/sub/a.phyk", ["problem/*.phyk"], False),
    ("problem/sub/a.phyk", ["problem/**/*.phyk"], True), ("problem/a.phyk", ["problem/**/*.phyk"], True),
    ("a.txt", ["*.txt"], True), ("d/a.txt", ["*.txt"], False), (".hidden", ["*"], True),
    ("p/a1.txt", ["p/a?.txt"], True), ("p/a12.txt", ["p/a?.txt"], False), ("p/b.txt", ["p/[ab].txt"], True),
    ("x", [], False)])
def test_matches_dialect(path: str, globs: list[str], ok: bool) -> None:
    assert matches(path, globs) is ok

def test_matches_agrees_with_pathlib_glob(tmp_path: Path) -> None:
    files = ["problem/a.phyk", "problem/sub/b.phyk", "problem/run.py", "obs.npz", "problem/.h"]
    # create them; for each pattern in ["problem/*", "problem/**/*.phyk", "*.npz", "problem/*.p?yk"]:
    # {p.relative_to(tmp_path).as_posix() for p in tmp_path.glob(pat) if p.is_file()} == {f for f in files if matches(f, [pat])}

@pytest.mark.parametrize("bad", ["", "/abs/*", "../x/*", "a/../*"])
def test_check_glob_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="glob"): check_glob(bad)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_tooling/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._tooling'`.

- [ ] **Step 3: Implement**

`RESERVED: Final = frozenset({"swarm", "parents", "candidate"})`. `check_relpath` rejects empty, `"."`, a leading `/`, any `..` part, NUL, and a first part in `RESERVED`, each as `ValueError(f"path {path!r}: {reason}")` with reason `empty`, `absolute`, `contains '..'`, `contains NUL`, or `uses the reserved name {part!r}`; then returns `PurePosixPath(path).as_posix()`. `check_key(path)` = `norm = check_relpath(path)`, plus `ValueError(f"path {path!r}: not in normal form (use {norm!r})")` when `norm != path`; dict-key paths use it, so two spellings of one path never both reach a stage. `check_glob` raises `ValueError(f"glob {pattern!r}: {reason}")`. `matches` compiles each glob once (`functools.cache`) to a regex: `*` → `[^/]*`, `?` → `[^/]`, `[...]` as in `fnmatch`, a whole `**/` segment → `(?:.*/)?`, a trailing `/**` → `(?:/.*)?`; `re.fullmatch`. Dotfiles match `*` (as `pathlib.Path.glob` does).

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_tooling tests/_tooling
git commit -m "feat(tooling): add workspace path rules and the editable-glob matcher" \
  -m "Relative paths only, no '..', reserved swarm/, parents/, candidate/ (spec §4.12); one glob dialect shared by snapshots and editable checks."
```

---

### Task 5: `Ref` and the component contract

**Files:**
- Create: `src/swarm_arena/_model/ref.py`, `tests/_fakes/tools.py`
- Modify: `src/swarm_arena/__init__.py`, `tests/test_public_api.py`, `CHANGELOG.md`, roadmap R4.0
- Test: `tests/_model/test_ref.py`

**Interfaces:**
- Consumes: Task 2 `FrozenModel`, `ConfigError`.
- Produces: R4.1 `Ref` (`of`, `load`, `build`, `component_name`); plus `EmptyConfig(BaseModel)`; `Configurable` with `Config = EmptyConfig` (a plain, unannotated class attribute), `__init__(self, config: BaseModel | Mapping[str, Any] | None = None, /) -> None`, read-only property `config -> BaseModel`; `T = TypeVar("T", bound=Configurable)` so `Ref.build(self, base: type[T]) -> T`; `import_path(obj: object) -> str`; `load_object(path: str) -> object` (cached); `coerce_ref(value: object) -> object`; `RefField = Annotated[Ref, BeforeValidator(coerce_ref)]`; `unique_names(refs: Sequence[Ref]) -> list[str]`. `tests/_fakes/tools.py`: `class Widget(Configurable)` with `name: ClassVar[str] = "widget"`, `Config(size: int = 1, label: str = "")`, and a `config` property typed `Widget.Config` (the built-in pattern); `class NoNameWidget(Configurable)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_path_format() -> None:
    assert Ref(path="a.b:C.D").path == "a.b:C.D"
    with pytest.raises(ValidationError): Ref(path="nocolon")

def test_of_instance_and_class() -> None:
    assert Ref.of(Widget(Widget.Config(size=3))) == Ref(path="_fakes.tools:Widget", config={"size": 3, "label": ""})
    assert Ref.of(Widget) == Ref(path="_fakes.tools:Widget")

def test_of_rejects_lambda_local_main_and_non_components() -> None:
    with pytest.raises(TypeError, match="lambda"): import_path(lambda: None)
    class Local(Configurable): ...
    with pytest.raises(TypeError, match="local"): Ref.of(Local)
    # a class whose __module__ is set to "__main__" → TypeError match "__main__"
    with pytest.raises(TypeError, match="component"): Ref.of(object())

def test_build_validates_config() -> None:
    w = Ref(path="_fakes.tools:Widget", config={"size": 2}).build(Configurable)
    assert isinstance(w, Widget) and w.config == Widget.Config(size=2)

def test_build_rejects_misspelled_key() -> None:
    with pytest.raises(ConfigError, match=r"(?s)_fakes\.tools:Widget.*sise"):
        Ref(path="_fakes.tools:Widget", config={"sise": 2}).build(Configurable)

def test_build_wrong_base_and_load_errors() -> None:
    with pytest.raises(TypeError, match="not a subclass of Widget"): Ref(path="_fakes.tools:NoNameWidget").build(Widget)
    with pytest.raises(ValueError, match="cannot import no_such_mod:X"): Ref(path="no_such_mod:X").load()
    with pytest.raises(TypeError, match="not a class"): Ref(path="json:dumps").load()

def test_load_is_cached() -> None:
    assert Ref(path="_fakes.tools:Widget").load() is Widget

def test_component_name() -> None:
    assert Ref(path="_fakes.tools:Widget", name="w2").component_name() == "w2"
    assert Ref(path="_fakes.tools:Widget").component_name() == "widget"
    with pytest.raises(TypeError, match="name"): Ref(path="_fakes.tools:NoNameWidget").component_name()

def test_unique_names() -> None:
    assert unique_names([Ref(path="_fakes.tools:Widget"), Ref(path="_fakes.tools:Widget", name="b")]) == ["widget", "b"]
    with pytest.raises(ValueError, match="duplicate component name 'widget'"):
        unique_names([Ref(path="_fakes.tools:Widget"), Ref(path="_fakes.tools:Widget")])

def test_ref_field_coerces_instances() -> None:
    class Holder(FrozenModel):
        ref: RefField
    assert Holder.model_validate({"ref": Widget()}).ref == Ref.of(Widget())

def test_configurable_config_forms() -> None:
    assert Configurable().config == EmptyConfig()
    assert Widget({"size": 5}).config.size == 5 and Widget(Widget.Config(size=6)).config.size == 6
    with pytest.raises(ValidationError): Configurable({"x": 1})
```
```python
# tests/test_public_api.py (added)
def test_doctest_finder_collects_method_examples() -> None:
    assert "swarm_arena.Ref.of" in {t.name for t in doctest.DocTestFinder().find(swarm_arena)}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_model/test_ref.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._model.ref'`.

- [ ] **Step 3: Implement**

- Path regex: `^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$`.
- `EmptyConfig(BaseModel)` sets `model_config = ConfigDict(extra="forbid", frozen=True)`, so `Configurable({"x": 1})` raises `ValidationError` (a plain `BaseModel` ignores extra keys).
- `import_path(obj)`: `f"{obj.__module__}:{obj.__qualname__}"`; `TypeError` when the qualname contains `<lambda>` ("lambda") or `<locals>` ("local"), or the module is `__main__`.
- `load_object(path)`: `importlib.import_module` on the module part, then `getattr` along the qualname; any import or attribute error → `ValueError(f"cannot import {path}: {err}")`. `Ref.load` returns it, `TypeError(f"{path} is not a class")` if it is not a class.
- `Ref.of`: class → `Ref(path=import_path(cls))`; `Configurable` instance → `Ref(path=import_path(type(obj)), config=obj.config.model_dump(mode="json"))`; anything else → `TypeError("not a component")`.
- `Ref.build(base)`: `cls = self.load()`; not a subclass of `base` → `TypeError(f"{path} is not a subclass of {base.__name__}")`; `ValidationError` from `cls.Config.model_validate(self.config)` → `ConfigError(f"{self.path}: {err}")`; return `cls(validated)`.
- `component_name()`: `self.name`, else the class's `name` attribute when it is a `str`, else `TypeError(f"{path} has no name; set Ref.name")`.
- `coerce_ref`: a `Configurable` instance or subclass → `Ref.of(value)`; anything else unchanged.
- Roadmap R4.0, "Component contract": replace "The M1a plan picks the typing pattern that passes `ty check` with no ignores." with: "Typing pattern (M1a, checked with ty 0.0.89): component classes inherit `_model/ref.py:Configurable`, whose `Config = EmptyConfig` is a plain class attribute with no annotation (ty rejects a nested `class Config` that overrides an annotated base attribute). `obj.config` is a read-only property; built-ins override it with their own `Config` as the return type. The `Config: ClassVar[type[BaseModel]]` lines in R4 describe this contract. Fields typed `Ref` are `RefField` (`Annotated[Ref, BeforeValidator(coerce_ref)]`): instances are converted at runtime; statically typed callers pass `Ref.of(obj)`."
- Docstring examples of `Ref.of` and `Ref.build` name `swarm_arena.executor:LocalExecutor` and carry `# doctest: +SKIP` (no public component exists before Task 17).
- Public surface: `swarm_arena.__all__` gains `Ref`; `PUBLIC["swarm_arena"]` too; CHANGELOG: "`Ref`: components referenced by import path, with validated `Config`."

- [ ] **Step 4: Run the task gate**

Expected: PASS, including `test_numpydoc[swarm_arena.Ref]`, `[swarm_arena.Ref.of]`, `.load`, `.build`, `.component_name`, and `test_doctest_finder_collects_method_examples`; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(model): add Ref and the Configurable component contract" \
  -m "Components are import paths plus a validated Config (spec §4.1). Roadmap R4.0 records the ty-compatible typing pattern the roadmap left to this plan."
```

---

### Task 6: Goal, FileSet, GoalKey, and tool-run data types

**Files:**
- Create: `src/swarm_arena/_model/{goal.py,tooling.py}`
- Modify: facades `__init__.py`, `tool.py`, `board.py`; `tests/_fakes/tools.py`; `tests/test_public_api.py`; `CHANGELOG.md`
- Test: `tests/_model/{test_goal.py,test_tooling.py}`

**Interfaces:**
- Consumes: Task 2 (`FrozenModel`, `UtcDatetime`, `canonical_json`, `sha256_hex`), Task 4 (`check_relpath`, `check_key`, `check_glob`).
- Produces: as R4.1/R4.2/R4.6: `FileSet`, `Goal` (`key()`), `GoalKey` (`of`), `RunStatus`, `Usage` (`total_tokens`, `__add__`), `Estimate`, `FileRef`, `ParentRef`, `ToolRequest`, `ToolOutput`, `Command`, `CompletedRun`; plus `check_parent_id(candidate_id: str) -> str` in `_model/tooling.py` (the `ParentRef.candidate_id` rule, reused by Task 14's `materialize`). Datetime fields are `UtcDatetime`. `tests/_fakes/tools.py` gains `GOAL = Goal(id="g", namespace="n")`, used by later tests.

- [ ] **Step 1: Write the failing tests**

```python
H = "a" * 64

def test_fileset_from_glob_list() -> None:
    fs = FileSet.model_validate(["problem/*.phyk"])
    assert fs.globs == ["problem/*.phyk"] and not fs.snapshotted
    assert FileSet().snapshotted and FileSet(globs=["x/*"], manifest={"x/a": H}).snapshotted

@pytest.mark.parametrize("bad", [{"globs": ["/abs/*"]}, {"globs": ["../x"]}, {"manifest": {"swarm/x": H}},
                                 {"manifest": {"a": "nothex"}}, {"manifest": {"./a": H}}])
def test_fileset_rejects(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError): FileSet.model_validate(bad)

def test_goal_slugs() -> None:
    assert Goal(id="ns-viscosity", namespace="physika").title == ""
    for bad in ({"id": "NS Visc", "namespace": "physika"}, {"id": "a", "namespace": "-x"}):
        with pytest.raises(ValidationError): Goal.model_validate(bad)

def test_goal_key_values() -> None:
    g = Goal(id="g", namespace="ns", inputs={"b": 1, "a": 2}, files=FileSet(globs=["p/*"], manifest={"p/x": H}))
    assert GoalKey.of(g) == g.key() == GoalKey(namespace="ns", id="g",
        inputs_sha256=sha256_hex(b'{"a":2,"b":1}'), files_sha256=sha256_hex(canonical_json({"p/x": H})))
    other = g.model_copy(update={"files": FileSet(globs=["p/*"], manifest={"p/x": "b" * 64})})
    assert other.key().files_sha256 != g.key().files_sha256

def test_goal_key_requires_snapshot() -> None:
    with pytest.raises(ValueError, match="snapshot"): Goal(id="g", namespace="n", files=FileSet(globs=["a/*"])).key()

def test_run_status_values() -> None:
    assert [s.value for s in RunStatus] == ["ok", "error", "timeout", "oom", "budget_exceeded",
                                            "cancelled", "provider_failed", "lost"]

def test_usage_add_and_total() -> None:
    u = Usage(input_tokens=10, output_tokens=5, cache_read_tokens=3, cache_write_tokens=2, cache_write_5m_tokens=2)
    assert u.total_tokens == 20                       # the 5m/1h fields split cache_write_tokens
    assert (u + u).input_tokens == 20 and (u + u).cache_write_5m_tokens == 4
    assert (Usage() + Usage(usd=0.5)).usd == 0.5 and (Usage() + Usage()).usd is None

def test_tool_request_edits_serialize_as_hashes() -> None:
    dumped = ToolRequest(edits={"a.txt": "hi"}).model_dump(mode="json")
    assert dumped["edits"] == {"a.txt": "sha256:" + sha256_hex(b"hi")}
    assert ToolRequest.model_validate(dumped).model_dump(mode="json") == dumped     # no double hashing

def test_tool_request_edit_content_with_sha_prefix() -> None:     # only an exact stored reference is kept
    for content in ("sha256:hello", "sha256:" + "A" * 64):
        dumped = ToolRequest(edits={"a.txt": content}).model_dump(mode="json")
        assert dumped["edits"] == {"a.txt": "sha256:" + sha256_hex(content.encode())}
        assert ToolRequest.model_validate(dumped).model_dump(mode="json") == dumped

def test_tool_request_reserved_form_is_a_reference() -> None:
    ref = "sha256:" + "a" * 64
    dumped = ToolRequest(edits={"a.txt": ref}).model_dump(mode="json")
    assert dumped["edits"] == {"a.txt": ref}                                          # dumped unchanged by definition
    assert ToolRequest.model_validate(dumped).model_dump(mode="json") == dumped     # stable

def test_parent_ref_rules() -> None:
    base = {"candidate_id": "c1", "run_id": "r", "tool_identity": "T"}
    for bad in ({"candidate_id": ".."}, {"candidate_id": "a/b"}, {"candidate_id": ""}, {"candidate_id": "../../outside"},
                {"overlay": {"/abs": H}}, {"overlay": {"../x": H}}, {"overlay": {"swarm/x": H}},
                {"overlay": {"./a": H}}, {"overlay": {"a": "nothex"}}):
        with pytest.raises(ValidationError): ParentRef.model_validate({**base, **bad})

def test_tool_output_and_command() -> None:
    out = ToolOutput(status=RunStatus.OK, started_at=datetime(2026, 10, 18, 13, 52, 1, 123456, tzinfo=UTC))
    assert out.model_dump(mode="json")["started_at"] == "2026-10-18T13:52:01.123Z"
    assert pickle.loads(pickle.dumps(out)) == out                                    # pickles by public path
    c = Command(argv=["python", "run.py"])
    assert (c.cwd, c.env, c.timeout_s, c.python, c.grace_s) == (None, [], None, None, 10.0)
    for bad in ({"argv": []}, {"argv": ["x"], "env": ["BAD-NAME"]}, {"argv": ["x"], "timeout_s": 0}):
        with pytest.raises(ValidationError): Command.model_validate(bad)

def test_file_ref_rules() -> None:
    for bad in ({"name": "a", "path": "/a", "sha256": H, "size": 1}, {"name": "a", "path": "a", "sha256": "x", "size": 1},
                {"name": "a", "path": "a", "sha256": H, "size": -1}):
        with pytest.raises(ValidationError): FileRef.model_validate(bad)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_model/test_goal.py tests/_model/test_tooling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._model.goal'`.

- [ ] **Step 3: Implement**

- `FileSet`: before-validator turns a list into `{"globs": list}`; `globs` items through `check_glob`; `manifest` keys through `check_key`, values `^[0-9a-f]{64}$`; `snapshotted = not globs or bool(manifest)`.
- `ParentRef`: `candidate_id` matches `^[A-Za-z0-9][A-Za-z0-9._-]*$` (one path component: it names `parents/<id>/`), checked by `check_parent_id`, which raises `ValueError(f"parent id {candidate_id!r}: not one path component")`; `overlay` keys through `check_key`, values 64 hex.
- `Goal`: `id` and `namespace` match `^[a-z0-9][a-z0-9._-]*$`; `key()` = `GoalKey.of(self)`. `GoalKey.of` raises `ValueError(f"goal {id}: files are not snapshotted")`.
- `Usage.total_tokens` = `input_tokens + output_tokens + cache_read_tokens + cache_write_tokens`; docstring: "`input_tokens` counts input tokens neither read from nor written to the prompt cache." `__add__` sums every field; `usd` sums the values that are not `None` and stays `None` when both are.
- `ToolRequest.edits`: field serializer (all modes) writes each value as `"sha256:" + sha256_hex(content.encode())`. It keeps a value as it is only when the value fully matches `^sha256:[0-9a-f]{64}$` (a stored reference from a loaded request); any other value, `"sha256:hello"` too, is content and is hashed. This rule, also in the `ToolRequest` docstring: "`sha256:` + 64 lowercase hex is the reserved stored-reference form. In any ToolRequest it means a reference (as `swarm/` is a reserved path), so it is dumped unchanged by definition. Live-content entry points (`validate_edits`, `edit_hashes`, `put_edits`, and so `testing.make_candidate`) refuse it as content." (So content and references never mix.) The R4.2 comments on `ParentRef` and `ToolRequest.edits` already say this: no roadmap edit.
- `Command`: `argv` non-empty; `env` names `^[A-Za-z_][A-Za-z0-9_]*$`; `timeout_s > 0` when set; `grace_s >= 0`; `cwd` through `check_relpath` when set. `FileRef`: `path` through `check_relpath`, `sha256` 64 hex, `size >= 0`.
- Public surface: `swarm_arena`: `Goal`; `swarm_arena.tool`: `Command`, `CompletedRun`, `ToolRequest`, `ParentRef`, `ToolOutput`, `FileRef`, `FileSet`, `Usage`, `Estimate`, `RunStatus`; `swarm_arena.board`: `GoalKey`. CHANGELOG: "`Goal`, `GoalKey`, and the tool-run data types of `swarm_arena.tool`."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md
git commit -m "feat(model): add Goal, GoalKey, FileSet, and the tool-run data types" \
  -m "GoalKey hashes inputs and the snapshotted file manifest, so results for changed files never mix (spec §4.2)."
```

---

### Task 7: Candidates, scores, learnings, LLM calls, and A2A types

**Files:**
- Create: `src/swarm_arena/_model/{results.py,protocol.py}`
- Modify: facades `__init__.py`, `scoring.py`, `board.py`, `manager.py`, `protocol.py`; `tests/_fakes/tools.py`; `tests/test_public_api.py`; `CHANGELOG.md`
- Test: `tests/_model/{test_results.py,test_protocol.py}`

**Interfaces:**
- Consumes: Task 2, Task 4 (`check_key`), Task 6 (`ToolRequest`, `ToolOutput`, `RunStatus`, `Usage`).
- Produces: as R4.3/R4.4/R4.6: `CheckResult`, `Score` (`minimize`, `maximize`, `identity`), `AgentStatus`, `Candidate` (kind `"candidate"`, property `gated_out`), `Learning` (kind `"learning"`, property `from_human`), `LLMCall` (kind `"llm_call"`); R4.11 `Role`, `TaskState`, `Part`, `Message`, `Artifact`, `TaskStatus`, `Task` and private `task_state_for(status: RunStatus) -> tuple[TaskState, dict[str, str]]`. `tests/_fakes/tools.py` gains `cand(*, data: dict[str, Any] | None = None, **fields: Any) -> Candidate` (defaults `id="c1"`, `run_id="r"`, `goal_id="g"`, `team="t"`, `agent="a"`, `round=1`, `request=ToolRequest()`, `output=ToolOutput(status=RunStatus.OK, data=data or {})`; `fields` override), used by every later test that needs a candidate.

- [ ] **Step 1: Write the failing tests**

```python
def test_gated_out_is_derived_and_never_stored() -> None:
    c = cand(checks=[CheckResult(check="a", passed=True), CheckResult(check="b", passed=False)])
    assert c.gated_out and not cand().gated_out and "gated_out" not in c.model_dump()
    assert Candidate.model_validate_json(c.model_dump_json()) == c

def test_candidate_newer_schema_refused() -> None:
    with pytest.raises(ValidationError, match="newer"):
        Candidate.model_validate({**cand().model_dump(mode="json"), "schema_version": 2})

def test_candidate_overlay_paths_checked() -> None:
    for overlay in ({"/abs": "a" * 64}, {"../x": "a" * 64}, {"swarm/x": "a" * 64}):
        with pytest.raises(ValidationError): cand(overlay=overlay)

def test_score_helpers() -> None:
    assert (Score.minimize(0.25).key, Score.minimize(0.25).value) == ((-0.25,), 0.25)
    assert Score.maximize(3.0, verdict="good", notes="n").key == (3.0,)
    assert Score(key=(1.0,), judge="j", rubric_hash="r", panel_hash="p").identity == ("j", "r", "p")

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_score_rejects_nonfinite(bad: float) -> None:
    with pytest.raises(ValidationError, match="finite"): Score(key=(1.0, bad))
    with pytest.raises(ValidationError, match="finite"): Score.minimize(bad)

def test_score_needs_a_key_value() -> None:
    with pytest.raises(ValidationError, match="at least one"): Score(key=())

def test_learning_limits() -> None:
    ok = {"id": "L1", "namespace": "n", "run_id": "r", "goal_id": "g", "round": 1, "author": "t/a",
          "title": "x" * 120, "text": "y" * 1500, "evidence": ["c1"]}
    assert Learning.model_validate(ok).title == "x" * 120
    for change in ({"title": "x" * 121}, {"title": "a\nb"}, {"title": "  "}, {"text": "y" * 1501}, {"evidence": []}):
        with pytest.raises(ValidationError): Learning.model_validate({**ok, **change})
    human = Learning.model_validate({**ok, "author": "human:alice", "evidence": []})
    assert human.from_human and not Learning.model_validate(ok).from_human

def test_agent_status_and_llm_call_defaults() -> None:
    assert [s.value for s in AgentStatus] == ["active", "finished", "starved", "pruned", "stopped", "watchdog_stopped"]
    call = LLMCall(id="llm1", run_id="r", goal_id="g", role="judge", model="openai/gpt-6", provider="openai")
    assert (call.status, call.cost_usd, call.cache_hit, call.usage) == ("ok", 0.0, False, Usage())
```
```python
# tests/_model/test_protocol.py
def test_enum_values_are_proto_names() -> None:
    assert {s.value for s in TaskState} == {f"TASK_STATE_{n}" for n in ("UNSPECIFIED", "SUBMITTED", "WORKING",
        "COMPLETED", "FAILED", "CANCELED", "REJECTED", "INPUT_REQUIRED", "AUTH_REQUIRED")}
    assert {r.value for r in Role} == {"ROLE_UNSPECIFIED", "ROLE_USER", "ROLE_AGENT"}

def test_part_exactly_one_content() -> None:
    assert Part(data=None).data is None and Part(raw=b"x").raw == b"x"
    for kwargs in ({}, {"text": "a", "url": "u"}):
        with pytest.raises(ValueError, match="exactly one"): Part(**kwargs)

def test_message_and_artifact_need_parts() -> None:
    with pytest.raises(ValueError, match="part"): Artifact(artifact_id="a", parts=())
    with pytest.raises(ValueError, match="part"): Message(message_id="m", role=Role.AGENT, parts=())

def test_task_status_needs_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="aware"): TaskStatus(state=TaskState.WORKING, timestamp=datetime(2026, 1, 1))

def test_frozen_slots() -> None:
    def mutate(obj: Any) -> None: obj.text = "b"
    with pytest.raises(dataclasses.FrozenInstanceError): mutate(Part(text="a"))
    assert hasattr(Part, "__slots__")

@pytest.mark.parametrize(("status", "expected"), [
    (RunStatus.OK, (TaskState.COMPLETED, {})), (RunStatus.CANCELLED, (TaskState.CANCELED, {})),
    *[(s, (TaskState.FAILED, {"swarm_arena.reason": s.value})) for s in (RunStatus.BUDGET_EXCEEDED,
      RunStatus.TIMEOUT, RunStatus.LOST, RunStatus.ERROR, RunStatus.OOM, RunStatus.PROVIDER_FAILED)]])
def test_task_state_for(status: RunStatus, expected: tuple[TaskState, dict[str, str]]) -> None:
    assert task_state_for(status) == expected
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_model/test_results.py tests/_model/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._model.results'`.

- [ ] **Step 3: Implement**

- `Score.key`: at least one value (`"key needs at least one value"`), each `math.isfinite` (`"score key values must be finite"`).
- `Candidate.overlay`: keys through `check_key`, values 64 hex.
- `Learning`: `title` stripped non-empty, `<= 120` characters, no `\n` or `\r`; `text <= 1500` characters; `evidence` needs at least one id unless `author.startswith("human:")`.
- Protocol (A2A v1.0.1, `api-a2a.md` "Recommendation"): `@dataclass(frozen=True, slots=True, kw_only=True)`. `Part(text: str | None = None, raw: bytes | None = None, url: str | None = None, data: JsonValue | _Unset = UNSET, media_type: str | None = None, filename: str | None = None, metadata: Mapping[str, JsonValue] = field(default_factory=dict))` with a private sentinel `UNSET`, so `data=None` is JSON null. `Message(message_id, role, parts: tuple[Part, ...], context_id=None, task_id=None, reference_task_ids=(), metadata, extensions=())`; `Artifact(artifact_id, parts, name=None, description=None, metadata, extensions=())`; `TaskStatus(state, timestamp: datetime, message: Message | None = None)`; `Task(id, context_id, status, artifacts=(), history=(), metadata)`. Validation in `__post_init__`.
- Public surface: `swarm_arena`: `Score`; `.scoring`: `CheckResult`; `.board`: `Candidate`, `Learning`, `LLMCall`; `.manager`: `AgentStatus`; `.protocol`: `Message`, `Part`, `Artifact`, `Task`, `TaskStatus`, `TaskState`, `Role`. CHANGELOG: "Result types (`Candidate`, `Score`, `CheckResult`, `Learning`, `LLMCall`, `AgentStatus`) and the A2A v1.0.1 types of `swarm_arena.protocol`."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md
git commit -m "feat(model): add candidates, scores, learnings, LLM calls, and A2A types" \
  -m "Non-finite score keys are rejected and learnings keep the D20 size limits; A2A records mirror v1.0.1 with the RunStatus→TaskState map (spec §4.2)."
```

---

### Task 8: `Plan`, its specs, and `Limits`

**Files:**
- Create: `src/swarm_arena/_model/{plan.py,limits.py}`
- Modify: facades `__init__.py`, `planning.py`; `tests/test_public_api.py`; `CHANGELOG.md`
- Test: `tests/_model/{test_plan.py,test_limits.py}`

**Interfaces:**
- Consumes: Task 3 `read_toml`, `dumps_toml`; Task 5 `Ref`, `RefField`, `unique_names`; Task 6 `Goal`, `Estimate`.
- Produces: as R4.1/R4.5: `Resources`, `PartSpec`, `TeamSpec`, `AgentSpec`, `RoundSpec` (`effective_fidelity`), `ConvergeSpec`, `FlowSpec`, `Plan` (kind `"plan"`; `load`, `loads(text, *, base_dir)`, `dumps`, property `base_dir`, `ranking_name`, `rounds_total`), `Limits` (`effective_round_timeout_s`). Every `Ref` field is a `RefField`.

- [ ] **Step 1: Write the failing tests**

```python
SPEC_TOML = '''...'''   # the §4.12 TOML example verbatim (goal, team fd-solver, agent tuner-1, ScriptCheck, ScriptJudge rel-error)

def test_limits_defaults_match_spec() -> None:
    assert Limits().model_dump() == {
        "max_teams": 4, "max_agents_per_team": 4, "max_agents_per_goal": 12, "max_concurrent_goals": 4,
        "max_agents_total": 32, "max_agents_per_fingerprint": 2, "max_rounds": 3, "max_turns_per_round": 20,
        "max_tool_runs_per_round": 3, "max_tokens": None, "max_usd": None,
        "default_run_reservation": {"seconds": None, "tokens": 100_000, "usd": 1.0},
        "llm_tranche_usd": 0.25, "llm_tranche_tokens": 25_000, "tool_timeout_s": 3600.0, "round_timeout_s": None,
        "run_timeout_s": 86_400.0, "max_learnings_per_agent_round": 3, "max_board_reads_per_round": 10,
        "digest_tokens": 1500, "max_edit_bytes": 262_144, "max_consecutive_failures": 3,
        "max_refused_repeats": 3, "watchdog_idle_rounds": 2, "max_notes_bytes": 16_384}

def test_effective_round_timeout() -> None:
    assert Limits().effective_round_timeout_s() == 14_400.0
    assert Limits(tool_timeout_s=100).effective_round_timeout_s() == 400.0
    assert Limits(round_timeout_s=50).effective_round_timeout_s() == 50.0

@pytest.mark.parametrize("bad", [{"max_rounds": 0}, {"max_usd": 0}, {"tool_timeout_s": 0}, {"max_board_reads_per_round": -1}])
def test_limits_bounds(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError): Limits.model_validate(bad)

def test_converge_defaults() -> None:
    c = ConvergeSpec()
    assert (c.mode, c.top_n, c.distinct, c.allow_partial, c.parents_per_part, c.max_join_attempts) == ("select", 2, True, False, 1, 1)

def test_effective_fidelity() -> None:
    assert RoundSpec(fidelity="full").effective_fidelity() == "full"
    assert RoundSpec(tool_overrides={"epochs": 100}).effective_fidelity() == sha256_hex(b'{"epochs":100}')[:12]

def test_rationale_required() -> None:
    for r in ("", "  "):
        with pytest.raises(ValidationError): AgentSpec(name="a", rationale=r, tool=Ref(path="m:T"), manager=Ref(path="m:M"))

def test_flows_reserved_for_v2() -> None:
    with pytest.raises(ValidationError, match="v2"):
        Plan(goal=GOAL, teams=TEAMS, judges=JUDGES, flows=[FlowSpec(sender="a", recipient="*", rationale="r")])

def test_load_spec_example(tmp_path: Path) -> None:
    (tmp_path / "plan.toml").write_text(SPEC_TOML)
    plan = Plan.load(tmp_path / "plan.toml")
    assert plan.goal.files.globs == ["problem/*.phyk", "problem/obs.npz", "problem/run.py"]
    tool = plan.teams[0].agents[0].tool
    assert tool.path == "swarm_arena:ScriptTool" and tool.config["params"]["lr"]["minimum"] == 1e-5
    assert plan.judges[0].name == "rel-error" and plan.origin == "human" and plan.base_dir == tmp_path
    assert plan.policy == Ref(path="swarm_arena:SuccessiveHalving")

@pytest.mark.parametrize("origin", ["llm", "together", "autonomous"])
def test_load_keeps_non_human_origin(tmp_path: Path, origin: str) -> None:
    (tmp_path / "p.toml").write_text(f'origin = "{origin}"\n' + SPEC_TOML)
    assert Plan.load(tmp_path / "p.toml").origin == origin

def test_dumps_roundtrip(tmp_path: Path) -> None:
    plan = Plan.loads(SPEC_TOML, base_dir=tmp_path)
    assert Plan.loads(plan.dumps(), base_dir=tmp_path) == plan
    assert 'origin = "human"' in plan.dumps() and plan.dumps() == plan.dumps()

def test_load_errors_name_the_file(tmp_path: Path) -> None:
    (tmp_path / "a.toml").write_text("teams = [")
    with pytest.raises(ConfigError, match="a.toml"): Plan.load(tmp_path / "a.toml")
    (tmp_path / "b.toml").write_text(SPEC_TOML + "\nteamz = 1\n")
    with pytest.raises(ConfigError, match=r"(?s)b\.toml.*teamz"): Plan.load(tmp_path / "b.toml")

def test_ranking_name() -> None:
    one = [Ref(path="m:J", name="rel-error")]
    two = [Ref(path="m:J", name="a"), Ref(path="m:J", name="b")]
    assert plan_with(judges=one).ranking_name() == "rel-error"
    assert plan_with(judges=two, ranking="a").ranking_name() == "a"
    with pytest.raises(ValueError, match="several judges"): plan_with(judges=two).ranking_name()
    with pytest.raises(ValueError, match="zz"): plan_with(judges=two, ranking="zz").ranking_name()
    with pytest.raises(ValueError, match="duplicate"): plan_with(judges=[two[0], two[0]], ranking="a").ranking_name()

def test_rounds_total() -> None:
    assert plan_with().rounds_total(Limits()) == 3
    assert plan_with(rounds=[RoundSpec(), RoundSpec()]).rounds_total(Limits()) == 2

def test_structure_rules() -> None: ...
    # duplicate team names → ValidationError "duplicate team"; duplicate agent names in a team → "duplicate agent";
    # team.part naming an undeclared part → "unknown part"; team name "a/b" → ValidationError (slug rule)

def test_base_dir_captured_at_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    plan = plan_with()
    monkeypatch.chdir("/")
    assert plan.base_dir == tmp_path and plan.model_copy(update={"rationale": "x"}).base_dir == tmp_path

def test_refs_accept_component_instances() -> None:
    spec = AgentSpec.model_validate({"name": "a1", "rationale": "r", "tool": Widget(), "manager": Widget()})
    assert spec.tool == Ref.of(Widget())
```
(`GOAL` comes from `_fakes.tools`; `TEAMS`, `JUDGES`, and `plan_with(**kw: Any) -> Plan` are small module-level helpers in the test file.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_model/test_plan.py tests/_model/test_limits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._model.plan'`.

- [ ] **Step 3: Implement**

- `Resources`: `cpus > 0`, `gpus >= 0`, `memory_gib > 0` when set, `custom` values `>= 0`. Team, agent, and part names use the goal slug rule. `replicates >= 1`; `RoundSpec.tool_timeout_s > 0` when set; `top_n >= 1`, `parents_per_part >= 1`, `max_join_attempts >= 0`.
- `Limits` bounds: `>= 1` for every count except `max_learnings_per_agent_round` and `max_board_reads_per_round` (`>= 0`); `> 0` for every time and money field when set; `max_tokens >= 1` when set.
- `Plan`: `flows` non-empty → `ValueError("Plan.flows is reserved for v2 messaging and must be empty")`; unique team names, unique agent names per team, unique part names, `team.part` in the declared parts. `_base_dir: Path = PrivateAttr(default_factory=Path.cwd)`. `loads(text, *, base_dir)` parses TOML, validates, and sets `_base_dir`; when the text has no `origin`, the default `"human"` applies; a non-human origin in the file is kept (R4.1). `load(path)` = `loads` with `base_dir = path.parent`; TOML and validation errors become `ConfigError(f"{path}: {err}")`. `dumps()` = `dumps_toml(self.model_dump(mode="json", exclude_none=True))`. `ranking_name()` uses `unique_names(self.judges)`; `""` with exactly one judge → its name. `rounds_total(limits)` = `len(self.rounds) or limits.max_rounds`.
- Public surface: `swarm_arena`: `Plan`, `PartSpec`, `TeamSpec`, `AgentSpec`, `RoundSpec`, `ConvergeSpec`, `Limits`, `Resources`; `.planning`: `FlowSpec`. CHANGELOG: "`Plan` and its specs with TOML load and dump, and `Limits` with the §8.1 defaults."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md
git commit -m "feat(model): add Plan, its specs, TOML load/dump, and Limits" \
  -m "Plan.load keeps a non-human origin found in the file and never raises trust (spec §4.6); flows stay reserved for v2; Limits carries the §8.1 defaults."
```

---

### Task 9: `RunConfig` and the `swarm_arena.config` models

**Files:**
- Create: `src/swarm_arena/_model/{settings.py,catalog.py,run_config.py}`
- Modify: facades `__init__.py`, `config.py`; `tests/_fakes/tools.py`; `tests/test_public_api.py`; `CHANGELOG.md`
- Test: `tests/_model/{test_settings.py,test_catalog.py,test_run_config.py}`

**Interfaces:**
- Consumes: Task 3 (`read_toml`, `dumps_toml`, `compile_object_schema`), Task 5 (`Ref`, `RefField`, `load_object`), Task 8 (`Limits`).
- Produces: as R4.1/R4.8: `SecretsPolicy`, `TracingConfig`, `GatewayConfig` (`hard_cap_required`), `CachePolicy` (with `RoleSwitch`), `ModelSpec` (`PromptCache`, `Prices`, property `provider`), `ModelCatalog` (`resolve`), `CatalogEntry` (`settable_model`), `PlannerCatalog` (`get`, `match`), `RunConfig` (kind `"run_config"`; `load`, `dumps`). Human-only hook: a component class attribute `_human_only: ClassVar[frozenset[str]]` (read with `getattr(cls, "_human_only", frozenset())`). `tests/_fakes/tools.py` gains `HumanOnlyWidget(Configurable)` with `_human_only = frozenset({"command"})` and `Config(command: list[str] = [], timeout_s: float | None = None)`.

- [ ] **Step 1: Write the failing tests**

```python
MODELS_TOML = '''...'''   # §4.13 [models.opus], [models.gpt], [models.flash] verbatim
CACHE_TOML = '''...'''    # §4.13 [cache] and [cache.roles] verbatim

def test_run_config_defaults() -> None:
    c = RunConfig()
    assert (c.reserve_fraction, c.board, c.curator, c.llm_gateway, c.cache) == (0.15, None, None, None, None)
    assert c.executor == Ref(path="swarm_arena.executor:LocalExecutor")
    assert (c.run_root, c.arena_root, c.artifact_root) == (".swarm-arena/runs", ".swarm-arena/arena", ".swarm-arena/artifacts")
    assert (c.schedule_timeout_s, c.sync_interval_s, c.lease_ttl_s, c.settle_timeout_s) == (600.0, 60.0, 60.0, 60.0)
    assert c.tracing.store_payloads and c.models == ModelCatalog({}) and c.limits == Limits()

def test_spec_model_catalog(tmp_path: Path) -> None:
    (tmp_path / "run.toml").write_text(MODELS_TOML)
    models = RunConfig.load(tmp_path / "run.toml").models
    opus = models.resolve("opus", "judge")
    assert (opus.model, opus.max_tokens, opus.prompt_cache.ttl, opus.provider) == ("anthropic/claude-opus-5-5", 8000, "5m", "anthropic")
    assert models.resolve("gpt", "manager").max_tokens == 4096
    assert models.resolve("flash", "curator").prices == ModelSpec.Prices(input=0.3, output=2.5)
    with pytest.raises(ValueError, match="role 'planner'"): models.resolve("gpt", "planner")
    with pytest.raises(ValueError, match="unknown model alias 'nope'"): models.resolve("nope", "judge")

def test_model_spec_rules() -> None:
    for bad in ({"model": "noslash", "roles": ["judge"]}, {"model": "a/b", "roles": []}, {"model": "a/b", "roles": ["judge", "judge"]}):
        with pytest.raises(ValidationError): ModelSpec.model_validate(bad)
    with pytest.raises(ValidationError): ModelCatalog.model_validate({"bad alias": {"model": "a/b", "roles": ["judge"]}})

def test_spec_cache_policy(tmp_path: Path) -> None:
    (tmp_path / "run.toml").write_text(CACHE_TOML)
    cache = RunConfig.load(tmp_path / "run.toml").cache
    assert cache is not None and (cache.backend, cache.ttl_s) == ("s3", 2_592_000)
    assert cache.roles == CachePolicy(backend="disk").roles == {
        "judge": CachePolicy.RoleSwitch(read=True, write=True), "planner": CachePolicy.RoleSwitch(read=False, write=True),
        "manager": CachePolicy.RoleSwitch(read=False, write=False), "curator": CachePolicy.RoleSwitch(read=True, write=True)}

def test_cache_backend_requirements() -> None:
    with pytest.raises(ValidationError, match="s3_path"): CachePolicy(backend="s3")
    with pytest.raises(ValidationError, match="redis_url_env"): CachePolicy(backend="redis")

def test_gateway() -> None:
    gw = GatewayConfig(url="http://gw:4000")
    assert (gw.master_key_env, gw.request_timeout_s, gw.key_duration_s, gw.default_estimated_output_tokens) == ("LITELLM_MASTER_KEY", 600.0, 86_400.0, 4096)
    assert gw.hard_cap_required(Limits(max_usd=5.0)) and not gw.hard_cap_required(Limits())
    assert not GatewayConfig(url="http://gw:4000", require_hard_cap=False).hard_cap_required(Limits(max_usd=5.0))
    with pytest.raises(ValidationError): GatewayConfig(url="gw:4000")

def test_secrets_allow_names() -> None:
    assert SecretsPolicy(allow=["ANTHROPIC_API_KEY"]).file is None
    with pytest.raises(ValidationError): SecretsPolicy(allow=["NOT-A-NAME"])

def test_catalog_settable_model() -> None:
    e = CatalogEntry(name="m", kind="manager", ref="_fakes.tools:Widget", description="d",
                     settable={"label": {"type": "string", "maxLength": 10}})
    M = e.settable_model()
    assert M().model_dump() == {"label": None} and M(label="x").model_dump() == {"label": "x"}
    for bad in ({"label": "x" * 11}, {"size": 1}):
        with pytest.raises(ValidationError): M.model_validate(bad)

def test_catalog_rejects_human_only_settable() -> None:
    entry = {"name": "s", "kind": "tool", "ref": "_fakes.tools:HumanOnlyWidget", "description": "d",
             "settable": {"command": {"type": "array"}}}
    with pytest.raises(ValidationError, match="human-only field 'command'"): PlannerCatalog.model_validate({"entries": [entry]})
    PlannerCatalog.model_validate({"entries": [{**entry, "settable": {"timeout_s": {"type": "number"}}}]})   # allowed

def test_catalog_get_and_match() -> None:
    cat = PlannerCatalog(entries=[CatalogEntry(name="w", kind="manager", ref="_fakes.tools:Widget", description="d",
                                               config={"size": 1}, settable={"label": {"type": "string"}})])
    assert cat.get("w").name == "w"
    with pytest.raises(ValueError, match="unknown catalog entry 'x'"): cat.get("x")
    assert cat.match(Ref(path="_fakes.tools:Widget", config={"size": 1, "label": "hi"}), "manager") is cat.entries[0]
    assert cat.match(Ref(path="_fakes.tools:Widget", config={"size": 2}), "manager") is None
    assert cat.match(Ref(path="_fakes.tools:Widget", config={"size": 1}), "tool") is None

def test_run_config_dumps_roundtrip_and_errors(tmp_path: Path) -> None:
    cfg = RunConfig(secrets=SecretsPolicy(file="/etc/sa.env", allow=["ANTHROPIC_API_KEY"]),
                    models=ModelCatalog({"gpt": ModelSpec(model="openai/gpt-6", roles=["judge"])}))
    assert RunConfig.model_validate(tomllib.loads(cfg.dumps())) == cfg
    (tmp_path / "r.toml").write_text("reserve_fraction = 1.5\n")
    with pytest.raises(ConfigError, match=r"(?s)r\.toml.*reserve_fraction"): RunConfig.load(tmp_path / "r.toml")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_model/test_settings.py tests/_model/test_catalog.py tests/_model/test_run_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._model.settings'`.

- [ ] **Step 3: Implement**

- `SecretsPolicy.allow` names `^[A-Za-z_][A-Za-z0-9_]*$`. `GatewayConfig.url` starts with `http://` or `https://`; `hard_cap_required(limits)` = `require_hard_cap` if not `None`, else `limits.max_usd is not None`. `CachePolicy`: `s3` needs `s3_path`; `redis` needs `redis_url_env`; `ttl_s >= 1`; `roles` default per R4.8.
- `ModelSpec`: `model` contains `/`; `roles` non-empty and unique; `max_tokens >= 1`; `Prices` values `>= 0`; `provider` = text before the first `/`. `ModelCatalog` aliases match `^[A-Za-z0-9][A-Za-z0-9._-]*$`; `resolve` raises `ValueError(f"unknown model alias {alias!r}")` or `ValueError(f"model {alias!r} may not be used for role {role!r}")`.
- `CatalogEntry`: `name` slug; `ref` uses the `Ref` path format; `settable` compiles at validation (bad schema → error at load). `settable_model()` returns a model with every settable field optional (`default None`), each value validated against its property schema when present.
- `PlannerCatalog`: unique entry names; for each entry with non-empty `settable`, load the class with `load_object(entry.ref)` and reject `settable` keys in its `_human_only` (`ValueError(f"catalog entry {name!r}: human-only field {key!r} cannot be settable")`). `match(ref, kind)`: first entry with `entry.kind == kind`, `entry.ref == ref.path`, and `{k: v for k, v in ref.config.items() if k not in entry.settable} == entry.config` (raw comparison; M1b's `check_catalog` does the full check).
- `RunConfig`: `reserve_fraction` in `[0, 1)`; `goal_weights` values `> 0`; `executor`, `board`, `curator` are `RefField`; `load(path)` → `ConfigError(f"{path}: {err}")` on errors; `dumps()` = `dumps_toml(self.model_dump(mode="json", exclude_none=True))`. Relative roots stay as written.
- Public surface: `swarm_arena`: `RunConfig`; `.config`: `ModelCatalog`, `ModelSpec`, `CachePolicy`, `PlannerCatalog`, `CatalogEntry`, `SecretsPolicy`, `TracingConfig`, `GatewayConfig`. CHANGELOG: "`RunConfig` and the `swarm_arena.config` models, with load-time catalog rules."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md
git commit -m "feat(model): add RunConfig, the model catalog, and the planner catalog" \
  -m "Human-only settings with load-time rules: aliases and roles, and script fields that can never be settable (spec §4.1, §4.13)."
```

---

### Task 10: `Store`, `LocalStore`, `ArtifactStore`

**Files:**
- Create: `src/swarm_arena/_store/{__init__.py,base.py,local.py,artifacts.py}`
- Modify: `tests/conftest.py` (fixtures `artifact_root(tmp_path: Path) -> str` = `str(tmp_path / "artifacts")`, `artifacts(artifact_root: str) -> ArtifactStore`)
- Test: `tests/_store/{test_base.py,test_local.py,test_artifacts.py}`

**Interfaces:**
- Consumes: Task 2 errors and `sha256_hex`; Task 6 `FileRef`.
- Produces: R4.13.2 `Blob`, `Store` (`url`, `read`, `create`, `replace`, `exists`, `list`, `child`, `probe_conditional_writes`), `LocalStore(root: Path)` (`replace` raises `NotImplementedError("Store.replace arrives in M1b")`), `open_store(url: str) -> Store`, `ArtifactStore(store)` (`put_bytes`, `put_file`, `get_bytes`, `fetch`).

- [ ] **Step 1: Write the failing tests**

```python
def test_create_then_read(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    assert s.create("a/b.json", b"x") == sha256_hex(b"x") and s.read("a/b.json") == Blob(b"x", sha256_hex(b"x"))

def test_create_is_write_once(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    s.create("k", b"1")
    with pytest.raises(AlreadyExists): s.create("k", b"2")
    assert s.read("k").data == b"1"

def test_create_race_has_one_winner(tmp_path: Path) -> None:
    # 8 threads call create("k", f"{i}".encode()); exactly one returns, 7 raise AlreadyExists; content == the winner's

def test_failed_create_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # monkeypatch os.link to raise OSError; create raises OSError; not exists("k"); no files left in tmp_path

def test_read_exists_list_child(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    for k in ("runs/a.json", "runs/b.json", "reviews/x/1.jsonl"): s.create(k, b"{}")
    assert s.list("runs/") == ["runs/a.json", "runs/b.json"] and s.exists("runs/a.json") and not s.exists("runs/c")
    s.child("runs").create("c.json", b"1")
    assert s.read("runs/c.json").data == b"1"
    with pytest.raises(NotFound): s.read("missing")
    with pytest.raises(NotImplementedError, match="M1b"): s.replace("runs/a.json", b"2", if_match="x")

@pytest.mark.parametrize("bad", ["", "/abs", "../x", "a/./b"])
def test_bad_keys(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="key"): LocalStore(tmp_path).create(bad, b"x")

def test_open_store(tmp_path: Path) -> None:
    s = open_store(str(tmp_path))
    assert isinstance(s, LocalStore) and s.url == str(tmp_path.resolve())
    s.probe_conditional_writes()
    with pytest.raises(ConditionalWritesUnsupported, match="s3"): open_store("s3://bucket/p")
    with pytest.raises(ConditionalWritesUnsupported, match="gs"): open_store("gs://bucket")

def test_artifacts_put_get_fetch(artifacts: ArtifactStore, tmp_path: Path) -> None:
    h = artifacts.put_bytes(b"hello")
    assert h == sha256_hex(b"hello") == artifacts.put_bytes(b"hello") and artifacts.get_bytes(h) == b"hello"
    (tmp_path / "a.txt").write_bytes(b"hi")
    ref = artifacts.put_file(tmp_path / "a.txt")
    assert (ref.name, ref.sha256, ref.size) == ("a.txt", sha256_hex(b"hi"), 2)
    artifacts.fetch(h, tmp_path / "out" / "x.txt")
    assert (tmp_path / "out" / "x.txt").read_bytes() == b"hello" and (tmp_path / "out" / "x.txt").stat().st_mode & 0o222 == 0
    artifacts.fetch(h, tmp_path / "w.txt", read_only=False)
    assert (tmp_path / "w.txt").stat().st_mode & 0o200
    with pytest.raises(NotFound): artifacts.get_bytes("0" * 64)

def test_get_bytes_verifies_hash(artifacts: ArtifactStore, artifact_root: str) -> None:
    # corrupt the stored object on disk (chmod u+w, rewrite) → StorageError match "hash"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_store -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._store'`.

- [ ] **Step 3: Implement**

- Keys: relative POSIX, non-empty, no leading `/`, no `.` or `..` part → else `ValueError(f"bad store key {key!r}")`.
- `LocalStore.create`: write to `<key>.tmp-<token_hex(6)>` with `O_CREAT | O_EXCL`, `fsync`, `os.link` to the key (`FileExistsError` → `AlreadyExists`), always unlink the temp; etag = `sha256_hex(data)`. `read` → `Blob(data, sha256_hex(data))`, missing → `NotFound`. `list(prefix)`: sorted keys of regular files under `root` that start with `prefix`, excluding `.tmp-` and `.lock` names. `child(prefix)` = `LocalStore(root / prefix)`. `url` = `str(root.resolve())`. `probe_conditional_writes` is a no-op on the base class (`S3Store` overrides it in M2).
- `open_store`: a URL with a scheme (`^[A-Za-z][A-Za-z0-9+.-]*://`) → `ConditionalWritesUnsupported(f"{scheme}:// storage is not available in this version")` (M2 adds `s3://`); otherwise `LocalStore(Path(url).expanduser().resolve())`.
- `ArtifactStore`: key = the hex sha256; `put_bytes` treats `AlreadyExists` as success; `put_file(path)` returns `FileRef(name=path.name, path=path.name, sha256, size)` (callers set `path` to the workspace-relative path with `model_copy`); `get_bytes` checks the hash (`StorageError(f"artifact {h} failed its hash check")`); `fetch` writes `dest` (parents created), mode `0o444` when `read_only` else `0o644`.

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_store tests/_store tests/conftest.py
git commit -m "feat(store): add the write-once Store, LocalStore, and content-addressed ArtifactStore" \
  -m "Conditional create only (O_EXCL + link); no unconditional put (R4.13.2). replace and the lease come in M1b."
```

---

### Task 11: File snapshots

**Files:**
- Create: `src/swarm_arena/_tooling/files.py`
- Modify: `tests/_fakes/tools.py` (`FilesWidget(Configurable)` with `name = "files-widget"` and `Config(files: FileSet = FileSet(), members: list[RefField] = [], label: str = "")`). Roadmap R4.13.9 already has `snapshot_ref`: no roadmap edit.
- Test: `tests/_tooling/test_files.py`

**Interfaces:**
- Consumes: Task 4 `check_relpath`, Task 5 `Ref`, Task 6 `FileSet`, `Goal`, Task 8 `Plan`, Task 10 `ArtifactStore`.
- Produces: R4.13.9 `snapshot_fileset(fs: FileSet, base_dir: Path, artifacts: ArtifactStore) -> FileSet`, `snapshot_plan(plan: Plan, artifacts: ArtifactStore) -> Plan`, `snapshot_ref(ref: Ref, base_dir: Path, artifacts: ArtifactStore) -> Ref` (used by Task 23 and by M1b for replanned revisions).

- [ ] **Step 1: Write the failing tests**

```python
def test_snapshot_fileset(tmp_path: Path, artifacts: ArtifactStore) -> None:
    # files: problem/a.phyk, problem/b.phyk, problem/run.py, problem/sub/c.phyk
    fs = snapshot_fileset(FileSet(globs=["problem/*.phyk", "problem/run.py"]), tmp_path, artifacts)
    assert list(fs.manifest) == ["problem/a.phyk", "problem/b.phyk", "problem/run.py"]
    assert fs.globs == ["problem/*.phyk", "problem/run.py"]
    assert artifacts.get_bytes(fs.manifest["problem/run.py"]) == (tmp_path / "problem/run.py").read_bytes()

def test_glob_matching_nothing_raises(tmp_path: Path, artifacts: ArtifactStore) -> None:   # Review Focus 2
    (tmp_path / "problem").mkdir(); (tmp_path / "problem/a.phyk").write_text("x")
    with pytest.raises(ValueError, match=r"glob 'problem/\*\.phky' matched no file"):
        snapshot_fileset(FileSet(globs=["problem/*.phky"]), tmp_path, artifacts)

def test_symlinks_rejected(tmp_path: Path, artifacts: ArtifactStore) -> None:
    # problem/link.txt → symlink to a file outside: ValueError match "symlink.*problem/link.txt"
    # problem/linkdir → symlink to a directory; glob "problem/linkdir/*": ValueError match "symlink"

def test_reserved_names_rejected(tmp_path: Path, artifacts: ArtifactStore) -> None:
    # file swarm/x.txt with glob "swarm/*" → ValueError match "reserved"

def test_already_snapshotted_reads_no_disk(tmp_path: Path, artifacts: ArtifactStore) -> None:
    # snapshot, delete the files, snapshot the result again → equal FileSet, no error

def test_snapshot_ref_nested(tmp_path: Path, artifacts: ArtifactStore) -> None:
    ref = Ref(path="_fakes.tools:FilesWidget", config={"files": ["data/*.csv"], "label": "x",
              "members": [{"path": "_fakes.tools:FilesWidget", "config": {"files": ["more/*.txt"]}}]})
    out = snapshot_ref(ref, tmp_path, artifacts)
    assert list(out.config["files"]["manifest"]) == ["data/a.csv"] and out.config["label"] == "x"
    assert list(out.config["members"][0]["config"]["files"]["manifest"]) == ["more/b.txt"]
    assert snapshot_ref(Ref(path="_fakes.tools:Widget", config={"size": 2}), tmp_path, artifacts) == Ref(path="_fakes.tools:Widget", config={"size": 2})

def test_snapshot_ref_keeps_omitted_fileset_omitted(tmp_path: Path, artifacts: ArtifactStore) -> None:
    out = snapshot_ref(Ref(path="_fakes.tools:FilesWidget", config={"label": "x"}), tmp_path, artifacts)
    assert out == Ref(path="_fakes.tools:FilesWidget", config={"label": "x"}) and "files" not in out.config

def test_snapshot_plan(tmp_path: Path, artifacts: ArtifactStore) -> None:
    # plan.toml in tmp_path: goal files ["problem/*"], one part whose goal has files ["part/*"], an agent whose tool and
    # manager are FilesWidget refs (tool files ["tool/*"]), a judge FilesWidget (files ["judge/*"]), and
    # policy = { path = "_fakes.tools:Widget" } (the default swarm_arena:SuccessiveHalving arrives in M1b)
    plan = snapshot_plan(Plan.load(tmp_path / "plan.toml"), artifacts)
    assert plan.goal.key().files_sha256 and plan.parts[0].goal.files.snapshotted
    assert plan.teams[0].agents[0].tool.config["files"]["manifest"] and plan.judges[0].config["files"]["manifest"]
    assert plan.base_dir == tmp_path and snapshot_plan(plan, artifacts) == plan

def test_snapshot_plan_unimportable_ref(tmp_path: Path, artifacts: ArtifactStore) -> None:
    # a judge ref "no_such_mod:J" → ValueError match "cannot import no_such_mod:J"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_tooling/test_files.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._tooling.files'`.

- [ ] **Step 3: Implement**

- `snapshot_fileset`: return `fs` unchanged when `fs.manifest` is non-empty or `fs.globs` is empty. For each glob, `sorted(base_dir.glob(g))`, regular files only; none → `ValueError(f"glob {g!r} matched no file under {base_dir}")`. For each match: `rel = check_relpath(path.relative_to(base_dir).as_posix())` (reserved names → `ValueError` containing "reserved"); the file or any parent directory below `base_dir` that is a symlink → `ValueError(f"symlink not allowed: {rel}")`; `artifacts.put_file`. Manifest keys sorted.
- `snapshot_ref`: load the class (`Configurable` subclass, else `TypeError`), validate the config with `cls.Config`, then for every field of the validated config: a `FileSet` value → `snapshot_fileset`; a `Ref` or a list of `Ref` → `snapshot_ref` recursively. Write back only those keys, and only when the raw `ref.config` has the key, into a copy of `ref.config` (`model_dump(mode="json")` of the new value); an omitted default `FileSet` stays omitted, and other keys keep their raw form (R4.13.9).
- `snapshot_plan`: `plan.goal.files`, every part goal's files, and `snapshot_ref` on every `Ref` site (each agent's `tool` and `manager`, plan and part `checks` and `judges`, `policy`, the converge team's agents); `base_dir = plan.base_dir`; return `plan.model_copy(update=...)` (keeps the private `_base_dir`).

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_tooling/files.py tests/_tooling/test_files.py tests/_fakes/tools.py
git commit -m "feat(tooling): snapshot goal and component files into the artifact store" \
  -m "Globs become a path→sha256 manifest so runs stay reproducible (spec §4.2); a glob that matches nothing and symlinks are errors. snapshot_ref writes back only keys the raw config has, so an omitted default FileSet stays omitted (R4.13.9)."
```

---

### Task 12: `Executor` ABC and `Job`

**Files:**
- Create: `src/swarm_arena/_exec/{__init__.py,executor.py}`
- Modify: facade `executor.py`; `tests/_fakes/tools.py`; `tests/test_public_api.py`; `CHANGELOG.md`; roadmap R4.7
- Test: `tests/_exec/test_executor.py`

**Interfaces:**
- Consumes: Task 5 `Configurable`, `Ref`; Task 6 `ToolRequest`, `Command`, `ToolOutput`, `ParentRef`; Task 8 `Resources`.
- Produces: R4.7 `Job` (nested `Stage`, `CandidateFiles`, `Started` with kind `"job_started"`), `Executor(Configurable, ABC)` with `preflight`, abstract `submit`, `capacity`, `aclose`, defaults `confirm_dead -> None`, `kill_orphan -> False`; plus the new field `Job.on_partial: Callable[[ToolOutput], Awaitable[None]] | None = None` (called inside the shielded stop sequence with the partial output when `submit` is cancelled; never serialized). `Job.Stage.model_rebuild()` runs after the class body. `tests/_fakes/tools.py` gains `make_job(**fields: Any) -> Job`: a command job with `id="x-1"`, `command=Command(argv=["true"])`, `stage=Job.Stage()`, `timeout_s=5.0`, `grace_s=1.0`, `resources=Resources()`, `env={}`, `secret_env=()`, `secrets_file=None`, `artifact_root="/tmp/a"`; `fields` override (its defaults dict is typed `dict[str, Any]`).

- [ ] **Step 1: Write the failing tests**

```python
def test_kind_rules() -> None:
    with pytest.raises(ValueError, match="kind 'tool' needs tool and request"): make_job(kind="tool", command=None)
    with pytest.raises(ValueError, match="kind 'command' needs command"): make_job(command=None)

def test_secret_values_hidden_from_repr() -> None:
    assert "sk-123" not in repr(make_job(secret_values={"OPENAI_API_KEY": "sk-123"}))

def test_stage_paths_checked() -> None:
    for root in ({"swarm/x": "a" * 64}, {"./a": "a" * 64}):
        with pytest.raises(ValidationError): Job.Stage(root=root)
    for files in ({"../x": "a" * 64}, {"/abs": "a" * 64}):
        with pytest.raises(ValidationError): Job.CandidateFiles(request={}, output={}, files=files)
    assert Job.Stage(candidate=Job.CandidateFiles(request={}, output={}, files={})).candidate is not None

def test_started_roundtrip() -> None:
    s = Job.Started(kind="process", job_id="c-1", host_id="h:1", pid=10, pgid=10, start_time="Sat Oct 18 13:52:01 2026")
    assert Job.Started.model_validate_json(s.model_dump_json()) == s and s.schema_version == 1

async def test_executor_defaults() -> None:
    class Minimal(Executor):                       # local subclass: only instantiated here, never built from a Ref
        async def submit(self, job: Job) -> ToolOutput: return ToolOutput(status=RunStatus.OK)
        async def capacity(self) -> Resources | None: return None
        async def aclose(self) -> None: return None
    m, started = Minimal(), Job.Started(kind="process", job_id="c-1")
    assert m.preflight() is None and await m.confirm_dead(started) is None and await m.kill_orphan(started) is False
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_exec/test_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._exec'`.

- [ ] **Step 3: Implement**

`Job` is `@dataclass(frozen=True, kw_only=True)` with the R4.7 fields plus `on_partial`; `secret_values`, `on_started`, `on_partial` use `field(..., repr=False)`. `__post_init__` enforces the kind rules. `Stage.root` and `CandidateFiles.files` keys through `check_key` (R4.7 "keys in normal form"). Roadmap R4.7: add the `on_partial` line under `on_started`, with the comment "the only channel by which an executor hands a cancel-path partial output to the caller (M1b's `execute_tool_run` sets it)", and under `LocalExecutor` the line "[plan, M1a] The executor runs with `Job.timeout_s` and `Job.grace_s` and writes them into `Command.timeout_s` and `Command.grace_s`; the driver computes them with `effective_tool_timeout` and `tool_grace` (`_tooling/tool.py`)." Public surface: `.executor`: `Executor`, `Job`. CHANGELOG: "`Executor` ABC and `Job` (`swarm_arena.executor`)."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(exec): add the Executor ABC and Job" \
  -m "Job carries the planned workspace, limits, and env; secret values never appear in repr. Roadmap R4.7 adds Job.on_partial so a cancelled run can still hand its partial output to the core (spec §4.8, §6.4)."
```

---

### Task 13: `Tool` ABC, `Action`, and result files

**Files:**
- Create: `src/swarm_arena/_tooling/{tool.py,result_file.py}`
- Modify: facade `tool.py`; `tests/_fakes/tools.py`; `tests/test_public_api.py`; `CHANGELOG.md`; roadmap R4.2, R4.13.9
- Test: `tests/_tooling/{test_tool.py,test_result_file.py}`

**Interfaces:**
- Consumes: Task 5 `Configurable`; Task 6 types; Task 4 `check_relpath`; Task 2 `StoredModel`, `canonical_json`, `sha256_hex`.
- Produces:
  - `Action(Generic[InputT])` (`InputT = TypeVar("InputT", bound=BaseModel)`), `@dataclass(frozen=True, kw_only=True)`: `name: str`, `description: str`, `input_model: type[InputT]`, `handler: Callable[[Any, InputT], Awaitable[JsonValue]]` (called with the agent's `AgentContext` and an `input_model` instance), `counts_as: Literal["board_read", "none"] = "none"`. `Tool.extra_actions() -> Sequence[Action[Any]]`.
  - `Tool(Configurable, ABC)` as R4.2, plus private hooks `_own_files(self) -> dict[str, str]` (default `{}`) and `_protected(self, root: AbstractSet[str]) -> frozenset[str]` (default `frozenset()`).
  - `effective_tool_timeout(tool: Tool, *, round_timeout_s: float | None, limit_s: float) -> float`; `tool_grace(tool: Tool) -> float`.
  - `request_json(request: ToolRequest) -> dict[str, Any]`; `ToolResultFile(StoredModel)` (kind `"tool_result"`: `data: dict[str, Any]`, `usage: Usage = Usage()`, `files: list[str] = []`, `failed_provider: str | None = Field(default=None, min_length=1)`, the result.json key of roadmap R4.2 `ToolOutput.failed_provider`); `read_regular_file(workspace: Path, rel: str) -> bytes` (the R4.13.8 `read_regular_file`: every workspace file read after a run goes through it); `read_result_json(workspace: Path) -> dict[str, Any]`; `tool_output_from_result(workspace: Path, run: CompletedRun) -> ToolOutput`; `command_output_from_result(workspace: Path, run: CompletedRun) -> ToolOutput`; `note_in_log(path: Path, message: str) -> None`.
  - `tests/_fakes/tools.py`: `PyTool(Tool)` (`name = "py"`, `description = "Runs Python code."`, `input_model = PyInputs` with `n: int = 0`; `Config(code: str, timeout_s: float | None = None, grace_s: float = 10.0)`; `config` property typed `PyTool.Config`; `command()` → `Command(argv=["python", "-c", code], timeout_s=..., grace_s=...)`; `collect()` → `tool_output_from_result`); `BadCollectTool(PyTool)` whose `collect` raises `RuntimeError("collect broke")`; `EchoTool(Tool)` (`name = "echo"`, `Config(text: str = "")`, no `timeout_s` or `grace_s` field); `WRITE = "import json, os; json.dump({0}, open(os.environ['SWARM_ARENA_RESULT'], 'w'))"` (code that writes the object literal given to `.format()` as `swarm/result.json`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/_tooling/test_tool.py
def test_tool_defaults() -> None:
    t = PyTool(PyTool.Config(code="pass"))
    assert (t.is_deterministic(ToolRequest()), tuple(t.editable()), t.estimate(ToolRequest()), t.providers(),
            tuple(t.extra_actions()), t.preflight()) == (True, (), None, frozenset(), (), None)
    assert t.get_input_model() is PyInputs and t.get_description() == "Runs Python code."
    assert t._own_files() == {} and t._protected({"a"}) == frozenset()

def test_default_fingerprint() -> None:
    t = PyTool(PyTool.Config(code="pass"))
    assert t.fingerprint(ToolOutput(status=RunStatus.OK, data={"b": 1, "a": 2})) == sha256_hex(b'{"a":2,"b":1}')

def test_abstract_methods_required() -> None:
    class Half(Tool):
        name = "h"
        description = "h"
        input_model = PyInputs
    with pytest.raises(TypeError, match="abstract"): cast(Any, Half)()

@pytest.mark.parametrize(("config_s", "round_s", "expected"), [
    (None, None, 3600.0), (1200.0, None, 1200.0), (1200.0, 2000.0, 2000.0), (None, 5000.0, 3600.0), (9000.0, None, 3600.0)])
def test_effective_tool_timeout(config_s: float | None, round_s: float | None, expected: float) -> None:
    tool = PyTool(PyTool.Config(code="pass", timeout_s=config_s))
    assert effective_tool_timeout(tool, round_timeout_s=round_s, limit_s=3600.0) == expected   # spec §6.1
    assert effective_tool_timeout(EchoTool(), round_timeout_s=None, limit_s=60.0) == 60.0        # no timeout_s field

def test_tool_grace() -> None:
    assert tool_grace(PyTool(PyTool.Config(code="pass", grace_s=2.5))) == 2.5 and tool_grace(EchoTool()) == 10.0
```
```python
# tests/_tooling/test_result_file.py  (helper run(tmp_path, status, rc) -> CompletedRun with stdout/stderr files)
def test_request_json_shape() -> None:
    req = ToolRequest(inputs={"lr": 0.1}, round=2, round_overrides={"epochs": 100}, replicate=1)
    assert request_json(req) == {"schema_version": 1, "params": {"lr": 0.1}, "round": 2,
                                 "round_overrides": {"epochs": 100}, "replicate": 1}

def test_ok_with_valid_result(tmp_path: Path) -> None:
    write_result(tmp_path, {"data": {"f": 1.5}, "usage": {"input_tokens": 10, "usd": 0.01}, "files": ["out/p.txt"]})
    (tmp_path / "out").mkdir(); (tmp_path / "out/p.txt").write_text("p")
    out = tool_output_from_result(tmp_path, run(tmp_path, RunStatus.OK, 0))
    assert (out.status, out.partial, out.data, out.usage.input_tokens) == (RunStatus.OK, False, {"f": 1.5}, 10)
    assert [(f.path, f.sha256) for f in out.files] == [("out/p.txt", sha256_hex(b"p"))]

@pytest.mark.parametrize(("content", "reason"), [
    (None, "swarm/result.json is missing"), ("{nope", "not valid JSON"), ('{"data": [1]}', "data"),
    ('{"data": {}, "usgae": {}}', "usgae"), ('{"schema_version": 2, "data": {}}', "newer"),
    ('{"data": {}, "files": ["swarm/x"]}', "reserved"), ('{"data": {}, "files": ["out/none.txt"]}', "out/none.txt"),
    ('{"data": {"f": NaN}}', "non-finite number"), ('{"data": {}, "failed_provider": ""}', "failed_provider"),
    ('{"data": {"f": 1e999}}', "non-finite number"), ('{"data": {"g": [-1e999]}}', "non-finite number"),
    ('{"data": {}, "usage": {"usd": 1e999}}', "non-finite number")])
def test_exit_zero_without_valid_result_is_error(tmp_path: Path, content: str | None, reason: str) -> None:   # Review Focus 3
    if content is not None: (tmp_path / "swarm").mkdir(exist_ok=True); (tmp_path / "swarm/result.json").write_text(content)
    r = run(tmp_path, RunStatus.OK, 0)
    out = tool_output_from_result(tmp_path, r)
    assert (out.status, out.data, out.files) == (RunStatus.ERROR, {}, [])
    assert "swarm-arena: " in r.stderr_path.read_text() and reason in r.stderr_path.read_text()

@pytest.mark.parametrize("status", [RunStatus.TIMEOUT, RunStatus.CANCELLED])
def test_killed_run_with_result_is_partial(tmp_path: Path, status: RunStatus) -> None:
    write_result(tmp_path, {"data": {"step": 3}})
    out = tool_output_from_result(tmp_path, run(tmp_path, status, None))
    assert (out.status, out.partial, out.data) == (status, True, {"step": 3})
    empty = tool_output_from_result(tmp_path / "none", run(tmp_path / "none", status, None))   # no result file
    assert (empty.partial, empty.data) == (False, {})

def test_error_keeps_usage_only(tmp_path: Path) -> None:
    write_result(tmp_path, {"data": {"x": 1}, "usage": {"output_tokens": 7}})
    out = tool_output_from_result(tmp_path, run(tmp_path, RunStatus.ERROR, 1))
    assert (out.status, out.data, out.usage.output_tokens) == (RunStatus.ERROR, {}, 7)

@pytest.mark.parametrize(("status", "rc"), [(RunStatus.OK, 0), (RunStatus.ERROR, 1)])
def test_failed_provider_key_is_provider_failed(tmp_path: Path, status: RunStatus, rc: int) -> None:   # roadmap R4.2
    write_result(tmp_path, {"data": {"x": 1}, "usage": {"usd": 0.2}, "failed_provider": "openai"})
    out = tool_output_from_result(tmp_path, run(tmp_path, status, rc))
    assert (out.status, out.failed_provider, out.partial, out.data, out.files, out.usage.usd) == (
        RunStatus.PROVIDER_FAILED, "openai", False, {}, [], 0.2)
    killed = tool_output_from_result(tmp_path, run(tmp_path, RunStatus.TIMEOUT, None))   # the kill status wins
    assert (killed.status, killed.partial, killed.data, killed.failed_provider) == (RunStatus.TIMEOUT, True, {"x": 1}, None)
    write_result(tmp_path, {"data": {}})
    assert tool_output_from_result(tmp_path, run(tmp_path, RunStatus.OK, 0)).failed_provider is None

def test_command_output_is_the_raw_object(tmp_path: Path) -> None:
    write_result(tmp_path, {"passed": True, "reasons": []})
    assert command_output_from_result(tmp_path, run(tmp_path, RunStatus.OK, 0)).data == {"passed": True, "reasons": []}
    missing = command_output_from_result(tmp_path / "none", run(tmp_path / "none", RunStatus.OK, 0))
    assert (missing.status, missing.data) == (RunStatus.ERROR, {})

def test_command_output_rejects_non_finite(tmp_path: Path) -> None:
    (tmp_path / "swarm").mkdir()
    (tmp_path / "swarm/result.json").write_text('{"passed": true, "x": 1e999}')   # raw text: json.dumps cannot write 1e999
    r = run(tmp_path, RunStatus.OK, 0)
    out = command_output_from_result(tmp_path, r)
    assert (out.status, out.data) == (RunStatus.ERROR, {}) and "non-finite number" in r.stderr_path.read_text()

@pytest.mark.parametrize("kind", ["fifo", "symlink", "symlinked_swarm_dir"])
async def test_result_must_be_a_regular_file(tmp_path: Path, kind: str) -> None:
    ws = tmp_path / "ws"
    # fifo: os.mkfifo(ws / "swarm/result.json"); symlink: ws/swarm/result.json → a valid result file in tmp_path;
    # symlinked_swarm_dir: ws/swarm → a directory in tmp_path (outside ws) that holds a valid result.json
    r = run(ws, RunStatus.OK, 0)
    with anyio.fail_after(5):                     # a FIFO opened without O_NONBLOCK would block forever
        out = await anyio.to_thread.run_sync(tool_output_from_result, ws, r, abandon_on_cancel=True)
    assert out.status is RunStatus.ERROR and "not a regular file" in r.stderr_path.read_text()

@pytest.mark.parametrize("kind", ["regular", "missing", "fifo", "symlink"])
async def test_note_in_log_never_blocks_or_follows(tmp_path: Path, kind: str) -> None:
    p = tmp_path / "stderr.log"
    # regular: p holds "x\n"; missing: no p; fifo: os.mkfifo(p), no reader; symlink: p → tmp_path / "target" ("t")
    with anyio.fail_after(5):
        await anyio.to_thread.run_sync(note_in_log, p, "m", abandon_on_cancel=True)   # returns None, raises nothing
    # regular → p.read_text().endswith("swarm-arena: m\n"); missing → p.read_text() == "swarm-arena: m\n";
    # fifo → returned with no exception; symlink → (tmp_path / "target").read_text() == "t"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_tooling/test_tool.py tests/_tooling/test_result_file.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._tooling.tool'`.

- [ ] **Step 3: Implement**

- `effective_tool_timeout`: `round_timeout_s` if set, else `getattr(tool.config, "timeout_s", None)` if set, else `limit_s`; then `min(value, limit_s)` (spec §6.1). `tool_grace` = `getattr(tool.config, "grace_s", 10.0)`.
- `request_json` writes `inputs` as `params` (spec §4.12).
- `read_regular_file(workspace, rel)`: refuses a symlink in any component below `workspace` (`lstat` of each parent) and anything that is not a regular file. It opens the last component with `os.open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)` and checks `fstat` `S_ISREG` before it reads, so a FIFO never blocks and a symlink is never followed. Any failure → `ValueError(f"{rel} is not a regular file")`.
- `read_result_json` reads `swarm/result.json` through `read_regular_file` and parses with `json.loads(text, parse_constant=_reject, parse_float=_finite_float)`. `_reject` refuses `NaN` and `Infinity`; `_finite_float(s)` returns `float(s)` and raises `ValueError("non-finite number in swarm/result.json")` when it is not finite (`json.loads("1e999")` gives `inf` without calling `parse_constant`). Results never carry non-finite numbers into `ToolOutput.data`, `usage`, or the journal.
- `tool_output_from_result(workspace, run)`, where "valid" means `read_result_json` succeeds and validates as `ToolResultFile`, and every declared file passes `check_relpath` and is read (and hashed) through `read_regular_file`:
  - `run.status` OK or ERROR, valid, and `failed_provider` set → `PROVIDER_FAILED`, `failed_provider` copied, `partial=False`, `data={}`, `files=[]`, `usage` from the file (roadmap R4.2; this is how a script reports "credits used up" or "provider unavailable" for spec §6.4, with `{"data": {}, "failed_provider": "<provider>"}` and any exit code). The rules below apply only when this one does not.
  - `run.status` OK and valid → `OK`, with `data`, `usage`, and `files` (`FileRef` hashed locally: `name` = base name, `path` = the declared path).
  - `run.status` OK and not valid → `ERROR`, `data={}`, `files=[]`; `note_in_log(run.stderr_path, reason)`.
  - `TIMEOUT` or `CANCELLED` and valid → that status, `partial=True`, `data`, `usage`, `files`; not valid → that status, `partial=False`, `data={}`.
  - `ERROR` or `OOM` → that status, `data={}`, `files=[]`, and `usage` from the file when it is valid (so reported spend still counts, spec §8.1).
  - `TIMEOUT`, `CANCELLED`, and `OOM` ignore `failed_provider`: the kill status wins and `ToolOutput.failed_provider` stays `None`.
  - Always `started_at=run.started_at`, `ended_at=run.ended_at`.
- `command_output_from_result`: parses with `read_result_json` too (so the regular-file and finite-number rules apply); `data` = the raw JSON object (`{}` when missing or invalid); status = `run.status`, except OK with no valid object → `ERROR` plus a log note.
- `note_in_log(path, message)`: `os.open(path, O_WRONLY | O_APPEND | O_CREAT | O_NOFOLLOW | O_NONBLOCK, 0o644)`, then `fstat` must be `S_ISREG`, then one `os.write` of `"swarm-arena: <message>\n"`, then close. A FIFO (ENXIO, or open with a reader present), a symlink (ELOOP), another non-regular file, or any `OSError` → the note is dropped. It never raises and never blocks. It is the write twin of `read_regular_file`, so a tool that replaces `swarm/stderr.log` cannot stall the stop sequence, `collect`, or the executor. (Checked on macOS: a FIFO open gives ENXIO and a symlink gives ELOOP.)
- Roadmap R4.2: `Action` becomes `class Action(Generic[InputT])` with `input_model: type[InputT]` and `handler: Callable[[Any, InputT], Awaitable[JsonValue]]` ("first argument: the agent's `AgentContext`; M1b may narrow `Any` to `AgentContext`"), `extra_actions() -> Sequence[Action[Any]]`, and the two private `Tool` hooks with the comment "`_own_files()`: the tool's own root files; the driver's protected set is `tool._protected(root) | private_paths` (R4.13.9)". Reason: a handler typed with its own input model is not assignable to `Callable[[AgentContext, BaseModel], ...]` (parameter contravariance), and `AgentContext` does not exist before M1b.
- Roadmap R4.13.9: add `effective_tool_timeout(tool: Tool, *, round_timeout_s: float | None, limit_s: float) -> float` and `tool_grace(tool: Tool) -> float` under a `# _tooling/tool.py` comment, with "the only tool timeout and grace computation; M1b passes `round_spec.tool_timeout_s` and `limits.tool_timeout_s`", and `read_regular_file(workspace: Path, rel: str) -> bytes` under a `# _tooling/result_file.py` comment, with "the `read_regular_file` of R4.13.8: O_NOFOLLOW | O_NONBLOCK, regular files only, no symlinked component", and, under the same comment, `note_in_log(path: Path, message: str) -> None` with "never blocks or follows: O_APPEND | O_NOFOLLOW | O_NONBLOCK, regular files only, otherwise the note is dropped; M4 job steps reuse it". Reason: the driver (M1b) and the conformance kit must compute one timeout the same way, and R4.13.8 names `read_regular_file` and `note_in_log` without a module or signature.
- Public surface: `.tool`: `Tool`, `Action`. CHANGELOG: "`Tool` ABC and `Action`; the `swarm/result.json` contract."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(tooling): add the Tool ABC, Action, and the result-file rules" \
  -m "Exit 0 plus a valid swarm/result.json is ok; anything else is error with the reason in the stderr log; killed runs keep a valid result as partial (spec §4.3, §4.12). A valid result that names failed_provider is provider_failed, so a script can report the provider that failed (roadmap R4.2, spec §6.4). Workspace files are read as regular files only (no FIFO, no symlink), log notes never block or follow a link, and non-finite numbers such as 1e999 are refused. Roadmap R4.2 makes Action generic over its input model and records the protected-set hooks; R4.13.9 records the one timeout and grace helper and the read_regular_file and note_in_log signatures."
```

---

### Task 14: Tool identity, staging, and materialize

**Files:**
- Create: `src/swarm_arena/_tooling/{staging.py,materialize.py}`
- Modify: roadmap R4.13.9
- Test: `tests/_tooling/{test_staging.py,test_materialize.py}`

**Interfaces:**
- Consumes: Task 4 `check_relpath`, `check_key`, `matches`; Task 5 `Ref`, `Configurable`; Task 6 `check_parent_id`; Task 9 `CatalogEntry`; Task 10 `ArtifactStore`; Task 12 `Job`; Task 13 `request_json`, `read_regular_file`, `PyTool`.
- Produces: R4.13.9 `fixed_config`, `tool_identity`, `plan_tool_stage`, `plan_script_stage`, `materialize`, `collect_output_files`; plus `parent_ref(candidate: Candidate) -> ParentRef`, `effective_files(goal: Goal, candidate: Candidate) -> dict[str, str]`, `validate_edits(edits: Mapping[str, str], *, editable: Sequence[str], protected: AbstractSet[str], max_edit_bytes: int) -> None` (raises `SoftError`), `edit_hashes(edits: Mapping[str, str]) -> dict[str, str]`, `put_edits(edits: Mapping[str, str], artifacts: ArtifactStore) -> None`. Callers run `validate_edits` before `plan_tool_stage`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_tooling/test_staging.py
H1, H2, H3 = "1" * 64, "2" * 64, "3" * 64

def test_tool_identity_ignores_spelling() -> None:                         # Review Focus 5
    toml_form = Ref(path="_fakes.tools:PyTool", config={"code": "x"})
    python_form = Ref.of(PyTool(PyTool.Config(code="x", grace_s=10.0)))
    assert tool_identity(toml_form, None) == tool_identity(python_form, None)
    assert tool_identity(Ref(path="_fakes.tools:PyTool", config={"code": "y"}), None) != tool_identity(toml_form, None)

def test_tool_identity_drops_settable_fields() -> None:
    entry = CatalogEntry(name="py", kind="tool", ref="_fakes.tools:PyTool", description="d",
                         settable={"timeout_s": {"type": "number"}})
    a = Ref(path="_fakes.tools:PyTool", config={"code": "x", "timeout_s": 5})
    b = Ref(path="_fakes.tools:PyTool", config={"code": "x", "timeout_s": 50})
    assert tool_identity(a, entry) == tool_identity(b, entry) and tool_identity(a, None) != tool_identity(b, None)
    assert "timeout_s" not in fixed_config(a, entry)

def stage(**kw: Any) -> tuple[Job.Stage, dict[str, str], list[str]]:
    base: dict[str, Any] = {"goal_files": {"p/run.py": H1, "p/a.txt": H2}, "tool_files": {}, "editable": ["p/*.txt"],
            "protected": {"p/run.py"}, "private_paths": set(), "parents": [], "receiving_identity": "T",
            "edits": {}, "request_json": None}
    return plan_tool_stage(**{**base, **kw})

def test_order_goal_then_same_identity_overlay_then_edits() -> None:
    parent = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay={"p/a.txt": H3})
    st, overlay, warnings = stage(parents=[parent], edits={"p/b.txt": "new"})
    hb = sha256_hex(b"new")
    assert st.root == {"p/run.py": H1, "p/a.txt": H3, "p/b.txt": hb} and overlay == {"p/a.txt": H3, "p/b.txt": hb}
    assert warnings == [] and st.parents == [parent]

def test_other_tool_identity_is_not_overlaid() -> None:
    st, overlay, warnings = stage(parents=[ParentRef(candidate_id="c1", run_id="r", tool_identity="U", overlay={"p/a.txt": H3})])
    assert st.root["p/a.txt"] == H2 and overlay == {}
    assert warnings == ["base parent c1 not overlaid: made by another tool; paths: p/a.txt"]

def test_overlay_is_all_or_nothing() -> None:
    parent = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay={"p/a.txt": H3, "p/run.py": H3})
    st, _, warnings = stage(parents=[parent])
    assert st.root == {"p/run.py": H1, "p/a.txt": H2}
    assert warnings == ["base parent c1 not overlaid: paths not editable here; paths: p/run.py"]

def test_empty_identity_and_later_parents_never_overlay() -> None:
    p0 = ParentRef(candidate_id="c0", run_id="r", tool_identity="", overlay={"p/a.txt": H3})
    p1 = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay={"p/a.txt": H3})
    assert stage(receiving_identity="", parents=[p0])[0].root["p/a.txt"] == H2
    st, _, _ = stage(parents=[ParentRef(candidate_id="c2", run_id="r", tool_identity="T"), p1])
    assert st.root["p/a.txt"] == H2 and len(st.parents) == 2

def test_private_path_in_root_is_internal_error() -> None:
    with pytest.raises(RuntimeError, match="judge/hidden/x.csv"):
        stage(goal_files={"judge/hidden/x.csv": H1}, private_paths={"judge/hidden/x.csv"})

def test_validate_edits_lists_every_reason() -> None:
    ref = "sha256:" + "a" * 64                                  # 71 bytes, under the limit, so its own reason shows
    with pytest.raises(SoftError) as exc:
        validate_edits({"p/run.py": "x", "q.txt": "y", "p/big.txt": "z" * 101, "swarm/x": "w", "p/ref.txt": ref},
                       editable=["p/*", "swarm/*"], protected={"p/run.py"}, max_edit_bytes=100)
    msg = str(exc.value)
    for part in ("p/run.py: protected", "q.txt: not editable", "p/big.txt: 101 bytes > max_edit_bytes 100",
                 "swarm/x: uses the reserved name 'swarm'", "p/ref.txt: content is a stored-edit reference"):
        assert part in msg

def test_validate_edits_rejects_alias_spellings() -> None:      # never normalized onto a protected path
    with pytest.raises(SoftError) as exc:
        validate_edits({"./p/run.py": "x", "p//run.py": "y"}, editable=["**/*"], protected={"p/run.py"}, max_edit_bytes=10)
    msg = str(exc.value)
    assert "./p/run.py: not in normal form (use 'p/run.py')" in msg and "p//run.py: not in normal form (use 'p/run.py')" in msg

def test_live_edit_helpers_refuse_stored_reference(artifacts: ArtifactStore) -> None:
    ref = "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="p/a.txt: content is a stored-edit reference"):
        edit_hashes({"p/a.txt": ref})
    with pytest.raises(ValueError, match="p/a.txt: content is a stored-edit reference"):
        put_edits({"p/a.txt": ref, "p/b.txt": "ok"}, artifacts)
    with pytest.raises(NotFound): artifacts.get_bytes(sha256_hex(b"ok"))   # every value is checked before any store
    assert edit_hashes({"p/a.txt": "sha256:hello"}) == {"p/a.txt": sha256_hex(b"sha256:hello")}

def test_plan_script_stage_and_helpers() -> None:
    out_file = FileRef(name="o.txt", path="out/o.txt", sha256=H3, size=1)
    c = cand(request=ToolRequest(inputs={"lr": 0.1}), parent_ids=["c0"], overlay={"p/a.txt": H2},
             output=ToolOutput(status=RunStatus.OK, data={"f": 1.0}, files=[out_file]))
    goal = Goal(id="g", namespace="n", files=FileSet(globs=["p/*"], manifest={"p/a.txt": H1, "p/run.py": H1}))
    assert effective_files(goal, c) == {"p/a.txt": H2, "p/run.py": H1}
    st = plan_script_stage(own_files={"judge/score.py": H1}, candidate=c, effective_files=effective_files(goal, c))
    assert st.root == {"judge/score.py": H1} and st.request_json is None and st.candidate is not None
    assert st.candidate.request == {**request_json(c.request), "parents": ["c0"]}
    assert st.candidate.output == {"schema_version": 1, "status": "ok", "partial": False, "data": {"f": 1.0},
                                   "usage": Usage().model_dump(mode="json"), "files": ["out/o.txt"]}
    assert st.candidate.files == {"p/a.txt": H2, "p/run.py": H1, "out/o.txt": H3}
    assert parent_ref(c) == ParentRef(candidate_id=c.id, run_id=c.run_id, tool_identity=c.tool_identity,
                                      data={"f": 1.0}, overlay={"p/a.txt": H2}, files=[out_file])
```
```python
# tests/_tooling/test_materialize.py
def test_materialize_tool_stage(tmp_path: Path, artifacts: ArtifactStore) -> None:
    ha, hb, hc = artifacts.put_bytes(b"A"), artifacts.put_bytes(b"B"), artifacts.put_bytes(b"C")
    out = FileRef(name="o.txt", path="out/o.txt", sha256=ha, size=1)
    clash = FileRef(name="b.txt", path="p/b.txt", sha256=hc, size=1)            # also an overlay path
    parent = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", data={"f": 1},
                       overlay={"p/a.txt": hb, "p/b.txt": hb}, files=[out, clash])
    materialize(Job.Stage(root={"p/a.txt": ha}, parents=[parent], request_json={"params": {}}), tmp_path / "ws", artifacts)
    ws = tmp_path / "ws"
    assert (ws / "p/a.txt").read_bytes() == b"A" and (ws / "parents/c1/p/a.txt").read_bytes() == b"B"
    assert (ws / "parents/c1/out/o.txt").read_bytes() == b"A"                  # the parent's output files are staged
    assert (ws / "parents/c1/p/b.txt").read_bytes() == b"C"                    # an output file wins over the overlay
    assert (ws / "parents/c1/data.json").read_bytes() == b'{"f":1}'
    for rel in ("parents/c1/p/a.txt", "parents/c1/p/b.txt", "parents/c1/out/o.txt", "parents/c1/data.json"):
        assert (ws / rel).stat().st_mode & 0o222 == 0
    assert (ws / "p/a.txt").stat().st_mode & 0o200
    assert json.loads((ws / "swarm/request.json").read_text()) == {"params": {}}

def test_materialize_script_stage(tmp_path: Path, artifacts: ArtifactStore) -> None:
    # stage with candidate=CandidateFiles(request={"params": {}}, output={"status": "ok"}, files={}):
    # candidate/request.json and candidate/output.json hold those objects; candidate/files/ exists and is empty; swarm/ exists

PARENT = ParentRef(candidate_id="c1", run_id="r", tool_identity="T")

@pytest.mark.parametrize("make", [                       # model_copy skips validators, so the sink checks again
    lambda h: Job.Stage(parents=[PARENT.model_copy(update={"candidate_id": "../../outside"})]),
    lambda h: Job.Stage(parents=[PARENT.model_copy(update={"overlay": {"/abs/x": h}})]),
    lambda h: Job.Stage(parents=[PARENT.model_copy(update={"overlay": {"../x": h}})]),
    lambda h: Job.Stage().model_copy(update={"root": {"../x": h}}),
    lambda h: Job.Stage(candidate=Job.CandidateFiles(request={}, output={}, files={}).model_copy(update={"files": {"../x": h}}))])
def test_materialize_rechecks_paths_at_the_sink(tmp_path: Path, artifacts: ArtifactStore,
                                                make: Callable[[str], Job.Stage]) -> None:
    ha = artifacts.put_bytes(b"A")
    with pytest.raises(ValueError, match=r"path|parent id"): materialize(make(ha), tmp_path / "ws", artifacts)
    assert not (tmp_path / "outside").exists() and not (tmp_path / "x").exists()

@pytest.mark.parametrize(("declared", "match"), [("../x", "path"), ("swarm/result.json", "reserved"),
                                                 ("missing.txt", "missing.txt"), ("link.txt", "symlink"),
                                                 ("fifo.txt", "regular file")])
def test_collect_output_files_rejects(tmp_path: Path, artifacts: ArtifactStore, declared: str, match: str) -> None:
    (tmp_path / "link.txt").symlink_to(tmp_path / "target"); (tmp_path / "target").write_text("t")
    os.mkfifo(tmp_path / "fifo.txt")
    with pytest.raises(ValueError, match=match): collect_output_files(tmp_path, [declared], artifacts)

def test_collect_output_files_uploads(tmp_path: Path, artifacts: ArtifactStore) -> None:
    (tmp_path / "out").mkdir(); (tmp_path / "out/p.txt").write_bytes(b"p")
    [ref] = collect_output_files(tmp_path, ["out/p.txt"], artifacts)
    assert (ref.name, ref.path, ref.size) == ("p.txt", "out/p.txt", 1) and artifacts.get_bytes(ref.sha256) == b"p"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_tooling/test_staging.py tests/_tooling/test_materialize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._tooling.staging'`.

- [ ] **Step 3: Implement**

- `fixed_config(ref, entry)` = `cls.Config.model_validate(ref.config).model_dump(mode="json")` (so a spelling that omits defaults and one that includes them agree), minus the keys of `entry.settable` when `entry` is given. `tool_identity` = `sha256_hex(canonical_json({"path": ref.path, "fixed_config": fixed_config(ref, entry)}))`.
- `validate_edits`: the first failing reason per path, in this order: a path rule (the `check_key` reason, the text after `path '<path>': `, for example `uses the reserved name 'swarm'` or `not in normal form (use 'p/run.py')`; an alias spelling is refused, never normalized onto a protected path), `protected`, `not editable` (`matches(path, editable)` is false), size (`f"{n} bytes > max_edit_bytes {max_edit_bytes}"` with `n = len(content.encode())`), then `content is a stored-edit reference` (content that fully matches `^sha256:[0-9a-f]{64}$`, the form Task 6's serializer keeps). One `SoftError` listing `"<path>: <reason>"` lines.
- `edit_hashes(edits)` = `{path: sha256_hex(content.encode())}`: it hashes the content and never reads a value as a reference; a value that fully matches `^sha256:[0-9a-f]{64}$` (the reserved stored-reference form, Task 6) → `ValueError(f"{path}: content is a stored-edit reference")`. `put_edits(edits, artifacts)` checks every value that way first, then stores each content with `put_bytes`, so a refused call stores nothing.
- `plan_tool_stage`: apply the §4.12 overlay rule to `parents[0]` only: overlay it when its identity equals a non-empty `receiving_identity` and every overlay path is editable, unprotected, and not private; otherwise add the warning strings the tests pin. Then add the edit hashes.
- `plan_script_stage`: `Stage(root=own_files, candidate=CandidateFiles(request={**request_json(c.request), "parents": c.parent_ids}, output={"schema_version": 1, "status", "partial", "data", "usage" (json dump), "files": [f.path ...]}, files={**effective_files, **{f.path: f.sha256 for f in c.output.files}}))`; output files win at the same path.
- `materialize`: before it writes anything, check every destination again at the sink (`model_copy` and `model_construct` skip validators): root keys, each parent's overlay keys and `FileRef.path` values, and candidate file keys through `check_key`, and each parent id through `check_parent_id`; a failure is the `ValueError` naming the path or the parent id. Then create `workspace` and `workspace/swarm`; root files with `fetch(read_only=False)`; for each parent, its data and its artifacts, read-only, under `parents/<id>/` (spec §4.2, §4.12): one map `{**p.overlay, **{f.path: f.sha256 for f in p.files}}` (an output file wins at a path that is also an overlay path, as in `plan_script_stage`), each entry fetched once to `parents/<id>/<path>` with `fetch(read_only=True)`, then `parents/<id>/data.json` (canonical JSON of `data`, written last; a map entry at `data.json` is skipped, so `data.json` always holds `data`); for a candidate stage `candidate/request.json` and `candidate/output.json` (`json.dumps(indent=2, sort_keys=True)`), `candidate/files/<path>` read-only, `candidate/files/` always created; `swarm/request.json` when `request_json` is set.
- `collect_output_files`: per declared path `check_relpath`; `lstat` must exist, and neither it nor a parent below `workspace` is a symlink (`ValueError(f"symlink not allowed: {rel}")`); the bytes are read through `read_regular_file` (a FIFO or other non-regular file → `"<rel> is not a regular file"`), then `put_bytes` and `FileRef(name=<base name>, path=rel, sha256, size)`; any failure → `ValueError` naming the path.
- Roadmap R4.13.9: add `check_key(path: str) -> str` after `check_relpath` with the comment "[plan] normal form for dict-key paths: check_relpath, and the input must already be normalized" (the R4.2 and R4.7 comments name it), and add `parent_ref`, `effective_files`, `validate_edits`, `edit_hashes`, and `put_edits` with the signatures of this task's Interfaces (`_tooling/staging.py`), and the comment "callers run `validate_edits` with `protected = tool._protected(root) | private_paths`, then `plan_tool_stage` with the same set", and on `edit_hashes` and `put_edits` the comment "refuse the reserved `sha256:<64 hex>` reference form". Reason: M1b's `execute_tool_run` and the conformance kit (Task 23) share these helpers; recorded so no later milestone re-implements them.

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_tooling tests/_tooling docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(tooling): add tool identity, the staging order and overlay rule, and materialize" \
  -m "Goal files, then a same-identity base parent's overlay on editable paths only, then edits; other parents stay read-only under parents/<id>/ with a soft warning (spec §4.12). materialize stages every parent's data.json, overlay, and output files read-only under parents/<id>/ (spec §4.2) and checks every destination path and parent id again first, because model_copy skips validators. Alias spellings and stored-edit references are refused as edits and by edit_hashes and put_edits. Roadmap R4.13.9 records check_key and the edit, parent, and effective-file helpers."
```

---

### Task 15: Launcher, process info, and `ProcessSupervisor`

**Files:**
- Create: `src/swarm_arena/_launch.py`, `src/swarm_arena/_exec/{procinfo.py,supervisor.py}`
- Test: `tests/test_launch.py`, `tests/_exec/{test_procinfo.py,test_supervisor.py}`
- Roadmap: none. R4.13.8 already has the `before_kill` type, `-I`, the ignored `BrokenPipeError` on the "go" write, the stop sequence with `BEFORE_KILL_TIMEOUT_S` (notes written only after the kill and reap), `read_regular_file`, and `note_in_log`; the R3.1 `_launch.py` row already says `python -I -m swarm_arena._launch`.

**Interfaces:**
- Consumes: Task 6 `CompletedRun`, `RunStatus`; Task 12 `Job.Started`; Task 2 `utc_now`; Task 13 `note_in_log`.
- Produces: `_launch.main(argv: Sequence[str] | None = None) -> int` with CLI `python -I -m swarm_arena._launch --go-fd FD [--rlimit-as BYTES] [--rlimit-cpu SECONDS] [--rlimit-nofile N] -- ARGV...`; R4.13.8 `host_id`, `process_start_time`, `group_alive`, `kill_orphan`, `RLimits`, `ProcessSupervisor(*, host_id: str)`, `run(...) -> CompletedRun` as registered: `before_kill: Callable[[CompletedRun], Awaitable[None]] | None` is called with a provisional `CompletedRun(returncode=None, status=TIMEOUT or CANCELLED, ended_at=<now>)` so the executor can run `collect()` before SIGKILL. Plus `BEFORE_KILL_TIMEOUT_S: Final = 30.0` in `_exec/supervisor.py`, read at call time (tests monkeypatch it).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_launch.py  (helper spawn(args, tmp_path, env=None) -> (Popen, write_fd): os.pipe(), Popen([sys.executable,
#  "-I", "-m", "swarm_arena._launch", "--go-fd", str(r), *args], pass_fds=(r,), cwd=tmp_path, env=env), closes r in the parent)
SHADOW = "open('shadow', 'w')"          # a workspace swarm_arena/__init__.py; -I keeps the cwd and PYTHONPATH off sys.path
def test_execs_after_go(tmp_path: Path) -> None:
    p, w = spawn(["--", sys.executable, "-c", "import os; open('pid', 'w').write(str(os.getpid()))"], tmp_path)
    os.write(w, b"go"); os.close(w)
    assert p.wait(10) == 0 and (tmp_path / "pid").read_text() == str(p.pid)      # exec keeps the pid

def test_exits_without_exec_when_pipe_closes(tmp_path: Path) -> None:
    p, w = spawn(["--", sys.executable, "-c", "open('m', 'w')"], tmp_path)
    os.close(w)
    assert p.wait(10) == 0 and not (tmp_path / "m").exists()

@pytest.mark.parametrize("pythonpath", [None, "."])
def test_workspace_package_never_imported(tmp_path: Path, pythonpath: str | None) -> None:
    (tmp_path / "swarm_arena").mkdir(); (tmp_path / "swarm_arena/__init__.py").write_text(SHADOW)
    env = {**os.environ, "PYTHONPATH": pythonpath} if pythonpath else None     # "." = an allowlisted PYTHONPATH
    p, w = spawn(["--", sys.executable, "-c", "pass"], tmp_path, env=env)
    os.close(w)                                                                   # no "go"
    assert p.wait(10) == 0 and not (tmp_path / "shadow").exists()

def test_exec_keeps_python_variables(tmp_path: Path) -> None:                     # -I does not change os.environ
    p, w = spawn(["--", sys.executable, "-c", "import os; open('pp', 'w').write(os.environ['PYTHONPATH'])"], tmp_path,
                 env={**os.environ, "PYTHONPATH": "."})
    os.write(w, b"go"); os.close(w)
    assert p.wait(10) == 0 and (tmp_path / "pp").read_text() == "."

def test_command_not_found_exits_127(tmp_path: Path) -> None: ...      # argv ["no-such-binary-xyz"] after "go" → 127

def test_applies_rlimits(tmp_path: Path) -> None: ...                  # --rlimit-nofile 64; tool writes getrlimit soft → "64"
```
```python
# tests/_exec/test_procinfo.py
def test_host_id_format_and_stable() -> None:
    assert re.fullmatch(r"[^:]+:.+", host_id()) and host_id() == host_id()

def test_process_start_time_and_group_alive() -> None:
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    st = process_start_time(p.pid)
    assert st and st == process_start_time(p.pid) and group_alive(p.pid)
    os.killpg(p.pid, signal.SIGKILL); p.wait()
    assert process_start_time(p.pid) is None and wait_until(lambda: not group_alive(p.pid), 5)

def test_kill_orphan_checks_host_and_start_time() -> None:
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    good = Job.Started(kind="process", job_id="c-1", host_id=host_id(), pid=p.pid, pgid=p.pid,
                       start_time=process_start_time(p.pid))
    assert not kill_orphan(good.model_copy(update={"start_time": "Thu Jan  1 00:00:00 1970"})) and p.poll() is None
    assert not kill_orphan(good.model_copy(update={"host_id": "other:boot"})) and p.poll() is None
    assert kill_orphan(good) and p.wait(5) == -signal.SIGKILL
```
```python
# tests/_exec/test_supervisor.py  (helper sup() = ProcessSupervisor(host_id=host_id()); py(code) = [sys.executable, "-c", code];
#  call(tmp_path, argv, **kw) runs sup().run with cwd=tmp_path, env=dict(os.environ), stdout=tmp_path / "stdout.log",
#  stderr=tmp_path / "stderr.log",
#  timeout_s=None, grace_s=1.0, rlimits=None, on_started=None, before_kill=None, overridden by kw)
async def test_exit_codes(tmp_path: Path) -> None:
    ok = await call(tmp_path, py("print('hi')"))
    assert (ok.status, ok.returncode, ok.stdout_path.read_text()) == (RunStatus.OK, 0, "hi\n")
    bad = await call(tmp_path, py("import sys; sys.stderr.write('e'); raise SystemExit(3)"))
    assert (bad.status, bad.returncode, bad.stderr_path.read_text()) == (RunStatus.ERROR, 3, "e")

async def test_on_started_receives_started(tmp_path: Path) -> None:
    seen: list[Job.Started] = []
    async def record(s: Job.Started) -> None: seen.append(s)
    await call(tmp_path, py("pass"), on_started=record)
    [s] = seen
    assert (s.kind, s.pgid == s.pid, s.host_id) == ("process", True, host_id()) and s.start_time

async def test_tool_never_starts_when_on_started_fails(tmp_path: Path) -> None:
    async def fail(s: Job.Started) -> None: raise RuntimeError("commit failed")
    with pytest.raises(RuntimeError, match="commit failed"):
        await call(tmp_path, py("open('ran', 'w')"), on_started=fail)
    await anyio.sleep(1)
    assert not (tmp_path / "ran").exists()

@pytest.mark.parametrize("pythonpath", [None, "."])                                # "." = an allowlisted PYTHONPATH
async def test_workspace_package_cannot_run_before_go(tmp_path: Path, pythonpath: str | None) -> None:   # spec §4.8
    (tmp_path / "swarm_arena").mkdir(); (tmp_path / "swarm_arena/__init__.py").write_text("open('shadow', 'w')")
    seen: list[bool] = []
    async def slow_fail(s: Job.Started) -> None:
        await anyio.sleep(1); seen.append((tmp_path / "shadow").exists()); raise RuntimeError("commit failed")
    env = {**os.environ, "PYTHONPATH": pythonpath} if pythonpath else dict(os.environ)
    with pytest.raises(RuntimeError, match="commit failed"):
        await call(tmp_path, py("pass"), on_started=slow_fail, env=env)
    await anyio.sleep(0.5)
    assert seen == [False] and not (tmp_path / "shadow").exists()

async def test_launcher_gone_before_go_is_error(tmp_path: Path) -> None:     # BrokenPipeError on the "go" write
    async def kill_it(s: Job.Started) -> None: os.kill(s.pid, signal.SIGKILL); await anyio.sleep(0.2)
    done = await call(tmp_path, py("open('ran', 'w')"), on_started=kill_it)   # raises nothing
    assert (done.status, done.returncode) == (RunStatus.ERROR, -signal.SIGKILL) and not (tmp_path / "ran").exists()

async def test_partial_collect_after_timeout(tmp_path: Path) -> None:               # exit criterion
    seen: list[tuple[CompletedRun, bool]] = []
    async def before_kill(r: CompletedRun) -> None: seen.append((r, (tmp_path / "partial.txt").exists()))
    started = time.monotonic()
    done = await call(tmp_path, py("open('partial.txt', 'w').write('x'); import time; time.sleep(60)"),
                      timeout_s=1.0, grace_s=0.5, before_kill=before_kill)
    assert done.status is RunStatus.TIMEOUT and time.monotonic() - started < 10
    [(provisional, partial_seen)] = seen
    assert (provisional.status, provisional.returncode, partial_seen) == (RunStatus.TIMEOUT, None, True)

async def test_sigterm_handler_runs_within_grace(tmp_path: Path) -> None: ...
    # tool traps SIGTERM, writes term.txt, exits; timeout_s=1, grace_s=3 → term.txt exists when before_kill runs

async def test_descendant_gets_full_grace_after_leader_exits(tmp_path: Path) -> None: ...
    # the leader keeps the default SIGTERM action and starts a child python whose SIGTERM handler sleeps 1.0 s,
    # writes child_partial.txt, and exits; timeout_s=1.5, grace_s=3.0; before_kill appends
    # (tmp_path / "child_partial.txt").exists() → seen == [True] (the wait covers the group, not only the leader)

async def test_stop_returns_early_when_group_is_gone(tmp_path: Path) -> None:
    started = time.monotonic()
    done = await call(tmp_path, py("import time; time.sleep(60)"), timeout_s=1.0, grace_s=30.0)
    assert done.status is RunStatus.TIMEOUT and time.monotonic() - started < 10

async def test_grandchildren_killed_on_cancel(tmp_path: Path) -> None:              # exit criterion
    code = "import subprocess, time; c = subprocess.Popen(['sleep', '60']); open('gc.pid', 'w').write(str(c.pid)); time.sleep(60)"
    statuses: list[RunStatus] = []
    async def before_kill(r: CompletedRun) -> None: statuses.append(r.status)
    async def go() -> None:
        await call(tmp_path, py(code), before_kill=before_kill)
    with anyio.move_on_after(30) as scope:
        async with anyio.create_task_group() as tg:
            tg.start_soon(go)
            await wait_for_file(tmp_path / "gc.pid", 10)
            tg.cancel_scope.cancel()
    assert not scope.cancelled_caught and statuses == [RunStatus.CANCELLED]
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)

async def test_raising_before_kill_still_kills_group(tmp_path: Path) -> None:
    async def broken(r: CompletedRun) -> None: raise RuntimeError("collector broke")
    done = await call(tmp_path, py(STUBBORN), timeout_s=1.0, grace_s=0.5, before_kill=broken)
    assert done.status is RunStatus.TIMEOUT
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)
    assert "before_kill failed: RuntimeError: collector broke" in done.stderr_path.read_text()

async def test_blocked_before_kill_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.5)
    async def blocked(r: CompletedRun) -> None: await anyio.sleep_forever()
    started = time.monotonic()
    done = await call(tmp_path, py(STUBBORN), timeout_s=1.0, grace_s=0.5, before_kill=blocked)
    assert done.status is RunStatus.TIMEOUT and time.monotonic() - started < 10
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)
    assert "before_kill timed out" in done.stderr_path.read_text()

@pytest.mark.parametrize("mode", ["blocked", "raises"])
async def test_fifo_stderr_cannot_block_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.5)
    err = str(tmp_path / "stderr.log")
    code = f"import os; os.remove({err!r}); os.mkfifo({err!r})\n" + STUBBORN     # the test never reads the FIFO
    async def blocked(r: CompletedRun) -> None: await anyio.sleep_forever()
    async def broken(r: CompletedRun) -> None: raise RuntimeError("collector broke")
    with anyio.fail_after(10):
        done = await call(tmp_path, py(code), timeout_s=1.0, grace_s=0.5,
                          before_kill=blocked if mode == "blocked" else broken)
    assert done.status is RunStatus.TIMEOUT
    assert wait_until(lambda: not pid_alive(int((tmp_path / "gc.pid").read_text())), 5)

async def test_cancel_propagates_when_before_kill_raises(tmp_path: Path) -> None: ...
    # the task-group cancel pattern of test_grandchildren_killed_on_cancel, with py(STUBBORN) and a before_kill that
    # raises RuntimeError → not scope.cancelled_caught (no ExceptionGroup leaves the group) and the grandchild is dead

async def test_background_child_killed_after_normal_exit(tmp_path: Path) -> None: ...
    # tool starts `sleep 60` in the background, writes its pid, exits 0 → status OK and that pid is dead within 5 s

async def test_oom_status_needs_memory_limit(tmp_path: Path) -> None:
    code = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"
    assert (await call(tmp_path, py(code), rlimits=RLimits(memory_bytes=8 * 2**30))).status is RunStatus.OOM
    assert (await call(tmp_path, py(code))).status is RunStatus.ERROR

async def test_default_rlimits_cap_open_files(tmp_path: Path) -> None: ...
    # rlimits=RLimits() → the tool's soft RLIMIT_NOFILE <= 4096

async def test_no_descriptor_or_transport_leak(tmp_path: Path) -> None:      # api-anyio-packaging.md §A
    async def fail(s: Job.Started) -> None: raise RuntimeError("commit failed")
    n0 = len(os.listdir("/dev/fd"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        for _ in range(10): await call(tmp_path, py("pass"))
        for _ in range(3):
            with pytest.raises(RuntimeError, match="commit failed"): await call(tmp_path, py("pass"), on_started=fail)
        for _ in range(2):
            with pytest.raises(FileNotFoundError, match="missing"): await call(tmp_path, py("pass"), cwd=tmp_path / "missing")
        gc.collect()
    assert len(os.listdir("/dev/fd")) == n0 and not [w for w in caught if issubclass(w.category, ResourceWarning)]
```
(Test-module helpers: `supervisor` is `swarm_arena._exec.supervisor`. `STUBBORN` is tool code that starts a grandchild `python -c` that ignores SIGTERM and sleeps 60, writes the grandchild's pid to `gc.pid`, and sleeps 60, so only the SIGKILL can end the grandchild. `wait_until(pred, timeout) -> bool` and `pid_alive(pid: int) -> bool` are sync; `pid_alive` uses `os.kill(pid, 0)`. `async def wait_for_file(path: Path, timeout: float) -> None` polls `await anyio.Path(path).exists()` with `await anyio.sleep(0.05)` between tries and raises `TimeoutError` after `timeout`; a sync wait would block the event loop before the `start_soon`'d supervisor task spawns the tool.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_launch.py tests/_exec/test_procinfo.py tests/_exec/test_supervisor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._launch'`.

- [ ] **Step 3: Implement**

- `_launch.main`: parse args; apply rlimits with `resource.setrlimit` (`RLIMIT_AS` and `RLIMIT_CPU` as `(v, v)`; `RLIMIT_NOFILE` as `(min(v, hard), hard)`); a limit the OS refuses prints `swarm-arena: rlimit <name> not supported: <err>` to stderr and continues. Read the go fd until 2 bytes or EOF; anything but `b"go"` → return 0. Resolve `argv[0]` with `shutil.which(argv[0], path=os.environ.get("PATH"))` when it has no `/`; not found → message on stderr, return 127. Then `os.execve(resolved, argv, os.environ)`. `if __name__ == "__main__": raise SystemExit(main())`.
- `host_id()`: `f"{socket.gethostname()}:{boot}"`, `boot` from `/proc/sys/kernel/random/boot_id` (Linux) or the `sec` value of `sysctl -n kern.boottime` (macOS), else `"unknown"`. `process_start_time(pid)`: stripped output of `ps -o lstart= -p <pid>`, `None` when `ps` fails or prints nothing. `group_alive(pgid)`: `os.killpg(pgid, 0)`; `ProcessLookupError` → False; `PermissionError` → True. `kill_orphan(started)`: only for `kind == "process"`, `host_id` equal to `host_id()` and not ending in `:unknown`, and `process_start_time(started.pid) == started.start_time`; then `os.killpg(pgid, SIGKILL)` → True; otherwise False and nothing is killed.
- `ProcessSupervisor.run`:
  1. Open `stdout` and `stderr` files (sync helper via `to_thread`); `r, w = os.pipe()`; `anyio.open_process([sys.executable, "-I", "-m", "swarm_arena._launch", "--go-fd", str(r), *rlimit_args, "--", *argv], stdin=DEVNULL, stdout=f_out, stderr=f_err, cwd=cwd, env=env, start_new_session=True, pass_fds=(r,))`; close `r` in the parent. `-I` (isolated mode = `-E -P -s`): the launcher ignores the cwd, every `PYTHON*` variable, and the user site. So neither the workspace nor an allowlisted `PYTHONPATH` (for example `.` or `src`) can put a staged `swarm_arena/` ahead of the installed package before "go". `os.environ` is unchanged, so the exec'd command still gets `PYTHONPATH` and every other name in `env`. (Checked with Python 3.12: `-P -m` with `PYTHONPATH=.` imports a workspace `swarm_arena/__init__.py`; `-I -m` does not; a uv editable install, which is a `.pth` file in the venv site-packages, still resolves under `-I`.) `f_out`, `f_err`, `r`, and `w` are closed in a `finally` on every path, including `open_process` raising (for example a missing `cwd`) and `on_started` failing or being cancelled.
  2. `Started(kind="process", job_id="", host_id, pid, pgid=pid, start_time=process_start_time(pid))` (via `to_thread`); `await on_started(started)` when `on_started` is set (the caller sets `job_id` before journaling; `LocalExecutor` passes a wrapper that fills it). On any exception or cancellation: in a shielded scope SIGKILL the group, close `w`, `await proc.wait()`, `await proc.aclose()`, re-raise.
  3. `started_at = utc_now()`; write `b"go"`, close `w`. A `BrokenPipeError` from writing `b"go"` (the launcher already exited, for example because it could not import swarm_arena under `-I`) is ignored; the reap gives its exit code, so the status is ERROR, and its traceback is in the stderr log.
  4. `with anyio.move_on_after(timeout_s)`: `await proc.wait()`. On timeout → stop sequence with `TIMEOUT`.
  5. Stop sequence (also on outer cancellation, with `CANCELLED`), all in `CancelScope(shield=True)` inside `finally`: `os.killpg(pgid, SIGTERM)`; then wait until `grace_s` has passed or the whole group is gone (spec §4.8 "waits grace_s"): under one `move_on_after(grace_s)`, `await proc.wait()` (reap the leader), then poll `group_alive(pgid)` every 0.05 s. Then, when `before_kill` is set, `await before_kill(provisional)` under `move_on_after(BEFORE_KILL_TIMEOUT_S)` (the module constant, read at call time). No note is written before the kill: the outcome is recorded as a string, `None`, `"before_kill failed: <Type>: <msg>"` (an `Exception`), or `"before_kill timed out after <n> s"`; neither stops the kill. In a nested `finally` that always runs: `os.killpg(pgid, SIGKILL)` (`ProcessLookupError` ignored), `await proc.wait()`, `await proc.aclose()`. Only after that, still shielded, the recorded note goes to the stderr log through `note_in_log` (thread), which never blocks (a FIFO or symlink there drops the note). Outer cancellation still propagates. After a normal exit, SIGKILL the group too if `group_alive(pgid)` (no leftover background children), then `await proc.aclose()`. Every path that started a process ends with `proc.aclose()` after the reap (the anyio cleanup of `api-anyio-packaging.md` §A).
  6. Status: timeout → `TIMEOUT`; cancelled → `CANCELLED` (then the cancellation propagates); `rc == 0` → `OK`; `rc > 0` → `ERROR`; `rc in (-SIGKILL, -SIGSEGV)` not sent by us while `rlimits.memory_bytes` is set → `OOM`; other negative → `ERROR`.
- Roadmap: no edit (R4.13.8 and the R3.1 `_launch.py` row already record `-I`, the ignored `BrokenPipeError`, the `before_kill` type, and this stop sequence with notes after the kill).

- [ ] **Step 4: Run the task gate**

Expected: PASS (supervisor tests complete in under 60 s); gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_launch.py src/swarm_arena/_exec tests/test_launch.py tests/_exec
git commit -m "feat(exec): add the launcher handshake and the process-group supervisor" \
  -m "The tool execs only after on_started returns, so every started tool has a committed start; python -I keeps a workspace swarm_arena/ and an allowlisted PYTHONPATH from running first. Stop is SIGTERM, grace for the whole group, a bounded collect, then SIGKILL to the group in a nested finally that always runs; no descriptor or transport leaks (spec §4.8, R4.13.8)."
```

---

### Task 16: Tool environment and resource pool

**Files:**
- Create: `src/swarm_arena/_exec/{env.py,pool.py}`
- Test: `tests/_exec/{test_env.py,test_pool.py}`

**Interfaces:**
- Consumes: Task 6 `Command`; Task 8 `Resources`; Task 12 `Job`.
- Produces: `BASE_ENV_NAMES: Final = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "TZ")`; `read_secrets_file(path: str) -> dict[str, str]`; `tool_env_vars(*, run_id: str, goal_id: str, candidate_id: str, team: str, agent: str, round: int) -> dict[str, str]`; `build_env(job: Job, command: Command, *, workspace: Path, node_env: Mapping[str, str]) -> dict[str, str]`; `ResourcePool(*, cpus: float, max_concurrent: int)` with property `capacity -> Resources` and `slot(need: Resources) -> AbstractAsyncContextManager[None]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_read_secrets_file(tmp_path: Path) -> None:
    (tmp_path / "s.env").write_text("# comment\n\nA=1\n B = x=y \n")
    assert read_secrets_file(str(tmp_path / "s.env")) == {"A": "1", "B": "x=y"}

def test_bad_secrets_line_never_echoes_value(tmp_path: Path) -> None:
    (tmp_path / "s.env").write_text("A=1\nSECRETVALUE\n")
    with pytest.raises(ValueError, match="line 2") as exc: read_secrets_file(str(tmp_path / "s.env"))
    assert "SECRETVALUE" not in str(exc.value)
    with pytest.raises(ValueError, match="secrets file not found"): read_secrets_file(str(tmp_path / "none.env"))

def test_tool_env_vars() -> None:
    assert tool_env_vars(run_id="r", goal_id="g", candidate_id="c", team="t", agent="a", round=2) == {
        "SWARM_ARENA_RUN_ID": "r", "SWARM_ARENA_GOAL_ID": "g", "SWARM_ARENA_CANDIDATE_ID": "c",
        "SWARM_ARENA_TEAM": "t", "SWARM_ARENA_AGENT": "a", "SWARM_ARENA_ROUND": "2"}

def test_build_env_layers(tmp_path: Path) -> None:
    (tmp_path / "s.env").write_text("FROM_FILE=f\nAPI_KEY=file\n")
    node = {"PATH": "/bin", "HOME": "/h", "SECRET_X": "s", "OTHER": "o", "API_KEY": "node"}
    job = make_job(secret_env=("SECRET_X",), secrets_file=str(tmp_path / "s.env"), env={"SWARM_ARENA_RUN_ID": "r"},
                   secret_values={"API_KEY": "virtual"})
    env = build_env(job, Command(argv=["x"], env=["API_KEY", "FROM_FILE", "UNSET_NAME"]), workspace=tmp_path, node_env=node)
    assert env == {"PATH": "/bin", "HOME": "/h", "SECRET_X": "s", "FROM_FILE": "f", "API_KEY": "virtual",
                   "SWARM_ARENA_RUN_ID": "r", "SWARM_ARENA_RESULT": str(tmp_path / "swarm" / "result.json")}
    plain = build_env(make_job(), Command(argv=["x"], env=["API_KEY"]), workspace=tmp_path, node_env=node)
    assert plain["API_KEY"] == "node" and "OTHER" not in plain

async def test_pool_limits_concurrency_and_cpus() -> None:
    # ResourcePool(cpus=8, max_concurrent=2): 4 tasks holding a slot 0.2 s → peak 2
    # ResourcePool(cpus=2, max_concurrent=8): tasks needing cpus=1.5 → peak 1

async def test_pool_rejects_impossible_requests() -> None:
    pool = ResourcePool(cpus=2.0, max_concurrent=4)
    assert pool.capacity == Resources(cpus=2.0)
    with pytest.raises(ValueError, match="needs 3.0 cpus"):
        async with pool.slot(Resources(cpus=3.0)): pass
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_exec/test_env.py tests/_exec/test_pool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._exec.env'`.

- [ ] **Step 3: Implement**

- `read_secrets_file`: `#` comments and blank lines skipped; `KEY=VALUE` split at the first `=`; key and value stripped; key must match `^[A-Za-z_][A-Za-z0-9_]*$`; a bad line → `ValueError(f"{path}: line {n}: expected KEY=VALUE")` (never the line text).
- `build_env` layers, later wins: (1) `BASE_ENV_NAMES` from `node_env`; (2) every name in `command.env` and `job.secret_env`, from `node_env`, else the secrets file (read once, only when needed); names found nowhere stay unset; (3) `job.env`; (4) `SWARM_ARENA_RESULT = str(workspace / "swarm" / "result.json")`; (5) `job.secret_values`.
- `ResourcePool`: lazily created `anyio.Condition`; `slot` waits until `running < max_concurrent` and `cpus_in_use + need.cpus <= cpus`; `need.cpus > cpus` → `ValueError(f"job needs {need.cpus} cpus; this executor has {cpus}")` at once. `gpus`, `memory_gib`, and `custom` are not scheduled locally (memory becomes an rlimit in Task 17).

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_exec tests/_exec
git commit -m "feat(exec): add the tool environment allowlist, secrets file, and resource pool" \
  -m "Tools see only base names, allowlisted names from the node or the secrets file, the SWARM_ARENA_* ids (spec §7.3 item 2b), and driver-made secret values, which win (R4.7)."
```

---

### Task 17: `LocalExecutor`

**Files:**
- Create: `src/swarm_arena/_exec/local.py`
- Modify: facade `executor.py`; `tests/conftest.py` (fixture `local_executor(tmp_path: Path) -> LocalExecutor` with `workspace_root=str(tmp_path / "ws")`); `tests/_fakes/tools.py` (`RELEASE = threading.Event()` and `BlockingCollectTool(PyTool)`, whose `collect` first calls `RELEASE.wait(30)`); `tests/test_public_api.py`; `CHANGELOG.md`
- Test: `tests/_exec/test_local.py`

**Interfaces:**
- Consumes: Tasks 10, 12–16 (`open_store`, `ArtifactStore`, `Job`, `Tool`, `tool_output_from_result`, `command_output_from_result`, `note_in_log`, `read_regular_file`, `materialize`, `collect_output_files`, `ProcessSupervisor`, `RLimits`, `host_id`, `group_alive`, `kill_orphan`, `build_env`, `ResourcePool`).
- Produces: R4.7 `LocalExecutor` with `Config(max_concurrent, workspace_root, keep_workspaces, rlimits)`, `__init__(config=None, /, **fields)`, `submit`, `capacity`, `aclose`, `preflight`, `confirm_dead`, `kill_orphan`, and the `config` property typed `LocalExecutor.Config`. Logs live at `<workspace>/swarm/stdout.log` and `swarm/stderr.log`; `logs_ref = "sha256:<stdout sha256>:<stderr sha256>"`.

- [ ] **Step 1: Write the failing tests**

```python
def tool_job(tmp_path: Path, code: str, **kw: Any) -> Job: ...      # kind "tool", tool=Ref.of(PyTool(...code)),
    # request=ToolRequest(), stage=Job.Stage(), timeout_s=30, grace_s=1, artifact_root=str(tmp_path / "artifacts")

async def test_tool_never_runs_when_start_commit_fails(local_executor: LocalExecutor, tmp_path: Path) -> None:   # exit criterion
    async def fail(s: Job.Started) -> None: raise RuntimeError("commit failed")
    marker = tmp_path / "ran"
    with pytest.raises(RuntimeError, match="commit failed"):
        await local_executor.submit(tool_job(tmp_path, f"open({str(marker)!r}, 'w')", on_started=fail))
    await anyio.sleep(1)
    assert not marker.exists() and list((tmp_path / "ws").iterdir()) == []

async def test_runs_a_tool_and_uploads_logs(local_executor: LocalExecutor, tmp_path: Path) -> None:
    out = await local_executor.submit(tool_job(tmp_path, "print('hello'); " + WRITE.format("{'data': {'f': 1.5}}")))
    assert (out.status, out.data) == (RunStatus.OK, {"f": 1.5})
    store = ArtifactStore(open_store(str(tmp_path / "artifacts")))
    _, out_h, err_h = out.logs_ref.split(":")
    assert out.logs_ref.startswith("sha256:") and store.get_bytes(out_h) == b"hello\n" and store.get_bytes(err_h) == b""

async def test_python_argv0_replaced(local_executor: LocalExecutor, tmp_path: Path) -> None:
    code = WRITE.format("{'exe': __import__('sys').executable}")
    out = await local_executor.submit(command_job(tmp_path, Command(argv=["python", "-c", code])))
    assert out.data["exe"] == sys.executable
    link = tmp_path / "py"; link.symlink_to(sys.executable)
    out2 = await local_executor.submit(command_job(tmp_path, Command(argv=["python", "-c", code], python=str(link))))
    assert out2.data["exe"] == str(link)

async def test_job_timeout_wins_and_partial_kept(local_executor: LocalExecutor, tmp_path: Path) -> None:
    code = WRITE.format("{'data': {'step': 1}}") + "; import time; time.sleep(60)"
    out = await local_executor.submit(tool_job(tmp_path, code, timeout_s=1.0, tool_timeout_in_config=100.0))
    assert (out.status, out.partial, out.data) == (RunStatus.TIMEOUT, True, {"step": 1})

async def test_cancel_delivers_partial_through_on_partial(local_executor: LocalExecutor, tmp_path: Path) -> None:
    got: list[ToolOutput] = []
    async def on_partial(o: ToolOutput) -> None: got.append(o)
    code = WRITE.format("{'data': {'step': 2}}") + "; open('ready', 'w'); import time; time.sleep(60)"
    # start submit in a task group; poll for <ws>/*/ready asynchronously (await anyio.sleep(0.05) between tries,
    # as Task 15's wait_for_file; never a sync wait), then cancel the group's scope
    assert [(o.status, o.partial, o.data) for o in got] == [(RunStatus.CANCELLED, True, {"step": 2})]

async def test_output_files_uploaded_or_error(local_executor: LocalExecutor, tmp_path: Path) -> None: ...
    # declared out/p.txt → FileRef whose sha256 is in the artifact store; declared "out/missing.txt" → status ERROR

async def test_env_allowlist(local_executor: LocalExecutor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISIBLE", "v"); monkeypatch.setenv("HIDDEN", "h")
    code = WRITE.format("{k: os.environ.get(k) for k in ('VISIBLE', 'HIDDEN', 'SWARM_ARENA_RUN_ID')}")
    out = await local_executor.submit(command_job(tmp_path, Command(argv=["python", "-c", code], env=["VISIBLE"]),
                                                  env={"SWARM_ARENA_RUN_ID": "local"}))
    assert out.data == {"VISIBLE": "v", "HIDDEN": None, "SWARM_ARENA_RUN_ID": "local"}

async def test_workspaces_removed_unless_kept(tmp_path: Path) -> None: ...
    # default → <ws> empty after submit; LocalExecutor(workspace_root=..., keep_workspaces=True) → one dir named "<job id>-…"

async def test_collect_exception_becomes_error(local_executor: LocalExecutor, tmp_path: Path) -> None: ...
    # BadCollectTool → status ERROR; the uploaded stderr log contains "collect broke"

@pytest.mark.parametrize("fifo_stderr", [False, True])
async def test_blocked_collect_cannot_stop_kill(local_executor: LocalExecutor, tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch, fifo_stderr: bool) -> None:
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.5)
    fifo = "import os; os.remove('swarm/stderr.log'); os.mkfifo('swarm/stderr.log'); " if fifo_stderr else ""
    code = fifo + f"import os, time; open({str(tmp_path / 'pid')!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    try:
        with anyio.fail_after(10):
            out = await local_executor.submit(blocking_job(tmp_path, code, timeout_s=1.0, grace_s=0.2))
        assert (out.status, out.partial) == (RunStatus.TIMEOUT, False)
        assert wait_until(lambda: not pid_alive(int((tmp_path / "pid").read_text())), 5)
    finally:
        RELEASE.set()

async def test_fifo_log_does_not_hang(local_executor: LocalExecutor, tmp_path: Path) -> None:
    code = "import os; os.remove('swarm/stdout.log'); os.mkfifo('swarm/stdout.log'); " + WRITE.format("{'data': {}}")
    with anyio.fail_after(10):
        out = await local_executor.submit(tool_job(tmp_path, code))
    assert out.logs_ref is not None and out.logs_ref.split(":")[1] == sha256_hex(b"")   # not a regular file → empty

async def test_fifo_stderr_with_invalid_result_does_not_hang(local_executor: LocalExecutor, tmp_path: Path) -> None:
    code = "import os; os.remove('swarm/stderr.log'); os.mkfifo('swarm/stderr.log')"      # exit 0, no result.json
    with anyio.fail_after(10):                       # the "missing result" note goes through note_in_log: dropped
        out = await local_executor.submit(tool_job(tmp_path, code))
    assert out.status is RunStatus.ERROR and out.logs_ref is not None and out.logs_ref.split(":")[2] == sha256_hex(b"")

async def test_command_job_without_on_started_runs(local_executor: LocalExecutor, tmp_path: Path) -> None:
    job = make_job(command=Command(argv=["python", "-c", WRITE.format("{'passed': True}")]),
                   artifact_root=str(tmp_path / "artifacts"))
    assert job.on_started is None and (await local_executor.submit(job)).status is RunStatus.OK

async def test_resources_and_concurrency(tmp_path: Path) -> None: ...
    # Resources(cpus=10_000) → ValueError match "cpus" before any process starts;
    # max_concurrent=1 with three 0.3 s jobs → their [start, end] intervals do not overlap

async def test_capacity_preflight_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ex = LocalExecutor(workspace_root=str(tmp_path / "never"))
    assert not (tmp_path / "never").exists()                                    # __init__ does no I/O
    assert await ex.capacity() == Resources(cpus=float(os.cpu_count() or 1))
    monkeypatch.setattr(os, "name", "nt")
    with pytest.raises(RuntimeError, match="POSIX"): ex.preflight()

async def test_confirm_dead_and_kill_orphan(local_executor: LocalExecutor) -> None: ...
    # live `sleep 30` group on this host: kill_orphan → True, then confirm_dead → True; host_id "other:x" → confirm_dead None
```
(`command_job(tmp_path, command, **kw)` builds a kind `"command"` job; `blocking_job` is `tool_job` with `BlockingCollectTool`; `tool_timeout_in_config` sets `PyTool.Config.timeout_s`; `supervisor`, `wait_until`, and `pid_alive` as in Task 15's test module.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_exec/test_local.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._exec.local'`.

- [ ] **Step 3: Implement `LocalExecutor.submit(job)`**

1. `async with pool.slot(job.resources)` (pool and `host_id()` made lazily on first submit; `max_concurrent` default `os.cpu_count()`; `workspace_root` default `Path(tempfile.gettempdir()) / "swarm-arena"`).
2. Workspace = `mkdtemp(prefix=f"{job.id}-", dir=workspace_root)`; `artifacts = ArtifactStore(open_store(job.artifact_root))`; `materialize(job.stage, workspace, artifacts)` (thread).
3. Command: `kind == "tool"` → `tool = job.tool.build(Tool)`, `cmd = tool.command(job.request, workspace)`; `kind == "command"` → `job.command`. Then `cmd = cmd.model_copy(update={"timeout_s": job.timeout_s, "grace_s": job.grace_s})`.
4. `argv[0] == "python"` → `cmd.python or sys.executable`. `cwd = workspace / cmd.cwd` when set. `env = build_env(job, cmd, workspace=workspace, node_env=os.environ)`. `rlimits = RLimits(memory_bytes=int(memory_gib * 2**30) if memory_gib else None)` when `config.rlimits`, else `None`.
5. `collect(run)` (sync): tool job → `tool.collect(job.request, workspace, run)`; command job → `command_output_from_result(workspace, run)`; an exception → `ToolOutput(status=ERROR)` and `note_in_log(stderr, traceback)`. `before_kill(provisional)` stores `await anyio.to_thread.run_sync(collect, provisional, abandon_on_cancel=True)`, so the supervisor's `BEFORE_KILL_TIMEOUT_S` deadline can end a blocked `collect`.
6. `run = await supervisor.run(argv, cwd=..., env=..., stdout=workspace/"swarm/stdout.log", stderr=workspace/"swarm/stderr.log", timeout_s=job.timeout_s, grace_s=job.grace_s, rlimits=..., on_started=<wrapper that sets job_id=job.id and awaits job.on_started, or None when job.on_started is None>, before_kill=before_kill)`.
7. `output` = the stored partial output when the run was stopped; when the run was stopped and no partial was stored (`collect` raised or timed out), `ToolOutput(status=run.status)` with a log note, and `collect` is not tried again; else `collect(run)` (thread). Then upload declared files with `collect_output_files(workspace, [f.path for f in output.files], artifacts)` (a `ValueError` → status `ERROR`, `files=[]`, log note), upload both logs read through `read_regular_file` (a log that is not a regular file → empty bytes uploaded for it, plus a log note), set `logs_ref`, `started_at`, `ended_at`.
8. On cancellation (caught as `anyio.get_cancelled_exc_class()`), inside `CancelScope(shield=True)`: finish step 7 with the stored partial output (or `ToolOutput(status=CANCELLED)`), `await job.on_partial(output)` when set, clean up, re-raise. In every case remove the workspace unless `keep_workspaces`.

`preflight()`: `os.name != "posix"` → `RuntimeError("LocalExecutor needs a POSIX system")`. `capacity()` → `Resources(cpus=float(os.cpu_count() or 1))`. `aclose()` does nothing. `confirm_dead(started)`: same host → `not group_alive(started.pgid)`, else `None`. `kill_orphan(started)` → `procinfo.kill_orphan(started)`. Every log note in steps 5–8 goes through `note_in_log`, so a tool that replaces `swarm/stderr.log` with a FIFO or symlink can not stall `collect` or the executor. The `LocalExecutor` docstring Notes say that swarm-arena must be installed in the interpreter's own environment (a venv or site-packages), because the launcher (`python -I`) reads neither `PYTHONPATH` nor the user site. Public surface: `.executor`: `LocalExecutor`. CHANGELOG: "`LocalExecutor`: process-group supervision, launcher handshake, env allowlist, rlimits, and logs by hash."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md
git commit -m "feat(exec): add LocalExecutor" \
  -m "Stages the workspace, runs the tool under the supervisor with the env allowlist and rlimits, keeps partial output on timeout or cancel, and uploads outputs and logs by sha256 (spec §4.8, §4.12). A blocked collect can not delay the kill past BEFORE_KILL_TIMEOUT_S, and a FIFO log can not hang the upload."
```

---

### Task 18: `FunctionTool` and `python -m swarm_arena.run_function`

**Files:**
- Create: `src/swarm_arena/_tooling/function_tool.py`, `src/swarm_arena/run_function.py`
- Modify: facade `__init__.py`; `tests/_fakes/tools.py`; `tests/test_public_api.py`; `CHANGELOG.md`; roadmap R4.2, R4.7
- Test: `tests/_tooling/test_function_tool.py`, `tests/test_run_function.py`

**Interfaces:**
- Consumes: Task 5 `import_path`, `load_object`; Task 13 `Tool`, `tool_output_from_result`; Task 17 `LocalExecutor`.
- Produces: R4.2 `FunctionTool` (`name = "function"`, `description = "Runs a Python function in a subprocess."`, `Config`, `__init__`, `config` property); `function_input_model(fn: Callable[..., Any]) -> type[BaseModel]`; `import_root(module_name: str) -> str`; `run_function.main(argv: Sequence[str] | None = None) -> int`. `command()` returns `Command(argv=["python", "-P", "-m", "swarm_arena.run_function", "--import-root", <root>, <function>], timeout_s=config.timeout_s, grace_s=config.grace_s)` (replaces the R4.2 PYTHONPATH note). `-P`, not `-I`: user code keeps `PYTHONPATH` and the user site, and `--import-root` already supplies the module path. `-P` suppresses only the implicit cwd/script-directory `sys.path` entry; explicit `PYTHONPATH` entries still apply (the launcher bootstrap itself uses `-I`). `tests/_fakes/tools.py` gains `gradient_descent(lr: float, steps: int, x0: float = -1.5, y0: float = 2.0) -> dict[str, float]` (plain gradient descent on f(x, y) = (1 − x)² + 100(y − x²)², returns `{"x", "y", "f"}`), `add(a: int, b: int = 1) -> dict[str, int]` (`{"sum": a + b}`, docstring "Add two numbers."), `not_a_dict() -> list[int]`, `boom() -> dict[str, int]` (raises `RuntimeError("boom")`), `nan_result() -> dict[str, float]`, `async def coro() -> dict[str, int]`, `star(*xs: int) -> dict[str, int]`, and `def unannotated(x) -> dict[str, int]:  # noqa: ANN001` (deliberately unannotated; the only `noqa` in the repo).

- [ ] **Step 1: Write the failing tests**

```python
def test_init_forms_and_flags() -> None:
    assert FunctionTool("_fakes.tools:add").config == FunctionTool(add).config == FunctionTool(
        FunctionTool.Config(function="_fakes.tools:add")).config
    with pytest.raises(TypeError, match="lambda"): FunctionTool(lambda: {})
    t = FunctionTool("_fakes.tools:add", deterministic=False, providers=["openai"])
    assert not t.is_deterministic(ToolRequest()) and t.providers() == frozenset({"openai"})

def test_init_imports_nothing() -> None:
    t = FunctionTool("no_such_mod:f")
    with pytest.raises(ValueError, match="cannot import no_such_mod:f"): t.get_input_model()

def test_input_model_and_description() -> None:
    m = FunctionTool(add).get_input_model()
    assert m(a=2).model_dump() == {"a": 2, "b": 1} and FunctionTool(add).get_description() == "Add two numbers."
    with pytest.raises(ValidationError): m(a=2, c=1)

@pytest.mark.parametrize("fn", ["_fakes.tools:coro", "_fakes.tools:star", "_fakes.tools:unannotated"])
def test_unsupported_signatures(fn: str) -> None:
    with pytest.raises(TypeError, match="FunctionTool"): FunctionTool(fn).preflight()

def test_command_argv() -> None:
    cmd = FunctionTool(add).command(ToolRequest(), Path("/ws"))
    tests_dir = str(Path(_fakes.tools.__file__).resolve().parents[1])
    assert cmd == Command(argv=["python", "-P", "-m", "swarm_arena.run_function", "--import-root", tests_dir,
                                "_fakes.tools:add"])
    assert import_root("swarm_arena._model.canonical") == str(Path(swarm_arena.__file__).resolve().parents[1])

async def test_runs_on_local_executor(local_executor: LocalExecutor, tmp_path: Path) -> None:
    req = ToolRequest(inputs={"lr": 0.001, "steps": 2000})
    out = await local_executor.submit(function_job(tmp_path, FunctionTool(gradient_descent), req))
    assert out.status is RunStatus.OK and set(out.data) == {"x", "y", "f"} and out.data["f"] < 12.5   # f(-1.5, 2.0) = 12.5
```
```python
# tests/test_run_function.py  (helper run(tmp_path, fn, params) writes swarm/request.json via request_json and runs
#  [sys.executable, "-P", "-m", "swarm_arena.run_function", "--import-root", <tests dir>, fn] with cwd=tmp_path)
def test_writes_result(tmp_path: Path) -> None:
    done = run(tmp_path, "_fakes.tools:add", {"a": 2, "b": 3})
    assert done.returncode == 0
    assert json.loads((tmp_path / "swarm/result.json").read_text()) == {"schema_version": 1, "data": {"sum": 5}}

def test_cwd_files_do_not_shadow_imports(tmp_path: Path) -> None:               # -P keeps the cwd off sys.path
    (tmp_path / "swarm_arena").mkdir(); (tmp_path / "swarm_arena/__init__.py").write_text("open('shadow', 'w')")
    (tmp_path / "json.py").write_text("raise RuntimeError('shadowed')")
    done = run(tmp_path, "_fakes.tools:add", {"a": 2, "b": 3})
    assert done.returncode == 0 and not (tmp_path / "shadow").exists()
    assert json.loads((tmp_path / "swarm/result.json").read_text())["data"] == {"sum": 5}

@pytest.mark.parametrize(("fn", "params", "stderr"), [
    ("_fakes.tools:add", {"a": "x"}, "validation error for add_inputs"), ("_fakes.tools:not_a_dict", {}, "JSON object"),
    ("_fakes.tools:boom", {}, "boom"), ("_fakes.tools:nan_result", {}, "finite")])
def test_failures_exit_1(tmp_path: Path, fn: str, params: dict[str, Any], stderr: str) -> None:
    done = run(tmp_path, fn, params)
    assert done.returncode == 1 and stderr in done.stderr and not (tmp_path / "swarm/result.json").exists()
```
(`function_job` builds a tool `Job` with `stage=Job.Stage(request_json=request_json(req))`.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_tooling/test_function_tool.py tests/test_run_function.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._tooling.function_tool'`.

- [ ] **Step 3: Implement**

- `__init__(function, /, **fields)`: a callable → `import_path(function)`; a `Config` with fields → `TypeError`. Stores the path only. `get_input_model()` loads the function (`load_object`, cached) and returns `function_input_model(fn)`; `preflight()` calls it.
- `function_input_model`: `inspect.signature`; every parameter must be positional-or-keyword or keyword-only and annotated (`typing.get_type_hints`); no `*args`/`**kwargs`; coroutine functions rejected; each violation → `TypeError(f"FunctionTool {path}: ...")`. `pydantic.create_model(f"{fn.__name__}_inputs", __config__=ConfigDict(extra="forbid"), **fields)`. `get_description()` = first line of the function's docstring, else the class `description`.
- `import_root(module_name)`: the module's resolved file path, up `len(module_name.split("."))` levels (one more for `__init__.py`).
- `collect()` = `tool_output_from_result(workspace, run)`.
- `run_function.main`: parse `[--import-root DIR] FUNCTION`; insert `DIR` at `sys.path[0]`; load the function; read `swarm/request.json`; validate `params` with `function_input_model`; call; the result must be a `dict` (`"must return a JSON object"`) that `canonical_json` accepts (`"values must be finite JSON"`); write `{"schema_version": 1, "data": result}` to `swarm/result.json` through a temp file and `os.replace`; return 0. Any failure → message or traceback on stderr, return 1.
- Roadmap R4.2 FunctionTool comment: replace the `command()` text from `[sys.executable, "-m", ...]` through "so the subprocess imports what the driver imported" (the PYTHONPATH-in-Job.env note) by "`python -P -m swarm_arena.run_function --import-root <dir> <function>` (`-P` suppresses only the implicit cwd/script-directory `sys.path` entry; explicit `PYTHONPATH` entries still apply; the launcher bootstrap itself uses `-I`) [plan], with `<dir>` the `sys.path` entry the module was loaded from, computed by `command()` next to the tool, so the subprocess imports what the driver imported; no `Job.env` entry is needed"; in R4.7, change the `Job.env` comment to "non-secret values (SWARM_ARENA_*, gateway URLs)". Reason: `command()` runs next to the tool and cannot write `Job.env`.
- Public surface: `swarm_arena`: `FunctionTool`. CHANGELOG: "`FunctionTool`: a module-level Python function run in a subprocess."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(tooling): add FunctionTool and the run_function entry point" \
  -m "A module-level function runs as python -P -m swarm_arena.run_function, so cancel and kill behave like any tool (spec §4.1). -P suppresses only the implicit cwd/script-directory sys.path entry; explicit PYTHONPATH entries still apply (the launcher bootstrap itself uses -I). Roadmap R4.2 and R4.7: the import root travels in argv, not Job.env."
```

---

### Task 19: `ScriptTool`

**Files:**
- Create: `src/swarm_arena/_tooling/script_tool.py`
- Modify: facade `__init__.py`; `tests/test_public_api.py`; `CHANGELOG.md`; roadmap R4.4
- Test: `tests/_tooling/test_script_tool.py`

**Interfaces:**
- Consumes: Task 3 `compile_object_schema`; Task 4 `check_relpath`; Task 13 `Tool`, `Action`, `tool_output_from_result`; Task 14 `effective_files`, `validate_edits`, `plan_tool_stage`; Task 9 `PlannerCatalog`, `CatalogEntry`; Task 2 `SoftError`.
- Produces: R4.2 `ScriptTool` (`name = "script"`, `description = "Runs your script on your files."`, `Config`, `__init__(config=None, /, **fields)`, `config` property), `_human_only = frozenset({"command", "python", "files", "env", "editable"})`, `_own_files()` = `config.files.manifest`, `_protected(root)` = `{n for a in (*config.command, config.python) if a and (n := _as_relpath(a)) in root} | set(config.files.manifest)`, where `_as_relpath(a)` returns `check_relpath(a)`, or `None` when that raises (so `./problem/run.py` and `problem//run.py` protect `problem/run.py`); actions `list_files` (`ListFilesArgs(candidate: str | None = None)` → `{"files": [sorted paths]}`) and `read_file` (`ReadFileArgs(path: str, candidate: str | None = None)` → `{"path", "content"}`), both `counts_as="board_read"`; private `FileViewContext(Protocol)` with property `goal -> Goal`, `get_candidate(id: str) -> Candidate`, `_read_artifact(sha256: str) -> bytes`.

- [ ] **Step 1: Write the failing tests**

```python
def test_config_forms() -> None:
    a = ScriptTool(command=["python", "run.py"])
    assert a.config == ScriptTool(ScriptTool.Config(command=["python", "run.py"])).config
    with pytest.raises(TypeError, match="config or fields"): ScriptTool(ScriptTool.Config(command=["x"]), timeout_s=1)
    with pytest.raises(ValidationError, match="timeout"): ScriptTool(command=["x"], timeout=10)

def test_methods_follow_config() -> None:
    t = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*.phyk"], deterministic=False,
                   providers=["anthropic"], params={"lr": {"type": "number", "minimum": 1e-5, "maximum": 1.0},
                                                    "epochs": {"type": "integer", "maximum": 400}}, description="Tune it.")
    assert tuple(t.editable()) == ("problem/*.phyk",) and not t.is_deterministic(ToolRequest())
    assert t.providers() == frozenset({"anthropic"}) and t.get_description() == "Tune it."
    with pytest.raises(ValidationError): t.get_input_model()(lr=0.1, epochs=401)
    assert ScriptTool(command=["x"]).get_description() == "Runs your script on your files."
    assert t.command(ToolRequest(), Path("/ws")) == Command(argv=["python", "problem/run.py"], grace_s=10.0)

@pytest.mark.parametrize("spelling", ["problem/run.py", "./problem/run.py", "problem//run.py"])
def test_editable_glob_never_unprotects_script(spelling: str) -> None:                  # Review Focus 1
    t = ScriptTool(command=["python", spelling], editable=["problem/*"],
                   files=FileSet(globs=["tool/*"], manifest={"tool/helper.py": "1" * 64}))
    root = {"problem/run.py", "problem/a.txt", "tool/helper.py"}
    protected = t._protected(root)
    assert protected == frozenset({"problem/run.py", "tool/helper.py"})
    with pytest.raises(SoftError, match="problem/run.py: protected"):
        validate_edits({"problem/run.py": "evil"}, editable=t.editable(), protected=protected, max_edit_bytes=262_144)
    parent = ParentRef(candidate_id="c1", run_id="r", tool_identity="T", overlay={"problem/run.py": "2" * 64})
    stage, overlay, warnings = plan_tool_stage(goal_files={"problem/run.py": "3" * 64, "problem/a.txt": "4" * 64},
        tool_files=t._own_files(), editable=t.editable(), protected=protected, private_paths=set(), parents=[parent],
        receiving_identity="T", edits={}, request_json=None)
    assert stage.root["problem/run.py"] == "3" * 64 and overlay == {} and "problem/run.py" in warnings[0]

def test_collect_uses_result_rules(tmp_path: Path) -> None: ...
    # valid swarm/result.json in tmp_path → ScriptTool(...).collect(ToolRequest(), tmp_path, ok_run) has status OK and that data

def test_extra_actions() -> None:
    names = [(a.name, a.counts_as, a.input_model) for a in ScriptTool(command=["x"]).extra_actions()]
    assert names == [("list_files", "board_read", ListFilesArgs), ("read_file", "board_read", ReadFileArgs)]

async def test_list_and_read_files() -> None:
    view = FakeView(goal_manifest={"problem/a.txt": HA, "problem/run.py": HR},
                    candidates={"c1": cand(overlay={"problem/a.txt": HA2, "problem/new.txt": HN})},
                    blobs={HA: b"a1", HA2: b"a2", HR: b"run", HN: b"\xff\xfe"})
    list_files, read_file = ScriptTool(command=["x"]).extra_actions()
    assert await list_files.handler(view, ListFilesArgs()) == {"files": ["problem/a.txt", "problem/run.py"]}
    assert await list_files.handler(view, ListFilesArgs(candidate="c1")) == {"files": ["problem/a.txt", "problem/new.txt", "problem/run.py"]}
    assert await read_file.handler(view, ReadFileArgs(path="problem/a.txt", candidate="c1")) == {"path": "problem/a.txt", "content": "a2"}
    with pytest.raises(SoftError, match="no such file: nope"): await read_file.handler(view, ReadFileArgs(path="nope"))
    with pytest.raises(SoftError, match="not UTF-8 text"): await read_file.handler(view, ReadFileArgs(path="problem/new.txt", candidate="c1"))

def test_catalog_refuses_settable_script_fields() -> None:
    entry = {"name": "s", "kind": "tool", "ref": "swarm_arena:ScriptTool", "description": "d"}
    for field in ("command", "python", "files", "env", "editable"):
        with pytest.raises(ValidationError, match=f"human-only field '{field}'"):
            PlannerCatalog.model_validate({"entries": [{**entry, "settable": {field: {"type": "string"}}}]})
    PlannerCatalog.model_validate({"entries": [{**entry, "settable": {"timeout_s": {"type": "number"}}}]})

def test_init_does_no_io() -> None:
    ScriptTool(command=["python", "missing.py"], files=["nowhere/*"])          # constructs; no file access
```
(`FakeView` is a small class in the test module implementing `FileViewContext`; `HA`, `HA2`, `HR`, `HN` are distinct 64-hex strings.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_tooling/test_script_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._tooling.script_tool'`.

- [ ] **Step 3: Implement**

`get_input_model()` = `compile_object_schema(f"{name}_params", config.params)` (cached). `command()` → `Command(argv=config.command, python=config.python, env=config.env, timeout_s=config.timeout_s, grace_s=config.grace_s)`. `collect()` = `tool_output_from_result`. Handlers: files = `ctx.goal.files.manifest` with no candidate, else `effective_files(ctx.goal, ctx.get_candidate(id))`; unknown path → `SoftError(f"no such file: {path}")`; bytes via `ctx._read_artifact(sha)`; undecodable → `SoftError(f"not UTF-8 text: {path}")`. Roadmap R4.4: under `AgentContext`, add "`def _read_artifact(self, sha256: str) -> bytes` [M1a plan, private]: bytes of a file in the agent's view only; `ScriptTool`'s `list_files`/`read_file` handlers use it through `_tooling/script_tool.py:FileViewContext`." Public surface: `swarm_arena`: `ScriptTool`. CHANGELOG: "`ScriptTool`: bring your own files and script, with `list_files` and `read_file` actions."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(tooling): add ScriptTool" \
  -m "TOML-only tools on the user's files; the script, its interpreter, and its own files stay protected even under a broad editable glob (spec §4.12). Roadmap R4.4: AgentContext gains a private _read_artifact for the file actions."
```

---

### Task 20: `Check` and `Judge` ABCs, contexts, score stamping

**Files:**
- Create: `src/swarm_arena/_scoring/{__init__.py,check.py,judge.py}`
- Modify: facade `scoring.py`; `tests/_fakes/tools.py` (`AlwaysPass(Check)` with `name = "always-pass"`; `FixedJudge(Judge)` with `name = "fixed"`, `Config(value: float)`, returns `Score.maximize(value)` with `candidate_id` set for each candidate; `RecordingExecutor(Executor)`, which appends each submitted `Job` to `jobs: list[Job]` and returns `ToolOutput(status=RunStatus.OK)`); `tests/test_public_api.py`; `CHANGELOG.md`; roadmap R4.3
- Test: `tests/_scoring/{test_check.py,test_judge.py}`

**Interfaces:**
- Consumes: Task 3 `IdGen`; Task 7 `Score`; Task 12 `Job`, `Executor`; Task 14 `plan_script_stage`, `effective_files`; Task 16 `tool_env_vars`; Task 8 `Resources`.
- Produces: R4.3 `Check`, `CheckContext`, `Judge`, `JudgeContext`, and `is_finite_score(score: Score) -> bool` (registered in R4.3: True when `score.key` is not empty and every value is `math.isfinite`; M1b Task 17 uses it for scores built with `model_construct` or `model_copy`, which skip the `Score` validator). Both contexts take the registered `on_started: Callable[[Job.Started], Awaitable[None]] | None = None`, kept as the attribute `on_started`; `execute` puts it on the command `Job` as `Job.on_started`. With these additions: `CheckContext.__init__` and `JudgeContext.__init__` gain `secrets_file: str | None = None`, `ids: IdGen | None = None` (`None` → `IdGen(run_id, 0)`), and `tool_timeout_s: float = 3600.0` (the `Limits.tool_timeout_s` default; the core passes `limits.tool_timeout_s`), kept as the attributes `secrets_file`, `ids`, and `tool_timeout_s`; `JudgeContext.errors: dict[str, str]` and `JudgeContext.judge_error(candidate_id: str, reason: str) -> None` (first reason per candidate wins); `stamp_scores(scores: Sequence[Score], *, judge: Judge, judge_name: str, ids: IdGen, batch_id: str) -> list[Score]`. `CheckContext.execute` builds a command `Job` with `id=ids.next("x")`, `stage` = `plan_script_stage(own_files=files or {}, candidate=c, effective_files=effective_files(goal, c))` when a candidate is given else `Job.Stage(root=dict(files or {}))`, `timeout_s = min(command.timeout_s or tool_timeout_s, tool_timeout_s)`, `grace_s = command.grace_s`, `resources = resources or Resources()`, `env = tool_env_vars(run_id=run_id, goal_id=goal.id, candidate_id=candidate.id if candidate else "", team="", agent="", round=round)`, `secret_env = tuple(command.env)`, `secrets_file`, `artifact_root`, `on_started = self.on_started`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_scoring/test_check.py
H1 = "1" * 64

def test_check_defaults() -> None:
    assert AlwaysPass().feedback == "reasons" and AlwaysPass().preflight() is None

async def test_execute_builds_a_command_job() -> None:
    ex = RecordingExecutor()
    ctx = CheckContext(goal=GOAL, executor=ex, artifact_root="/a", run_id="r", round=2, secrets_file="/s.env", ids=IdGen("r", 1))
    c = cand(id="c9")
    await ctx.execute(Command(argv=["python", "checks/check.py"], env=["K"], grace_s=2.0), files={"checks/check.py": H1},
                      candidate=c)
    job = ex.jobs[0]
    assert (job.kind, job.id, job.timeout_s, job.grace_s, job.secret_env, job.secrets_file, job.resources) == (
        "command", f"x-{run_tag('r')}-e1-0001", 3600.0, 2.0, ("K",), "/s.env", Resources())
    assert job.stage.root == {"checks/check.py": H1} and job.stage.candidate is not None
    assert job.env["SWARM_ARENA_CANDIDATE_ID"] == "c9" and job.env["SWARM_ARENA_ROUND"] == "2"

@pytest.mark.parametrize(("configured", "expected"), [(None, 60.0), (30.0, 30.0), (99_999.0, 60.0)])
async def test_execute_timeout_follows_and_is_capped_by_limit(configured: float | None, expected: float) -> None:
    ex = RecordingExecutor()
    ctx = CheckContext(goal=GOAL, executor=ex, artifact_root="/a", tool_timeout_s=60.0)
    await ctx.execute(Command(argv=["x"], timeout_s=configured))
    assert ex.jobs[0].timeout_s == expected

async def test_execute_passes_on_started_to_job() -> None:
    async def cb(s: Job.Started) -> None: return None
    ex = RecordingExecutor()
    await CheckContext(goal=GOAL, executor=ex, artifact_root="/a", on_started=cb).execute(Command(argv=["x"]))
    await CheckContext(goal=GOAL, executor=ex, artifact_root="/a").execute(Command(argv=["x"]))
    await JudgeContext(goal=GOAL, executor=ex, artifact_root="/a", on_started=cb).execute(Command(argv=["x"]))
    assert [j.on_started for j in ex.jobs] == [cb, None, cb]

async def test_execute_on_local_executor(local_executor: LocalExecutor, artifacts: ArtifactStore, artifact_root: str) -> None: ...
    # own file checks/x.txt and a candidate with an output file; the command reads checks/x.txt and candidate/output.json,
    # lists candidate/files, and writes them into result.json → ToolOutput.data shows all three
```
```python
# tests/_scoring/test_judge.py
def test_judge_defaults() -> None:
    j = FixedJudge(FixedJudge.Config(value=1.0))
    assert j.rubric_hash() == sha256_hex(canonical_json({"path": "_fakes.tools:FixedJudge", "config": {"value": 1.0}}))
    assert (j.panel_hash(), j.anchor_count(), j.providers(), j.models()) == ("", 0, frozenset(), ())

def test_judge_context() -> None:
    ctx = JudgeContext(goal=GOAL, executor=RecordingExecutor(), artifact_root="/a", fidelity="f", batch_id="b",
                       anchors=frozenset({"c0"}), final=True, read_cache=False)
    ctx.judge_error("c1", "first"); ctx.judge_error("c1", "second")
    assert ctx.errors == {"c1": "first"} and (ctx.fidelity, ctx.final, ctx.read_cache) == ("f", True, False)

def test_stamp_scores() -> None:
    j = FixedJudge(FixedJudge.Config(value=1.0))
    raw = [Score.maximize(1.0).model_copy(update={"candidate_id": "c1"}),
           Score(key=(2.0,), candidate_id="c2", panel_hash="p", batch_id="own")]
    a, b = stamp_scores(raw, judge=j, judge_name="rel-error", ids=IdGen("r", 1), batch_id="batch-1")
    tag = run_tag("r")
    assert (a.id, a.judge, a.rubric_hash, a.panel_hash, a.batch_id, a.candidate_id) == (
        f"s-{tag}-e1-0001", "rel-error", j.rubric_hash(), "", "batch-1", "c1")
    assert (b.id, b.panel_hash, b.batch_id) == (f"s-{tag}-e1-0002", "p", "own")

def test_is_finite_score() -> None:
    assert is_finite_score(Score.maximize(1.0))
    assert not is_finite_score(Score.model_construct(key=(float("nan"),)))
    assert not is_finite_score(Score.model_construct(key=()))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_scoring -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._scoring'`.

- [ ] **Step 3: Implement**

`Check(Configurable, ABC)` and `Judge(Configurable, ABC)` per R4.3. `Judge.rubric_hash()` = `sha256_hex(canonical_json({"path": f"{cls.__module__}:{cls.__qualname__}", "config": self.config.model_dump(mode="json")}))`. `stamp_scores` sets `id = ids.next("s")`, `judge = judge_name`, `rubric_hash = judge.rubric_hash()`, `panel_hash = score.panel_hash or judge.panel_hash()`, and `batch_id` when the score's is empty; keeps `candidate_id`. `is_finite_score(score)` = `bool(score.key) and all(math.isfinite(v) for v in score.key)`. Roadmap R4.3 already has `on_started` on both contexts and `is_finite_score` with the comment "the finite-key check of M1b judge-error rule 3": no edit for those. Roadmap R4.3: add the three keyword arguments (`secrets_file`, `ids`, `tool_timeout_s`) and their attributes to both context signatures, the comment "`execute` uses `min(command.timeout_s or tool_timeout_s, tool_timeout_s)`; `ctx.ids` is the only id source for command jobs and score stamping (M1b's scoring runner calls `stamp_scores(..., ids=ctx.ids)` and turns `ctx.errors` into `judge_error` events)", `errors` and `judge_error` to `JudgeContext`, and `stamp_scores` — reason: command-job ids need the run's `IdGen` (R2.7 item 7), script checks need the secrets file, script commands must stay under the human tool-time limit (spec §6.1, §8.1), and per-candidate judge errors need a channel (spec §4.4 "recorded as a judge error for that candidate"). Public surface: `.scoring`: `Check`, `CheckContext`, `Judge`, `JudgeContext`. CHANGELOG: "`Check` and `Judge` ABCs with `CheckContext.execute` and score stamping."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat(scoring): add the Check and Judge ABCs, contexts, and score stamping" \
  -m "ctx.execute sends a command job through the executor with the candidate staged under candidate/ and the context's on_started on the job (spec §4.4, §4.12); is_finite_score checks keys that skipped the Score validator. Roadmap R4.3: contexts take the run IdGen, the secrets file, and the tool-time limit that caps script commands, and JudgeContext records per-candidate judge errors."
```

---

### Task 21: `ProgramJudge` and `Lexicographic`

**Files:**
- Create: `src/swarm_arena/_scoring/{program_judge.py,lexicographic.py}`
- Modify: facades `__init__.py`, `scoring.py`; `tests/_fakes/tools.py` (`rosen_score(goal, candidate) -> Score` = `Score.minimize(candidate.output.data["f"])`; `small_b(goal, candidate) -> Score` = `Score.minimize(abs(candidate.output.data["b"]))`; `raising_score` raises `RuntimeError("no")`; `nan_score` returns `Score.minimize(float("nan"))`; `list_score` returns `[1]`; `async def async_score`; `PanelJudge(Judge)` with `name = "panel"`, `Config(panel: str = "", per_candidate: dict[str, str] = {}, anchors: int = 0, providers: list[str] = [], models: list[str] = [])`, returning `Score.maximize(1.0)` with `candidate_id=c.id` and `panel_hash=per_candidate.get(c.id, "")`, and whose `panel_hash()`, `anchor_count()`, `providers()`, `models()` return its config); `tests/test_public_api.py`; `CHANGELOG.md`. Roadmap: none (the R4.3 `Lexicographic` comment already records the composite panel hash and the member requirements).
- Test: `tests/_scoring/{conftest.py,test_program_judge.py,test_lexicographic.py}`

**Interfaces:**
- Consumes: Task 20 `Judge`, `JudgeContext`, `stamp_scores` (tests); Task 5 `import_path`, `load_object`, `Ref`, `RefField`; Task 7 `Score`.
- Produces: R4.3 `ProgramJudge` (`name = "program"`), `Lexicographic` (`name = "lexicographic"`), each with its `config` property.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_scoring/test_program_judge.py
async def test_scores_each_candidate(judge_ctx: JudgeContext) -> None:
    c1, c2 = cand(id="c1", data={"f": 2.0}), cand(id="c2", data={"f": 0.5})
    scores = await ProgramJudge(rosen_score).judge(GOAL, [c1, c2], judge_ctx)
    assert [(s.candidate_id, s.key, s.value) for s in scores] == [("c1", (-2.0,), 2.0), ("c2", (-0.5,), 0.5)]
    assert judge_ctx.errors == {}

@pytest.mark.parametrize(("fn", "reason"), [("_fakes.tools:raising_score", "RuntimeError: no"),
    ("_fakes.tools:nan_score", "finite"), ("_fakes.tools:list_score", "returned list, expected Score")])
async def test_failures_are_judge_errors(judge_ctx: JudgeContext, fn: str, reason: str) -> None:
    assert await ProgramJudge(fn).judge(GOAL, [cand(id="c1", data={"f": 1.0})], judge_ctx) == []
    assert reason in judge_ctx.errors["c1"]

def test_forms_and_preflight() -> None:
    assert ProgramJudge(rosen_score).config == ProgramJudge("_fakes.tools:rosen_score").config
    with pytest.raises(TypeError, match="lambda"): ProgramJudge(lambda g, c: Score.minimize(0))
    with pytest.raises(TypeError, match="sync"): ProgramJudge("_fakes.tools:async_score").preflight()
```
```python
# tests/_scoring/test_lexicographic.py
async def test_key_is_member_keys_in_order(judge_ctx: JudgeContext) -> None:
    lex = Lexicographic([FixedJudge(FixedJudge.Config(value=1.0)), FixedJudge(FixedJudge.Config(value=2.0))])
    [s] = await lex.judge(GOAL, [cand(id="c1")], judge_ctx)
    assert (s.key, s.value, s.candidate_id) == ((1.0, 2.0), 1.0, "c1")

async def test_second_member_breaks_ties(judge_ctx: JudgeContext) -> None:
    lex = Lexicographic([ProgramJudge(rosen_score), ProgramJudge(small_b)])
    a, b = cand(id="a", data={"f": 1.0, "b": 3.0}), cand(id="b", data={"f": 1.0, "b": 0.5})
    scores = {s.candidate_id: s.key for s in await lex.judge(GOAL, [a, b], judge_ctx)}
    assert scores == {"a": (-1.0, -3.0), "b": (-1.0, -0.5)} and max(scores, key=scores.__getitem__) == "b"

async def test_member_error_leaves_candidate_unjudged(judge_ctx: JudgeContext) -> None:
    lex = Lexicographic([ProgramJudge(rosen_score), ProgramJudge("_fakes.tools:raising_score")])
    assert await lex.judge(GOAL, [cand(id="c1", data={"f": 1.0})], judge_ctx) == []
    assert judge_ctx.errors["c1"].startswith("RuntimeError")          # the member's own reason is kept

def test_hashes_and_forms() -> None:
    one = Lexicographic([FixedJudge(FixedJudge.Config(value=1.0))])
    assert one.rubric_hash() != Lexicographic([FixedJudge(FixedJudge.Config(value=2.0))]).rubric_hash()
    assert one.panel_hash() == ""
    ref = Ref.of(FixedJudge(FixedJudge.Config(value=1.0)))
    assert Lexicographic([ref]).config == one.config == Lexicographic(Lexicographic.Config(judges=[ref])).config

def test_init_imports_nothing() -> None:
    lex = Lexicographic([Ref(path="no_such_mod:J")])
    with pytest.raises(ValueError, match="cannot import no_such_mod:J"): lex.preflight()

async def test_composite_panel_hash_follows_member_scores(judge_ctx: JudgeContext) -> None:   # §4.9, §6.4
    lex = Lexicographic([PanelJudge(PanelJudge.Config(panel="p", per_candidate={"c2": "dropped"})),
                         FixedJudge(FixedJudge.Config(value=1.0))])
    a, b = stamp_scores(await lex.judge(GOAL, [cand(id="c1"), cand(id="c2")], judge_ctx),
                        judge=lex, judge_name="lex", ids=IdGen("r", 0), batch_id="b")
    assert a.panel_hash == lex.panel_hash() == sha256_hex(canonical_json(["p", ""]))
    assert b.panel_hash == sha256_hex(canonical_json(["dropped", ""])) != lex.panel_hash()

def test_member_requirements_propagate() -> None:
    lex = Lexicographic([PanelJudge(PanelJudge.Config(anchors=2, providers=["openai"], models=["gpt"])),
                         FixedJudge(FixedJudge.Config(value=1.0))])
    assert (lex.anchor_count(), lex.providers(), lex.models()) == (2, frozenset({"openai"}), ("gpt",))
    assert Lexicographic([FixedJudge(FixedJudge.Config(value=1.0))]).anchor_count() == 0
```
(`tests/_scoring/conftest.py` holds the fixture `judge_ctx(artifact_root: str) -> JudgeContext` = `JudgeContext(goal=GOAL, executor=RecordingExecutor(), artifact_root=artifact_root)`.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_scoring/test_program_judge.py tests/_scoring/test_lexicographic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._scoring.program_judge'`.

- [ ] **Step 3: Implement**

- `ProgramJudge.judge`: per candidate, `await anyio.to_thread.run_sync(fn, goal, c)`; a non-`Score` result → `judge_error(c.id, f"returned {type(r).__name__}, expected Score")`; any exception → `judge_error(c.id, f"{type(e).__name__}: {e}")`; else `score.model_copy(update={"candidate_id": c.id})`. `preflight()` loads the function and rejects coroutine functions (`TypeError("ProgramJudge needs a sync function")`).
- `Lexicographic`: `__init__` stores the refs (instances via `Ref.of`); members are built lazily (`Ref.build(Judge)`, cached); `preflight()` builds them and calls each member's `preflight()`. `judge()` runs each member on all candidates in order; a candidate with every member's score gets `Score(candidate_id, key=sum of member keys in order, value=first member's value, verdict=first non-None verdict, notes="; ".join(non-empty notes), llm_call_ids=concatenated)`; a candidate missing a member score gets `judge_error(c.id, f"member {name} gave no score")` unless the member already recorded a reason. `rubric_hash()` = `sha256_hex(canonical_json([[component name, member.rubric_hash()] for each member]))`; `panel_hash()` = `""` when every member's is `""`, else `sha256_hex(canonical_json([member.panel_hash() ...]))`. A composite score's `panel_hash` = `""` when every member's effective hash is `""`, else `sha256_hex(canonical_json([eff_1, ..., eff_n]))`, where `eff_i` is member score i's own `panel_hash`, else `member_i.panel_hash()`. `stamp_scores` keeps it, so a member panel that drops a model (M3, spec §6.4) gives a separate identity; with no drop it equals `panel_hash()`. `anchor_count()` = `max(m.anchor_count() for m in members)`, default 0; `providers()` = the union of the members' providers; `models()` = the members' models concatenated with duplicates removed in order. These build the members lazily and cache them, as `preflight()` does.
- Public surface: `swarm_arena`: `ProgramJudge`; `.scoring`: `Lexicographic`. CHANGELOG: "`ProgramJudge` and `Lexicographic` judges."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests CHANGELOG.md
git commit -m "feat(scoring): add ProgramJudge and Lexicographic" \
  -m "Program scores run in the driver; a raise or a non-finite key is a judge error for that candidate only; Lexicographic concatenates member keys in order, its composite panel hash follows each member score, and it passes on its members' anchors, providers, and models (spec §4.4, §4.9, §6.4; R4.3)."
```

---

### Task 22: `ScriptCheck`, `ScriptJudge`, and the "Use your own files and script" page

**Files:**
- Create: `src/swarm_arena/_scoring/{script_check.py,script_judge.py}`
- Modify: facade `__init__.py`; `docs/own-files.md`; `tests/test_public_api.py`; `CHANGELOG.md`
- Test: `tests/_scoring/{test_script_check.py,test_script_judge.py}`; `tests/_scoring/conftest.py` gains fixtures `ctx(local_executor: LocalExecutor, artifact_root: str) -> CheckContext` and `jctx(local_executor: LocalExecutor, artifact_root: str) -> JudgeContext`

**Interfaces:**
- Consumes: Task 20 `Check`, `Judge`, contexts, `execute`; Task 11 `snapshot_fileset` (tests); Task 17 `LocalExecutor`; Task 9 `PlannerCatalog` (tests).
- Produces: R4.3 `ScriptCheck` (`name = "script-check"`), `ScriptJudge` (`name = "script-judge"`), each with `__init__(config=None, /, **fields)`, `config` property, `_human_only = frozenset({"command", "python", "files", "env"})`, and `preflight()` that raises `RuntimeError(f"{name}: files are not snapshotted")` for an unsnapshotted `files`. `ScriptCheck.__init__` sets `self.feedback = config.feedback`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_scoring/test_script_check.py  (helper script_check(tmp_path, artifacts, body, **cfg) writes checks/check.py with
#  `body`, snapshots ["checks/check.py"], returns ScriptCheck(command=["python", "checks/check.py"], files=<snapshotted>, **cfg))
async def test_pass_and_reasons(tmp_path: Path, artifacts: ArtifactStore, ctx: CheckContext) -> None:
    chk = script_check(tmp_path, artifacts, WRITE.format("{'passed': False, 'reasons': ['x too big']}"))
    assert await chk.check(cand(), ctx) == CheckResult(check="script-check", passed=False, reasons=["x too big"])

@pytest.mark.parametrize(("body", "cfg", "reason"), [
    ("raise SystemExit(2)", {}, "check script failed: error"),
    ("pass", {}, "check script failed: error"),                      # exit 0, no result file
    (WRITE.format("{'passed': 'yes'}"), {}, "passed"),                # strict: lax mode would read 'yes' as True
    (WRITE.format("{'passed': 1}"), {}, "passed"),
    (WRITE.format("{'passed': True, 'reasons': 'x'}"), {}, "reasons"),
    (WRITE.format("{'passed': True, 'reason': ['x']}"), {}, "reason"),  # misspelled key: extra="forbid"
    ("import time; time.sleep(60)", {"timeout_s": 1.0, "grace_s": 0.5}, "check script failed: timeout")])
async def test_crash_fails_closed(tmp_path: Path, artifacts: ArtifactStore, ctx: CheckContext,
                                  body: str, cfg: dict[str, Any], reason: str) -> None:        # Review Focus 4
    result = await script_check(tmp_path, artifacts, body, **cfg).check(cand(), ctx)
    assert result.passed is False and reason in " ".join(result.reasons)

async def test_sees_own_files_and_candidate_layout(tmp_path: Path, artifacts: ArtifactStore, ctx: CheckContext) -> None: ...
    # the script asserts checks/check.py exists, candidate/output.json has status "ok", and candidate/files holds the
    # candidate's effective files plus its output file; it writes {"passed": true} → passed True

def test_feedback_and_preflight() -> None:
    assert ScriptCheck(command=["x"], feedback="pass_fail").feedback == "pass_fail"
    with pytest.raises(RuntimeError, match="not snapshotted"): ScriptCheck(command=["x"], files=["a/*"]).preflight()

@pytest.mark.parametrize(("ref", "kind"), [("swarm_arena:ScriptCheck", "check"), ("swarm_arena:ScriptJudge", "judge")])
@pytest.mark.parametrize("field", ["command", "python", "files", "env"])
def test_catalog_refuses_settable_script_fields(ref: str, kind: str, field: str) -> None:   # covers both classes
    entry = {"name": "s", "kind": kind, "ref": ref, "description": "d"}
    with pytest.raises(ValidationError, match=f"human-only field '{field}'"):
        PlannerCatalog.model_validate({"entries": [{**entry, "settable": {field: {"type": "string"}}}]})
    PlannerCatalog.model_validate({"entries": [{**entry, "settable": {"timeout_s": {"type": "number"}}}]})   # allowed
```
```python
# tests/_scoring/test_script_judge.py  (helper sj(tmp_path, artifacts, body, **cfg) like script_check, writing judge/score.py)
@pytest.mark.parametrize(("direction", "key"), [("min", (-0.25,)), ("max", (0.25,))])
async def test_value_with_direction(tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext,
                                    direction: str, key: tuple[float, ...]) -> None:
    [s] = await sj(tmp_path, artifacts, WRITE.format("{'value': 0.25}"), direction=direction).judge(GOAL, [cand(id="c1")], jctx)
    assert (s.candidate_id, s.key, s.value) == ("c1", key, 0.25)

async def test_key_list_and_extras(tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext) -> None:
    body = WRITE.format("{'key': [1, 2], 'verdict': 'ok', 'notes': 'n'}")
    [s] = await sj(tmp_path, artifacts, body).judge(GOAL, [cand(id="c1")], jctx)
    assert (s.key, s.value, s.verdict, s.notes) == ((1.0, 2.0), None, "ok", "n")

async def test_nonfinite_value_is_judge_error(tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext) -> None:
    body = "import os; open(os.environ['SWARM_ARENA_RESULT'], 'w').write('{\"value\": NaN}')"
    assert await sj(tmp_path, artifacts, body).judge(GOAL, [cand(id="c1")], jctx) == []
    assert jctx.errors["c1"] == "judge script failed: error"          # NaN never reaches a Score (Task 13 rule)

async def test_one_workspace_per_candidate(tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext) -> None:
    body = ("import json; cid = json.load(open('candidate/request.json'))['params']['cid']; "
            + WRITE.format("{'value': 1.0, 'notes': cid}"))
    cands = [cand(id=f"c{i}", request=ToolRequest(inputs={"cid": f"c{i}"})) for i in range(3)]
    scores = await sj(tmp_path, artifacts, body).judge(GOAL, cands, jctx)
    assert [(s.candidate_id, s.notes) for s in scores] == [("c0", "c0"), ("c1", "c1"), ("c2", "c2")]

async def test_hidden_file_staged_for_the_judge(tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext) -> None: ...
    # files ["judge/score.py", "judge/hidden/points.csv"]; the script returns the number stored in points.csv as its value

async def test_failed_script_or_bad_result_is_judge_error(tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext) -> None: ...
    # exit 1 → errors["c1"] == "judge script failed: error"; {"other": 1} → errors["c1"] == "result needs key or value"

RESULTS = {"c0": {"value": 1.0}, "c1": {"value": None}, "c2": {"key": None}, "c3": {"value": "0.5"},
           "c4": {"key": [1, "a"]}, "c5": {"value": True}}

async def test_malformed_results_are_per_candidate_errors(tmp_path: Path, artifacts: ArtifactStore, jctx: JudgeContext) -> None:
    body = ("import json, os; cid = json.load(open('candidate/request.json'))['params']['cid']; "
            f"json.dump({RESULTS!r}[cid], open(os.environ['SWARM_ARENA_RESULT'], 'w'))")
    cands = [cand(id=c, request=ToolRequest(inputs={"cid": c})) for c in RESULTS]
    scores = await sj(tmp_path, artifacts, body).judge(GOAL, cands, jctx)
    assert [s.candidate_id for s in scores] == ["c0"]
    assert jctx.errors["c1"] == jctx.errors["c2"] == "result needs key or value"
    assert all("invalid judge result" in jctx.errors[c] for c in ("c3", "c4", "c5"))   # strict: no '0.5' or True

async def test_execute_errors_propagate() -> None:                    # so M1b's Fenced is never a judge error
    class Refusing(RecordingExecutor):
        async def submit(self, job: Job) -> ToolOutput: raise RuntimeError("fenced")
    def ctx_args() -> dict[str, Any]: return {"goal": GOAL, "executor": Refusing(), "artifact_root": "/a"}
    with pytest.raises(RuntimeError, match="fenced"):
        await ScriptCheck(command=["x"]).check(cand(), CheckContext(**ctx_args()))
    with pytest.raises(RuntimeError, match="fenced"):                   # bare, not an exception group
        await ScriptJudge(command=["x"]).judge(GOAL, [cand(id="c1"), cand(id="c2")], JudgeContext(**ctx_args()))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/_scoring/test_script_check.py tests/_scoring/test_script_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._scoring.script_check'`.

- [ ] **Step 3: Implement**

- Shared: `command = Command(argv=config.command, python=config.python, env=config.env, timeout_s=config.timeout_s, grace_s=config.grace_s)`; `out = await ctx.execute(command, files=config.files.manifest, candidate=c, resources=config.resources)`. Neither class catches an exception raised by `ctx.execute` (M1b's `Fenced` from `on_started` must propagate, R4.13.11); a script failure is a status, not an exception.
- `ScriptCheck.check`: `out.status != OK` → `CheckResult(check=name, passed=False, reasons=[f"check script failed: {status}"])` (the reason for a missing or invalid result file is in the uploaded stderr log); else validate `out.data` with a private `_CheckResultFile(BaseModel)`, `model_config = ConfigDict(strict=True, extra="forbid", frozen=True)`, fields `passed: bool`, `reasons: list[str] = []`, `details: dict[str, Any] = {}` (strict, because lax mode reads `'yes'` as `True`). A `ValidationError` → `CheckResult(check=name, passed=False, reasons=[f"invalid check result: {loc}: {msg}"])` from the first error (`loc` = its `loc` parts joined with `.`). A check never raises for a script failure.
- `ScriptJudge.judge`: all candidates at once in a task group (the executor's pool limits concurrency), results in candidate order. An exception from `ctx.execute` cancels the other candidates' runs and leaves `judge()` bare (the first one, taken out of the task group's exception group). Per result: status not OK → `judge_error(c.id, f"judge script failed: {status}")`; else validate `out.data` with a private `_JudgeResultFile(BaseModel)`, `ConfigDict(strict=True, extra="ignore", frozen=True)`, fields `value: float | None = None`, `key: list[float] | None = None`, `verdict: str | None = None`, `notes: str = ""`; a `ValidationError` → `judge_error(c.id, f"invalid judge result: {loc}: {msg}")` from the first error. `value` only → `Score.minimize(x)` for `direction="min"`, `Score.maximize(x)` for `"max"`; `key` set → `Score(key=tuple(key), value=value)`; `verdict` and `notes` copied; neither key nor value → `judge_error(c.id, "result needs key or value")`; `candidate_id = c.id`. Building one candidate's `Score` catches `Exception` (a non-finite value's `ValueError` too) and turns it into `judge_error(c.id, str(e))` for that candidate only; cancellation is never caught.
- `docs/own-files.md` ("Use your own files and script"): the §4.12 TOML example; the workspace tree; staging order; the overlay rule and the soft warning; protected paths, with this sentence and example: "Only the paths named in `command` and `python`, and the tool's own `files`, are protected. A helper module your script imports (for example `problem/helpers.py`) can be edited when an `editable` glob matches it. To keep it fixed, list it in the tool's `files` or narrow `editable`."; path rules (relative, no symlinks, reserved `swarm/`, `parents/`, `candidate/`); `swarm/request.json` (shape from `request_json`); `swarm/result.json` (`data` required, `usage` optional, `files` optional, `failed_provider` optional, `schema_version` optional; `{"data": {}, "failed_provider": "openai"}` gives status `provider_failed` with that provider, at any exit code except after a timeout, cancel, or OOM, spec §6.4); exit-code and timeout rules with partial output; `parents/<id>/`: `data.json`, the overlay copy, and the parent's output files (an output file wins at an overlay path); the `SWARM_ARENA_*` variables and `SWARM_ARENA_RESULT`; edit rules and `max_edit_bytes` (262,144 bytes); the check and judge workspace (`candidate/request.json`, `candidate/output.json`, `candidate/files/`) and their result formats; "not strong isolation" (spec §8.4). Each section shows a short example.
- Public surface: `swarm_arena`: `ScriptCheck`, `ScriptJudge`. CHANGELOG: "`ScriptCheck` and `ScriptJudge`, and the \"Use your own files and script\" docs page."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0 (the Sphinx build renders the new page).

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests docs/own-files.md CHANGELOG.md
git commit -m "feat(scoring): add ScriptCheck and ScriptJudge and document the workspace contract" \
  -m "Script checks fail closed and script judges run once per candidate in their own workspace with their private files (spec §4.4, §4.12). Result files are validated strictly, a malformed judge result is a judge error for that candidate only, and an exception from ctx.execute propagates (R4.13.11)."
```

---

### Task 23: Conformance kit (`swarm_arena.testing`) and connector docs

**Files:**
- Create: `src/swarm_arena/testing/{_conformance.py,_candidates.py}`
- Modify: `src/swarm_arena/testing/__init__.py`; `docs/writing-a-connector.md`; `docs/concepts.md`; `tests/test_public_api.py`; `CHANGELOG.md`
- Test: `tests/testing/{test_conformance.py,test_candidates.py}`

**Interfaces:**
- Consumes: Tasks 10, 11 (`snapshot_ref`, `snapshot_fileset`), 13 (`effective_tool_timeout`, `tool_grace`, `request_json`), 14 (`tool_identity`, `parent_ref`, `validate_edits`, `put_edits`, `plan_tool_stage`), 16 (`tool_env_vars`), 17, 20 (`stamp_scores`), 8 (`Limits`).
- Produces: R4.12 `check_tool`, `check_check`, `check_judge`, `make_candidate`; private `_artifact_root() -> str` (one `tempfile.mkdtemp(prefix="swarm-arena-testing-")` per process, cached) and `_ids() -> IdGen` (one `IdGen(run_id="local", epoch=0)` per process, cached), both in `_conformance.py`. The default goal is `Goal(id="check", namespace="testing")`; unsnapshotted file sets are snapshotted relative to `Path.cwd()` into `_artifact_root()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/testing/test_conformance.py  (fixture ws(tmp_path, monkeypatch): chdir tmp_path; writes problem/run.py (prints
#  {"a": <problem/a.txt>, "b": <problem/b.txt>} into data), problem/a.txt "1", problem/b.txt "1")
async def test_function_tool(local_executor: LocalExecutor) -> None:
    out = await check_tool(FunctionTool(add), {"a": 2}, executor=local_executor)
    assert (out.status, out.data) == (RunStatus.OK, {"sum": 3})
    assert (await check_tool(Ref(path="swarm_arena:FunctionTool", config={"function": "_fakes.tools:add"}), {"a": 1},
                             executor=local_executor)).data == {"sum": 2}
    with pytest.raises(ValidationError): await check_tool(FunctionTool(add), {"a": "x"}, executor=local_executor)

async def test_script_tool_edit_and_same_identity_overlay(ws: Path, local_executor: LocalExecutor) -> None:
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*.txt"])
    goal = Goal(id="g", namespace="ns", files=FileSet(globs=["problem/*"]))
    parent = make_candidate(ToolOutput(status=RunStatus.OK), request=ToolRequest(edits={"problem/a.txt": "2"}), tool=tool)
    out = await check_tool(tool, {}, goal=goal, parents=[parent], edits={"problem/b.txt": "3"}, executor=local_executor)
    assert out.data == {"a": "2", "b": "3"}

async def test_other_tool_parent_warns(ws: Path, local_executor: LocalExecutor) -> None:
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*.txt"])
    other = ScriptTool(command=["python", "problem/run.py", "--other"], editable=["problem/*.txt"])
    parent = make_candidate(ToolOutput(status=RunStatus.OK), request=ToolRequest(edits={"problem/a.txt": "2"}), tool=other)
    with pytest.warns(UserWarning, match="not overlaid: made by another tool"):
        out = await check_tool(tool, {}, goal=Goal(id="g", namespace="ns", files=FileSet(globs=["problem/*"])), parents=[parent],
                               executor=local_executor)
    assert out.data["a"] == "1"

async def test_protected_edit_refused(ws: Path, local_executor: LocalExecutor) -> None:
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*"])
    with pytest.raises(SoftError, match="problem/run.py: protected"):
        await check_tool(tool, {}, goal=Goal(id="g", namespace="ns", files=FileSet(globs=["problem/*"])),
                         edits={"problem/run.py": "x"}, executor=local_executor)

async def test_conformance_violation(local_executor: LocalExecutor) -> None:
    with pytest.raises(AssertionError, match="fingerprint"):
        await check_tool(EmptyFingerprintTool(PyTool.Config(code=WRITE.format("{'data': {}}"))), {}, executor=local_executor)

async def test_check_check(ws: Path, local_executor: LocalExecutor) -> None: ...
    # check_check(ScriptCheck(command=["python", "checks/ok.py"], files=["checks/ok.py"]), make_candidate(...)) → passed, check == "script-check"

async def test_check_judge_stamps_and_warns(local_executor: LocalExecutor) -> None:
    c1, c2 = make_candidate(ToolOutput(status=RunStatus.OK, data={"f": 2.0})), make_candidate(ToolOutput(status=RunStatus.OK, data={"f": 1.0}))
    scores = await check_judge(ProgramJudge(rosen_score), GOAL, [c1, c2], executor=local_executor)
    assert [s.candidate_id for s in scores] == [c1.id, c2.id] and {s.judge for s in scores} == {"program"}
    assert all(s.id.startswith(f"s-{run_tag('local')}-e0-") and s.rubric_hash for s in scores)
    with pytest.warns(UserWarning, match=f"judge error for {c1.id}"):
        assert await check_judge(ProgramJudge(nan_score), GOAL, [c1], executor=local_executor) == []

async def test_check_judge_rejects_unknown_candidate(local_executor: LocalExecutor) -> None:
    with pytest.raises(AssertionError, match="unknown candidate"):
        await check_judge(StrayJudge(), GOAL, [make_candidate(ToolOutput(status=RunStatus.OK))], executor=local_executor)
```
```python
# tests/testing/test_candidates.py
def test_defaults_and_unique_ids() -> None:
    a, b = make_candidate(ToolOutput(status=RunStatus.OK)), make_candidate(ToolOutput(status=RunStatus.OK))
    assert (a.run_id, a.goal_id, a.team, a.agent, a.round, a.origin, a.tool_identity) == ("local", "check", "team", "agent", 0, "agent", "")
    assert a.id != b.id and a.id.startswith(f"c-{run_tag('local')}-e0-")
    assert make_candidate(ToolOutput(status=RunStatus.OK), team="x").team == "x"

def test_overlay_from_edits_is_stored() -> None:
    c = make_candidate(ToolOutput(status=RunStatus.OK), request=ToolRequest(edits={"p/a.txt": "2"}))
    assert c.overlay == {"p/a.txt": sha256_hex(b"2")}
    assert ArtifactStore(open_store(_artifact_root())).get_bytes(sha256_hex(b"2")) == b"2"

def test_make_candidate_refuses_stored_reference_edit() -> None:   # no new code: edit_hashes and put_edits refuse it
    with pytest.raises(ValueError, match="stored-edit reference"):
        make_candidate(ToolOutput(status=RunStatus.OK), request=ToolRequest(edits={"p/a.txt": "sha256:" + "a" * 64}))

def test_tool_sets_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    tool = PyTool(PyTool.Config(code="pass"))
    assert make_candidate(ToolOutput(status=RunStatus.OK), tool=tool).tool_identity == tool_identity(Ref.of(tool), None)
```
(`EmptyFingerprintTool` (fingerprint returns `""`) and `StrayJudge` (returns a score for `"nope"`) are added to `tests/_fakes/tools.py`.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/testing -v`
Expected: FAIL with `ImportError: cannot import name 'check_tool' from 'swarm_arena.testing'`.

- [ ] **Step 3: Implement**

`check_tool(tool, inputs, *, goal=None, parents=(), edits=None, executor=None, timeout_s=60.0)`:
1. `ref = snapshot_ref(Ref.of(tool) if not a Ref, Path.cwd(), artifacts)`; `tool = ref.build(Tool)`; `tool.preflight()`; goal default, then `snapshot_fileset(goal.files, Path.cwd(), artifacts)`.
2. Conformance asserts (`AssertionError` with a message naming the method): `get_input_model()` is a `BaseModel` subclass; `get_description()` is a `str`; every `editable()` item is a `str`; `providers()` is a `frozenset` of `str`.
3. `request = ToolRequest(inputs=tool.get_input_model().model_validate(inputs).model_dump(mode="json"), parents=[parent_ref(p) for p in parents], edits=dict(edits or {}))`; `root = {**goal.files.manifest, **tool._own_files()}`; `protected = tool._protected(root.keys())`; `validate_edits(request.edits, editable=tool.editable(), protected=protected, max_edit_bytes=Limits().max_edit_bytes)`; `put_edits`.
4. `stage, _, warnings = plan_tool_stage(goal_files=goal.files.manifest, tool_files=tool._own_files(), editable=tool.editable(), protected=protected, private_paths=frozenset(), parents=request.parents, receiving_identity=tool_identity(ref, None), edits=request.edits, request_json=request_json(request))`; each warning → `warnings.warn(w, UserWarning, stacklevel=2)`.
5. `job_id = _ids().next("c")`; `Job(id=job_id, kind="tool", tool=ref, request=request, stage=stage, timeout_s=effective_tool_timeout(tool, round_timeout_s=None, limit_s=timeout_s), grace_s=tool_grace(tool), resources=Resources(), env=tool_env_vars(run_id="local", goal_id=goal.id, candidate_id=job_id, team="", agent="", round=0), secret_env=(), secrets_file=None, artifact_root=_artifact_root())`; `executor = executor or LocalExecutor()` (closed with `aclose()` when created here); `out = await executor.submit(job)`.
6. Asserts: `out` is a `ToolOutput`; every `FileRef` sha256 is in the artifact store; `tool.fingerprint(out)` is a non-empty `str` and equal on a second call; `is_deterministic(request)` is a `bool`; `estimate(request)` is `None` or an `Estimate`. Return `out`.

`check_check(check, candidate, *, goal=None, executor=None)`: snapshot and build like step 1 (`Check`); `CheckContext(goal=goal, executor=..., artifact_root=_artifact_root(), ids=_ids())`; assert `check.feedback in ("pass_fail", "reasons")`; return the result with `check = ref.component_name()`.
`check_judge(judge, goal, candidates, *, executor=None)`: snapshot and build (`Judge`); `JudgeContext(goal=goal, executor=..., artifact_root=_artifact_root(), ids=_ids(), batch_id="b-" + secrets.token_hex(4))`; assert every score names a candidate of the batch (`"unknown candidate"`) at most once; warn `UserWarning(f"judge error for {cid}: {reason}")` per `ctx.errors` entry; return `stamp_scores(scores, judge=judge, judge_name=ref.component_name(), ids=_ids(), batch_id=ctx.batch_id)`.
`make_candidate(output, *, request=None, tool=None, **fields)`: defaults `id=_ids().next("c")`, `run_id="local"`, `goal_id="check"`, `team="team"`, `agent="agent"`, `round=0`, `request=request or ToolRequest()`; `overlay` (when not in `fields`) = `edit_hashes(request.edits)` with the bytes stored by `put_edits` into `_artifact_root()`; `tool` → `tool_identity(snapshot_ref(Ref.of(tool) or the Ref, Path.cwd(), artifacts), None)`; `fields` override.
Docs: `docs/writing-a-connector.md`: subclassing `Tool` (nested `Config`, the `config` property override, `command`/`collect` next to the program, the optional methods and their defaults, `swarm/result.json`), a `Check` and a `Judge` (set `candidate_id`, use `ctx.judge_error`, `ctx.execute`), and running `check_tool`/`check_check`/`check_judge` in a connector's pytest CI (one complete example test). `docs/concepts.md` (draft): Goal (inputs, files, `GoalKey`), Tool (`ScriptTool`, `FunctionTool`, tool identity), Check (gated out, `feedback`), Judge (larger key is better, judge identity, finite keys, `Lexicographic`), Executor (`LocalExecutor`, process groups, launcher handshake, workspaces, env allowlist); a closing line that managers, plans, and runs arrive in M1b. Public surface: `.testing`: `check_tool`, `check_check`, `check_judge`, `make_candidate`. CHANGELOG: "Conformance kit `swarm_arena.testing` and the \"Writing a connector\" and \"Concepts\" docs pages."

- [ ] **Step 4: Run the task gate**

Expected: PASS; gate exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena tests docs/writing-a-connector.md docs/concepts.md CHANGELOG.md
git commit -m "feat(testing): add the conformance kit and connector docs" \
  -m "check_tool stages and runs a tool exactly as the core does, with the overlay rule and conformance asserts; check_check and check_judge run checks and judges offline (spec §9 rule 4)."
```

---

### Task 24: End-to-end single run and README status

**Files:**
- Create: `tests/e2e/test_single_run.py`, `tests/e2e/data/single_run/{problem/model_a.txt,problem/model_b.txt,problem/run.py,checks/check.py,judge/score.py,judge/hidden/points.csv}`
- Modify: `README.md`, `CHANGELOG.md`
- Test: `tests/e2e/test_single_run.py`

**Interfaces:**
- Consumes: public API only: `swarm_arena` (`FunctionTool`, `Goal`, `ProgramJudge`, `ScriptCheck`, `ScriptJudge`, `ScriptTool`), `swarm_arena.executor.LocalExecutor`, `swarm_arena.scoring.Lexicographic`, `swarm_arena.testing` (`check_check`, `check_judge`, `check_tool`, `make_candidate`), `swarm_arena.tool` (`FileSet`, `RunStatus`, `ToolRequest`); test helpers `_fakes.tools` (`gradient_descent`, `rosen_score`, `small_b`).
- Produces: the M1a exit test.

- [ ] **Step 1: Write the failing test**

The test below, in `tests/e2e/test_single_run.py`. The fixture files it names come in Step 3.
```python
DATA = Path(__file__).parent / "data" / "single_run"
CHECK = {"command": ["python", "checks/check.py"], "files": ["checks/check.py"]}

@pytest.fixture
def executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalExecutor:
    monkeypatch.chdir(DATA)
    return LocalExecutor(workspace_root=str(tmp_path / "ws"))

async def test_function_tool_run_checked_and_judged(executor: LocalExecutor) -> None:
    goal = Goal(id="rosenbrock", namespace="e2e")
    tool = FunctionTool(gradient_descent)
    out = await check_tool(tool, {"lr": 0.001, "steps": 2000}, goal=goal, executor=executor)
    assert out.status is RunStatus.OK and set(out.data) == {"x", "y", "f"} and out.data["f"] < 12.5
    cand = make_candidate(out, request=ToolRequest(inputs={"lr": 0.001, "steps": 2000}), tool=tool, goal_id="rosenbrock")
    result = await check_check(ScriptCheck(**CHECK), cand, goal=goal, executor=executor)
    assert (result.passed, result.check) == (True, "script-check")
    [score] = await check_judge(ProgramJudge(rosen_score), goal, [cand], executor=executor)
    assert (score.candidate_id, score.value, score.key, score.judge) == (cand.id, out.data["f"], (-out.data["f"],), "program")

async def test_script_tool_edit_overlay_check_and_lexicographic_judge(executor: LocalExecutor) -> None:
    goal = Goal(id="fit", namespace="e2e", files=FileSet(globs=["problem/*"]))
    tool = ScriptTool(command=["python", "problem/run.py"], editable=["problem/*.txt"])
    first = await check_tool(tool, {}, goal=goal, edits={"problem/model_a.txt": "2.0"}, executor=executor)
    assert (first.status, first.data["a"], first.data["b"]) == (RunStatus.OK, 2.0, 0.0)
    parent = make_candidate(first, request=ToolRequest(edits={"problem/model_a.txt": "2.0"}), tool=tool, goal_id="fit")
    out = await check_tool(tool, {}, goal=goal, parents=[parent], edits={"problem/model_b.txt": "1.0"}, executor=executor)
    assert (out.status, out.data["a"], out.data["b"]) == (RunStatus.OK, 2.0, 1.0)      # base overlay + this edit
    assert "judge/hidden/points.csv" not in out.data["files_seen"]                  # judge data never in a tool workspace
    assert "judge/score.py" not in out.data["files_seen"] and "problem/run.py" in out.data["files_seen"]
    child = make_candidate(out, tool=tool, goal_id="fit", parent_ids=[parent.id])
    baseline = make_candidate(first, tool=tool, goal_id="fit")
    for c in (child, baseline):
        assert (await check_check(ScriptCheck(**CHECK), c, goal=goal, executor=executor)).passed
    judge = Lexicographic([ScriptJudge(command=["python", "judge/score.py"],
                                       files=["judge/score.py", "judge/hidden/points.csv"], direction="min"),
                           ProgramJudge(small_b)])
    scores = {s.candidate_id: s for s in await check_judge(judge, goal, [child, baseline], executor=executor)}
    assert scores[child.id].key == (-0.0, -1.0) and scores[baseline.id].key == (-1.0, -0.0)   # mse 0 vs 1, then |b|
    assert max(scores.values(), key=lambda s: s.key).candidate_id == child.id
    assert {s.judge for s in scores.values()} == {"lexicographic"}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/e2e/test_single_run.py -v`
Expected: FAIL, both tests, with `FileNotFoundError` from `monkeypatch.chdir(DATA)` (the data folder does not exist yet).

- [ ] **Step 3: Write the fixture files**

Fixture data (stdlib only, every function annotated, each script guarded by `if __name__ == "__main__":`):
- `problem/model_a.txt` = `1.0`; `problem/model_b.txt` = `0.0`; `judge/hidden/points.csv` = `x,y\n0,1\n1,3\n2,5\n` (the line y = 2x + 1).
- `problem/run.py`: reads `a` and `b` as floats from the two model files; writes `{"data": {"a": a, "b": b, "files_seen": sorted workspace-relative paths of regular files outside swarm/ and parents/}}` to `$SWARM_ARENA_RESULT`.
- `checks/check.py`: reads `candidate/output.json`; `passed` = status `"ok"`, at least one number in `data`, and every number finite (lists and strings skipped); writes `{"passed": ..., "reasons": [] or ["status ..., non-finite ..."]}`.
- `judge/score.py`: reads `a`, `b` from `candidate/output.json` and the rows of `judge/hidden/points.csv`; writes `{"value": mean((a*x + b - y)**2)}`.

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/e2e/test_single_run.py -v`
Expected: 2 passed. A failure now is a defect in an earlier task: fix it in that task's module with a new unit test there, then rerun.

- [ ] **Step 5: Update the README**

Add a "Status" section: "M1a (foundations): tools, checks, judges, `LocalExecutor`, and the `swarm_arena.testing` conformance kit work; plans, managers, and full runs arrive in M1b." Add a short `check_tool` example with `FunctionTool` (from the e2e test, no fixtures). CHANGELOG: "End-to-end offline test of one tool run, checked and judged through the public API."

- [ ] **Step 6: Run the task gate**

Run: `grep -nE "swarm_arena\._|from swarm_arena\.[a-z_]*\._" tests/e2e/test_single_run.py` — Expected: no output (public imports only). Then the task gate. Expected: gate exits 0.

- [ ] **Step 7: Commit**

```bash
git add tests/e2e README.md CHANGELOG.md
git commit -m "test(e2e): run, check, and judge one FunctionTool and one ScriptTool offline" \
  -m "The M1a exit test: an edit plus a same-identity base parent overlay, a ScriptCheck, and a hidden-data ScriptJudge combined with a ProgramJudge by Lexicographic, all through public imports."
```

---

### Task 25: Milestone exit

**Files:**
- Modify: nothing unless a check below fails (a fix goes in a new commit with its own test).

**Interfaces:**
- Consumes: everything above.
- Produces: the M1a exit evidence for the PR.

- [ ] **Step 1: Run the common gate (R2.0) from a clean sync**

```bash
uv sync --all-extras --all-groups
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```
Expected: every command exits 0; pytest shows no failures and no errors.

- [ ] **Step 2: Run the M1a exit tests**

```bash
uv run pytest tests/e2e/test_single_run.py -v
uv run pytest "tests/_exec/test_supervisor.py::test_grandchildren_killed_on_cancel" \
  "tests/_exec/test_supervisor.py::test_partial_collect_after_timeout" \
  "tests/_exec/test_local.py::test_tool_never_runs_when_start_commit_fails" -v
```
Expected: 2 passed, then 3 passed.

- [ ] **Step 3: Check the public surface and the "Must NOT" list**

Run: `uv run pytest tests/test_public_api.py tests/test_docstrings.py tests/test_import_isolation.py -v`. Expected: PASS, with `PUBLIC` equal to:

| Module | M1a `__all__` |
|---|---|
| `swarm_arena` | `Goal`, `Plan`, `RunConfig`, `PartSpec`, `TeamSpec`, `AgentSpec`, `RoundSpec`, `ConvergeSpec`, `Limits`, `Resources`, `Ref`, `ScriptTool`, `ScriptCheck`, `ScriptJudge`, `FunctionTool`, `ProgramJudge`, `Score` |
| `swarm_arena.tool` | `Tool`, `Command`, `CompletedRun`, `ToolRequest`, `ParentRef`, `ToolOutput`, `FileRef`, `FileSet`, `Usage`, `Estimate`, `Action`, `RunStatus` |
| `swarm_arena.scoring` | `Check`, `CheckContext`, `CheckResult`, `Judge`, `JudgeContext`, `Lexicographic` |
| `swarm_arena.manager` | `AgentStatus` |
| `swarm_arena.planning` | `FlowSpec` |
| `swarm_arena.board` | `Candidate`, `GoalKey`, `LLMCall`, `Learning` |
| `swarm_arena.executor` | `Executor`, `Job`, `LocalExecutor` |
| `swarm_arena.config` | `ModelCatalog`, `ModelSpec`, `CachePolicy`, `PlannerCatalog`, `CatalogEntry`, `SecretsPolicy`, `TracingConfig`, `GatewayConfig` |
| `swarm_arena.protocol` | `Message`, `Part`, `Artifact`, `Task`, `TaskStatus`, `TaskState`, `Role` |
| `swarm_arena.testing` | `check_tool`, `check_check`, `check_judge`, `make_candidate` |

Run: `grep -n "optional-dependencies" pyproject.toml; ls src/swarm_arena/_journal src/swarm_arena/_budget src/swarm_arena/_engine src/swarm_arena/_board 2>&1 | grep -v "No such file"`. Expected: no output (no extras, no journal, ledger, engine, or board code in M1a). Run: `git log --format=%B m1a-foundations | grep -inE "co-authored-by|generated with"`. Expected: no output.

- [ ] **Step 4: Prepare the PR (owner go-ahead required)**

Creating the remote, pushing, and opening the PR reach an audience, so ask the owner first (also ask for the repository owner and visibility). The repo has no remote yet (`git remote -v` prints nothing) and `main` holds only local commits. After approval:
```bash
gh repo create <owner>/swarm-arena --source . --remote origin --private   # or --public; or: git remote add origin <url>
git push -u origin main                       # gh pr create --base main needs main on the remote
git push -u origin m1a-foundations
gh pr create --base main --title "M1a: foundations" --body "Implements roadmap milestone M1a (docs/superpowers/plans/2026-10-18-swarm-arena-m1a-foundations.md): models, storage, tools, LocalExecutor, checks, judges, and the conformance kit. Exit criteria: the common gate, tests/e2e/test_single_run.py, and the three supervisor/executor tests, all green."
```
The PR body has no AI attribution and no "generated with" line.
