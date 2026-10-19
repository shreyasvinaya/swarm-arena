# swarm-arena M5 Release 0.1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make swarm-arena ready to publish as 0.1.0. The public surface must equal spec §9, the 12 docs pages and the README must be complete with examples that a test runs, and the wheel must be built and checked, with release automation that waits for the owner to push the tag.

**Architecture:** M5 adds no features and no public names (R2 M5 "Must NOT"). It adds release-level tests that pin the public surface, docstrings, the §9 compatibility rules, the docs, the packaging metadata, and the CI config. A small test harness in `tests/test_docs.py` runs the code blocks of every docs page and of the README, so examples cannot go stale. Fixes that these tests force on M1a–M4 code are in scope: facade `__all__` order, `__module__`, extras error messages, and I/O in component `__init__`.

**Tech Stack:** uv and uv_build, pytest 9, `numpydoc.validate`, Sphinx 9 with myst-parser and sphinx-rtd-theme, PyYAML (test-only), GitHub Actions, Read the Docs, PyPI trusted publishing.

**Spec:** `docs/superpowers/specs/2026-10-16-swarm-arena-design.md` (APPROVED r8). Binding roadmap: `docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md` (R2 M5, R3 module map, R4 interface registry, R5 constraints). Read §9, §11, §12, and roadmap R2.0, R2 M5, R3.2, R3.3, R4.12, and R5 before Task 1.

## Global Constraints

- Package `swarm-arena`, import `swarm_arena`, license MIT, version `0.1.0` (set in Task 13). Python `>=3.11`, POSIX only (§11, R5.1).
- Core dependencies are exactly `anyio>=4.15,<5`, `pydantic>=2.11,<3`, `typing-extensions>=4.12` (R5.1). Extras are exactly `llm`, `cache`, `ray`, `s3`, `langfuse`. `a2a` is v2 and is not declared (§11).
- The public surface is §9 exactly: the top level plus 11 submodules. `swarm_arena.protocol` is `Message, Part, Artifact, Task, TaskStatus, TaskState, Role` (§4.2). `swarm_arena.testing` is the R4.12 list. "Everything else is private" (§9).
- M5 must NOT add features or public names. Only docs, packaging, tests, and fixes (R2 M5). 0.1.0 is the first release, so a facade fix that removes a name not in §9 needs no deprecation.
- Facades hold only a docstring, imports, `__all__`, and one `_publish(__name__, __all__)` call (R0). `_publish` is M1a's `_facade.publish`, imported as `from swarm_arena._facade import publish as _publish`. It sets `__module__ = "swarm_arena.<facade>"` on each exported class and function, on their public nested classes, and on their methods, classmethods, staticmethods, and properties, so autodoc and `--doctest-modules` find all of them. It is the only mechanism: no facade sets `__module__` by hand. Each `__all__` is in §9 order, which is not isort order, so ruff `RUF022` is ignored for exactly the 12 facade files (Task 1, roadmap R5.3).
- Docstrings: numpydoc, with an `Examples` section on every public object and its public methods. `numpydoc.validate` runs all checks except `SA01` and `ES01` (R5.3). As in M1a, `PR01` is also skipped for `Enum` classes, because numpydoc reads the `Enum` constructor as parameters.
- Docs: Sphinx pages in Markdown (myst), `sphinx-rtd-theme`, and `exclude_patterns = ["_build", "superpowers"]` (R2 M1a, R5.3). The 12 pages of §11, in this order and with these H1 titles: Getting started, Concepts, Use your own files and script, Writing a connector, Planning modes, Scoring and review, Limits and safety, Running on Ray, The arena, Versioning policy, Roadmap, API reference.
- Docs page files: the stubs that M1a Task 1 creates (roadmap R3.3), which M1b–M4 edit. In §11 order: `docs/getting-started.md`, `docs/concepts.md`, `docs/own-files.md`, `docs/writing-a-connector.md`, `docs/planning-modes.md`, `docs/scoring-and-review.md`, `docs/limits-and-safety.md`, `docs/running-on-ray.md`, `docs/arena.md`, `docs/versioning.md`, `docs/roadmap.md`, `docs/api/index.md`. Do not rename them. Tests find a page by its H1 title (`page_path`), and the commit steps name these paths.
- URLs [plan]: repository `https://github.com/shreyasvinaya/swarm-arena`, docs `https://swarm-arena.readthedocs.io/`, PyPI `https://pypi.org/p/swarm-arena`.
- GitHub Actions versions: `actions/checkout@v7`, `astral-sh/setup-uv@v10`, `actions/upload-artifact@v7`, `actions/download-artifact@v8`, `pypa/gh-action-pypi-publish@release/v1` (research note `api-anyio-packaging.md`; artifact v7 and v8 were checked on 2026-10-18).
- Every task is TDD: write the failing test, run it, implement, run it again. Before each commit, run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run ty check`. All must be green. Tests are fully type-annotated (ruff `ANN`).
- Text I/O in M5 tests and in the harness passes `encoding="utf-8"`. This covers every `read_text()`, `write_text()`, and `open()`. `subprocess.run` calls that capture text use `encoding="utf-8"`, not bare `text=True`. Reason: the §1 picture has box-drawing characters, and the pytest process does not run in UTF-8 mode, unlike the docs blocks, which get `PYTHONUTF8=1`. The code samples below are not all rewritten; implementers apply the rule (for example in `spec_picture`, the `page_path` users, the README tests, and the `run_blocks` block files).
- R0: a task that changes R3, R4, or R5 of the roadmap edits the roadmap in the same commit and says why in the commit body.
- Commits: one per task, on branch `m5-release`. Use an imperative subject and a body that says what and why. **No AI or Claude co-author line and no "generated with" text** in commits or in the PR.
- No tag push and no PyPI upload without the owner's explicit go-ahead. An upload reaches an audience and cannot be undone (R2 M5).

## Review Focus

1. **A public name is missing from the API reference.** Autodoc lists module members from `dir()`, so it skips a lazily exported name that `__dir__` does not return, and `sphinx-build -W` still passes. A private `__module__` does not hide a name when `__all__` is set. It does add a private-path alias such as `swarm_arena.tool._models.FileRef` to `objects.inv`, and it hides method examples from `--doctest-modules`. Expected: every §9 name is in `objects.inv`, and the inventory has no private path. Tests: Task 1 `test_public_names_report_facade_module`, Task 3 `test_api_reference_documents_every_public_name`.
2. **The README quickstart works only inside the repo checkout.** It may depend on `examples/`, on `tests/_fakes`, on the editable source, on a dev dependency, or on an API key. Expected: it runs from the built wheel in an empty folder, with no keys and no network. Tests: Task 9 `test_readme_quickstart`, which uses an empty work dir and a minimal environment and can target the wheel venv through `SWARM_ARENA_DOCS_VENV`. Task 11's release build job runs that test against the wheel it builds.
3. **A plug-in class change breaks connectors in other repos (§9 rule 3).** This happens when an ABC gains an abstract method, or a component's `__init__` does I/O. Such I/O breaks `Ref.build` on Ray workers and the pre-spend config checks of `validate_plan`. Expected: the abstract method sets are frozen, and construction opens no file for reading or writing and creates no folder (`open`, `io.open`, `os.open`, `os.mkdir`), connects no socket, and starts no subprocess. Tests: Task 4 `test_abstract_methods_frozen`, `test_every_public_abc_is_frozen`, `test_component_init_does_no_io`.
4. **The core install fails without extras, or the error does not help.** Without the extra, `import swarm_arena.llm` must not give a bare `ModuleNotFoundError: No module named 'agents'`, and the wheel must not declare an optional package as a hard dependency. Expected: a `ModuleNotFoundError` (an `ImportError` subclass, so `pytest.importorskip` skips on it) that names `swarm-arena[llm]` or `swarm-arena[ray]`, and exactly three core requirements. An import error that is not a missing extra (for example `cannot import name X from 'agents'` after an upstream release, or `No module named 'agents.models'` while `agents` is installed) comes through unchanged, not as an "install the extra" message. Tests: Task 4 `test_extra_facade_names_its_extra`, `test_fake_model_names_the_llm_extra`, `test_extra_facade_keeps_other_import_errors`, `test_extra_facade_keeps_missing_vendor_submodule`, Task 13 `test_core_requirements_are_exactly_three`.
5. **A release publishes the wrong version, and PyPI never lets that version number be used again.** This happens when tag `vX.Y.Z` disagrees with `pyproject.toml`, `__version__`, `CHANGELOG.md`, or the docs `release`. Expected: all of them agree, and the publish job refuses a mismatch. Tests: Task 13 `test_version_matches_everywhere`, Task 11 `test_release_checks_tag_against_version`, `test_publish_job_is_least_privilege`.

---

## File map

| Path | Task | Responsibility |
|---|---|---|
| `tests/_surface.py` (new) | 1 | The §9 table and import helpers that the surface tests share |
| `tests/test_public_api.py` | 1 | `__all__` snapshots = §9; `__module__`; facades hold nothing else |
| `src/swarm_arena/**/__init__.py`, `src/swarm_arena/{tool,scoring,manager,planning,board,executor,config,protocol}.py` | 1, 4 | Facade fixes only |
| `tests/test_docstrings.py` | 2 | numpydoc over the whole surface; Examples contain doctests |
| `tests/test_docs.py` (new) | 3, 4–9 | Code-block harness, site structure, API inventory, required facts per page, README |
| `docs/conf.py`, `docs/index.md`, `docs/api/*.md` | 3 | Sphinx settings, landing page, one autodoc page per public module |
| `tests/test_compat.py` (new), `tests/test_import_isolation.py`, `tests/_fakes/tools.py` | 4 | §9 rules 2, 3, 5 |
| `docs/versioning.md` | 4 | "Versioning policy" |
| `docs/getting-started.md`, `docs/concepts.md` | 5 | Pages 1–2 |
| `docs/own-files.md`, `docs/writing-a-connector.md` | 6 | Pages 3–4 |
| `docs/planning-modes.md`, `docs/scoring-and-review.md`, `docs/limits-and-safety.md` | 7 | Pages 5–7 |
| `docs/running-on-ray.md`, `docs/arena.md`, `docs/roadmap.md` | 8 | Pages 8, 9, 11 |
| `README.md` | 9 | Pitch, picture, offline quickstart |
| `.readthedocs.yaml` (new), `tests/test_ci_config.py` (new) | 10 | RTD build with uv; config tests |
| `.github/workflows/release.yml` (new), `RELEASING.md` (new) | 11 | Build, check, publish with trusted publishing |
| `.github/workflows/upstream.yml` (new), `upstream-overrides.txt` (new), `CONTRIBUTING.md` | 12 | Weekly test against the newest `openai-agents` and `litellm` |
| `pyproject.toml`, `uv.lock`, `CHANGELOG.md`, `tests/test_packaging.py` (new) | 1, 10, 13 | `RUF022` ignore for the facades, merged into M1a's `__init__.py` `E402` entry (1); PyYAML dev dependency (10); version 0.1.0, metadata, wheel contents, changelog (13) |
| `docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md` | 1, 3, 4, 10, 13 | R3.2 new test files; R5.3 `RUF022` facade ignore (with the merged `__init__.py` `["E402", "RUF022"]` entry); R5.4 test-only PyYAML |

---

### Task 1: Public API equals §9

**Files:**
- Create: `tests/_surface.py`
- Modify: `tests/test_public_api.py`: replace M1a's `PUBLIC` table and `test_all_snapshot` with `_surface.PUBLIC_ALL` and the tests below. `test_public_names_report_facade_module` replaces `test_names_report_public_module` and keeps its `__pydantic_complete__` check. Keep every other test in the file, among them M1a's `test_version_is_package_metadata`, `test_publish_lets_doctest_find_method_examples`, and `test_doctest_finder_collects_method_examples`, and M3's `test_llm_facade_*` tests.
- Modify (only as the tests demand): `src/swarm_arena/__init__.py`, `tool.py`, `scoring.py`, `manager.py`, `planning.py`, `board.py`, `executor.py`, `config.py`, `protocol.py`, `llm/__init__.py`, `ray/__init__.py`, `testing/__init__.py`
- Modify: `pyproject.toml` (`[tool.ruff.lint.per-file-ignores]`: `RUF022` for the 12 facade files; the existing M1a `"src/swarm_arena/__init__.py" = ["E402"]` entry becomes `["E402", "RUF022"]`, no second key, Step 3)
- Modify: roadmap R3.2 (add `tests/_surface.py`: "the §9 table and import helpers shared by the surface tests, M5") and R5.3 (add the same three per-file ignore lines to the ruff config, including the merged `"src/swarm_arena/__init__.py" = ["E402", "RUF022"]` entry that M1a did not write to R5.3, with the reasons "core facades imported after `_publish`" for `E402` and "§9 order is the snapshot order, and it is not isort order" for `RUF022`), and, in the R3.2 `tests/conftest.py` row, replace the sentence "`test_public_api.py` and `test_docstrings.py` skip the `llm` and `ray` modules with `pytest.importorskip`" with: "`test_public_api.py`, `test_docstrings.py`, and `test_compat.py` skip the `llm` and `ray` modules and the lazy `swarm_arena.testing` names through `tests/_surface.py` `deps_present` (`importlib.metadata`; never `importorskip` or `find_spec`), M5". The roadmap update of 2026-10-18 already made this conftest-row edit, so it is a no-op: check that the row holds the new sentence, and edit it only if it does not.

**Interfaces:**
- Consumes: the facades of M1a–M4 (R3.1) and the `testing.__all__` list (R4.12).
- Produces (`tests/_surface.py`, imported as `_surface` through `pythonpath = ["tests"]`):
  - `PUBLIC_ALL: dict[str, list[str]]`: module name → `__all__` in §9 order (below).
  - `PUBLIC_SUBMODULES: frozenset[str] = frozenset({"tool", "scoring", "manager", "planning", "board", "executor", "config", "protocol", "testing", "llm", "ray"})`
  - `ENTRY_MODULES: frozenset[str] = frozenset({"run_function"})`: a public module name fixed by §4.1, with no `__all__`.
  - `EXTRA_DEPS: dict[str, tuple[str, ...]] = {"swarm_arena.llm": ("agents", "litellm"), "swarm_arena.ray": ("ray",)}`
  - `LAZY_DEPS: dict[tuple[str, str], str] = {("swarm_arena.testing", "FakeModel"): "agents", ("swarm_arena.testing", "FakeTurn"): "agents"}`
  - `DISTRIBUTIONS: dict[str, str] = {"agents": "openai-agents", "litellm": "litellm", "ray": "ray"}`: import package → distribution name.
  - `def deps_present(module: str, name: str | None = None) -> bool`: True when `importlib.metadata.version(DISTRIBUTIONS[dep])` raises no `PackageNotFoundError` for every `dep` in `EXTRA_DEPS.get(module, ())` and, when `name` is given, in `LAZY_DEPS.get((module, name))`. This is the rule of M1a's `_missing` in the conftests. Do not use `importlib.util.find_spec`: `pythonpath = ["tests"]` puts `tests/` on `sys.path`, so in the core job (no Ray) `find_spec("ray")` finds the `tests/ray/` namespace directory, and `import_public("swarm_arena.ray")` and `public_object` would not skip.
  - `def import_public(module: str) -> ModuleType`: `pytest.skip(f"needs the {extra} extra")` when deps are missing, else imports the module.
  - `def public_object(module: str, name: str) -> Any`: skips like `import_public`, then `getattr`.
  - `def resolve(dotted: str) -> Any`: imports the longest importable module prefix, then follows `getattr` for the rest.

- [ ] **Step 0: Branch**

```bash
git switch main && git pull --ff-only && git switch -c m5-release
```

- [ ] **Step 1: Write `tests/_surface.py` and the failing tests**

```python
# tests/_surface.py — PUBLIC_ALL (order = §9; protocol = §4.2; testing = R4.12)
PUBLIC_ALL: dict[str, list[str]] = {
    "swarm_arena": ["Goal", "Plan", "RunConfig", "PartSpec", "TeamSpec", "AgentSpec", "RoundSpec",
                    "ConvergeSpec", "Limits", "Resources", "Ref", "Swarm", "validate_plan", "ScriptTool",
                    "ScriptCheck", "ScriptJudge", "FunctionTool", "ProgramJudge", "Score", "ScriptedManager",
                    "KeepTopTeams", "SuccessiveHalving", "Patience", "Chain"],
    "swarm_arena.tool": ["Tool", "Command", "CompletedRun", "ToolRequest", "ParentRef", "ToolOutput",
                         "FileRef", "FileSet", "Usage", "Estimate", "Action", "RunStatus"],
    "swarm_arena.scoring": ["Check", "CheckContext", "CheckResult", "Judge", "JudgeContext", "Lexicographic",
                            "HumanReview"],
    "swarm_arena.manager": ["Manager", "AgentContext", "RoundReport", "LimitReached", "AgentStatus"],
    "swarm_arena.planning": ["Planner", "History", "Policy", "RoundState", "Decision", "FlowSpec"],
    "swarm_arena.board": ["Board", "BoardView", "ArenaBoard", "Frontier", "Learning", "Candidate", "LLMCall",
                          "GoalKey", "Row", "LineageGraph", "Curator"],
    "swarm_arena.executor": ["Executor", "LocalExecutor", "Job"],
    "swarm_arena.config": ["ModelCatalog", "ModelSpec", "CachePolicy", "PlannerCatalog", "CatalogEntry",
                           "SecretsPolicy", "TracingConfig", "GatewayConfig"],
    "swarm_arena.llm": ["AgentsSDKManager", "LLMPlanner", "CollaborativePlanner", "Reviewer", "Approve",
                        "Feedback", "Edited", "LLMJudge", "LLMCurator", "BudgetExceeded"],
    "swarm_arena.ray": ["RayExecutor"],
    "swarm_arena.protocol": ["Message", "Part", "Artifact", "Task", "TaskStatus", "TaskState", "Role"],
    "swarm_arena.testing": ["check_tool", "check_check", "check_judge", "check_manager", "make_candidate",
                            "FakeModel", "FakeTurn"],
}
```

```python
# tests/test_public_api.py (new and replaced tests; the M1a and M3 tests named in Files stay)
@pytest.mark.parametrize("module", PUBLIC_ALL)
def test_all_matches_spec(module: str) -> None:
    assert import_public(module).__all__ == PUBLIC_ALL[module]

@pytest.mark.parametrize("module", PUBLIC_ALL)
def test_public_names_report_facade_module(module: str) -> None:
    import_public(module)
    objs = [(n, public_object(module, n)) for n in PUBLIC_ALL[module] if deps_present(module, n)]
    assert [n for n, o in objs if o.__module__ != module] == []
    assert [n for n, o in objs
            if isinstance(o, type) and issubclass(o, BaseModel) and not o.__pydantic_complete__] == []

def test_each_public_name_has_one_home() -> None:
    names = [n for ns in PUBLIC_ALL.values() for n in ns]
    assert sorted(n for n in set(names) if names.count(n) > 1) == []

@pytest.mark.parametrize("module", PUBLIC_ALL)
def test_facades_hold_only_imports_and_all(module: str) -> None:
    mod = import_public(module)
    allowed = set(mod.__all__) | (PUBLIC_SUBMODULES | ENTRY_MODULES if module == "swarm_arena" else set())
    assert {n for n in vars(mod) if not n.startswith("_")} - allowed == set()

def test_no_other_public_modules() -> None:
    top = {m.name for m in pkgutil.iter_modules(swarm_arena.__path__) if not m.name.startswith("_")}
    assert top == PUBLIC_SUBMODULES | ENTRY_MODULES
    for pkg in ("swarm_arena.llm", "swarm_arena.ray", "swarm_arena.testing"):
        spec = importlib.util.find_spec(pkg)           # imports only swarm_arena, not the extra
        assert spec is not None
        assert spec.submodule_search_locations is not None
        subs = {m.name for m in pkgutil.iter_modules(spec.submodule_search_locations)}
        assert {s for s in subs if not s.startswith("_")} == set(), pkg
```

M1a's `test_version_is_package_metadata` already checks `__version__`, so this task adds no version test.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_public_api.py -v`
Expected: FAIL in every facade that differs from §9. Typical causes: `__all__` order (`test_all_matches_spec`), a stray public name such as `annotations` from `from __future__ import annotations` or a module imported without an `_` alias (`test_facades_hold_only_imports_and_all`), and a private `__module__`. A case that already passes stays as it is.

- [ ] **Step 3: Fix the facades**

Each facade keeps this shape: a one-line module docstring, then imports from private modules, then `__all__` in the `PUBLIC_ALL` order, then one `_publish(__name__, __all__)` call (Global Constraints). If a facade imports `publish` without the underscore alias, change the import to `from swarm_arena._facade import publish as _publish`, so `test_facades_hold_only_imports_and_all` passes. If a facade sets `__module__` by hand (for example `RayExecutor.__module__ = "swarm_arena.ray"`), replace those lines with the `_publish` call: a hand-set class keeps its methods on the private module, so `--doctest-modules` skips their examples. Name any other helper import with a leading underscore (for example `import os as _os`). Remove from `__all__` (and from the facade namespace) every name that is not in `PUBLIC_ALL`. The object stays importable from its private module.

The §9 order is not isort order, and the R5.3 `RUF` set includes `RUF022` (unsorted `__all__`). M1a Task 1 Step 2 already put `"src/swarm_arena/__init__.py" = ["E402"]` in `[tool.ruff.lint.per-file-ignores]`. A repeated key in one TOML table is a parse error, so do not add a second `"src/swarm_arena/__init__.py"` line: extend that entry to `["E402", "RUF022"]`, and add the other two lines. The table then holds these lines:

```toml
"src/swarm_arena/__init__.py" = ["E402", "RUF022"]   # E402: core facades imported after _publish (M1a); RUF022: §9 order is the snapshot order (tests/test_public_api.py)
"src/swarm_arena/{tool,scoring,manager,planning,board,executor,config,protocol}.py" = ["RUF022"]
"src/swarm_arena/{llm,ray,testing}/__init__.py" = ["RUF022"]
```

Write the same three lines in the R5.3 `[tool.ruff.lint.per-file-ignores]` block of the roadmap, after `"examples/**"` (R0). M1a did not add its `E402` entry to R5.3, so this edit adds the merged `["E402", "RUF022"]` entry there. If R5.3 already has a `"src/swarm_arena/__init__.py"` line, change that line to the merged entry and do not add a second one. Then check that each key occurs only once in `pyproject.toml`: `uv run python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"` exits 0.

Ruff merges these with the existing `TID251` ignores for `llm/**` and `ray/**`. Do not sort any `__all__`, and do not use `ruff check --fix` on the facades.

`swarm_arena.testing` exports `FakeModel` and `FakeTurn` lazily (R4.12). Its `__getattr__(name: str) -> Any` loads the module with `_importlib.import_module("swarm_arena.testing._fake_model")` (`import importlib as _importlib`), never with `from swarm_arena.testing import _fake_model`: the `from` form reuses the package attribute `_fake_model` when it is still set, so the Task 4 isolation test could not force a fresh import. It then stores the object in `globals()` and calls `_publish(__name__, [name])`, so the class and its methods report `swarm_arena.testing`. The module-level call stays `_publish(__name__, [n for n in __all__ if n not in _LAZY])` (M3), because `publish` reads every name it gets and would import `agents`. Its `__dir__() -> list[str]` returns `sorted({*globals(), *__all__})`, so autodoc finds both names (Task 3).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_public_api.py -v && uv run pytest -q`
Expected: PASS. The whole suite is green, because a facade change can break imports elsewhere. With all extras installed, there are no skips in `test_public_api.py`.
Then: `uv run ruff check . && uv run ruff format --check . && uv run ty check` → no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/_surface.py tests/test_public_api.py src/swarm_arena pyproject.toml docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "test: pin every public __all__ to spec §9" -m "Snapshot tests now equal §9 exactly, in order, for the top level and all 11 submodules. They also check that each name reports its facade as __module__ and that facades export nothing else. Every facade sets __module__ only through _facade.publish, imported as _publish. Facades fixed: <list>. Roadmap R3.2 lists tests/_surface.py. Roadmap R3.2 now says the surface tests skip extras through deps_present. Ruff RUF022 is ignored for the 12 facades, and roadmap R5.3 says so, because §9 order is the snapshot order and is not isort order. The top-level __init__.py entry is merged into one key (E402 and RUF022) in pyproject.toml and in R5.3."
```

---

### Task 2: Docstrings and doctests across the whole surface

**Files:**
- Modify: `tests/test_docstrings.py`: `test_numpydoc_valid` replaces M1a's `test_numpydoc` and keeps its `PR01` skip for enums. Keep M1a's `_is_enum` helper (True when the object that `dotted` names is an `Enum` subclass).
- Modify (only as the tests demand): the docstrings in the private modules that define the public objects

**Interfaces:**
- Consumes: `_surface.PUBLIC_ALL`, `LAZY_DEPS`, `deps_present`, `import_public`, `resolve` (Task 1); M1a `_facade.publish`, which makes `--doctest-modules` collect method examples.
- Produces: `def public_objects() -> list[ParameterSet]` in `tests/test_docstrings.py`. It returns one param per `"<module>.<name>"`, and one per `"<module>.<Class>.<member>"` for each own member of a public class, and, recursively, one per public nested class defined in the class body and per own member of it (M1a rule): `"<module>.<Class>.<Nested>"` and `"<module>.<Class>.<Nested>.<member>"`. A module or lazy name whose extra is missing gives one param marked `pytest.mark.skip(reason="needs the <extra> extra")`. It is never dropped silently. It never calls `importorskip` at collection time.

- [ ] **Step 1: Write the failing tests**

```python
IGNORED = frozenset({"SA01", "ES01"})

def own_members(cls: type) -> list[str]:
    """Public methods, properties, and nested classes defined on the class itself (not inherited)."""
    # the __qualname__ test keeps only classes defined in the body: an alias such as Config = EmptyConfig is out;
    # public_objects() recurses into each nested class it returns
    return [n for n, v in vars(cls).items()
            if not n.startswith("_") and (isinstance(v, FunctionType | classmethod | staticmethod | property)
                                          or (isinstance(v, type) and v.__qualname__ == f"{cls.__qualname__}.{n}"))]

def test_public_objects_cover_nested_classes() -> None:
    names = {p.values[0] for p in public_objects()}
    assert {"swarm_arena.ScriptTool.Config", "swarm_arena.ScriptedManager.Request",
            "swarm_arena.executor.Job.Stage"} <= names

@pytest.mark.parametrize("dotted", public_objects())
def test_numpydoc_valid(dotted: str) -> None:
    errors = [(code, msg) for code, msg in validate(dotted)["errors"]
              if code not in IGNORED and not (code == "PR01" and _is_enum(dotted))]
    assert errors == []          # M1a rule: numpydoc reads Enum's constructor as parameters, hence the PR01 skip

DOCTEST_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE      # = R5.3 doctest_optionflags

@pytest.mark.parametrize("dotted", public_objects())
def test_examples_are_doctests(dotted: str) -> None:
    module = max((m for m in PUBLIC_ALL if dotted.startswith(f"{m}.")), key=len)
    found = doctest.DocTestFinder(recurse=False).find(resolve(dotted), dotted,
                                                     globs=dict(vars(import_public(module))))
    assert any(t.examples for t in found), f"{dotted}: the Examples section has no >>> line"
    if (module, dotted[len(module) + 1:].partition(".")[0]) in LAZY_DEPS:   # see the note below
        runner = doctest.DocTestRunner(optionflags=DOCTEST_FLAGS)
        for test in found:
            runner.run(test)
        assert runner.failures == 0, f"{dotted}: an Examples line failed (see the captured stdout)"

@pytest.mark.parametrize("module", PUBLIC_ALL)
def test_facade_docstring(module: str) -> None:
    doc = (import_public(module).__doc__ or "").strip()
    assert doc != ""
    assert doc.splitlines()[0].endswith(".")
```

Where the examples run: `--doctest-modules` (`uv run pytest src`) runs the examples of every exported name and of its methods, classmethods, staticmethods, properties, and nested classes, because `_publish` sets their `__module__` to the facade (M1a `test_publish_lets_doctest_find_method_examples` and `test_doctest_finder_collects_method_examples` pin this). So `test_examples_are_doctests` only checks that each Examples section holds a `>>>` line. The one exception is the lazy names of `LAZY_DEPS` (`FakeModel`, `FakeTurn`): they are not in `vars(swarm_arena.testing)` until first access, so `--doctest-modules` does not see them, and this test runs their examples and their members' examples, with the facade's globals.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_docstrings.py -v`
Expected: FAIL. `test_examples_are_doctests` is new and fails on every Examples section that holds only prose, and on every wrong output in a `FakeModel` or `FakeTurn` example. `test_numpydoc_valid` now covers M3 and M4 names, the own members of every public class, and the public nested classes (`ScriptTool.Config`, `ScriptedManager.Request`, `Job.Stage`) with their members, so it also fails wherever a section is missing. `test_public_objects_cover_nested_classes` pins the helper: it fails whenever `own_members` drops nested classes or `public_objects` does not recurse into them.

- [ ] **Step 3: Fix the docstrings**

Edit the docstrings where each object is defined. Every Examples section gets at least one `>>>` line. An example that needs I/O or a long run marks those lines `# doctest: +SKIP`, and a test covers that behavior instead (R5.3). Do not change any signature.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_docstrings.py -v && uv run pytest src -q`
Expected: PASS. The `src` doctests pass too. `--doctest-modules` collects, from the facades, the examples of the public classes and functions and of their methods and properties, because `_publish` sets their `__module__` to the facade. A wrong method or property example fails here; fix it in Step 3.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_docstrings.py src/swarm_arena
git commit -m "test: validate numpydoc and doctest examples on the whole public surface" -m "Covers every §9 name and each public class's own methods, including llm, ray, and testing. Every Examples section must hold a runnable >>> line. Method examples run under --doctest-modules, because _facade.publish sets their __module__; the test runs only the lazy FakeModel and FakeTurn examples itself, because they are not in the facade namespace when --doctest-modules scans it."
```

---

### Task 3: Docs harness, site structure, and API reference

**Files:**
- Create: `tests/test_docs.py`
- Modify: `docs/conf.py`, `docs/index.md`, `docs/api/index.md`
- Create or modify: `docs/api/swarm_arena.md`, `tool.md`, `scoring.md`, `manager.md`, `planning.md`, `board.md`, `executor.md`, `config.md`, `llm.md`, `ray.md`, `protocol.md`, `testing.md`
- Modify: the existing code blocks of any docs page, until they run or carry a marker
- Modify: roadmap R3.2 (add `tests/test_docs.py`: "docs code-block harness, site structure, required facts, README quickstart, M5")

**Interfaces:**
- Consumes: `_surface.PUBLIC_ALL`, `deps_present` (Task 1).
- Produces (in `tests/test_docs.py`, used by Tasks 4–9 and 11):
  - `REPO: Path = Path(__file__).resolve().parents[1]`
  - `PAGES: list[tuple[str, str]]`: (H1 title, test id) in §11 order: `("Getting started", "getting_started")`, `("Concepts", "concepts")`, `("Use your own files and script", "own_files")`, `("Writing a connector", "connector")`, `("Planning modes", "planning")`, `("Scoring and review", "scoring")`, `("Limits and safety", "limits")`, `("Running on Ray", "ray")`, `("The arena", "arena")`, `("Versioning policy", "versioning")`, `("Roadmap", "roadmap")`, `("API reference", "api")`
  - `REQUIRED_FACTS: dict[str, list[str]] = {}`: title → exact substrings the page source must contain. It is one dict literal, placed above `test_required_facts`, and Tasks 4–8 add their entries inside this literal. `@pytest.mark.parametrize` reads the dict when the module loads, so an assignment such as `REQUIRED_FACTS["X"] = [...]` placed after the test gives an empty param set, and that page's facts are never checked.
  - `EXTRA_IMPORTS: dict[str, str] = {"llm": "agents", "cache": "litellm", "ray": "ray", "s3": "s3fs", "langfuse": "langfuse"}`
  - `@dataclass(frozen=True) class Block: lang: str; code: str; kind: Literal["none", "skip", "requires", "file", "run"]; arg: str; line: int` (`line` = 1-based line number of the opening fence)
  - `def parse_blocks(text: str) -> list[Block]`
  - `def run_blocks(blocks: Sequence[Block], *, source: str, workdir: Path, bindir: Path, timeout_s: float = 300.0) -> None`
  - `def block_env(workdir: Path, bindir: Path) -> dict[str, str]`
  - `def docs_bindir() -> Path`: `Path(os.environ["SWARM_ARENA_DOCS_VENV"]) / "bin"` when that variable is set, else `Path(sys.executable).parent`.
  - `def page_path(title: str) -> Path`: the one file under `docs/` (not `docs/superpowers/`, not `docs/_build/`) whose first `# ` line equals `title`.
  - `def make_workdir(tmp_path: Path, *, with_examples: bool) -> Path`: `tmp_path / "work"`. With `with_examples`, `examples/` is copied in with `ignore=shutil.ignore_patterns(".swarm-arena", "__pycache__")`.
  - `def spec_picture() -> str`: the content of the first fenced block in the spec file (the §1 tree).
  - `def inventory_names(path: Path) -> set[str]`: the first token of each line of a Sphinx v2 `objects.inv`. Skip the 4 header lines, then `zlib.decompress` the rest.

**Marker grammar [plan].** A marker is one HTML comment on the line directly above an opening fence. HTML comments are hidden in Sphinx, on GitHub, and on PyPI.

| Marker | Allowed on | Effect |
|---|---|---|
| `<!-- docs-test: skip <reason> -->` | `python` | Not run. The reason is required. |
| `<!-- docs-test: requires <extra> -->` | `python` | Runs only if the target interpreter can import `EXTRA_IMPORTS[extra]`, checked with `<bindir>/python -c` and `importlib.util.find_spec`. |
| `<!-- docs-test: file <relpath> -->` | any language | The block content is written to `<workdir>/<relpath>` and is never run. |
| `<!-- docs-test: run -->` | `bash` | Runs with `bash -euo pipefail -c <code>`. |
| (none) | — | A `python` block runs. Every other language is ignored. |

Fences are lines at column 0 with 3 or more backticks. The first word of the info string is the language. A language that starts with `{` is a MyST directive (for example `{literalinclude}` or `{toctree}`) and is never a block. The fence closes on a column-0 line that holds only at least as many backticks. `parse_blocks` raises `ValueError` for the cases below. Every message contains `docs-test` and the 1-based line number of the marker:
- a line that contains `docs-test:` but does not match `^<!-- docs-test: (skip|requires|file|run)(?: (.*\S))? -->$`;
- `skip` with no reason;
- `requires` with an extra that is not in `EXTRA_IMPORTS`;
- `file` with no path, an absolute path, or a `..` part;
- `run` with an argument, or `run` on a block that is not `bash`;
- `skip` or `requires` on a block that is not `python`;
- a marker that is not directly followed by a fence.

`run_blocks` handles the blocks in document order. A `python` block is written to `<workdir>/.docs_block_L<line>.py` and run as `<bindir>/python <file>` with `cwd=workdir`. A `bash` `run` block is run the same way. Both use `env=block_env(...)` and `timeout_s`. A non-zero exit raises `AssertionError(f"{source}:{line} exited {rc}\n{stdout}\n{stderr}")`. So the blocks share state only through files in `workdir`, and each block must be self-contained, as a copy-paste example is.

`block_env` returns exactly these keys:
- `PATH`: `f"{bindir}{os.pathsep}{os.environ['PATH']}"`
- `HOME`: `<workdir>/.home`
- `TMPDIR`: `<workdir>/.tmp` (both folders are created)
- `PYTHONUTF8`: `"1"`
- `PYTHONDONTWRITEBYTECODE`: `"1"`
- `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`: `"http://127.0.0.1:9"`, so an accidental network call fails fast
- `NO_PROXY`: `""`

No API keys, `PYTHONPATH`, or `VIRTUAL_ENV` pass through.

- [ ] **Step 1: Write the failing tests**

````python
SAMPLE = """\
```python
print(1)
```
<!-- docs-test: skip needs a key -->
```python
print(2)
```
<!-- docs-test: file m.py -->
```python
X = 41
```
<!-- docs-test: run -->
```bash
echo ok
```
```{literalinclude} ../examples/x.py
```
"""

def test_parse_blocks_reads_markers() -> None:
    got = [(b.lang, b.kind, b.arg) for b in parse_blocks(SAMPLE)]
    assert got == [("python", "none", ""), ("python", "skip", "needs a key"), ("python", "file", "m.py"),
                   ("bash", "run", "")]

@pytest.mark.parametrize("marker", ["<!-- docs-test: skip -->", "<!-- docs-test: maybe -->",
                                    "<!-- docs-test: requires gpu -->", "<!-- docs-test: file ../x.py -->",
                                    "<!-- docs-test: run -->"])
def test_parse_blocks_rejects_bad_markers(marker: str) -> None:
    with pytest.raises(ValueError, match="docs-test"):
        parse_blocks(f"{marker}\n```python\nx = 1\n```\n")

def test_run_blocks_shares_files_between_blocks(tmp_path: Path) -> None:
    blocks = parse_blocks("<!-- docs-test: file m.py -->\n```python\nX = 41\n```\n"
                          "```python\nimport m\nassert m.X + 1 == 42\n```\n")
    run_blocks(blocks, source="t.md", workdir=make_workdir(tmp_path, with_examples=False), bindir=docs_bindir())

def test_run_blocks_reports_failing_line(tmp_path: Path) -> None:
    blocks = parse_blocks("text\n```python\nraise SystemExit(3)\n```\n")
    with pytest.raises(AssertionError, match=r"t\.md:2 exited 3"):
        run_blocks(blocks, source="t.md", workdir=make_workdir(tmp_path, with_examples=False),
                   bindir=docs_bindir())

def test_blocks_get_no_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    code = ('import os\nassert "OPENAI_API_KEY" not in os.environ\n'
            'assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:9"\n')
    run_blocks(parse_blocks(f"```python\n{code}```\n"), source="t.md",
               workdir=make_workdir(tmp_path, with_examples=False), bindir=docs_bindir())

def test_index_lists_the_12_pages_in_order() -> None:
    # entries of the ```{toctree} block in docs/index.md (lines not starting with ":"), each resolved to
    # docs/<entry>.md; their first "# " lines in order:
    assert toctree_titles(REPO / "docs" / "index.md") == [t for t, _ in PAGES]

def test_index_shows_the_picture() -> None:
    assert spec_picture() in (REPO / "docs" / "index.md").read_text()

def test_conf_settings() -> None:
    conf = runpy.run_path(str(REPO / "docs" / "conf.py"))
    assert {"_build", "superpowers"} <= set(conf["exclude_patterns"])
    assert {"sphinx.ext.autodoc", "numpydoc", "myst_parser"} <= set(conf["extensions"])
    assert conf["html_theme"] == "sphinx_rtd_theme"
    assert conf["autodoc_member_order"] == "bysource"
    assert conf["numpydoc_show_class_members"] is False
    assert conf["release"] == swarm_arena.__version__

def test_api_reference_has_one_page_per_public_module() -> None:
    pages = list((REPO / "docs" / "api").glob("*.md"))
    for module in PUBLIC_ALL:
        # the M1a form: a ```{eval-rst} fence, then ".. automodule:: <module>", then an indented ":members:"
        pattern = re.compile(rf"^```\{{eval-rst\}}\n\.\. automodule:: {re.escape(module)}\n +:members:$", re.M)
        assert sum(bool(pattern.search(p.read_text())) for p in pages) == 1, module

@pytest.mark.slow
def test_api_reference_documents_every_public_name(tmp_path: Path) -> None:
    if not (deps_present("swarm_arena.llm") and deps_present("swarm_arena.ray")):   # not find_spec (Task 1)
        pytest.skip("needs the llm and ray extras")
    out = tmp_path / "html"
    done = subprocess.run([sys.executable, "-m", "sphinx", "-W", "--keep-going", "-q", "-b", "html",
                           str(REPO / "docs"), str(out)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    names = inventory_names(out / "objects.inv")
    assert [f"{m}.{n}" for m, ns in PUBLIC_ALL.items() for n in ns if f"{m}.{n}" not in names] == []
    assert sorted(n for n in names if "._" in n) == []      # a private __module__ adds a :canonical: alias

@pytest.mark.parametrize(("title", "pid"), PAGES, ids=[pid for _, pid in PAGES])
def test_page_blocks_run(title: str, pid: str, tmp_path: Path) -> None:
    path = page_path(title)
    run_blocks(parse_blocks(path.read_text()), source=str(path.relative_to(REPO)),
               workdir=make_workdir(tmp_path, with_examples=True), bindir=docs_bindir())

REQUIRED_FACTS: dict[str, list[str]] = {
    # Tasks 4-8 add one entry per page here, inside this literal
}

@pytest.mark.parametrize("title", list(REQUIRED_FACTS), ids=[dict(PAGES)[t] for t in REQUIRED_FACTS])
def test_required_facts(title: str) -> None:
    text = page_path(title).read_text()
    assert [p for p in REQUIRED_FACTS[title] if p not in text] == []
````

Sphinx runs in a subprocess, so pytest's `error::DeprecationWarning` filter does not apply to it. `test_required_facts` has no params until Task 4, so pytest reports it as skipped.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_docs.py -v`
Expected: FAIL. `test_conf_settings` fails on `autodoc_member_order` or `numpydoc_show_class_members`. `test_api_reference_has_one_page_per_public_module` fails for module pages that do not exist yet, and for any page that does not use the M1a `{eval-rst}` form exactly (for example a top-level page that names `swarm_arena.swarm_arena`, or an `llm.md` or `ray.md` written as a MyST ```` ```{automodule} ```` fence). The M1a pages that follow the form pass. `test_index_shows_the_picture` fails if the index has no picture. `test_page_blocks_run` fails on any existing block that does not run.

- [ ] **Step 3: Implement the harness and the site structure**

- `tests/test_docs.py`: the helpers in Interfaces, as specified above. Add `def toctree_titles(index: Path) -> list[str]`.
- `docs/conf.py`:
  - `project = "swarm-arena"`, `author = "Shreyas Vinaya"`, `copyright = "2026, Shreyas Vinaya"`
  - `release = swarm_arena.__version__`, `version = ".".join(release.split(".")[:2])`
  - `autodoc_member_order = "bysource"`, so the API reference follows the §9 order
  - `numpydoc_show_class_members = False`, which avoids the duplicate-member warnings of numpydoc with autodoc
  - keep the M1a extensions, theme, and `exclude_patterns`
- `docs/index.md`:
  - H1 `swarm-arena`
  - the §1 pitch sentence
  - the §1 picture as a ```` ```text ```` block, copied byte for byte from the spec
  - one install line
  - a ```` ```{toctree} ```` with `:maxdepth: 2` and the 12 entries in §11 order (the last is `api/index`)
- `docs/api/index.md`:
  - H1 `API reference`
  - one paragraph that states the public surface rule of §9 ("Everything in the `__all__` of the top level and of the public submodules is public. Everything else is private.")
  - a toctree of the 12 module pages
- Each `docs/api/<slug>.md` keeps the M1a form (M1a Task 1; M3 Task 1 and M4 Task 7 write `llm.md` and `ray.md` the same way):
  - H1 = the module name: `# swarm_arena` for `swarm_arena.md`, `# swarm_arena.<module>` for the others
  - exactly one ```` ```{eval-rst} ```` block that holds `.. automodule:: <module>` and, on the next line, `   :members:` (3 spaces), then a closing fence. `<module>` is the `PUBLIC_ALL` key: `swarm_arena` for the top-level page, never `swarm_arena.swarm_arena`
  - a page that uses another form (for example a MyST ```` ```{automodule} ```` fence) is rewritten to this form
  - `llm.md` and `ray.md` add one line: "Needs the `llm` extra" or "Needs the `ray` extra"
- Existing blocks that fail: fix them if the fix is local. If not, mark them `<!-- docs-test: skip rewritten in M5 task <n> -->`, where `<n>` is the task that rewrites that page (5–8). Task 8 rejects any skip reason that still contains `M5 task`.
- If `test_api_reference_documents_every_public_name` lists lazy names as missing, the Task 1 `__dir__` in `swarm_arena.testing` is wrong. Fix it there.
- If it lists a private path, `_publish` did not set that object's `__module__`. Fix the facade.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_docs.py -v && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: PASS. Sphinx exits 0 with no warnings.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_docs.py docs/conf.py docs/index.md docs/api docs/*.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "test: run docs code blocks and check the API reference inventory" -m "Adds a harness that runs every python block and every marked bash block of a page, in a clean work dir with no secrets. Adds a check that every §9 name is in the built objects.inv, and one autodoc page per public module. Roadmap R3.2 lists tests/test_docs.py."
```

---

### Task 4: Compatibility rules (§9 rules 2, 3, 5) and the "Versioning policy" page

**Files:**
- Create: `tests/test_compat.py`
- Modify: `tests/test_import_isolation.py`, `tests/_fakes/tools.py`, `tests/test_docs.py` (facts and one test), `docs/versioning.md`
- Modify (only as the tests demand):
  - `src/swarm_arena/llm/__init__.py`, `ray/__init__.py`, `testing/__init__.py`: the extra error (a `ModuleNotFoundError`, exact-name rule) and its message
  - any component `__init__` that does I/O: move the I/O into `preflight()` (§9 rule 3)
- Modify: roadmap R3.2 (add `tests/test_compat.py`: "§9 rules 2, 3: frozen abstract methods, no I/O in `__init__`, newer `schema_version` refused, M5")

**Interfaces:**
- Consumes: `_surface` (Task 1), `CURRENT` and `upgrade` from `swarm_arena._model.migrate` (R4.13.0), `Plan.loads`, `RunConfig.load` (R4.1), harness and `REQUIRED_FACTS` (Task 3).
- Produces: two module-level functions in `tests/_fakes/tools.py`, used only here:
  - `def compat_probe(x: float) -> dict[str, float]`, which returns `{"x": x}`
  - `def compat_judge(goal: Goal, candidate: Candidate) -> Score`, which returns `Score.maximize(1.0)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_compat.py
ABSTRACT: dict[str, frozenset[str]] = {       # R4.2-R4.7; §9 rule 3: ABCs grow only by optional methods
    "swarm_arena.tool.Tool": frozenset({"command", "collect"}),
    "swarm_arena.scoring.Check": frozenset({"check"}),
    "swarm_arena.scoring.Judge": frozenset({"judge"}),
    "swarm_arena.manager.Manager": frozenset({"run_round"}),
    "swarm_arena.planning.Planner": frozenset({"plan"}),
    "swarm_arena.planning.Policy": frozenset({"after_round"}),
    "swarm_arena.executor.Executor": frozenset({"submit", "capacity", "aclose"}),
    "swarm_arena.board.Board": frozenset({"put_candidate", "put_score", "put_learning", "put_decision",
        "frontier", "snapshot", "best", "learnings", "search_learnings", "get_learning", "get_candidate",
        "scores", "lineage", "leaderboard"}),
    "swarm_arena.board.BoardView": frozenset({"best", "learnings", "search_learnings", "get_learning",
        "get_candidate", "scores", "lineage", "leaderboard"}),
    "swarm_arena.board.Curator": frozenset({"curate"}),
}

@pytest.mark.parametrize("dotted", ABSTRACT)
def test_abstract_methods_frozen(dotted: str) -> None:
    assert resolve(dotted).__abstractmethods__ == ABSTRACT[dotted]

def test_every_public_abc_is_frozen() -> None:
    found = {f"{m}.{n}" for m, ns in PUBLIC_ALL.items() if deps_present(m) for n in ns
             if deps_present(m, n) and isinstance(o := public_object(m, n), type)
             and getattr(o, "__abstractmethods__", frozenset())}
    assert found == set(ABSTRACT)

# (id, module to import before the I/O guard, builder); llm and ray params carry
# pytest.mark.skipif(not deps_present("swarm_arena.llm" | "swarm_arena.ray"), reason="needs the extra")
COMPONENTS: list[ParameterSet] = [
    pytest.param("swarm_arena", lambda tmp: ScriptTool(command=["python", "run.py"]), id="ScriptTool"),
    pytest.param("swarm_arena", lambda tmp: FunctionTool("_fakes.tools:compat_probe"), id="FunctionTool"),
    pytest.param("swarm_arena", lambda tmp: ScriptCheck(command=["python", "check.py"]), id="ScriptCheck"),
    pytest.param("swarm_arena", lambda tmp: ScriptJudge(command=["python", "score.py"]), id="ScriptJudge"),
    pytest.param("swarm_arena", lambda tmp: ProgramJudge("_fakes.tools:compat_judge"), id="ProgramJudge"),
    pytest.param("swarm_arena.scoring",
                 lambda tmp: Lexicographic([ProgramJudge("_fakes.tools:compat_judge")]), id="Lexicographic"),
    pytest.param("swarm_arena", lambda tmp: ScriptedManager(requests=[{"inputs": {"x": 1.0}}]),
                 id="ScriptedManager"),
    pytest.param("swarm_arena", lambda tmp: KeepTopTeams(2), id="KeepTopTeams"),
    pytest.param("swarm_arena", lambda tmp: SuccessiveHalving(keep=0.5, min_per_team=1), id="SuccessiveHalving"),
    pytest.param("swarm_arena", lambda tmp: Patience(rounds=2), id="Patience"),
    pytest.param("swarm_arena", lambda tmp: Chain(KeepTopTeams(2), Patience(rounds=2)), id="Chain"),
    pytest.param("swarm_arena.executor", lambda tmp: LocalExecutor(), id="LocalExecutor"),
    pytest.param("swarm_arena.board", lambda tmp: ArenaBoard(arena_root=str(tmp / "arena")), id="ArenaBoard"),
    # extras: AgentsSDKManager(model="opus"), LLMJudge("Score 0 to 10.", ["opus"]), LLMPlanner("opus"),
    #         LLMCurator(model="flash") from swarm_arena.llm; RayExecutor() from swarm_arena.ray. Also, with the
    #         llm skipif mark and imported the same way as the other llm params:
    #         pytest.param("swarm_arena.llm", lambda tmp: CollaborativePlanner(LLMPlanner("opus"), _NoReview()),
    #                      id="CollaborativePlanner")
]

@pytest.mark.parametrize(("module", "build"), COMPONENTS)
def test_component_init_does_no_io(module: str, build: Callable[[Path], object], tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    importlib.import_module(module)                      # imports happen before the guard
    dirs = [tmp_path / d for d in ("cwd", "home", "tmp", "cache")]
    for d in dirs:
        d.mkdir()
    monkeypatch.chdir(dirs[0])
    for var, d in zip(("HOME", "TMPDIR", "XDG_CACHE_HOME"), dirs[1:], strict=True):
        monkeypatch.setenv(var, str(d))
    monkeypatch.setattr(tempfile, "tempdir", None)
    with monkeypatch.context() as guard:                 # the guards end before the asserts and the pytest report
        for target, attr in ((socket.socket, "connect"), (subprocess.Popen, "__init__"), (builtins, "open"),
                             (io, "open"), (os, "open"), (os, "mkdir")):
            guard.setattr(target, attr, _forbid)
        build(tmp_path)
    assert [d.name for d in dirs if any(d.iterdir())] == []
    assert not (tmp_path / "arena").exists()

@pytest.mark.parametrize("kind", sorted(CURRENT))
def test_newer_schema_version_refused(kind: str) -> None:
    newer = CURRENT[kind] + 1
    with pytest.raises(ValueError, match="schema_version") as err:
        upgrade(kind, {"schema_version": newer})
    assert str(newer) in str(err.value)

def test_plan_and_run_files_refuse_newer_versions(tmp_path: Path) -> None:
    # read examples/rosenbrock/plan.toml and run.toml, drop any top-level "schema_version = ..." line,
    # prepend "schema_version = 999\n"; Plan.loads(text, base_dir=tmp_path) and RunConfig.load(<written file>)
    # each raise ValueError whose message contains "schema_version" and "999"
```

`_forbid(*args: object, **kwargs: object) -> NoReturn` raises `AssertionError("I/O in a component __init__ (§9 rule 3)")`. `_NoReview` is a module-level class with `async def review(self, plan: Plan) -> Any`, which raises `AssertionError` (it is never called). The guards catch every file open and folder creation: pathlib's `open`, `read_text`, `write_text`, and `touch` go through `io.open` and `os.open`, and `Path.mkdir` and `os.makedirs` go through `os.mkdir`. Imports are not affected, because importlib uses `_io.open_code` and `posix` directly. A realistic failure: hashing a script file for `fingerprint()` in `__init__`.

```python
# tests/test_import_isolation.py (add)
class _Hide:                       # wraps one sys.meta_path finder, so dep and dep.* look uninstalled
    def __init__(self, finder: Any, dep: str) -> None: ...
    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any: ...
        # None for dep and dep.*; otherwise the wrapped finder's find_spec (None if it has none)
    def __getattr__(self, attr: str) -> Any: ...
        # delegates the rest: invalidate_caches, and find_distributions, which importlib.metadata reads
        # from sys.meta_path, so version() still finds installed distributions (M4's version("ray") check)

def _hide(prefix: str) -> None: ...  # fresh interpreter only: sys.meta_path[:] = [_Hide(f, prefix) for f in ...]

HIDE_SOURCE: str = ...               # "import sys\nfrom typing import Any\n" + inspect.getsource of _Hide and _hide

def _block(dep: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Imitates an uninstalled distribution, which the exact-name rule of the facades needs (Step 3).
    # sys.modules[dep] = None is not enough: `from dep.sub import x` then raises with name="dep.sub",
    # which the facade rightly re-raises. After this, find_spec(dep) is None, and every import of dep
    # or dep.x raises ModuleNotFoundError("No module named '<dep>'", name=dep).
    for name in [m for m in sys.modules if m == dep or m.startswith(f"{dep}.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_Hide(f, dep) for f in sys.meta_path])

@pytest.mark.parametrize(("module", "dep", "extra"), [("swarm_arena.llm", "agents", "llm"),
    ("swarm_arena.llm", "litellm", "llm"), ("swarm_arena.ray", "ray", "ray")])
def test_extra_facade_names_its_extra(module: str, dep: str, extra: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [m for m in sys.modules if m == module or m.startswith(f"{module}.")]:
        monkeypatch.delitem(sys.modules, name)
    _block(dep, monkeypatch)
    with pytest.raises(ModuleNotFoundError, match=re.escape(f"swarm-arena[{extra}]")):
        importlib.import_module(module)

@pytest.mark.parametrize("name", [n for m, n in LAZY_DEPS if m == "swarm_arena.testing"])   # FakeModel, FakeTurn
def test_fake_model_names_the_llm_extra(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import swarm_arena.testing as testing
    monkeypatch.delitem(sys.modules, "swarm_arena.testing._fake_model", raising=False)
    for attr in ("_fake_model", "FakeModel", "FakeTurn"):
        monkeypatch.delitem(vars(testing), attr, raising=False)   # not delattr: its hasattr() runs the lazy __getattr__
    _block("agents", monkeypatch)
    with pytest.raises(ModuleNotFoundError, match=re.escape("swarm-arena[llm]")):
        getattr(testing, name)

@pytest.mark.parametrize(("module", "sub"), [("swarm_arena.llm", "agents.models"), ("swarm_arena.ray", "ray._private")])
def test_extra_facade_keeps_missing_vendor_submodule(module: str, sub: str) -> None:
    public_object(module, PUBLIC_ALL[module][0])            # skips when the extra is missing
    code = HIDE_SOURCE + f"_hide({sub!r})\nimport {module}\n"   # fresh interpreter: no vendor module re-runs in-process
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, encoding="utf-8", check=False)
    assert done.returncode != 0
    assert f"No module named '{sub}'" in done.stderr
    assert "swarm-arena[" not in done.stderr

@pytest.mark.parametrize("module", ["swarm_arena.llm", "swarm_arena.ray"])
def test_extra_facade_keeps_other_import_errors(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    name = PUBLIC_ALL[module][0]
    obj = public_object(module, name)                   # skips when the extra is missing
    homes = [m for k, m in list(sys.modules.items())
             if k.startswith("swarm_arena.") and k != module and getattr(m, name, None) is obj]
    assert homes != []
    for home in homes:                                  # the deps are present, but a private import breaks,
        monkeypatch.delattr(home, name)                 # as after an incompatible upstream release
    monkeypatch.delitem(sys.modules, module)
    with pytest.raises(ImportError, match=re.escape(f"cannot import name '{name}'")) as err:
        importlib.import_module(module)
    assert "swarm-arena[" not in str(err.value)
```

Never use `monkeypatch.delattr`, `hasattr`, or `getattr(..., default)` on a lazy name. All three run the module `__getattr__`, which imports `agents` or raises the extra error. Remove cached names through `vars(module)`.

The pinned submodules `agents.models` and `ray._private` are imported by each package's own `__init__`. If an upstream release renames one, pick another submodule that the `__init__` imports.

The `ray` case of `test_extra_facade_keeps_other_import_errors` must reach the private import, so M4's `importlib.metadata.version("ray")` check in the facade (inbound M4) must pass, or the extra message fires first. `public_object` lets the test run only where `version("ray")` finds the distribution, which is the same check, so the facade check passes wherever the case runs, and the core job skips it. Keep that skip ahead of the import, and do not use `_block` or hide the distribution in this test.

```python
# tests/test_docs.py: the entry goes inside the REQUIRED_FACTS literal of Task 3
REQUIRED_FACTS: dict[str, list[str]] = {
    "Versioning policy": [
        "SemVer", "breaking changes bump the minor version", "CHANGELOG.md", "one minor release",
        "DeprecationWarning", "typing_extensions.deprecated",
        "Release 1.0 after both connectors have run real campaigns", "schema_version",
        "Newer versions are refused", "optional methods with defaults", "does no I/O", "preflight()",
        "check_tool", "check_check", "check_judge", "check_manager", "FakeModel",
        "Extras keep vendor churn out of the core", "snapshot test", "Everything else is private",
        "upper bound is raised only after"],
}

# tests/test_docs.py (new test, anywhere in the module)
def test_versioning_page_lists_public_modules() -> None:
    text = page_path("Versioning policy").read_text()
    assert [m for m in PUBLIC_ALL if f"`{m}`" not in text] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_compat.py tests/test_import_isolation.py tests/test_docs.py -v -k "compat or extra or fake_model or versioning or required_facts"`
Expected: FAIL. `test_required_facts[versioning]` and `test_versioning_page_lists_public_modules` fail on the stub page. `test_extra_facade_names_its_extra` and `test_fake_model_names_the_llm_extra` fail wherever a facade lets a bare `ModuleNotFoundError` through or raises a plain `ImportError` (which `pytest.importorskip` does not skip on). `test_extra_facade_keeps_other_import_errors` fails wherever a facade turns every `ImportError` into the extra message. `test_extra_facade_keeps_missing_vendor_submodule` fails while a facade translates by the root of `_exc.name`, or while M4's ray facade catches every `ImportError` around `import ray`. `test_component_init_does_no_io` fails for any component that opens a file, creates a folder, connects a socket, or starts a subprocess in `__init__`. `test_newer_schema_version_refused` and `test_plan_and_run_files_refuse_newer_versions` are expected to pass already, because M1a's `upgrade()` message holds `schema_version` and the found version.

- [ ] **Step 3: Implement the fixes and the page**

- Extra facades: wrap the imports of the private modules in `try` / `except ModuleNotFoundError as _exc`. Catch nothing broader. Translate only when `_exc.name` is exactly in the facade's private tuple `_EXTRA_DEPS` (the `_surface` sets: `("agents", "litellm")` for `llm`, `("ray",)` for `ray`). Re-raise every other `ModuleNotFoundError` unchanged, `agents.<sub>` while `agents` is installed included. Translate with `raise ModuleNotFoundError("swarm_arena.<name> needs the <name> extra: pip install 'swarm-arena[<name>]'", name=_exc.name) from _exc`. It must be a `ModuleNotFoundError`: `pytest.importorskip` defaults to `exc_type=ModuleNotFoundError`, so a plain `ImportError` propagates, and the core-job tests that skip on it error instead (M3's `test_llm_facade_publishes_its_names`, M4's `pytest.importorskip("swarm_arena.ray")` tests in `tests/_engine/test_resume.py` and `tests/testing/test_conformance.py`). So `cannot import name X from 'agents'` after an incompatible upstream release (a plain `ImportError`), a missing vendor submodule, and an internal import bug come through as they are, and the user is not told to install an extra they already have.
  - `llm/__init__.py`: if M3's `find_spec` pre-check stays, it raises the same `ModuleNotFoundError`, with `name=` the missing package.
  - `ray/__init__.py` already follows the exact-name rule (M4 Task 4). Keep its `importlib.metadata.version("ray")` check ahead of `import ray`: under pytest, the `tests/ray/` namespace package makes `import ray` succeed without Ray, and `PackageNotFoundError` gives the extra error. M5 changes only the message text. If the file still catches every `ImportError` around `import ray`, narrow it to the same exact-name rule.
  - `swarm_arena.testing.__getattr__` uses the same exact rule for `FakeModel` and `FakeTurn`, with `("agents",)` and the `llm` extra.
- `upgrade()`: keep M1a's message (`plan schema_version 2 is newer than this swarm-arena supports (1)`), which `tests/_model/test_migrate.py::test_newer_version_refused` pins. Do not reword it. If `test_plan_and_run_files_refuse_newer_versions` fails, the loader (`Plan.loads` or `RunConfig.load`) hides that message; fix the loader so the `upgrade()` message reaches the user.
- Components: move any I/O from `__init__` into `preflight()`. Signatures stay as they are.
- `docs/versioning.md` (H1 `Versioning policy`). Sections:
  1. "Public surface": the 12 modules, each written as `` `swarm_arena.x` ``; the §9 rule; "Everything else is private".
  2. "Version numbers": §9 rule 1, and the upstream rule of §11 ("An upper bound is raised only after the weekly upstream job passes on the newer release").
  3. "Deprecations": rule 1, with the runnable example below.
  4. "Stored formats": rule 2.
  5. "Plug-in classes": rule 3.
  6. "Conformance kit": rule 4.
  7. "Extras": rule 5.
  8. "How this is checked": rule 6, naming `tests/test_public_api.py`, `tests/test_docstrings.py`, and `tests/test_compat.py`.

  The runnable block is a `python` block. It defines a function with `@deprecated("old() is replaced by new(); removed in 0.3", category=DeprecationWarning)` from `typing_extensions`, calls it inside `warnings.catch_warnings(record=True)` with `simplefilter("always")`, and asserts that the warning category is `DeprecationWarning`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_compat.py tests/test_import_isolation.py tests/test_docs.py -v && uv run pytest -q`
Expected: PASS. With all extras installed, there are no skips in `test_compat.py`.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_compat.py tests/test_import_isolation.py tests/_fakes/tools.py tests/test_docs.py docs/versioning.md src/swarm_arena docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "test: enforce §9 compatibility rules and write the versioning policy" -m "Freezes the abstract methods of every public ABC. Checks that no component __init__ does I/O, that newer schema_version values are refused with a clear message, and that the extra facades name their extra only when the extra is missing; other import errors come through unchanged. The Versioning policy page states §9 rules 1-6. Roadmap R3.2 lists tests/test_compat.py."
```

---

### Task 5: "Getting started" and "Concepts" pages

**Files:**
- Modify: `docs/getting-started.md`, `docs/concepts.md`, `tests/test_docs.py` (facts)

**Interfaces:**
- Consumes: harness, `REQUIRED_FACTS` (Task 3); `examples/rosenbrock/` (M1b, R3.3).
- Produces: nothing for later tasks.

- [ ] **Step 1: Write the failing tests** (the entries go inside the `REQUIRED_FACTS` literal of Task 3)

```python
REQUIRED_FACTS: dict[str, list[str]] = {
    # entries of earlier tasks stay here
    "Getting started": [
        "POSIX", "pip install swarm-arena", "gradient_descent(lr, steps, x0, y0)",
        "random_search(n, radius, seed)", "deterministic = false", "SuccessiveHalving", "ScriptedManager",
        "AgentsSDKManager", "solution.json", "swarm-arena run --yes"],
    "Concepts": [
        "Goal", "RunConfig", "Plan", "Tool", "Check", "Judge", "Manager", "Planner", "Policy", "Executor",
        "Board", "Curator", "Swarm", "import path", "A lambda or local function is rejected", "GoalKey",
        "Plan.parts", "stable address", "model catalog", "on by default", "shorter than the provider's minimum",
        "response cache", "nothing goes to any vendor"],
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_docs.py -v -k "getting_started or concepts"`
Expected: FAIL in `test_required_facts[getting_started]` and `test_required_facts[concepts]`, with the missing phrases listed.

- [ ] **Step 3: Write the pages**

Extend the existing file. Keep the sections that M1a-M4 wrote, and change them only where a fact or a test needs it. Then add or complete what the table lists.

| Page | Content (spec sources) | Runnable blocks |
|---|---|---|
| Getting started (§12, §11, §2 non-goals) | Install, with the extras named. Python 3.11+, POSIX only. The Rosenbrock example: the two tools, `random_search` declared `deterministic = false`, the check, the `ProgramJudge` with `Score.minimize(f)`, 2 teams of `ScriptedManager` agents, `SuccessiveHalving`, select-top-2 converge. The example files are shown with ```` ```{literalinclude} ../examples/rosenbrock/<file> ````. The run folder layout and `solution.json` (§7.1). "With an API key, change one line to use `AgentsSDKManager`", shown as a `toml` block. | A `bash` `run` block: `swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml`. A `python` block runs the same plan through `Swarm(RunConfig.load(...)).run(Plan.load(...))` with `anyio.run(..., backend="asyncio")` and prints the best finalist's id and `output.data`. |
| Concepts (§1, §4 table, §4.1, §4.2, §4.10, §4.11, §4.13, §7.3 item 3) | The 13 concepts in the §4 table. Refs by import path, and why a lambda or local function is rejected. Data types in brief. Goal identity (`GoalKey`). Many goals in one run. `Plan.parts` (split, solve, join). Stable addresses. The model catalog and aliases. Prompt caching (on by default, the provider minimum). The optional response cache. Tracing: nothing goes to any vendor until Langfuse is configured. | A `file` block `concepts_tools.py` with one module-level function. A `python` block builds a one-team `Plan` in Python from `FunctionTool` and `ScriptedManager` instances and prints `plan.dumps()`. It also shows `Ref.of(lambda x: x)` raising `TypeError` (M1a: a lambda or local function is "not a component"), caught and printed. |

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_docs.py -v -k "getting_started or concepts" && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: PASS. Sphinx gives no warnings.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add docs/getting-started.md docs/concepts.md tests/test_docs.py
git commit -m "docs: complete Getting started and Concepts" -m "Both pages state the spec facts that the test pins, and their examples run in the docs harness."
```

---

### Task 6: "Use your own files and script" and "Writing a connector" pages

**Files:**
- Modify: `docs/own-files.md`, `docs/writing-a-connector.md`, `tests/test_docs.py` (facts)

**Interfaces:**
- Consumes: harness (Task 3); `examples/own_script/` (M1b); `check_tool`, `check_check`, `make_candidate` (R4.12).
- Produces: nothing for later tasks.

- [ ] **Step 1: Write the failing tests** (the entries go inside the `REQUIRED_FACTS` literal of Task 3)

```python
REQUIRED_FACTS: dict[str, list[str]] = {
    # entries of earlier tasks stay here
    "Use your own files and script": [
        "stable format", "swarm/request.json", "swarm/result.json", "SWARM_ARENA_RESULT",
        "parents/<candidate-id>/", "candidate/request.json", "candidate/output.json", "candidate/files/",
        "swarm/llm_calls.jsonl", "SWARM_ARENA_CANDIDATE_ID", "overlay rule", "max_edit_bytes", "256 KiB",
        "never editable", "Symlinks are rejected"],
    "Writing a connector": [
        "swarm_arena.testing", "check_tool", "check_check", "check_judge", "check_manager", "FakeModel",
        "preflight()", "does no I/O", "is_deterministic", "editable", "estimate", "providers", "fingerprint",
        "extra_actions", "never imported", "executor=RayExecutor("],
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_docs.py -v -k "own_files or connector"`
Expected: FAIL in `test_required_facts[own_files]` and `test_required_facts[connector]`.

- [ ] **Step 3: Write the pages**

Extend the existing file. Keep the sections that M1a-M4 wrote, and change them only where a fact or a test needs it. Then add or complete what the table lists.

| Page | Content (spec sources) | Runnable blocks |
|---|---|---|
| Use your own files and script (§4.12, §7.3 item 2b, §8.1 `max_edit_bytes`) | The TOML-only setup. The workspace contract, marked as a stable format: the tool workspace tree, the staging order, the overlay rule, protected paths, path rules, reserved names, script output, exit codes, timeout and partial output, parents, the `list_files` and `read_file` actions, edit rules (256 KiB per edit; script, checks, and judges are never editable), and the check and judge workspace with `candidate/`. Tool environment variables `SWARM_ARENA_*`. The `swarm/llm_calls.jsonl` row format, as M4 defines it. Judge-private data. | ```` ```{literalinclude} ```` of `examples/own_script/plan.toml` and `problem/run.py`. A `bash` `run` block: `swarm-arena run --yes examples/own_script/run.toml examples/own_script/plan.toml`. A `python` block finds the newest `solution.json` with `glob.glob("**/.swarm-arena/runs/*/goals/*/solution.json", recursive=True)` (newest by `os.path.getmtime`), asserts that one exists, and prints it. The pattern matches whether the default roots resolve against the cwd or against the folder of `run.toml`, so the block does not depend on that rule. |
| Writing a connector (§4.3, §4.4, §9 rules 3–4, §4.5 extra actions, M4 Ray option) | When to subclass `Tool` rather than use `ScriptTool`. The `Tool` methods and their defaults. `command()` and `collect()` run next to the tool, and the tool program is never imported. `Check` and `Judge` subclasses. A component's `__init__` does no I/O; side effects go in `preflight()`. The conformance kit, with the Ray option `check_tool(..., executor=RayExecutor(...))`. A connector CI job (a `yaml` block, not run) that runs `uv run pytest` with the kit. | A `file` block `echo_tool.py`: an `EchoTool(Tool)` with a `Config`, an `input_model` with `text: str`, a `command()` that runs `python -c` to write `swarm/result.json`, and a `collect()`. A `python` block runs `anyio.run(check_tool, EchoTool(), {"text": "hi"}, backend="asyncio")` and asserts `status == "ok"`. A second `file` block `nonempty_check.py` holds a `Check` subclass. A `python` block runs `check_check` on a `make_candidate(...)` and prints the result. |

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_docs.py -v -k "own_files or connector" && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: PASS.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add docs/own-files.md docs/writing-a-connector.md tests/test_docs.py
git commit -m "docs: complete the own-script and connector pages" -m "Documents the workspace contract as a stable format and the conformance kit. Both pages carry examples that the docs harness runs."
```

---

### Task 7: "Planning modes", "Scoring and review", and "Limits and safety" pages

**Files:**
- Modify: `docs/planning-modes.md`, `docs/scoring-and-review.md`, `docs/limits-and-safety.md`, `tests/test_docs.py` (facts and one test)

**Interfaces:**
- Consumes: harness (Task 3); `validate_plan`, `PlannerCatalog`, `CatalogEntry`, `Limits`, `RunConfig` (R4.1, R4.8); `check_judge`, `make_candidate` (R4.12); the M2 CLI commands `review export|import` and `leaderboard`; the M3 `swarm-arena plan`.
- Produces: nothing for later tasks.

- [ ] **Step 1: Write the failing tests** (the entries go inside the `REQUIRED_FACTS` literal of Task 3)

```python
REQUIRED_FACTS: dict[str, list[str]] = {
    # entries of earlier tasks stay here
    "Planning modes": [
        "Manual", "Together", "Autonomous", "LLMPlanner", "CollaborativePlanner", "replanner=", "Approve",
        "Feedback", "Edited", "planner_catalog", "settable", "never settable", "set by the core",
        "validate_plan", "max_agents_per_fingerprint", "only between rounds", "swarm-arena plan"],
    "Scoring and review": [
        "gated_out", "Plan.ranking", "rubric_hash", "panel_hash", "anchor", "final judge", "NaN",
        "ProgramJudge", "Lexicographic", "ScriptJudge", "LLMJudge", "temperature 0", "review.csv", "--top 20",
        "--reviewer alice", "human:alice", "--by human:alice", "NO_RESULT", "KeepTopTeams",
        "SuccessiveHalving", "Patience", "Chain", "only passing candidate", "starved", "last team"],
    "Limits and safety": [
        "This is not strong isolation",
        ("Without a hard cap, the worst-case overspend is the error in the estimates "
         "of runs in flight"),
        "Total spend can never go over the run cap because of a crash", "reserve_fraction", "final slice",
        "tranche", "watchdog_stopped", "rlimits", "--yes"],
}

# tests/test_docs.py (new test, anywhere in the module)
def test_limits_page_lists_every_limit() -> None:
    text = page_path("Limits and safety").read_text()
    assert [f for f in Limits.model_fields if f"`{f}`" not in text] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_docs.py -v -k "planning or scoring or limits"`
Expected: FAIL in the three `test_required_facts` cases and in `test_limits_page_lists_every_limit`.

- [ ] **Step 3: Write the pages**

Extend the existing file. Keep the sections that M1a-M4 wrote, and change them only where a fact or a test needs it. Then add or complete what the table lists.

| Page | Content (spec sources) | Runnable blocks |
|---|---|---|
| Planning modes (§4.1, §4.6, D8, D9, §8.3) | The four modes table. Provenance set by the core. The planner catalog: entries, `settable`, and the script fields that are never settable (`command`, `python`, `files`, `env`, `editable`). Carried-over human parts. Every `validate_plan` rule of §4.6. What a replanner can and cannot change. `swarm-arena plan` (manual, `--llm`, `--together`). | A core `python` block loads `examples/rosenbrock/plan.toml` and `run.toml`, runs `validate_plan`, and prints the warnings. A second core `python` block shows the planning boundary: the same plan copied with `origin="llm"`, a non-catalog tool ref, and an empty `planner_catalog`. It catches the `ValueError` from `validate_plan` and prints it. A `bash` `run` block runs M3's manual `swarm-arena plan` on the Rosenbrock files. The LLM, together, and autonomous examples are `skip` blocks whose reasons name `tests/e2e/test_llm_plan_file.py` and `tests/e2e/test_autonomous_replan.py`. |
| Scoring and review (§4.4, §4.7, §4.9 scope, §6.6, D10, D11) | Checks and `gated_out`. Judges and `Plan.ranking`. Judge identity. One scale across rounds and anchors. Bad scores (NaN or inf). The final judge. The built-in judges, including the `LLMJudge` panel and its cache rules (temperature 0, anchors never read the cache). Judge-private data. Policies and their guarantees (only passing candidate, starved agents, a part's last team, `NO_RESULT`). Human review: export, import, re-rank. | A `file` block `judges.py` with two module-level judge functions. A `python` block scores two `make_candidate(...)` outputs with `check_judge(Lexicographic([ProgramJudge("judges:exact"), ProgramJudge("judges:short")]), goal, candidates)` and prints each `Score.key`. A `bash` `run` block runs the Rosenbrock example and finds the run id with `run="$(basename "$(find . -type d -path '*/.swarm-arena/runs/*' -print -quit)")"`, then `test -n "$run"`. `find` visits a folder before its contents, so the first match is the run folder, wherever the roots resolve. It runs `swarm-arena review export "$run" --top 20 --out review`. It sets a verdict in `review/review.csv` with an inline `python - <<'EOF'` step, then runs `swarm-arena review import "$run" review/review.csv --reviewer alice` and `swarm-arena leaderboard "$run" --by human:alice`. Each arena command gets `--config examples/rosenbrock/run.toml`, because the M2 default is `run.toml` in the cwd. |
| Limits and safety (§8, §6.1, §6.4, §6.5 stop modes, §4.5 soft errors, §5 step 2) | The full §8.1 limits table: every `Limits` field in backticks with its default. How the budget is split. The reservation ledger and tranches. Crash and takeover bounds. The overspend statement. Gateway caps in short, with a link to "Running on Ray". The watchdog. The right number of agents. Clocks. Mid-round stops. Stop and resume. Tool sandboxing ("This is not strong isolation"). The estimate and approval step (`--yes`). | A `python` block prints `Limits()`, then `RunConfig(limits=Limits(max_usd=5.0, max_tokens=2_000_000)).dumps()`. A `bash` `run` block shows the approval rule: `swarm-arena run examples/rosenbrock/run.toml examples/rosenbrock/plan.toml < /dev/null` must exit with code 2 (checked with `if ...; then exit 1; else test $? -eq 2; fi`). Then `test -z "$(find . -path '*/.swarm-arena/runs' -print -quit)"` checks that no run folder exists anywhere in the work dir, so the check can fail whether the roots resolve against the cwd or against the folder of `run.toml`. |

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_docs.py -v -k "planning or scoring or limits" && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: PASS.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add docs/planning-modes.md docs/scoring-and-review.md docs/limits-and-safety.md tests/test_docs.py
git commit -m "docs: complete planning, scoring, and limits pages" -m "The Limits page lists every Limits field and the overspend and isolation statements that §8 requires the docs to make. The examples cover the planning boundary, judge composition, human review, and the approval exit code."
```

---

### Task 8: "Running on Ray", "The arena", and "Roadmap" pages

**Files:**
- Modify: `docs/running-on-ray.md`, `docs/arena.md`, `docs/roadmap.md`, `tests/test_docs.py` (facts and two tests)

**Interfaces:**
- Consumes: harness (Task 3); `RayExecutor` (R4.10); the M2 CLI commands `leaderboard`, `board search`, and `reindex`.
- Produces: nothing for later tasks.

- [ ] **Step 1: Write the failing tests** (the entries go inside the `REQUIRED_FACTS` literal of Task 3)

```python
REQUIRED_FACTS: dict[str, list[str]] = {
    # entries of earlier tasks stay here
    "Running on Ray": [
        "swarm-arena[ray]", 'ray.init(address="auto")', "swarm-arena submit", "RAY_AUTH_MODE=token",
        "RAY_AUTH_TOKEN", "runtime_env", '{"llm_keys": 1}', "max_retries=0", "s3://", "worker nodes",
        "strict profile", "require_hard_cap", "Blocking outbound traffic from workers to provider hosts",
        "swarm-arena reconcile", "observed (completeness unverified)"],
    "The arena": [
        "swarm-arena[s3]", "runs/<run-id>.json", "reviews/", "curation/", "control/owner.json", "frontier",
        "at most one round", "rounds/round-NN.md", "digest_tokens", "search_learnings", "swarm-arena reindex",
        "swarm-arena board serve", "swarm-arena board search", "swarm-arena board show",
        "swarm-arena learnings pin", "swarm-arena learnings hide", "swarm-arena learnings add",
        "swarm-arena leaderboard", "swarm-arena trace"],
    "Roadmap": [
        "Direct and broadcast messages", "Live sharing with no barrier", "Outside agents joining", "Forum",
        "Ray-sandbox executors", "Pareto ranking", "A database arena backend", "Nested teams",
        "Mid-run release of unused gateway reservations", "DeepRetro connector", "physika"],
}

# tests/test_docs.py (new tests and constant, anywhere in the module)
STUB_TEXT = re.compile(r"written in M\d|under construction|M5 task", re.I)    # Task 9 reuses it for README.md

def test_no_stub_text() -> None:
    # docs/ only: README.md is rewritten in Task 9, which adds test_readme_has_no_stub_text
    files = [p for p in (REPO / "docs").rglob("*.md") if "superpowers" not in p.parts and "_build" not in p.parts]
    bad = [str(p) for p in files if STUB_TEXT.search(p.read_text())]
    assert bad == []

def test_every_content_page_has_a_runnable_block() -> None:
    content = [t for t, _ in PAGES if t not in {"Roadmap", "API reference"}]
    empty = [t for t in content if not any(b.kind in {"none", "requires", "run"} and b.lang in {"python", "bash"}
                                           for b in parse_blocks(page_path(t).read_text()))]
    assert empty == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_docs.py -v -k "ray or arena or roadmap or stub or runnable"`
Expected: FAIL in the three `test_required_facts` cases. `test_no_stub_text` fails while any stub or temporary skip from Task 3 remains.

- [ ] **Step 3: Write the pages**

Extend the existing file. Keep the sections that M1a-M4 wrote, and change them only where a fact or a test needs it. Then add or complete what the table lists.

| Page | Content (spec sources) | Runnable blocks |
|---|---|---|
| Running on Ray (§11 Ray mechanics and secrets, §4.8, §7.3 item 2, §8.1 gateway runs) | Install `swarm-arena[ray]`. The driver as a Ray Job (`ray.init(address="auto")`, `swarm-arena submit`, no Ray Client). Token auth (`RAY_AUTH_MODE=token`, `RAY_AUTH_TOKEN`; on by default for local clusters in 2.59). Worker images: `swarm-arena` plus the connector. VMs as worker nodes. `artifact_root` on `s3://`. Task rules (`max_retries=0`, `max_calls=1`) and cancel. Secrets: never in `runtime_env`, and the custom resource `{"llm_keys": 1}`. The LLM gateway: strict profile, `require_hard_cap`, full reservations, `swarm-arena reconcile` and its labels, and the network rule for provider hosts. | A `requires ray` `python` block builds `RunConfig(executor=RayExecutor(), run_root="s3://bucket/runs", arena_root="s3://bucket/arena", artifact_root="s3://bucket/artifacts")` and prints `dumps()`. No cluster and no I/O are needed (§9 rule 3). The cluster commands are `bash` blocks with no marker. |
| The arena (§4.9, §6.3, §6.3.1, §7.1 to "Control from a laptop", §7.2, §7.3 item 4) | What the arena holds. The `arena_root` layout. One journal per run, and no locks. Frontiers and the delay of at most one round. Scope (reads across the namespace; decisions use this run only). The digest and on-demand reads. Posting rules and the curator. Round pages. The board viewer and the terminal commands. Human control (`learnings pin|hide|add`). `reindex`. `trace`. S3 with `swarm-arena[s3]`. | A `bash` `run` block runs the Rosenbrock example twice (one arena, two runs), then `swarm-arena leaderboard <namespace>`, `swarm-arena board search "<word>" --namespace <namespace>`, and `swarm-arena reindex`, each with `--config examples/rosenbrock/run.toml` (the M2 default is `run.toml` in the cwd). The namespace and the word come from the example's plan. `board serve` is a `bash` block with no marker, because it blocks. |
| Roadmap (§10, §13, §9 rule 1) | The §10 table: each future feature and what v1 already reserves. The follow-up projects (DeepRetro connector, physika). The 1.0 condition. | None (exempt). |

Remove the temporary skips of Task 3 from every page that this task or Tasks 5–7 rewrote.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_docs.py -v && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: PASS for the whole of `test_docs.py`, all 12 `test_page_blocks_run` cases included.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add docs/running-on-ray.md docs/arena.md docs/roadmap.md docs tests/test_docs.py
git commit -m "docs: complete the Ray, arena, and roadmap pages" -m "All 12 §11 pages are now complete. No stub text is left, and every content page has an example that runs."
```

---

### Task 9: README with the pitch, the picture, and an offline quickstart

**Files:**
- Modify: `README.md`, `tests/test_docs.py` (README tests)

**Interfaces:**
- Consumes: harness (`parse_blocks`, `run_blocks`, `make_workdir`, `docs_bindir`, `spec_picture`), Task 3.
- Produces: `test_readme_quickstart` in `tests/test_docs.py`, with the node id `tests/test_docs.py::test_readme_quickstart`. Task 11 and Task 14 run it with `SWARM_ARENA_DOCS_VENV=/tmp/sa-wheel`.

- [ ] **Step 1: Write the failing tests**

```python
README = REPO / "README.md"

def test_readme_quickstart(tmp_path: Path) -> None:
    work = make_workdir(tmp_path, with_examples=False)          # no repo files: it must run from the wheel
    blocks = parse_blocks(README.read_text())
    assert [b for b in blocks if b.kind in {"requires", "skip"}] == []
    run_blocks(blocks, source="README.md", workdir=work, bindir=docs_bindir())
    solutions = list(work.glob(".swarm-arena/runs/*/goals/*/solution.json"))
    assert len(solutions) == 1
    assert "source_head" in json.loads(solutions[0].read_text())

def test_readme_has_picture() -> None:
    assert spec_picture() in README.read_text()

def test_readme_has_no_stub_text() -> None:
    assert STUB_TEXT.search(README.read_text()) is None          # STUB_TEXT from Task 8

def test_readme_sections() -> None:
    assert re.findall(r"^## (.+)$", README.read_text(), re.M) == [
        "Install", "Quickstart (offline)", "Documentation", "Status and versioning", "Contributing", "License"]

def test_readme_links_are_absolute() -> None:
    targets = re.findall(r"\]\(([^)\s]+)\)", README.read_text())
    assert [t for t in targets if not t.startswith(("https://", "#"))] == []    # PyPI breaks relative links

def test_readme_names_every_extra() -> None:
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert [e for e in extras if f"swarm-arena[{e}]" not in README.read_text()] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_docs.py -v -k readme`
Expected: FAIL. The M1b draft has no marked file blocks, so `test_readme_quickstart` fails on the missing `solution.json`. `test_readme_sections` and `test_readme_links_are_absolute` fail on the draft. `test_readme_has_no_stub_text` fails while the draft still holds stub text, such as the M1a "under construction" status line.

- [ ] **Step 3: Write `README.md`**

- H1 `swarm-arena`, then the §1 pitch sentence and the §1 picture (a ```` ```text ```` block, byte-identical to the spec), then the four numbered §1 steps.
- `## Install`: `pip install swarm-arena` and `uv add swarm-arena`, both unmarked `bash` blocks. Then a table of the five extras, each written `swarm-arena[<extra>]` with one line on what it adds. Then "Python 3.11+, Linux or macOS (POSIX)".
- `## Quickstart (offline)`: a Rosenbrock goal that needs no repo files, no key, and no network.
  - Three `file` blocks: `rosen.py` (module-level `gradient_descent`, `random_search`, and `score(goal, candidate) -> Score` using `Score.minimize`), `plan.toml` (2 teams of `ScriptedManager` agents, `ProgramJudge` `rosen:score`, `SuccessiveHalving`, select-top-2 converge), and `run.toml`. `run.toml` keeps the default roots, so the run lands in `.swarm-arena/runs/`, and it sets `[limits] max_rounds = 2`.
  - One `bash` `run` block: `swarm-arena run --yes run.toml plan.toml`.
  - A short note on where `solution.json` and the round pages are.
  - A one-line pointer to Getting started for the full example and for `AgentsSDKManager`.
- `## Documentation`: `https://swarm-arena.readthedocs.io/`.
- `## Status and versioning`: 0.x alpha; the §9 rule in two sentences; a link to the Versioning policy page.
- `## Contributing`: an absolute link to `CONTRIBUTING.md` on GitHub.
- `## License`: MIT, with an absolute link to `LICENSE`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_docs.py -v -k readme`
Expected: PASS. The quickstart finishes well inside the 300 s block timeout.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_docs.py
git commit -m "docs: finish the README with an offline quickstart that a test runs" -m "The quickstart runs in an empty folder with no keys and no network. SWARM_ARENA_DOCS_VENV lets the same test run it from a venv that has only the built wheel. Links are absolute so the PyPI page works."
```

---

### Task 10: Read the Docs build config

**Files:**
- Create: `.readthedocs.yaml`, `tests/test_ci_config.py`
- Modify: `pyproject.toml` and `uv.lock` (`uv add --dev "pyyaml>=6"`), `tasks/research/api-anyio-packaging.md` (§B, one fact)
- Modify: roadmap R5.4 (new item 19, after items 14-18 of the 2026-10-18 roadmap update; use the next free number if more exist: "Test-only dev dependency `pyyaml>=6` parses `.readthedocs.yaml` and the workflow files, M5") and R3.2 (add `tests/test_ci_config.py`)

**Interfaces:**
- Consumes: the `docs` job of `.github/workflows/ci.yml` (R2.0) and `[project.optional-dependencies]`.
- Produces (in `tests/test_ci_config.py`, used by Tasks 11–12):
  - `REPO: Path`
  - `def load_yaml(path: Path) -> dict[str, Any]`, which uses `yaml.safe_load`
  - `def triggers(workflow: dict[str, Any]) -> dict[str, Any]`, which returns `workflow["on"]` if present, else `workflow[True]`. PyYAML reads the bare key `on` as the boolean `True` (YAML 1.1).
  - `def steps_run(job: dict[str, Any]) -> list[str]`: the `run` strings of the job's steps.
  - `def steps_uses(job: dict[str, Any]) -> list[str]`: the `uses` strings of the job's steps.

**[verify] result, checked while planning (2026-10-18, docs.readthedocs.com config file v2).** `python.install` with `method: uv` and `command: sync` accepts `groups` (a list or `all`) and `extras` (a list or `all`). With uv, only one install entry is allowed. The page does not say that `extras: all` equals `--all-extras`, so list the extras by name. The `build.jobs` fallback is not needed.

- [ ] **Step 1: Write the failing tests**

```python
def test_rtd_config() -> None:
    cfg = load_yaml(REPO / ".readthedocs.yaml")
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert cfg["version"] == 2
    assert cfg["build"] == {"os": "ubuntu-24.04", "tools": {"python": "3.13"}}
    [install] = cfg["python"]["install"]
    assert {k: v for k, v in install.items() if k != "extras"} == {"method": "uv", "command": "sync",
                                                                    "groups": ["docs"]}
    assert sorted(install["extras"]) == sorted(extras)          # a new extra cannot be left out of the RTD build
    assert cfg["sphinx"] == {"configuration": "docs/conf.py", "fail_on_warning": True}

def test_rtd_matches_docs_ci_job() -> None:
    job = load_yaml(REPO / ".github" / "workflows" / "ci.yml")["jobs"]["docs"]
    assert any("uv sync --all-extras --group docs" in r for r in steps_run(job))
    assert "3.13" in json.dumps(job)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv add --dev "pyyaml>=6" && uv run pytest tests/test_ci_config.py -v`
Expected: FAIL in `test_rtd_config` with `FileNotFoundError: .readthedocs.yaml`.

- [ ] **Step 3: Write `.readthedocs.yaml`**

```yaml
version: 2
build:
  os: ubuntu-24.04
  tools:
    python: "3.13"
python:
  install:
    - method: uv
      command: sync
      groups: [docs]
      extras: [llm, cache, ray, s3, langfuse]
sphinx:
  configuration: docs/conf.py
  fail_on_warning: true
```

Add the [verify] result above to `tasks/research/api-anyio-packaging.md` §B, under the existing `.readthedocs.yaml` example.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ci_config.py -v`
Expected: PASS.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add .readthedocs.yaml tests/test_ci_config.py pyproject.toml uv.lock tasks/research/api-anyio-packaging.md docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "build: add the Read the Docs config with uv and all extras" -m "RTD runs uv sync with the docs group and every extra by name, as the docs CI job does, so autodoc can import swarm_arena.llm and swarm_arena.ray. PyYAML joins the dev group to parse config files in tests (roadmap R5.4 item 19)."
```

---

### Task 11: Release workflow and `RELEASING.md`

**Files:**
- Create: `.github/workflows/release.yml`, `RELEASING.md`
- Modify: `tests/test_ci_config.py`, `tasks/research/api-anyio-packaging.md` (the artifact action versions)

**Interfaces:**
- Consumes: `load_yaml`, `triggers`, `steps_run`, `steps_uses` (Task 10); `tests/test_docs.py::test_readme_quickstart` and `SWARM_ARENA_DOCS_VENV` (Task 9).
- Produces: the GitHub environment name `pypi` and the workflow file name `release.yml`, which the PyPI trusted publisher must name (`RELEASING.md`, one-time setup).

- [ ] **Step 1: Write the failing tests**

```python
RELEASE = REPO / ".github" / "workflows" / "release.yml"

def test_release_triggers() -> None:
    on = triggers(load_yaml(RELEASE))
    assert on["push"] == {"tags": ["v*"]}
    assert "pull_request" in on
    assert "workflow_dispatch" in on

def test_release_build_job_checks_the_wheel() -> None:
    wf = load_yaml(RELEASE)
    assert wf["permissions"] == {"contents": "read"}
    runs = "\n".join(steps_run(wf["jobs"]["build"]))
    for needed in ("uv build", "uvx twine check --strict dist/*", "uv venv /tmp/sa-wheel --python 3.11",
                   "/tmp/sa-wheel/bin/swarm-arena --help",
                   ("SWARM_ARENA_DOCS_VENV=/tmp/sa-wheel uv run pytest "
                    "tests/test_docs.py::test_readme_quickstart")):
        assert needed in runs
    assert "actions/upload-artifact@v7" in steps_uses(wf["jobs"]["build"])

def test_release_checks_tag_against_version() -> None:
    [step] = [s for s in load_yaml(RELEASE)["jobs"]["build"]["steps"] if "uv version --short" in s.get("run", "")]
    assert step["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert "CHANGELOG.md" in step["run"]

def test_publish_job_is_least_privilege() -> None:
    wf = load_yaml(RELEASE)
    job = wf["jobs"]["publish"]
    assert job["needs"] == "build"
    assert job["if"] == "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    assert job["environment"] == {"name": "pypi", "url": "https://pypi.org/p/swarm-arena"}
    assert job["permissions"] == {"id-token": "write"}
    assert steps_uses(job) == ["actions/download-artifact@v8", "pypa/gh-action-pypi-publish@release/v1"]
    assert "id-token" not in json.dumps(wf["jobs"]["build"])
    assert "secrets." not in RELEASE.read_text()           # trusted publishing needs no token

def test_releasing_checklist() -> None:
    text = (REPO / "RELEASING.md").read_text().lower()                 # "Yank it" counts as "yank"
    for phrase in ("trusted publisher", "release.yml", "environment `pypi`", "uv version", "CHANGELOG.md",
                   "uv build", "twine check", "git tag -a v", "git push origin v", "go-ahead", "yank",
                   "required reviewer", "v*"):
        assert phrase.lower() in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ci_config.py -v -k "release or publish"`
Expected: FAIL with `FileNotFoundError` for `release.yml`.

- [ ] **Step 3: Write the workflow and the checklist**

```yaml
name: release
on:
  push:
    tags: ["v*"]
  pull_request:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10
      - name: Tag matches version and changelog
        if: startsWith(github.ref, 'refs/tags/v')
        run: |
          version="$(uv version --short)"
          test "${GITHUB_REF_NAME}" = "v${version}"
          grep -q "^## \[${version}\] - " CHANGELOG.md
      - run: uv build
      - run: uvx twine check --strict dist/*
      - name: Smoke-test the wheel in a clean venv
        run: |
          uv venv /tmp/sa-wheel --python 3.11
          uv pip install --python /tmp/sa-wheel dist/*.whl
          /tmp/sa-wheel/bin/python -c "import swarm_arena; print(swarm_arena.__version__)"
          /tmp/sa-wheel/bin/swarm-arena --help
      - name: README quickstart from the wheel
        run: SWARM_ARENA_DOCS_VENV=/tmp/sa-wheel uv run pytest tests/test_docs.py::test_readme_quickstart -v
      - uses: actions/upload-artifact@v7
        with: {name: dist, path: dist/}
  publish:
    needs: build
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-24.04
    environment: {name: pypi, url: "https://pypi.org/p/swarm-arena"}
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v8
        with: {name: dist, path: dist/}
      - uses: pypa/gh-action-pypi-publish@release/v1
```

`RELEASING.md` is a numbered checklist with these sections:
1. **One-time setup.**
   - On PyPI, add a trusted publisher for project `swarm-arena`: owner `shreyasvinaya`, repository `swarm-arena`, workflow `release.yml`, environment `pypi`. The name was reserved with `0.0.0a1` (D14).
   - On GitHub, create the environment `pypi`. Set the owner as required reviewer. Under Deployment branches and tags, select "Selected branches and tags" and add only the tag rule `v*`.
   - On Read the Docs, import the repository as project `swarm-arena`.
   - Check that `[project.urls]` matches the real repository URL.
2. **Prepare.**
   - CI and the weekly upstream job are green on `main`.
   - Run `uv version X.Y.Z` and `uv lock`.
   - In `CHANGELOG.md`, move the entries under `## [Unreleased]` into a new `## [X.Y.Z] - YYYY-MM-DD` section directly below it, using the tag date. Keep the `## [Unreleased]` heading, now empty, as the first `## [` heading. Update both link references: `[Unreleased]: https://github.com/shreyasvinaya/swarm-arena/compare/vX.Y.Z...HEAD`, and a new `[X.Y.Z]: https://github.com/shreyasvinaya/swarm-arena/releases/tag/vX.Y.Z`. This is the layout that `test_changelog_format` requires, so the common gate below passes on the commit that gets the tag.
   - Run the local checks: the common gate (R2.0); `uv build`; `uv run --with twine twine check dist/*`; the clean-venv smoke test; and `SWARM_ARENA_DOCS_VENV=/tmp/sa-wheel uv run pytest tests/test_docs.py::test_readme_quickstart -v`.
   - Commit with the message `release: X.Y.Z`.
3. **Tag (only with the owner's go-ahead).**
   - Check Settings → Environments → `pypi`: the required reviewer and the `v*` tag rule are set. A workflow that names a missing environment creates it with no protection.
   - Run `git tag -a vX.Y.Z -m "X.Y.Z"` and `git push origin vX.Y.Z`. Approve the `pypi` environment run.
4. **After.**
   - Check the PyPI page, the RTD build of the tag, and `pip install swarm-arena==X.Y.Z` in a clean venv.
5. **A bad release.** Yank it on PyPI (a version number can never be reused), then ship X.Y.(Z+1).
6. **Deprecations.** Follow the Versioning policy page (§9 rule 1).

Add the artifact action versions (`upload-artifact` v7, `download-artifact` v8, checked 2026-10-18) to `tasks/research/api-anyio-packaging.md`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ci_config.py -v`
Expected: PASS.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/release.yml RELEASING.md tests/test_ci_config.py tasks/research/api-anyio-packaging.md
git commit -m "ci: add the release workflow with PyPI trusted publishing" -m "Every PR builds and checks the wheel and runs the README quickstart from it. A v* tag also checks that the tag, pyproject, and CHANGELOG agree, then publishes through the pypi environment with id-token: write on the publish job only. RELEASING.md is the checklist; the tag push needs the owner's go-ahead."
```

---

### Task 12: Weekly upstream workflow

**Files:**
- Create: `.github/workflows/upstream.yml`, `upstream-overrides.txt`
- Modify: `tests/test_ci_config.py`, `CONTRIBUTING.md` (new section `## Upstream bounds`)

**Interfaces:**
- Consumes: `load_yaml`, `triggers`, `steps_run` (Task 10); the `llm` and `cache` extras in `pyproject.toml`; `tests/llm` (M3).
- Produces: nothing for later tasks.

- [ ] **Step 1: Write the failing tests**

```python
UPSTREAM = REPO / ".github" / "workflows" / "upstream.yml"

def test_overrides_lift_only_upper_bounds() -> None:
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    reqs = [parse_req(r) for r in extras["llm"] + extras["cache"]]     # (name, extras set, specifier set)
    expected = [override_line("openai-agents", reqs), override_line("litellm", reqs)]
    lines = [ln.strip() for ln in (REPO / "upstream-overrides.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert lines == expected
```

`parse_req(text: str) -> tuple[str, frozenset[str], frozenset[str]]` splits `name[extra,...]spec,spec`. `override_line(name, reqs) -> str` builds `name[<sorted union of extras>]` followed by the comma-joined specifiers with every `<`, `<=`, and `==` removed. It keeps `>=` and `!=`, sorted. With the M3 pins, the expected lines are `openai-agents[litellm]>=0.24` and `litellm[caching,proxy]>=1.105.1`.

```python
def test_upstream_workflow() -> None:
    wf = load_yaml(UPSTREAM)
    on = triggers(wf)
    assert on["schedule"] == [{"cron": "0 6 * * 1"}]
    assert "workflow_dispatch" in on
    [job] = wf["jobs"].values()
    runs = steps_run(job)
    assert "uv venv --python 3.13" in runs
    assert any("uv pip install" in r and "--override upstream-overrides.txt" in r
               and '-e ".[llm,cache,langfuse]"' in r and "--group dev" in r for r in runs)
    assert ".venv/bin/python -m pytest tests/llm -v" in runs

def test_upstream_never_resyncs_to_lock() -> None:
    assert not any("uv run" in r or "uv sync" in r for r in steps_run(next(iter(load_yaml(UPSTREAM)["jobs"].values()))))

def test_contributing_states_the_bound_rule() -> None:
    text = (REPO / "CONTRIBUTING.md").read_text()
    assert "## Upstream bounds" in text
    assert "upstream-overrides.txt" in text
    assert "!=" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_ci_config.py -v -k "upstream or overrides or contributing"`
Expected: FAIL with `FileNotFoundError` for `upstream-overrides.txt` and `upstream.yml`.

- [ ] **Step 3: Write the workflow, the overrides, and the rule**

```yaml
name: upstream
on:
  schedule:
    - cron: "0 6 * * 1"
  workflow_dispatch:
permissions:
  contents: read
jobs:
  newest-llm-deps:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v10
      - run: uv venv --python 3.13
      - run: uv pip install -e ".[llm,cache,langfuse]" --group dev --override upstream-overrides.txt
      - run: uv pip show openai-agents litellm
      - run: .venv/bin/python -m pytest tests/llm -v
```

`uv run` and `uv sync` are not used here, because they sync the environment back to `uv.lock` and would test the locked versions instead of the newest ones.

`upstream-overrides.txt` holds a comment line naming its purpose and the test that guards it, then the two expected lines.

The `## Upstream bounds` section of `CONTRIBUTING.md` says:
- The weekly job tests the newest `openai-agents` and `litellm`.
- An upper bound in `pyproject.toml` is raised only after that job passes on the newer release (§11).
- A release found bad gets a `!=` pin.
- `upstream-overrides.txt` follows the lower bounds of `pyproject.toml`, and `test_overrides_lift_only_upper_bounds` enforces this.
- A failed scheduled run emails the person who last changed the workflow.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_ci_config.py -v`
Expected: PASS.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/upstream.yml upstream-overrides.txt tests/test_ci_config.py CONTRIBUTING.md
git commit -m "ci: test the llm extra weekly against the newest openai-agents and litellm" -m "An override file lifts only the upper bounds, and a test keeps it in step with pyproject. The job never calls uv run or uv sync, which would restore the locked versions. CONTRIBUTING states when a bound may be raised (§11)."
```

---

### Task 13: Version 0.1.0, package metadata, and the 0.1.0 changelog

**Files:**
- Create: `tests/test_packaging.py`
- Modify: `pyproject.toml` (`[project]` metadata and `version`), `uv.lock`, `CHANGELOG.md`
- Modify: roadmap R3.2 (add `tests/test_packaging.py`: "wheel contents and metadata, version agreement, changelog format, M5")

**Interfaces:**
- Consumes: `swarm_arena.__version__` (R3.1 `_version.py`), `docs/conf.py` `release` (Task 3), the `uv` binary.
- Produces: `dist/swarm_arena-0.1.0-py3-none-any.whl`, which Task 14 uses.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    out = tmp_path_factory.mktemp("dist")
    subprocess.run([uv, "build", "--wheel", "--out-dir", str(out)], cwd=REPO, check=True)
    [path] = out.glob("swarm_arena-*-py3-none-any.whl")
    return zipfile.ZipFile(path)

def metadata(wheel: zipfile.ZipFile) -> Message:
    [name] = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
    return email.parser.Parser().parsestr(wheel.read(name).decode())

def test_wheel_files(wheel: zipfile.ZipFile) -> None:
    names = set(wheel.namelist())
    for needed in ("swarm_arena/__init__.py", "swarm_arena/py.typed", "swarm_arena/_launch.py",
                   "swarm_arena/run_function.py", "swarm_arena/llm/__init__.py", "swarm_arena/ray/__init__.py",
                   "swarm_arena/testing/__init__.py"):
        assert needed in names
    assert [n for n in names if n.split("/")[0] in {"tests", "docs", "examples", "tasks"}] == []

def test_wheel_metadata(wheel: zipfile.ZipFile) -> None:
    md = metadata(wheel)
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert (md["Name"], md["Version"], md["Requires-Python"]) == ("swarm-arena", project["version"], ">=3.11")
    assert md["License-Expression"] == "MIT"
    assert md["Description-Content-Type"] == "text/markdown"
    assert md["Author"] == "Shreyas Vinaya"
    assert md["Author-email"] is None                                          # no email published
    assert set(md.get_all("Provides-Extra", [])) == {"llm", "cache", "ray", "s3", "langfuse"}
    urls = dict(u.split(", ", 1) for u in md.get_all("Project-URL", []))
    assert urls["Documentation"] == "https://swarm-arena.readthedocs.io/"
    assert urls["Repository"] == "https://github.com/shreyasvinaya/swarm-arena"
    classifiers = md.get_all("Classifier", [])
    assert {"Operating System :: POSIX", "Typing :: Typed", "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3.11", "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13"} <= set(classifiers)
    assert [c for c in classifiers if c.startswith("License ::")] == []        # PEP 639: expression only

def test_core_requirements_are_exactly_three(wheel: zipfile.ZipFile) -> None:
    core: dict[str, set[str]] = {}
    for req in metadata(wheel).get_all("Requires-Dist", []):
        if "extra ==" not in req:
            m = re.match(r"([A-Za-z0-9._-]+)\s*(.*)", req.split(";")[0])
            assert m is not None, req
            name, spec = m.groups()
            core[name.lower()] = {s.strip() for s in spec.split(",") if s.strip()}
    assert core == {"anyio": {">=4.15", "<5"}, "pydantic": {">=2.11", "<3"}, "typing-extensions": {">=4.12"}}

def test_entry_point(wheel: zipfile.ZipFile) -> None:
    [name] = [n for n in wheel.namelist() if n.endswith(".dist-info/entry_points.txt")]
    assert "swarm-arena = swarm_arena._cli.main:main" in wheel.read(name).decode()

def test_version_matches_everywhere() -> None:
    version = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    released = re.findall(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", (REPO / "CHANGELOG.md").read_text(), re.M)
    assert swarm_arena.__version__ == version == released[0] == runpy.run_path(str(REPO / "docs/conf.py"))["release"]

def test_changelog_format() -> None:
    text = (REPO / "CHANGELOG.md").read_text()
    heads = re.findall(r"^## \[([^\]]+)\]", text, re.M)
    assert heads[0] == "Unreleased"
    assert re.fullmatch(r"\d+\.\d+\.\d+", heads[1])
    base = "https://github.com/shreyasvinaya/swarm-arena"
    assert f"[Unreleased]: {base}/compare/v{heads[1]}...HEAD" in text
    assert [v for v in heads[1:] if f"[{v}]: {base}/releases/tag/v{v}" not in text] == []
    section = text.split(f"## [{heads[1]}]", 1)[1].split("\n## [", 1)[0]
    assert "### Added" in section
    assert re.search(r"^- ", section, re.M)

def test_license_is_mit() -> None:
    text = (REPO / "LICENSE").read_text()
    assert "Permission is hereby granted, free of charge" in text      # M1a pins only "MIT, Copyright (c) 2026 ..."
    assert "Shreyas Vinaya" in text
```

The literal `0.1.0` is checked by the Task 14 exit commands, not here, so these tests stay valid for later releases.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_packaging.py -v`
Expected: FAIL. `test_version_matches_everywhere` fails, because there is no released heading yet. `test_wheel_metadata` fails on `License-Expression`, `Author`, or `Project-URL`, wherever M1a's minimal metadata lacks them.

- [ ] **Step 3: Set the metadata, the version, and the changelog**

- `[project]` in `pyproject.toml`:
  - `version = "0.1.0"`, set with `uv version 0.1.0`, then `uv lock`
  - `description = "Run a swarm of agents on hard goals: teams, pruning with reasons, and answers you can trace."`
  - `readme = "README.md"`, `license = "MIT"`, `license-files = ["LICENSE"]`, `authors = [{ name = "Shreyas Vinaya" }]`
  - `keywords = ["agents", "multi-agent", "swarm", "llm", "optimization", "research"]`
  - `classifiers`: `Development Status :: 3 - Alpha`, `Framework :: AsyncIO`, `Intended Audience :: Developers`, `Intended Audience :: Science/Research`, `Operating System :: POSIX`, `Programming Language :: Python :: 3 :: Only`, `Programming Language :: Python :: 3.11`, `Programming Language :: Python :: 3.12`, `Programming Language :: Python :: 3.13`, `Topic :: Scientific/Engineering :: Artificial Intelligence`, `Typing :: Typed`
- `[project.urls]`:
  - `Homepage` and `Repository`: `https://github.com/shreyasvinaya/swarm-arena`
  - `Documentation`: `https://swarm-arena.readthedocs.io/`
  - `Changelog`: `https://github.com/shreyasvinaya/swarm-arena/blob/main/CHANGELOG.md`
  - `Issues`: `https://github.com/shreyasvinaya/swarm-arena/issues`
- `CHANGELOG.md`:
  - Keep the Keep a Changelog header.
  - Add an empty `## [Unreleased]`.
  - Add `## [0.1.0] - <today's date, YYYY-MM-DD>` with `### Added`. Merge the M1a–M4 `Unreleased` entries into one bullet per area: data models and TOML plans; tools (`ScriptTool`, `FunctionTool`) and the workspace contract; `LocalExecutor`; checks and judges; the engine (rounds, policies, parts, converge, the final judge, stop and resume); the run journal and lease; the budget ledger; the arena and board; human review; the CLI; the `llm` extra; the `ray` extra and the LLM gateway; the conformance kit; the docs.
  - Add the link references that `test_changelog_format` expects.
  - The date is set again at tag time (`RELEASING.md` step 2).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_packaging.py tests/test_docs.py::test_conf_settings -v && uv lock --check`
Expected: PASS. The lock is up to date.
Then: ruff check, ruff format --check, ty check → no errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock CHANGELOG.md tests/test_packaging.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "release: set version 0.1.0 with full package metadata and changelog" -m "The wheel must hold the entry modules and py.typed, declare exactly the three core requirements and five extras, use a PEP 639 license expression, and publish no email. Version, __version__, the newest CHANGELOG heading, and the docs release must agree. Roadmap R3.2 lists tests/test_packaging.py."
```

---

### Task 14: Milestone exit

**Files:** none changed, unless a check fails. A fix goes in its own commit, with the failing command named in the body.

**Interfaces:**
- Consumes: everything above; the exit criteria of roadmap R2 M5 and the common gate R2.0.
- Produces: branch `m5-release` with every exit check green, and a PR. No tag.

- [ ] **Step 1: Clean build outputs**

```bash
rm -rf dist /tmp/sa-wheel docs/_build
```

- [ ] **Step 2: Common gate (R2.0)**

```bash
uv sync --all-extras --all-groups
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv lock --check
```

Expected: every command exits 0. pytest reports no failures, and no skips except the deselected `ray`, `s3`, `llm_live`, and `gateway_live` markers. Sphinx gives no warnings.

- [ ] **Step 3: M5 exit criteria (R2 M5)**

```bash
uv run pytest tests/test_public_api.py tests/test_docstrings.py -v
uv build
uv run --with twine twine check dist/*
uv venv /tmp/sa-wheel && uv pip install --python /tmp/sa-wheel dist/swarm_arena-0.1.0-py3-none-any.whl
/tmp/sa-wheel/bin/python -c "import swarm_arena; print(swarm_arena.__version__)"
/tmp/sa-wheel/bin/swarm-arena --help
SWARM_ARENA_DOCS_VENV=/tmp/sa-wheel uv run pytest tests/test_docs.py::test_readme_quickstart -v
```

Expected:
- `test_public_api.py` and `test_docstrings.py` PASS with no skips.
- `uv build` writes `dist/swarm_arena-0.1.0.tar.gz` and `dist/swarm_arena-0.1.0-py3-none-any.whl`.
- twine prints `PASSED` for both files.
- The import prints exactly `0.1.0`.
- `--help` exits 0 and lists the §14 commands.
- The README quickstart PASSES against the wheel-only venv, which proves it runs offline from the published artifact.

- [ ] **Step 4: Earlier milestones' exit tests still pass**

```bash
uv run pytest tests/e2e -v
uv run pytest -m "s3 and not ray" -v
uv run pytest -m "ray and not s3" -v
uv run pytest tests/ray/test_executor.py -m "ray and s3" -v     # optional: skips unless SWARM_ARENA_TEST_S3_ENDPOINT is set
```

Expected: PASS. The e2e tests prove that the M5 fixes changed no behavior. These are M4's commands (M4 Tasks 4 and 7, roadmap M4 exit criteria): a command-line `-m` replaces the `addopts` expression, so plain `-m s3` would start a Ray cluster for the one `ray` and `s3` test, and plain `-m ray` would run that test without an S3 endpoint. `-m "s3 and not ray"` uses moto or MinIO, as M2 chose. `-m "ray and not s3"` uses the local Ray session fixture. The `ray and s3` test needs `SWARM_ARENA_TEST_S3_ENDPOINT` set before the session.

- [ ] **Step 5: Push and open the PR (no tag)**

```bash
git push -u origin m5-release
gh pr create --base main --head m5-release --title "M5: release 0.1.0" --body "$(cat <<'EOF'
Completes milestone M5 of the roadmap: public API frozen to spec §9, all 12 docs pages and the README with tested examples, compatibility tests for §9 rules 2, 3, and 5, Read the Docs config, release and weekly upstream workflows, and package metadata for 0.1.0.

Exit criteria (roadmap R2 M5) are green, including the README quickstart run from the built wheel.

Not done here: the v0.1.0 tag push. It publishes to PyPI and cannot be undone, so it waits for the owner's go-ahead (RELEASING.md step 3).
EOF
)"
```

Expected: the PR is open, and the `ci` and `release` (build job) checks pass on it. Do not merge, and do not tag. Both wait for the owner.
