# Contributing to swarm-arena

## Set up

Install [uv](https://docs.astral.sh/uv/). Then, in the repo root:

```bash
uv sync --all-extras --all-groups
```

Use uv for everything: `uv add` for a dependency, `uv add --dev` for a tool, and `uv run` to run a
tool. `pyproject.toml` is the single source of truth, and `uv.lock` is committed.

## The task gate

Every change must pass all of these commands (each must exit 0):

```bash
uv sync --all-extras --all-groups
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Test output must be clean: no failures, no errors, and no warnings.

## Test first (TDD)

1. Write a failing test.
2. Run it, and make sure that it fails for the reason you expect.
3. Write the code.
4. Run the test again, and make sure that it passes.

Tests mirror `src/swarm_arena/`: the module `src/swarm_arena/<path>/<name>.py` is tested in
`tests/<path>/test_<name>.py` (drop a leading underscore from the file name only). Every function
and method is fully type-annotated, tests included.

## Docstrings

Use numpydoc style. Every public object (each name in a public `__all__`, its public methods and
properties, and its public nested classes) has a docstring with an `Examples` section. The examples
run as doctests. An example that needs a subprocess or files uses `# doctest: +SKIP`, and a test
covers it instead. `tests/test_docstrings.py` checks every public docstring with `numpydoc.validate`.

## Public modules and the snapshot test

The public modules (`swarm_arena`, `swarm_arena.tool`, `.scoring`, `.manager`, `.planning`,
`.board`, `.executor`, `.config`, `.protocol`, `.testing`) are facades. A facade holds only a
docstring, imports, `__all__`, and one `_publish(__name__, __all__)` call. All code lives in private
modules (a leading underscore), and a private module never imports from a facade.

When you add a public name:

1. Add it to the facade's `__all__`.
2. Add it to `PUBLIC` in `tests/test_public_api.py` (the `__all__` snapshot test).
3. Add one line to `CHANGELOG.md` under `## [Unreleased]`.

## Tools never start with `uv run`

Never start a tool program with `uv run`. uv runs the program as a child process, and it cannot
forward SIGKILL, so a killed run can leave the tool running. Call the Python of the environment
directly.

## Commits

- One commit per completed task.
- The subject line is in the imperative mood (for example "add the store").
- The body says what changed and why.
- No AI co-author lines and no "generated with" text in commits or pull request descriptions.
