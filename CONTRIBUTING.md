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
2. Add it to `PUBLIC_ALL` in `tests/_surface.py` (the `__all__` snapshot, in spec §9 order).
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


## Arena implementation notes

Board preparation, curation reads, and page storage use abandonable daemon workers
under the owning run's supervision. Normalize connector-returned data inside the
worker before the loop consumes it. An abandoned wait must never start a later
publication. Shared outcome containment retains caller cancellation and registered
core controls; untrusted failures are reconstructed with type-only diagnostics.

Round-page rendering produces the Markdown body. `ViewWriter` adds one source-head
comment. Live closure probes only the newly closed round; resume checks every
closed round for a missing page. Page creation is conditional and preserves any
existing bytes.

Score batch IDs include the barrier or final phase and their durable recovery key.
This keeps ordinary and final judging of the same candidate distinct during replay.

## CI timing and isolation

CI uses separate pytest-xdist worker processes inside pytest-split shards.
Branch pushes use 20 jobs: core (two Python versions × two shards), extras
(two versions × three shards), lowest-direct core (three shards), lowest-direct
with all extras (two shards), macOS (four shards), and docs. Both lowest-direct
configurations use Python 3.11 and preserve their resolved environment with
`uv run --no-sync`. Main also tests core Python 3.12 and macOS Python 3.11;
two shards per macOS version keep its total at four jobs. Keep
filesystem writes under `tmp_path`, bind test servers to port 0, and reap children
in fixture teardown or `finally`. Collection must produce identical node IDs in
every worker; inventories built from weak subclass registries must discard
unreachable classes and use stable ordering. Process-local monkeypatches do not
require serial execution. Tests that assert authoritative LiteLLM accounting
after `acompletion` must flush its background logging worker before draining
the scope; completion itself does not await the callback. Tests of timeout
accounting must establish call admission with an event before expiring the real
round timer; short startup deadlines can expire before any call exists. Unknown-write
renewal tests must observe takeover explicitly and leave enough lease lifetime
for CI scheduling noise. The moto development
floor includes conditional S3 `IfMatch` support, required by store CAS tests.

The committed `.test_durations` balances shards with `least_duration`.
Each CI shard refreshes its measured subset using `--store-durations
--clean-durations --durations-path test-durations.json` and uploads a uniquely
named `durations-*` artifact. Refresh after substantial test changes from a
completed green CI run (do not run the full suite on the development laptop):

```bash
uv run gh run download RUN_ID -R shreyasvinaya/swarm-arena \
  --pattern 'durations-*' --dir /tmp/swarm-ci-durations
uv run python - <<'PYTHON'
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

samples = defaultdict(list)
for artifact in Path("/tmp/swarm-ci-durations").glob("*/test-durations.json"):
    for nodeid, seconds in json.loads(artifact.read_text()).items():
        samples[nodeid].append(seconds)
assert samples, "No duration artifacts found"
Path(".test_durations").write_text(
    json.dumps({key: round(median(values), 6) for key, values in samples.items()},
               indent=2, sort_keys=True) + "\n"
)
PYTHON
```

Use an empty download directory for each refresh. Review and commit the resulting
file. Durations are scheduling hints, never test timeouts or grounds for skipping
a test. New tests without measurements receive pytest-split's mean estimate.

## Upstream bounds

The weekly `upstream` workflow tests the newest resolvable `openai-agents` and
`litellm` every Monday at 06:00 UTC, and can also be dispatched manually. Raise an
upper bound in `pyproject.toml` only after this job passes on the newer release
(design §11). A release found bad gets a `!=` pin.

`upstream-overrides.txt` follows the lower bounds of `pyproject.toml`, unions the
`llm` and `cache` extras, and retains every exclusion. Only `<`, `<=`, and `==`
constraints are lifted. `test_overrides_lift_only_upper_bounds` enforces this;
production requirements and `uv.lock` stay unchanged by a canary run.

The job creates a Python 3.13 environment, installs with `uv pip install` and the
overrides, and reports the installed versions. It then calls `.venv/bin/python`
directly: `uv run` or `uv sync` would restore the locked environment. The test-only
`SWARM_ARENA_UPSTREAM_CANARY=upstream-installed.json` setting selects the lifted
version policy only for that pytest step. The record is written immediately
after installation and must match the test interpreter's exact installed
versions, so a later resync fails the check. If the newest versions are still
within the production bounds, that is a valid canary too. Normal tests enforce
the production pins; both modes retain all retry, fallback, inventory, and
unrelated dependency checks.

With GitHub Actions email notifications enabled, a failed scheduled run emails
the person who last changed its cron schedule (initially the workflow creator).
Re-enabling a disabled schedule transfers notifications to the person who
re-enabled it. See [GitHub's notification rules](https://docs.github.com/en/actions/concepts/workflows-and-actions/notifications-for-workflow-runs).
