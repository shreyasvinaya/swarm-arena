# Versioning policy

## Public surface

Everything in the `__all__` of the top level and these public modules is public:

| Module | Purpose |
| --- | --- |
| `swarm_arena` | Plans, run configuration, built-in components, and running a swarm |
| `swarm_arena.tool` | Tool contracts, requests, outputs, and file references |
| `swarm_arena.scoring` | Checks, judges, combined scoring, and human review |
| `swarm_arena.manager` | Manager contracts and round reports |
| `swarm_arena.planning` | Planners, policies, history, and decisions |
| `swarm_arena.board` | Arena storage, board views, candidates, and learnings |
| `swarm_arena.executor` | Executor contracts, local execution, and jobs |
| `swarm_arena.config` | Model catalogs and optional run settings |
| `swarm_arena.llm` | Optional LLM managers, planners, judges, and curators |
| `swarm_arena.ray` | Optional Ray execution |
| `swarm_arena.protocol` | Messages, parts, artifacts, and task states |
| `swarm_arena.testing` | Connector conformance checks and offline test helpers |

Everything else is private. Import through these modules rather than relying on
implementation paths. The [API reference](api/index.md) lists their exports.
Each module has an `__all__` snapshot test. In 0.x, public names can change only
after the deprecation period below.

## Version numbers

Follow SemVer. In 0.x, breaking changes bump the minor version and are listed in
`CHANGELOG.md`. Release 1.0 after both connectors have run real campaigns.

For upstream dependencies, an upper bound is raised only after the weekly
upstream job passes on the newer release. A passing job is evidence for reviewing
the bound; it does not itself change the supported dependency range.

## Deprecations

Before removing a public name, ship one minor release with a
`DeprecationWarning`. Use `typing_extensions.deprecated` so callers and type
checkers can see the replacement and planned removal. For example, a name
deprecated in 0.2 can be removed in 0.3:

```python
import warnings

from typing_extensions import deprecated


def new() -> str:
    return "result"


@deprecated("old() is replaced by new(); removed in 0.3", category=DeprecationWarning)
def old() -> str:
    return new()


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert old() == "result"

assert len(caught) == 1
assert caught[0].category is DeprecationWarning
```

This is an illustrative deprecation, not an additional package API.

## Stored formats

Every stored format has a `schema_version`. Older versions load through
migration functions. Newer versions are refused with a clear error identifying
`schema_version`; public loaders omit the rejected input value from diagnostics.
This applies to plan and run configuration files as well as stored records. See
the [board API](api/board.md) for the current record contracts.

## Plug-in classes

ABCs grow only by optional methods with defaults. Adding a required abstract
method would break existing connector subclasses, so the abstract method sets
are frozen by compatibility tests.

A component's `__init__` does no I/O. It records and validates configuration;
side effects such as uploads, network checks, and model loading belong in
`preflight()`. Construction must not open files, create folders, connect sockets,
or start subprocesses. This keeps component construction usable during plan
validation and on workers. See [Writing a connector](writing-a-connector.md).

## Conformance kit

Connectors run `check_tool`, `check_check`, `check_judge`, and `check_manager`
from `swarm_arena.testing` in their CI. Its scripted `FakeModel` and `FakeTurn`
let managers and judges run offline with no API key. These lazy fake exports
require the `llm` extra, including both the Agents SDK and LiteLLM; importing the
core conformance kit does not load those dependencies.

## Extras

Extras keep vendor churn out of the core. Optional integrations are installed
through the `llm`, `cache`, `ray`, `s3`, and `langfuse` extras; plain run
configuration models remain available without them. See
[Getting started](getting-started.md) for installation.

The LLM and Ray entry points report a missing extra with a
`ModuleNotFoundError` that names `swarm-arena[llm]` or `swarm-arena[ray]` and
includes the install command. Distribution metadata checks distinguish an
installed dependency from a same-named namespace directory. Only an exactly
missing dependency is translated: a missing vendor submodule, an incompatible
import, or an internal import bug propagates unchanged. Such failures are not
evidence that reinstalling an already present extra is the right remedy.

## How this is checked

- `tests/test_public_api.py` has a snapshot test for each public module's
  `__all__`, and checks that public objects report their public import paths.
- `tests/test_docstrings.py` validates public docstrings and their examples.
- `tests/test_compat.py` freezes every public ABC's abstract methods, guards
  component construction against I/O, and tests newer stored-format rejection.
- `tests/test_import_isolation.py` checks core import isolation and distinguishes
  missing extras from broken installed dependencies.
- `tests/test_docs_versioning.py` checks this policy's facts and public modules,
  and executes the deprecation example above.

## Python compatibility

The package requires Python 3.11 or newer. CI tests Python 3.11, 3.12, and 3.13.
Public classes report their facade module as their import path. Publishing clears
the original class line metadata on Python 3.13, so source inspection does not
mistake lines in that facade for the class definition. Docstring validation still
checks every public object.
