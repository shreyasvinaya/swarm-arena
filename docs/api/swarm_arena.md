# swarm_arena

```{eval-rst}
.. automodule:: swarm_arena
   :members:
```

## Function tools

`FunctionTool` wraps an importable, synchronous Python function. Put the function
in a module, for example `my_tools.py`:

```python
def add(a: int, b: int = 1) -> dict[str, int]:
    """Add two numbers."""
    return {"sum": a + b}
```

Construct the tool with the function, its import path, or a `FunctionTool.Config`:

```python
from swarm_arena import FunctionTool
from my_tools import add

tool = FunctionTool(add, timeout_s=30, grace_s=2)
tool.preflight()
inputs = tool.get_input_model().model_validate({"a": 2})
assert inputs.model_dump() == {"a": 2, "b": 1}
assert FunctionTool("my_tools:add").config.function == "my_tools:add"
```

Construction does not import the referenced module. Preflight validates the
annotated parameter signature. Positional-or-keyword and keyword-only parameters
are supported; positional-only parameters, `*args`, `**kwargs`, missing parameter
annotations, and coroutine functions are rejected. Extra request parameters are
rejected. The first docstring line supplies the description, and the return value
must be a dictionary accepted by the finite JSON encoder.

`LocalExecutor` runs the generated command through its launcher:
`python -P -m swarm_arena.run_function --import-root <dir> <function>`.
The runner validates `params` from `swarm/request.json` and atomically publishes
`{"schema_version": 1, "data": ...}` to `swarm/result.json`. Failures exit with
status 1 and sanitized diagnostics. The executor collects the standard result
envelope and supervises cancellation and timeouts as for other tools.

Install `swarm-arena` in the interpreter's own environment: the launcher uses
`python -I`. The function command uses `-P`, which suppresses only the implicit
cwd/script-directory import entry; explicit `PYTHONPATH` entries still apply.
The function's import root travels in argv, so no `Job.env` entry is required.
