"""Preserve bounded core collection diagnostics across output detachment."""

from collections.abc import Sequence
from types import MappingProxyType
from typing import Final

from swarm_arena._model.detach import _detach_model
from swarm_arena._model.tooling import ToolOutput

COLLECTION_DIAGNOSTICS: Final = MappingProxyType(
    {
        "llm_calls_unreadable": "swarm/llm_calls.jsonl could not be read; not collected",
        "llm_calls_symlink": "swarm/llm_calls.jsonl has a symlink component; not collected",
        "llm_calls_special_file": "swarm/llm_calls.jsonl is not a regular file; not collected",
        "llm_calls_changed": "swarm/llm_calls.jsonl changed while being read; not collected",
        "llm_calls_hash_limit": (
            "swarm/llm_calls.jsonl is over the 1 GiB full-hash limit; not collected"
        ),
    }
)
MAX_COLLECTION_DIAGNOSTICS: Final = 32


def _diagnostics(values: Sequence[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
    """Validate transport pairs and retain only registered core diagnostic codes."""
    if len(values) > MAX_COLLECTION_DIAGNOSTICS:
        raise ValueError("invalid collection diagnostics")
    counts: dict[str, int] = {}
    for pair in values:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("invalid collection diagnostics")
        code, count = pair
        if type(code) is not str or type(count) is not int or not 1 <= count <= 10_000:
            raise ValueError("invalid collection diagnostics")
        if code in COLLECTION_DIAGNOSTICS:
            counts[code] = min(10_000, counts.get(code, 0) + count)
    return tuple(counts.items())


def with_collection_diagnostics(
    output: ToolOutput, diagnostics: Sequence[tuple[str, int]]
) -> ToolOutput:
    """Copy an output with validated private diagnostics, excluding unknown codes.

    Parameters
    ----------
    output : ToolOutput
        Output whose ordinary fields are retained.
    diagnostics : Sequence[tuple[str, int]]
        At most 32 code/count pairs; counts are exact integers in 1..10000.

    Returns
    -------
    ToolOutput
        Copy whose private carrier is absent from JSON and repr.

    Examples
    --------
    >>> from swarm_arena.tool import RunStatus
    >>> value = with_collection_diagnostics(ToolOutput(status=RunStatus.OK), [])
    >>> value._collection_diagnostics
    ()
    """
    copied = output.model_copy()
    copied._collection_diagnostics = _diagnostics(diagnostics)
    return copied


def detach_tool_output(value: object) -> ToolOutput:
    """Detach an extension-owned output and preserve its validated diagnostics.

    Parameters
    ----------
    value : object
        ToolOutput to normalize inside the owner's preparation daemon.

    Returns
    -------
    ToolOutput
        Core-owned, revalidated output and private code/count carrier.

    Examples
    --------
    >>> from swarm_arena.tool import RunStatus
    >>> detach_tool_output(ToolOutput(status=RunStatus.OK)).data
    {}
    """
    output = _detach_model(ToolOutput, value)
    assert isinstance(value, ToolOutput)
    return with_collection_diagnostics(output, value._collection_diagnostics)


def collection_warnings(output: ToolOutput) -> list[str]:
    """Render private diagnostics as bounded value-free candidate warnings.

    Parameters
    ----------
    output : ToolOutput
        Detached core-owned output.

    Returns
    -------
    list[str]
        One fixed diagnostic per recognized code, with an optional bounded count.

    Examples
    --------
    >>> from swarm_arena.tool import RunStatus
    >>> collection_warnings(ToolOutput(status=RunStatus.OK))
    []
    """
    return [
        COLLECTION_DIAGNOSTICS[code] + (f" ({count} occurrences)" if count > 1 else "")
        for code, count in _diagnostics(output._collection_diagnostics)
    ]
