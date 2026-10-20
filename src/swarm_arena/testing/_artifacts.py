"""Read verified artifacts produced by the process-local conformance kit."""

import re

from swarm_arena._model.tooling import ToolOutput
from swarm_arena.testing._conformance import _artifacts


def read_output_file(output: ToolOutput, path: str) -> bytes:
    """
    Read one declared output file from the conformance kit's artifact store.

    Match the exact workspace-relative ``FileRef.path``, not its display
    name. The bytes remain available after workspace cleanup, until this
    Python process exits. The stored content is verified against its SHA-256.

    Parameters
    ----------
    output : ToolOutput
        Result returned by ``check_tool`` in this Python process.
    path : str
        Exact path of a file declared in ``output.files``.

    Returns
    -------
    bytes
        Verified file contents, without decoding.

    Raises
    ------
    ValueError
        If the path does not identify exactly one declared output file.
    NotFound
        If the file is absent from this process's artifact store.
    StorageError
        If the stored content cannot be read or fails its hash check.

    Examples
    --------
    >>> import anyio
    >>> from swarm_arena import ScriptTool
    >>> from swarm_arena.testing import check_tool, read_output_file
    >>> code = ("from pathlib import Path; import json; "
    ...         "Path('answer.txt').write_text('42'); "
    ...         "Path('swarm/result.json').write_text("
    ...         "json.dumps({'data': {}, 'files': ['answer.txt']}))")
    >>> out = anyio.run(check_tool, ScriptTool(command=["python", "-c", code]), {})
    >>> read_output_file(out, "answer.txt")
    b'42'
    """
    valid = ToolOutput.model_validate(output.model_dump(warnings=False))
    matches = [file for file in valid.files if file.path == path]
    if len(matches) != 1:
        raise ValueError("path must identify exactly one declared output file")
    return _artifacts().get_bytes(matches[0].sha256)


def read_logs(output: ToolOutput) -> tuple[bytes, bytes]:
    r"""
    Read a kit run's stored stdout and stderr logs as bytes, in that order.

    These are the complete logs retained by the executor, which may truncate
    them to its configured log limit. They remain available after workspace
    cleanup until this Python process exits. Both digests are verified.

    Parameters
    ----------
    output : ToolOutput
        Result returned by ``check_tool`` in this Python process, with a
        stored ``logs_ref``. Custom executors must use the kit's artifact store.

    Returns
    -------
    tuple[bytes, bytes]
        Verified stdout and stderr bytes; decoding is left to the caller.

    Raises
    ------
    ValueError
        If ``logs_ref`` is missing or malformed.
    NotFound
        If either log is absent from this process's artifact store.
    StorageError
        If stored content cannot be read or fails its hash check.

    Examples
    --------
    >>> import anyio
    >>> from swarm_arena import ScriptTool
    >>> from swarm_arena.testing import check_tool, read_logs
    >>> code = ("from pathlib import Path; import sys; "
    ...         "print('hello'); print('diagnostic', file=sys.stderr); "
    ...         "Path('swarm/result.json').write_text('{\"data\": {}}')")
    >>> out = anyio.run(check_tool, ScriptTool(command=["python", "-c", code]), {})
    >>> stdout, stderr = read_logs(out)
    >>> stdout.decode().strip(), stderr.decode().strip()
    ('hello', 'diagnostic')
    """
    ref = getattr(output, "logs_ref", None)
    if not isinstance(ref, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}:[0-9a-f]{64}", ref):
        raise ValueError("logs_ref must identify stored stdout and stderr logs")
    _, stdout, stderr = ref.split(":")
    artifacts = _artifacts()
    return artifacts.get_bytes(stdout), artifacts.get_bytes(stderr)
