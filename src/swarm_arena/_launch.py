"""
Hold a tool inert until the driver commits its start, then exec it.

Run as ``python -I -m swarm_arena._launch --go-fd FD [--rlimit-as BYTES]
[--rlimit-cpu SECONDS] [--rlimit-nofile N] -- ARGV...``. Isolated mode (``-I``)
keeps the working directory, every ``PYTHON*`` variable, and the user site off
the launcher's ``sys.path``, so workspace files never shadow this module.
``os.environ`` is unchanged and goes to the command as is.
"""

import argparse
import os
import resource
import shutil
import signal
import sys
from collections.abc import Sequence

_GO = b"go"
# Python ignores these at start-up and an ignored signal survives exec; subprocess's
# restore_signals resets the same set, so the command sees what a direct spawn would give it.
_RESTORED_SIGNALS = ("SIGPIPE", "SIGXFZ", "SIGXFSZ")


def _parser() -> argparse.ArgumentParser:
    """Build the parser for the options before ``--``."""
    parser = argparse.ArgumentParser(prog="python -I -m swarm_arena._launch")
    parser.add_argument("--go-fd", type=int, required=True)
    parser.add_argument("--rlimit-as", type=int)
    parser.add_argument("--rlimit-cpu", type=int)
    parser.add_argument("--rlimit-nofile", type=int)
    return parser


def _set_limit(name: str, which: int, value: int) -> None:
    """Apply one resource limit; report a refusal on stderr and continue."""
    try:
        if which == resource.RLIMIT_NOFILE:
            hard = resource.getrlimit(which)[1]
            soft = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (soft, hard))
        else:
            resource.setrlimit(which, (value, value))
    except (ValueError, OSError) as exc:
        print(f"swarm-arena: rlimit {name} not supported: {exc}", file=sys.stderr)


def _read_go(fd: int) -> bytes:
    """Read the go descriptor until two bytes or end of file, then close it."""
    data = b""
    try:
        while len(data) < len(_GO):
            chunk = os.read(fd, len(_GO) - len(data))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    return data


def main(argv: Sequence[str] | None = None) -> int:
    """
    Apply limits, wait for "go", and replace this process with the command.

    The launcher stays inert until it reads ``b"go"`` from the go descriptor.
    End of file or any other bytes mean the driver did not commit the start,
    so it exits 0 and the command never runs. After "go" it resolves the
    command on ``PATH`` and calls ``os.execve``, which keeps the pid, the
    process group, and the process start time.

    Parameters
    ----------
    argv : Sequence[str] or None, default=None
        Options, ``--``, then the command argv. None reads ``sys.argv[1:]``.

    Returns
    -------
    int
        0 when the go descriptor closes without "go", 127 when the command is
        not found, 126 when it cannot be executed. On success it does not return.

    Examples
    --------
    >>> import os
    >>> r, w = os.pipe()
    >>> os.close(w)  # the driver closed the pipe without "go"
    >>> main(["--go-fd", str(r), "--", "true"])
    0
    """
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if "--" not in args:
        parser.error("expected -- before the command")
    split = args.index("--")
    options, command = parser.parse_args(args[:split]), args[split + 1 :]
    if not command:
        parser.error("expected a command after --")
    limits = (
        ("as", resource.RLIMIT_AS, options.rlimit_as),
        ("cpu", resource.RLIMIT_CPU, options.rlimit_cpu),
        ("nofile", resource.RLIMIT_NOFILE, options.rlimit_nofile),
    )
    for name, which, value in limits:
        if value is not None:
            _set_limit(name, which, value)
    if _read_go(options.go_fd) != _GO:
        return 0
    program = command[0]
    resolved = program if "/" in program else shutil.which(program, path=os.environ.get("PATH"))
    if resolved is None:
        print(f"swarm-arena: command not found: {program}", file=sys.stderr)
        return 127
    for name in _RESTORED_SIGNALS:
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_DFL)
    sys.stderr.flush()
    try:
        os.execve(resolved, command, os.environ)
    except OSError as exc:
        print(f"swarm-arena: cannot run {program}: {exc.strerror}", file=sys.stderr)
        return 127 if isinstance(exc, FileNotFoundError) else 126


if __name__ == "__main__":
    raise SystemExit(main())
