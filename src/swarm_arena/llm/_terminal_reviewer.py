"""Cancellable terminal adapter for collaborative planning."""

import os
import subprocess
import sys
import termios
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TextIO

import anyio

from swarm_arena._agent.context import _local_timer_scope
from swarm_arena._cli.run_cmds import _readline
from swarm_arena._errors import _core_error
from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.plan import Plan
from swarm_arena._outcomes import _Outcome
from swarm_arena.llm._hooks import _prepare
from swarm_arena.llm._reviewer import Approve, Edited, Feedback


class TerminalReviewer:
    r"""Review TOML with terminal input and an awaited editor.

    Parameters
    ----------
    stdin : TextIO or None, optional
        Input descriptor, or an in-memory StringIO for scripted use.
    stdout : TextIO or None, optional
        Plan and prompt output, defaulting to standard output.
    editor : str or None, optional
        Executable path, ahead of VISUAL, EDITOR and vi. No shell is used.

    Examples
    --------
    >>> import io
    >>> reviewer = TerminalReviewer(stdin=io.StringIO("a\n"), stdout=io.StringIO())
    """

    def __init__(
        self,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        editor: str | None = None,
    ) -> None:
        self._stdin = sys.stdin if stdin is None else stdin
        self._stdout = sys.stdout if stdout is None else stdout
        self._editor = editor

    async def review(self, plan: Plan) -> Approve | Feedback | Edited:
        r"""Display a plan and await approval, feedback or a valid TOML edit.

        Parameters
        ----------
        plan : Plan
            Draft whose source directory is retained through editing.

        Returns
        -------
        Approve or Feedback or Edited
            Human decision; malformed edits and editor failures retry.

        Raises
        ------
        RuntimeError
            Input ends before a decision or feedback line.

        Examples
        --------
        >>> import anyio, io
        >>> from swarm_arena import Goal, Plan
        >>> reviewer = TerminalReviewer(stdin=io.StringIO("a\n"), stdout=io.StringIO())
        >>> anyio.run(reviewer.review, Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[]))
        Approve()
        """
        boundary = _Outcome()
        try:
            text = await _prepare(lambda: _context_string(plan.dumps(), "plan TOML"))
            print(text, file=self._stdout)
            while True:
                print(
                    "Approve [a], feedback [f], edit [e]: ", end="", file=self._stdout, flush=True
                )
                choice = (await self._line()).strip().lower()
                if choice == "a":
                    return Approve()
                if choice == "f":
                    return Feedback((await self._line()).rstrip("\r\n"))
                if choice == "e":
                    edited = await self._edit(plan, text)
                    if edited is not None:
                        return edited
        except BaseException as error:
            boundary.contain(error, RuntimeError, "terminal review")

    async def _line(self) -> str:
        line = await _readline(self._stdin)
        if not line:
            raise _core_error(RuntimeError, "terminal input ended")
        return line

    def _retry(self, boundary: _Outcome, error: BaseException) -> None:
        try:
            boundary.contain(error, RuntimeError, "editor")
        except Exception as safe:
            boundary.propagate_control(safe)
            print(str(safe), file=self._stdout)

    async def _edit(self, plan: Plan, text: str) -> Edited | None:
        boundary = _Outcome()
        # The result is exposed only after temporary-file and process cleanup.
        with TemporaryDirectory(prefix="swarm-review-") as directory:
            path = Path(directory) / "plan.toml"
            await _prepare(lambda: path.write_text(text, encoding="utf-8"))
            editor = self._editor or os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
            tty_fd = next((fd for fd in (0, 1, 2) if os.isatty(fd)), None)
            saved = termios.tcgetattr(tty_fd) if tty_fd is not None else None
            try:
                try:
                    process = await anyio.open_process(
                        [editor, str(path)], stdin=None, stdout=None, stderr=subprocess.DEVNULL
                    )
                except BaseException as error:
                    self._retry(boundary, error)
                    return None
                async with process:
                    try:
                        code = await process.wait()
                    finally:
                        with anyio.CancelScope(shield=True):
                            if process.returncode is None:
                                with suppress(ProcessLookupError):
                                    process.terminate()
                                grace = anyio.CancelScope(deadline=anyio.current_time() + 1.0)
                                with _local_timer_scope(grace), grace:
                                    await process.wait()
                                if process.returncode is None:
                                    process.kill()
                                await process.wait()
            finally:
                if tty_fd is not None and saved is not None:
                    termios.tcsetattr(tty_fd, termios.TCSANOW, saved)
            if code != 0:
                print("editor failed", file=self._stdout)
                return None
            try:
                result = await _prepare(
                    lambda: Plan.loads(path.read_text(encoding="utf-8"), base_dir=plan.base_dir)
                )
            except BaseException as error:
                self._retry(boundary, error)
                return None
        return Edited(result)
