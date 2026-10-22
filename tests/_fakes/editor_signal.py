"""Exercise a worker-thread interrupt while the editor wait is asleep."""

import asyncio
import io
import json
import os
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Any, cast

from swarm_arena import Goal, Plan
from swarm_arena.llm._terminal_reviewer import TerminalReviewer


def main(root: Path) -> None:
    """Run an isolated selector/socket handshake and print its evidence."""
    evidence: dict[str, Any] = {}
    armed = threading.Event()
    selecting = threading.Event()
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(15)
    editor = root / "editor"
    editor.write_text(
        "#!" + sys.executable + "\n"
        "import os, signal, socket\n"
        "connection = socket.socket()\n"
        "def terminate(*args):\n"
        "    connection.sendall(b'terminated\\n')\n"
        "    os._exit(0)\n"
        "signal.signal(signal.SIGTERM, terminate)\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        f"connection.connect({listener.getsockname()!r})\n"
        "connection.sendall((str(os.getpid()) + '\\n').encode())\n"
        "while True: signal.pause()\n"
    )
    editor.chmod(0o700)
    loop: Any = None

    def sender() -> None:
        try:
            with listener.accept()[0] as connection:
                connection.settimeout(5)
                with connection.makefile("rb") as stream:
                    evidence["pid"] = int(stream.readline())
                    loop.call_soon_threadsafe(armed.set)
                    assert selecting.wait(10), "selector did not enter its idle wait"
                    signal.pthread_kill(threading.get_ident(), signal.SIGINT)
                    try:
                        evidence["terminated_before_fallback"] = (
                            stream.readline() == b"terminated\n"
                        )
                    except TimeoutError:
                        evidence["terminated_before_fallback"] = False
                    if not evidence["terminated_before_fallback"]:
                        loop.call_soon_threadsafe(lambda: None)
        except BaseException as error:
            evidence["sender_error"] = repr(error)
            loop.call_soon_threadsafe(loop.stop)

    thread = threading.Thread(target=sender)

    async def review() -> None:
        nonlocal loop
        loop = asyncio.get_running_loop()
        selector = cast(Any, loop)._selector
        select = selector.select

        def select_when_armed(timeout: float | None = None) -> Any:
            if armed.is_set() and timeout is None:
                selecting.set()
            return select(timeout)

        selector.select = select_when_armed
        thread.start()
        await TerminalReviewer(
            stdin=io.StringIO("e\n"), stdout=io.StringIO(), editor=str(editor)
        ).review(Plan(goal=Goal(id="g", namespace="n"), teams=[], judges=[]))

    interval = sys.getswitchinterval()
    # The selector releases the GIL: the sender cannot run between setting the
    # event and entering select, which would turn the missing wakeup into a race.
    sys.setswitchinterval(60)
    try:
        try:
            asyncio.run(review())
        except KeyboardInterrupt:
            evidence["interrupted"] = True
    finally:
        sys.setswitchinterval(interval)
        thread.join(15)
        listener.close()
    assert not thread.is_alive()
    try:
        os.kill(evidence["pid"], 0)
    except ProcessLookupError:
        evidence["reaped"] = True
    print(json.dumps(evidence))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
