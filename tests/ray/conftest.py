"""Connect selected worker tests to the single process-wide Ray head."""

import signal
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session")
def ray_driver(ray_cluster: str) -> Iterator[None]:
    """Bound initialization and release the driver before head teardown."""
    import ray

    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum: int, frame: object) -> None:
        raise TimeoutError("Ray test driver operation timed out")

    try:
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(45)
        try:
            ray.init(address=ray_cluster, ignore_reinit_error=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        yield
    finally:
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(30)
        try:
            ray.shutdown()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
