"""Limit concurrent local jobs by CPU capacity and slot count."""

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from fractions import Fraction

import anyio

from swarm_arena._model.plan import Resources


class ResourcePool:
    """
    Limit concurrent local jobs and their combined CPU reservations.

    Construct outside an async backend; create the condition lazily on first
    acquisition. GPU, memory, and custom requests do not affect local scheduling.
    Release reservations even when the slot body raises or is cancelled.

    Parameters
    ----------
    cpus : float
        Total CPU capacity, including fractional capacity.
    max_concurrent : int
        Maximum number of simultaneously running jobs.

    Examples
    --------
    >>> pool = ResourcePool(cpus=2.0, max_concurrent=4)
    >>> pool.capacity.cpus
    2.0
    """

    def __init__(self, *, cpus: float, max_concurrent: int) -> None:
        """Store scheduling limits without creating backend-bound resources."""
        self._cpus = cpus
        self._cpu_capacity = Fraction(str(float(cpus)))
        self._max_concurrent = max_concurrent
        self._running = 0
        self._cpus_in_use = Fraction(0)
        self._condition: anyio.Condition | None = None

    @property
    def capacity(self) -> Resources:
        """
        Return total CPU capacity rather than currently unreserved capacity.

        Returns
        -------
        Resources
            Total CPU capacity; other resource fields retain their defaults.

        Examples
        --------
        >>> ResourcePool(cpus=2.0, max_concurrent=4).capacity.cpus
        2.0
        """
        return Resources(cpus=self._cpus)

    def slot(self, need: Resources) -> AbstractAsyncContextManager[None]:
        """
        Reserve a job slot and CPUs until the context exits.

        Wait for both concurrency and CPU capacity. An impossible CPU request
        fails immediately, without waiting for other jobs to release resources.

        Parameters
        ----------
        need : Resources
            Resource request; only ``cpus`` participates in local scheduling.

        Returns
        -------
        AbstractAsyncContextManager[None]
            Async context manager yielding None while holding the reservation.

        Raises
        ------
        ValueError
            If the requested CPUs exceed this pool's total capacity.

        Examples
        --------
        >>> async with pool.slot(need):  # doctest: +SKIP
        ...     await run_job()
        """
        if need.cpus > self._cpus:
            raise ValueError("job needs more cpus than this executor has")
        return self._slot(need)

    @asynccontextmanager
    async def _slot(self, need: Resources) -> AsyncGenerator[None]:
        """Acquire atomically and shield release from caller cancellation."""
        if self._condition is None:
            self._condition = anyio.Condition()
        condition = self._condition
        cpus = Fraction(str(need.cpus))
        async with condition:
            while (
                self._running >= self._max_concurrent
                or self._cpus_in_use + cpus > self._cpu_capacity
            ):
                await condition.wait()
            self._running += 1
            self._cpus_in_use += cpus
        try:
            yield
        finally:
            with anyio.CancelScope(shield=True):
                async with condition:
                    self._running -= 1
                    self._cpus_in_use -= cpus
                    if self._running == 0:
                        self._cpus_in_use = Fraction(0)
                    condition.notify_all()
