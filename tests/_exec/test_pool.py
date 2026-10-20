import anyio
import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import Resources
from swarm_arena._exec.pool import ResourcePool


@pytest.mark.parametrize(
    ("cpus", "max_concurrent", "need_cpus", "expected_peak"),
    [(8.0, 2, 1.0, 2), (2.0, 8, 1.5, 1)],
)
async def test_pool_limits_concurrency_and_cpus(
    cpus: float, max_concurrent: int, need_cpus: float, expected_peak: int
) -> None:
    pool = ResourcePool(cpus=cpus, max_concurrent=max_concurrent)
    running = peak = completed = 0

    async def worker() -> None:
        nonlocal running, peak, completed
        async with pool.slot(Resources(cpus=need_cpus)):
            running += 1
            peak = max(peak, running)
            assert running * need_cpus <= cpus
            await anyio.sleep(0.2)
            running -= 1
            completed += 1

    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            for _ in range(4):
                group.start_soon(worker)
    assert peak == expected_peak
    assert completed == 4


async def test_pool_rejects_impossible_requests() -> None:
    pool = ResourcePool(cpus=2.0, max_concurrent=4)
    assert pool.capacity == Resources(cpus=2.0)
    need = Resources(cpus=3.0)
    with pytest.raises(ValueError, match="job needs more cpus") as exc:
        async with pool.slot(need):
            pytest.fail("impossible request acquired a slot")
    assert str(exc.value) == "job needs more cpus than this executor has"


def test_pool_constructs_without_async_backend() -> None:
    assert ResourcePool(cpus=2.0, max_concurrent=4).capacity == Resources(cpus=2.0)


def test_pool_error_never_formats_capacity() -> None:
    class SecretCapacity(float):
        def __str__(self) -> str:
            return "dd4c657e-e627-49a8-bab5-c83fa99d8e66"

    pool = ResourcePool(cpus=SecretCapacity(2.0), max_concurrent=4)
    with pytest.raises(ValueError, match="job needs more cpus") as exc:
        pool.slot(Resources(cpus=3.0))
    assert_secret_absent(exc.value, "dd4c657e-e627-49a8-bab5-c83fa99d8e66")


async def test_pool_ignores_other_resources() -> None:
    pool = ResourcePool(cpus=1.0, max_concurrent=1)
    with anyio.fail_after(1):
        async with pool.slot(
            Resources(cpus=1.0, gpus=8, memory_gib=100, custom={"gpu": 9})
        ) as slot:
            assert slot is None


async def test_pool_releases_after_error() -> None:
    pool = ResourcePool(cpus=1.0, max_concurrent=1)
    with pytest.raises(RuntimeError, match="worker failed"):
        async with pool.slot(Resources()):
            raise RuntimeError("worker failed")
    with anyio.fail_after(1):
        async with pool.slot(Resources()):
            pass


async def test_pool_releases_after_cancellation() -> None:
    pool = ResourcePool(cpus=1.0, max_concurrent=1)
    with anyio.CancelScope() as scope:
        async with pool.slot(Resources()):
            scope.cancel()
            await anyio.sleep_forever()
    assert scope.cancelled_caught
    with anyio.fail_after(1):
        async with pool.slot(Resources()):
            pass


async def test_cancelled_waiter_does_not_consume_capacity() -> None:
    pool = ResourcePool(cpus=2.0, max_concurrent=1)

    async def waiter(*, task_status: anyio.abc.TaskStatus[anyio.CancelScope]) -> None:
        with anyio.CancelScope() as scope:
            task_status.started(scope)
            async with pool.slot(Resources()):
                pytest.fail("waiter exceeded concurrency limit")
        assert scope.cancelled_caught

    with anyio.fail_after(1):
        async with pool.slot(Resources()), anyio.create_task_group() as group:
            scope = await group.start(waiter)
            await anyio.wait_all_tasks_blocked()
            scope.cancel()
        async with pool.slot(Resources(cpus=2.0)):
            pass


async def test_fractional_cpu_reservations_fit_exactly() -> None:
    pool = ResourcePool(cpus=0.3, max_concurrent=3)
    with anyio.fail_after(1):
        async with pool.slot(Resources(cpus=0.1)), pool.slot(Resources(cpus=0.2)):
            pass


async def test_cpu_release_does_not_accumulate_rounding_error() -> None:
    pool = ResourcePool(cpus=1.0, max_concurrent=3)
    with anyio.fail_after(1):
        async with pool.slot(Resources(cpus=0.66)):
            for _ in range(20):
                async with pool.slot(Resources(cpus=0.18)):
                    pass
            async with pool.slot(Resources(cpus=0.34)):
                pass
