import anyio
import anyio.lowlevel
import pytest

from swarm_arena._engine.admission import RunAdmission
from swarm_arena._errors import PlanError


async def test_atomic_growth_does_not_exceed_total() -> None:
    admission = RunAdmission(3)
    await admission.update("a", 1)
    await admission.update("b", 1)
    accepted: list[str] = []
    refused: list[str] = []

    async def grow(goal: str) -> None:
        async def publish() -> None:
            await anyio.lowlevel.checkpoint()
            accepted.append(goal)

        try:
            await admission.update(goal, 2, publish=publish)
        except PlanError:
            refused.append(goal)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(grow, "a")
        tasks.start_soon(grow, "b")
    assert len(accepted) == len(refused) == 1
    await admission.release(accepted[0])
    await admission.update(refused[0], 3)


async def test_failed_publication_retains_previous_peak() -> None:
    admission = RunAdmission(2)
    await admission.update("a", 1)

    async def fail() -> None:
        raise RuntimeError("publication failed")

    with pytest.raises(RuntimeError):
        await admission.update("a", 2, publish=fail)
    await admission.update("b", 1)
