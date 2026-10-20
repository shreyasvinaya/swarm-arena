"""Exercise the validation entry points required at the dependency floor."""

import pytest

from swarm_arena._model.base import FrozenModel
from swarm_arena._model.ref import Ref


@pytest.mark.parametrize("entry", ["python", "json", "strings"])
def test_validation_entry_points(entry: str) -> None:
    class Count(FrozenModel):
        count: int

    if entry == "python":
        result = Count.model_validate({"count": 2})
    elif entry == "json":
        result = Count.model_validate_json('{"count":2}')
    else:
        result = Count.model_validate_strings({"count": "2"})
    assert result.count == 2


def test_component_build_uses_supported_validation() -> None:
    from swarm_arena.executor import LocalExecutor

    assert isinstance(Ref.of(LocalExecutor).build(LocalExecutor), LocalExecutor)
