from typing import Any, ClassVar, get_args

import pytest
from pydantic import JsonValue as PydanticJsonValue
from pydantic import ValidationError

from swarm_arena._model import migrate
from swarm_arena._model.base import (
    CallRole,
    CatalogKind,
    FrozenModel,
    JsonValue,
    LLMRole,
    Origin,
    StoredModel,
)
from swarm_arena._model.migrate import migration


@pytest.fixture(autouse=True)
def isolated_migrations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(migrate, "_MIGRATIONS", {})


def test_frozen_model_rejects_extra_and_mutation() -> None:
    class Demo(FrozenModel):
        x: int

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Demo.model_validate({"x": 1, "extra": 2})
    m = Demo(x=1)
    field_name = "x"
    with pytest.raises(ValidationError, match="frozen_instance"):
        setattr(m, field_name, 2)
    assert m.model_copy(update={"x": 2}).x == 2


def test_stored_model_upgrades_and_refuses_newer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(migrate.CURRENT, "demo", 2)

    @migration("demo", 1)
    def one(raw: dict[str, Any]) -> dict[str, Any]:
        return {**raw, "y": 0}

    class Demo(StoredModel):
        schema_kind: ClassVar[str] = "demo"
        x: int
        y: int

    assert Demo.model_validate({"x": 1}) == Demo(x=1, y=0, schema_version=2)
    assert Demo.model_validate_json('{"x": 1}').schema_version == 2
    with pytest.raises(ValidationError, match="newer"):
        Demo.model_validate({"schema_version": 3})


def test_stored_model_without_kind_fails() -> None:
    with pytest.raises(ValidationError, match="schema_kind"):
        StoredModel.model_validate({})


def test_stored_model_current_version_and_instance() -> None:
    class Demo(StoredModel):
        schema_kind: ClassVar[str] = "plan"
        x: int

    m = Demo(x=1)
    assert m.schema_version == 1
    assert Demo.model_validate(m) is m
    assert m.model_dump() == {"schema_version": 1, "x": 1}


def test_type_aliases() -> None:
    assert get_args(Origin) == ("human", "llm", "together", "autonomous")
    assert get_args(LLMRole) == ("manager", "judge", "planner", "curator")
    assert get_args(CallRole) == ("manager", "judge", "planner", "curator", "tool")
    assert get_args(CatalogKind) == ("tool", "manager", "check", "judge", "policy")
    assert JsonValue is PydanticJsonValue
