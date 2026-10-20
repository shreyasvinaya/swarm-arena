"""Fail loudly when Pydantic changes assumptions used by safe validation."""

from typing import Annotated, Any, cast

import pytest
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    WrapValidator,
)

from swarm_arena._model.base import _SafeModel
from swarm_arena._model.compat import _alias_paths, _core_schema, _union_branch


def test_nested_models_use_core_schema_instead_of_custom_init() -> None:
    class Child(_SafeModel):
        count: int

    class Parent(_SafeModel):
        child: Child

    assert Child.__pydantic_custom_init__ is False
    assert Parent(child={"count": 2}).child.count == 2
    schema = _core_schema(Child)
    assert schema["type"] == "function-wrap"
    assert schema["schema"]["custom_init"] is False


@pytest.mark.parametrize(
    "alias", ["renamed", AliasPath("payload", 0), AliasChoices("first", AliasPath("payload", 0))]
)
def test_alias_normalization_matches_core_error_locations(alias: Any) -> None:
    class Model(BaseModel):
        count: int = Field(validation_alias=alias)

    schema = _core_schema(Model)["schema"]["fields"]["count"]
    paths = _alias_paths(schema["validation_alias"])
    with pytest.raises(ValidationError) as caught:
        Model.model_validate({})
    assert caught.value.errors()[0]["loc"] == paths[0]


def test_inferred_union_labels_match_core_error_locations() -> None:
    class Alpha(BaseModel):
        count: int

    class Beta(BaseModel):
        count: int

    def wrap(value: Any, handler: Any) -> Any:
        return handler(value)

    adapter = TypeAdapter(Annotated[Alpha, WrapValidator(wrap)] | Beta)
    labels = [_union_branch(choice)[1] for choice in cast(Any, adapter.core_schema)["choices"]]
    with pytest.raises(ValidationError) as caught:
        adapter.validate_python({"count": "bad"})
    assert [item["loc"][0] for item in caught.value.errors()] == labels
    assert labels == ["function-wrap[wrap()]", "Beta"]
