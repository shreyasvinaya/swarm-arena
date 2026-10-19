"""Define frozen models, stored-format validation, and shared type aliases."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import JsonValue as JsonValue

from swarm_arena._model.migrate import upgrade

Origin = Literal["human", "llm", "together", "autonomous"]
LLMRole = Literal["manager", "judge", "planner", "curator"]
CallRole = Literal["manager", "judge", "planner", "curator", "tool"]
CatalogKind = Literal["tool", "manager", "check", "judge", "policy"]


class FrozenModel(BaseModel):
    """
    Validate immutable model fields and reject unknown fields.

    Examples
    --------
    >>> class Example(FrozenModel):
    ...     x: int
    >>> Example(x=1).model_copy(update={"x": 2}).x
    2
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class _SafeModel(FrozenModel):
    """Hide rejected values in validation and frozen-assignment messages."""

    model_config = ConfigDict(hide_input_in_errors=True)


class StoredModel(FrozenModel):
    """
    Upgrade stored data before validating its fields.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version, upgraded to the current version when loaded.

    Attributes
    ----------
    schema_kind : ClassVar[str]
        Registry kind that every concrete stored model must declare.

    Examples
    --------
    >>> class Example(StoredModel):
    ...     schema_kind: ClassVar[str] = "plan"
    >>> Example.model_validate({}).schema_version
    1
    """

    schema_version: int = 1
    schema_kind: ClassVar[str] = ""

    @model_validator(mode="before")
    @classmethod
    def _upgrade(cls, data: object) -> object:
        """Upgrade dictionary inputs through the model's registered schema kind."""
        if isinstance(data, dict):
            if not cls.schema_kind:
                raise ValueError(f"{cls.__name__} has no schema_kind")
            return upgrade(cls.schema_kind, data)
        return data
