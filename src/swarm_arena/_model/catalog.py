"""Define permitted models and planner-selectable component catalogs."""

from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    create_model,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from swarm_arena._model.base import CatalogKind, LLMRole, _SafeModel
from swarm_arena._model.data import TomlJsonObject, json_equal
from swarm_arena._model.goal import _Slug
from swarm_arena._model.ref import Ref, load_object
from swarm_arena._model.schema import compile_object_schema
from swarm_arena._model.settings import _ConfigModel


class ModelSpec(_ConfigModel):
    """
    Declare a provider model and the roles allowed to use it.

    Parameters
    ----------
    model : str
        LiteLLM provider/model identifier containing a slash.
    max_tokens : int, default=4096
        Positive output token limit.
    roles : list[str]
        Nonempty, unique list of manager, judge, planner, or curator roles.
    prompt_cache : PromptCache, optional
        Provider prompt caching settings, enabled by default.
    prices : Prices or None, default=None
        Optional dollar prices per million tokens.

    Examples
    --------
    >>> from swarm_arena.config import ModelSpec
    >>> ModelSpec(model='openai/gpt-6', roles=['judge']).provider
    'openai'
    """

    class PromptCache(_ConfigModel):
        """
        Configure provider-side prompt caching.

        Parameters
        ----------
        enabled : bool, default=True
            Enable prompt caching.
        ttl : {'5m', '1h'}, default='5m'
            Requested provider cache lifetime.

        Examples
        --------
        >>> from swarm_arena.config import ModelSpec
        >>> ModelSpec.PromptCache().ttl
        '5m'
        """

        enabled: bool = True
        ttl: Literal["5m", "1h"] = "5m"

    class Prices(_ConfigModel):
        """
        Specify nonnegative dollar prices per million tokens.

        Parameters
        ----------
        input : float
            Price of input tokens.
        output : float
            Price of output tokens.
        cache_read : float or None, default=None
            Optional price of cached input tokens.
        cache_write : float or None, default=None
            Optional price of prompt cache writes.

        Examples
        --------
        >>> from swarm_arena.config import ModelSpec
        >>> ModelSpec.Prices(input=0.3, output=2.5).output
        2.5
        """

        input: float = Field(ge=0)
        output: float = Field(ge=0)
        cache_read: float | None = Field(default=None, ge=0)
        cache_write: float | None = Field(default=None, ge=0)

    model: str = Field(pattern="/")
    max_tokens: int = Field(default=4096, ge=1)
    roles: list[LLMRole] = Field(min_length=1)
    prompt_cache: PromptCache = PromptCache()
    prices: Prices | None = None

    @field_validator("roles")
    @classmethod
    def _unique_roles(cls, roles: list[LLMRole]) -> list[LLMRole]:
        """Reject duplicate role permissions."""
        if len(set(roles)) != len(roles):
            raise PydanticCustomError("arena_config", "roles must be unique")
        return roles

    @property
    def provider(self) -> str:
        """
        Return the provider prefix before the first slash.

        Returns
        -------
        str
            Provider portion of the model identifier.

        Examples
        --------
        >>> from swarm_arena.config import ModelSpec
        >>> ModelSpec(model='openai/gpt-6', roles=['judge']).provider
        'openai'
        """
        return self.model.split("/", 1)[0]


class ModelCatalog(
    RootModel[dict[Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")], ModelSpec]],
    _SafeModel,
):
    """
    Map human-approved aliases to model specifications.

    Parameters
    ----------
    root : dict[str, ModelSpec]
        Model specifications keyed by alphanumeric, dot, underscore, or hyphen
        aliases beginning with an alphanumeric character.

    Examples
    --------
    >>> from swarm_arena.config import ModelCatalog, ModelSpec
    >>> catalog = ModelCatalog({'gpt': ModelSpec(model='openai/gpt-6', roles=['judge'])})
    >>> catalog.resolve('gpt', 'judge').max_tokens
    4096
    """

    model_config = ConfigDict(frozen=True, extra=None, hide_input_in_errors=True)

    def resolve(self, alias: str, role: LLMRole) -> ModelSpec:
        """
        Resolve an alias and check its role permission.

        Parameters
        ----------
        alias : str
            Human-defined model alias.
        role : str
            Required LLM role.

        Returns
        -------
        ModelSpec
            The permitted model specification.

        Raises
        ------
        ValueError
            If the alias is unknown or its roles exclude the requested role.

        Examples
        --------
        >>> from swarm_arena.config import ModelCatalog, ModelSpec
        >>> catalog = ModelCatalog({'gpt': ModelSpec(model='openai/gpt-6', roles=['judge'])})
        >>> catalog.resolve('gpt', 'judge').provider
        'openai'
        """
        if alias not in self.root:
            raise ValueError(f"unknown model alias {alias!r}")
        spec = self.root[alias]
        if role not in spec.roles:
            raise ValueError(f"model {alias!r} may not be used for role {role!r}")
        return spec


class CatalogEntry(_ConfigModel):
    """
    Describe a component and its planner-settable configuration fields.

    Parameters
    ----------
    name : str
        Lowercase slug identifying this entry.
    kind : {'tool', 'manager', 'check', 'judge', 'policy'}
        Position in which the component may be used.
    ref : str
        Component import path in module:QualName form.
    description : str
        Human-supplied explanation for the planner.
    config : dict[str, Any], default={}
        Fixed component settings.
    settable : dict[str, Any], default={}
        Supported JSON-schema property definitions for planner overrides.
        Omitted override fields default to None, regardless of schema defaults.

    Examples
    --------
    >>> from swarm_arena.config import CatalogEntry
    >>> entry = CatalogEntry(name='tool', kind='tool', ref='demo:Tool', description='Demo')
    >>> entry.settable_model()().model_dump()
    {}
    """

    name: _Slug
    kind: CatalogKind
    ref: str
    description: str
    config: TomlJsonObject = {}
    settable: TomlJsonObject = {}

    @field_validator("ref")
    @classmethod
    def _ref_path(cls, value: str) -> str:
        """Apply the shared reference path format without importing code."""
        return Ref(path=value).path

    def _compile_settable(self) -> type[BaseModel]:
        """Compile schemas while suppressing their potentially sensitive diagnostics."""
        try:
            return compile_object_schema("Settable", self.settable)
        except Exception:
            raise PydanticCustomError("arena_config", "invalid settable schema") from None

    @model_validator(mode="after")
    def _check_schema(self) -> Self:
        """Reject invalid property schemas as soon as the entry is loaded."""
        self._compile_settable()
        return self

    def settable_model(self) -> type[BaseModel]:
        """
        Build a strict model of optional planner overrides.

        Returns
        -------
        type[BaseModel]
            Frozen model rejecting unknown keys. Omitted fields default to None;
            supplied values must satisfy the property schema, including nulls.

        Examples
        --------
        >>> from swarm_arena.config import CatalogEntry
        >>> entry = CatalogEntry(name='m', kind='manager', ref='demo:Manager',
        ...     description='Demo', settable={'label': {'type': 'string', 'maxLength': 10}})
        >>> entry.settable_model()(label='hello').model_dump()
        {'label': 'hello'}
        """
        compiled = self._compile_settable()
        fields: dict[str, Any] = {
            name: (field.rebuild_annotation(), None)
            for name, field in compiled.model_fields.items()
        }
        return create_model(
            "Settable",
            __base__=_ConfigModel,
            __config__=ConfigDict(strict=True),
            **fields,
        )


class PlannerCatalog(_ConfigModel):
    """
    Restrict planner component choices and protect human-only settings.

    Parameters
    ----------
    entries : list[CatalogEntry], default=[]
        Entries with unique names. Entries exposing settable fields import their
        component class during validation to inspect its human-only field names.

    Examples
    --------
    >>> from swarm_arena.config import PlannerCatalog
    >>> PlannerCatalog().entries
    []
    """

    entries: list[CatalogEntry] = []

    @model_validator(mode="after")
    def _check_entries(self) -> Self:
        """Check name uniqueness and reject exposed human-only fields."""
        if len({entry.name for entry in self.entries}) != len(self.entries):
            raise PydanticCustomError("arena_config", "duplicate catalog entry names")
        for entry in self.entries:
            if not entry.settable:
                continue
            cls = load_object(entry.ref)
            if not isinstance(cls, type):
                raise ValueError("catalog reference must be a class")
            human_only = getattr(cls, "_human_only", frozenset())
            for key in entry.settable:
                if key in human_only:
                    raise PydanticCustomError(
                        "arena_config",
                        "catalog entry '{entry}': human-only field '{field}' cannot be settable",
                        {"entry": entry.name, "field": key},
                    )
        return self

    def get(self, name: str) -> CatalogEntry:
        """
        Find an entry by its unique name.

        Parameters
        ----------
        name : str
            Catalog entry name.

        Returns
        -------
        CatalogEntry
            Matching catalog entry.

        Raises
        ------
        ValueError
            If the entry name is unknown.

        Examples
        --------
        >>> from swarm_arena.config import CatalogEntry, PlannerCatalog
        >>> entry = CatalogEntry(name='w', kind='tool', ref='demo:Tool', description='Demo')
        >>> PlannerCatalog(entries=[entry]).get('w') is entry
        True
        """
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise ValueError(f"unknown catalog entry {name!r}")

    def match(self, ref: Ref, kind: CatalogKind) -> CatalogEntry | None:
        """
        Find the first entry matching a reference's kind, path, and fixed settings.

        Parameters
        ----------
        ref : Ref
            Component reference with raw configuration values.
        kind : str
            Required component kind.

        Returns
        -------
        CatalogEntry or None
            First match after removing settable keys from the reference config,
            or None. Defaults and settable values are not validated here.

        Notes
        -----
        Fixed JSON values must have matching types and values recursively:
        booleans, integers, and floats are distinct. Mapping order is ignored;
        list order is preserved. Only top-level settable keys are removed.
        Matching does not import components or perform I/O.

        Examples
        --------
        >>> from swarm_arena import Ref
        >>> from swarm_arena.config import CatalogEntry, PlannerCatalog
        >>> entry = CatalogEntry(name='w', kind='tool', ref='demo:Tool', description='Demo')
        >>> PlannerCatalog(entries=[entry]).match(Ref(path='demo:Tool'), 'tool') is entry
        True
        >>> fixed = entry.model_copy(update={"config": {"enabled": True}})
        >>> catalog = PlannerCatalog(entries=[fixed])
        >>> catalog.match(Ref(path='demo:Tool', config={"enabled": 1}), 'tool') is None
        True
        """
        for entry in self.entries:
            if (
                entry.kind == kind
                and entry.ref == ref.path
                and json_equal(
                    {k: v for k, v in ref.config.items() if k not in entry.settable}, entry.config
                )
            ):
                return entry
        return None
