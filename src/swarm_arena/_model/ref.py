"""Reference components by import path and validate their configuration."""

import importlib
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import Annotated, Any, LiteralString, Self, TypeVar, cast, get_args

from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
from pydantic.config import ExtraValues
from pydantic_core import ErrorType, InitErrorDetails, PydanticCustomError

from swarm_arena._errors import ConfigError
from swarm_arena._model.base import FrozenModel, _SafeModel

_CONFIG_ERROR_TYPES = frozenset(get_args(ErrorType))


class EmptyConfig(BaseModel):
    """
    Reject settings for a component with no configurable fields.

    Examples
    --------
    >>> EmptyConfig().model_dump()
    {}
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Configurable:
    """
    Validate component settings without importing referenced code or doing I/O.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None, optional
        Settings validated by the component's nested ``Config`` model.
        None selects that model's defaults.

    Examples
    --------
    >>> Configurable().config.model_dump()
    {}
    """

    Config = EmptyConfig

    def __init__(self, config: BaseModel | Mapping[str, Any] | None = None, /) -> None:
        """Validate supplied settings or construct the default configuration."""
        self._config: BaseModel = self.Config.model_validate({} if config is None else config)

    @property
    def config(self) -> BaseModel:
        """
        Return the component's validated configuration.

        Returns
        -------
        BaseModel
            Configuration model exposed through a read-only property.

        Examples
        --------
        >>> Configurable().config == EmptyConfig()
        True
        """
        return self._config


T = TypeVar("T", bound=Configurable)


def config_or_fields(
    config: BaseModel | Mapping[str, Any] | None, fields: Mapping[str, Any]
) -> BaseModel | Mapping[str, Any]:
    """
    Select explicit configuration or keyword fields for a component constructor.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None
        Explicit configuration, or None when using keyword fields.
    fields : Mapping[str, Any]
        Keyword settings supplied to the constructor.

    Returns
    -------
    BaseModel or Mapping[str, Any]
        The supplied configuration or keyword settings.

    Raises
    ------
    TypeError
        If both an explicit configuration and keyword settings are supplied.

    Examples
    --------
    >>> config_or_fields(None, {"size": 2})
    {'size': 2}
    """
    if config is not None and fields:
        raise TypeError("pass a config or fields, not both")
    return fields if config is None else config


def import_path(obj: object) -> str:
    """
    Return the module and qualified name of an importable object.

    Parameters
    ----------
    obj : object
        A module-level class or function, including nested classes.

    Returns
    -------
    str
        The import path in ``module:QualName`` form.

    Raises
    ------
    TypeError
        If the object is a lambda, local, defined in ``__main__``, or unnamed.

    Examples
    --------
    >>> import_path(str)
    'builtins:str'
    """
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise TypeError("object has no import path")
    if "<lambda>" in qualname:
        raise TypeError("cannot reference a lambda")
    if "<locals>" in qualname:
        raise TypeError("cannot reference a local object")
    if module == "__main__":
        raise TypeError("cannot reference an object defined in __main__")
    return f"{module}:{qualname}"


@cache
def load_object(path: str) -> object:
    """
    Import an object by path and cache successful lookups.

    Parameters
    ----------
    path : str
        The object's ``module:QualName`` path.

    Returns
    -------
    object
        The imported class, function, or attribute.

    Raises
    ------
    ValueError
        If the module or an attribute cannot be imported.

    Examples
    --------
    >>> load_object("builtins:str") is str
    True
    """
    try:
        module, qualname = path.split(":", 1)
        obj: object = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError, ValueError) as err:
        raise ValueError(f"cannot import {path}: {err}") from err
    return obj


def _safe_config_location(
    config_model: type[BaseModel], location: tuple[str | int, ...]
) -> tuple[str | int, ...]:
    """Show only a declared top-level field or alias, redacting all deeper segments."""
    names = set(config_model.model_fields)
    for field in config_model.model_fields.values():
        if field.alias is not None:
            names.add(field.alias)
        alias = field.validation_alias
        choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
        for choice in choices:
            if isinstance(choice, str):
                names.add(choice)
            elif isinstance(choice, AliasPath) and isinstance(choice.path[0], str):
                names.add(choice.path[0])
    first = location[0] if location else None
    name = first if isinstance(first, str) and first in names else "<field>"
    return (name, "<…>") if len(location) > 1 else (name,)


def _config_error_details(model_cls: type[BaseModel], error: Exception) -> list[InitErrorDetails]:
    """Share the component location and built-in error-type rule across entry points."""
    if not isinstance(error, ValidationError):
        return [
            InitErrorDetails(type=PydanticCustomError("invalid", "invalid"), loc=(), input=None)
        ]
    details: list[InitErrorDetails] = []
    for item in error.errors(include_input=False, include_url=False, include_context=False):
        code = cast(
            LiteralString, item["type"] if item["type"] in _CONFIG_ERROR_TYPES else "invalid"
        )
        details.append(
            InitErrorDetails(
                type=PydanticCustomError(code, code),
                loc=_safe_config_location(model_cls, item["loc"]),
                input=None,
            )
        )
    return details


def _sanitize_config_validation(
    cls: type[BaseModel], data: Any, handler: Callable[[Any], Any]
) -> Any:
    """Rebuild validation errors without input, custom messages, or dynamic locations."""
    try:
        return handler(data)
    except Exception as error:
        raise ValidationError.from_exception_data(
            cls.__name__, _config_error_details(cls, error), hide_input=True
        ) from None


class _ConfigInputModel(_SafeModel):
    """Sanitize component and tool-input validation, including JSON parse failures."""

    _sanitize_errors = model_validator(mode="wrap")(_sanitize_config_validation)

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """
        Validate JSON without exposing rejected input in error details.

        Parameters
        ----------
        json_data : str or bytes or bytearray
            JSON document to validate.
        strict : bool or None, optional
            Override strict validation.
        extra : {'allow', 'ignore', 'forbid'} or None, optional
            Override handling of extra fields.
        context : Any, optional
            Context passed to validators.
        by_alias : bool or None, optional
            Accept validation aliases.
        by_name : bool or None, optional
            Accept model field names.

        Returns
        -------
        Self
            Validated model.

        Raises
        ------
        ValidationError
            If parsing or validation fails; diagnostics omit rejected values.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> ScriptTool.Config.model_validate_json('{"command": ["x"]}').command
        ['x']
        """
        validate = super().model_validate_json
        return _sanitize_config_validation(
            cls,
            json_data,
            lambda data: validate(
                data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            ),
        )


def _validate_config(model_cls: type[BaseModel], data: object, *, where: str) -> BaseModel:
    """Validate config while exposing only schema locations and built-in error types.

    Parameters
    ----------
    model_cls : type[BaseModel]
        Component configuration schema.
    data : object
        Untrusted settings to validate.
    where : str
        Safe component identifier for diagnostics, never a configuration value.

    Returns
    -------
    BaseModel
        Validated configuration.

    Raises
    ------
    ConfigError
        If validation raises an exception. Custom error types become ``invalid``;
        input, messages, context, and original exception chains are suppressed.

    Examples
    --------
    >>> _validate_config(EmptyConfig, {}, where="example:Component").model_dump()
    {}
    """
    try:
        return model_cls.model_validate(data)
    except Exception as error:
        details = _config_error_details(model_cls, error)
        summary = "; ".join(
            (".".join(str(part) for part in item.get("loc", ())) + ": " if item.get("loc") else "")
            + str(item["type"])
            for item in details
        )
        raise ConfigError(f"{where}: {summary}") from None


class Ref(FrozenModel):
    """
    Reference a component by its import path and configuration.

    Parameters
    ----------
    path : str
        Component class path in ``module:QualName`` form.
    config : dict[str, Any], optional
        Component settings, validated when the component is built.
    name : str or None, optional
        Explicit component name, overriding the class's name when supplied.

    Examples
    --------
    >>> from swarm_arena import Ref
    >>> Ref(path="my_package.tools:MyTool").config
    {}
    """

    path: str = Field(pattern=r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
    config: dict[str, Any] = {}
    name: str | None = None

    @classmethod
    def of(cls, component: object) -> "Ref":
        """
        Create a reference from a component class or configured instance.

        Parameters
        ----------
        component : object
            Component class or instance implementing the configuration contract.

        Returns
        -------
        Ref
            Import path and, for an instance, its JSON-serializable settings.

        Raises
        ------
        TypeError
            If the value is not a component or has no usable import path.

        Examples
        --------
        >>> from swarm_arena import Ref
        >>> from swarm_arena.executor import LocalExecutor  # doctest: +SKIP
        >>> Ref.of(LocalExecutor).path  # doctest: +SKIP
        'swarm_arena.executor:LocalExecutor'
        """
        if isinstance(component, type):
            return cls(path=import_path(component))
        if isinstance(component, Configurable):
            return cls(
                path=import_path(type(component)), config=component.config.model_dump(mode="json")
            )
        raise TypeError("not a component")

    def load(self) -> type:
        """
        Load the referenced class using the shared import cache.

        Returns
        -------
        type
            The class at this reference's import path.

        Raises
        ------
        ValueError
            If the referenced object cannot be imported.
        TypeError
            If the referenced object is not a class.

        Examples
        --------
        >>> from swarm_arena import Ref
        >>> Ref(path="builtins:str").load() is str
        True
        """
        obj = load_object(self.path)
        if not isinstance(obj, type):
            raise TypeError(f"{self.path} is not a class")
        return obj

    def build(self, base: type[T]) -> T:
        """
        Validate configuration and construct a component of the required base.

        Parameters
        ----------
        base : type[T]
            Required component base class.

        Returns
        -------
        T
            The component initialized with its validated configuration model.

        Raises
        ------
        TypeError
            If the referenced class does not inherit from the required base.
        ValueError
            If importing fails or configuration validation raises ConfigError.
            ConfigError reports the component path, field locations, and error
            types, omitting input values and validator messages or context.
            Locations show only a declared top-level field name or alias,
            or ``<field>`` when unrecognized. All deeper segments are replaced
            together by a single ``<…>`` marker. Only built-in Pydantic error
            types are shown; custom types and other exceptions become ``invalid``.

        Examples
        --------
        >>> from swarm_arena import Ref
        >>> from swarm_arena.executor import LocalExecutor  # doctest: +SKIP
        >>> ref = Ref(path="swarm_arena.executor:LocalExecutor")
        >>> executor = ref.build(LocalExecutor)  # doctest: +SKIP
        """
        cls = self.load()
        if not issubclass(cls, base):
            raise TypeError(f"{self.path} is not a subclass of {base.__name__}")
        validated = _validate_config(cls.Config, self.config, where=self.path)
        return cls(validated)

    def component_name(self) -> str:
        """
        Resolve the explicit name or the component class's default name.

        Returns
        -------
        str
            The name used to identify the component within a plan.

        Raises
        ------
        TypeError
            If no explicit name or string class name exists.
        ValueError
            If the component class cannot be imported.

        Examples
        --------
        >>> from swarm_arena import Ref
        >>> Ref(path="my_package.tools:MyTool", name="worker").component_name()
        'worker'
        """
        if self.name is not None:
            return self.name
        name = getattr(self.load(), "name", None)
        if isinstance(name, str):
            return name
        raise TypeError(f"{self.path} has no name; set Ref.name")


def coerce_ref(value: object) -> object:
    """
    Convert component instances and subclasses to references.

    Parameters
    ----------
    value : object
        Input to a reference field's validation.

    Returns
    -------
    object
        A reference for components, or the original input for other values.

    Examples
    --------
    >>> from swarm_arena import Ref
    >>> ref = Ref(path="my_package.tools:MyTool")
    >>> coerce_ref(ref) is ref
    True
    """
    if isinstance(value, Configurable) or (
        isinstance(value, type) and issubclass(value, Configurable)
    ):
        return Ref.of(value)
    return value


RefField = Annotated[Ref, BeforeValidator(coerce_ref)]


def unique_names(refs: Sequence[Ref]) -> list[str]:
    """
    Resolve component names in order and reject duplicates.

    Parameters
    ----------
    refs : Sequence[Ref]
        References whose component names must be unique.

    Returns
    -------
    list[str]
        Resolved names in the same order as the references.

    Raises
    ------
    ValueError
        If a component name occurs more than once.

    Examples
    --------
    >>> from swarm_arena import Ref
    >>> unique_names([Ref(path="my_package.tools:MyTool", name="worker")])
    ['worker']
    """
    names: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        name = ref.component_name()
        if name in seen:
            raise ValueError(f"duplicate component name {name!r}")
        seen.add(name)
        names.append(name)
    return names
