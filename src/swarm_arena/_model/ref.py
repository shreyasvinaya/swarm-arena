"""Reference components by import path and validate their configuration."""

import importlib
from collections.abc import Mapping, Sequence
from functools import cache
from threading import get_ident
from typing import Annotated, Any, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from pydantic.errors import PydanticErrorMixin
from pydantic_core import PydanticCustomError

from swarm_arena._errors import ConfigError
from swarm_arena._model.base import (
    _VALIDATION_ACTIVE,
    FrozenModel,
    _minimal_validation_error,
    _SafeModel,
    _sanitize_validation_error,
    _ValidationOwner,
)
from swarm_arena._model.data import TomlJsonObject
from swarm_arena._model.diagnostics import validation_summary


class EmptyConfig(_SafeModel):
    """
    Reject settings for a component with no configurable fields.

    Examples
    --------
    >>> EmptyConfig().model_dump()
    {}
    """


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
    except (Exception, SystemExit):
        raise ValueError(f"cannot import {path}") from None
    return obj


class _ConfigInputModel(_SafeModel):
    """Inherit shared component/input validation with hidden unknown keys."""


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
        If input validation fails. Custom error types become ``invalid``;
        input, untrusted messages, context, and original exception chains are
        suppressed. At most 20 details are shown, followed by ``and N more errors``
        when truncated, including errors omitted by explicit nested validation.
        Pydantic developer errors propagate unchanged with their fix-it hints.

    Examples
    --------
    >>> _validate_config(EmptyConfig, {}, where="example:Component").model_dump()
    {}
    """
    # Track explicit safe boundaries so a second sanitization retains their
    # schema locations and counts without trusting arbitrary validator messages.
    owner = _ValidationOwner(get_ident())
    token = _VALIDATION_ACTIVE.set(owner)
    try:
        try:
            return model_cls.model_validate(data)
        except Exception as error:
            if isinstance(error, PydanticErrorMixin):
                raise
            try:
                safe = _sanitize_validation_error(model_cls, error)
            except Exception:
                safe = None
    finally:
        owner.thread_id = None
        _VALIDATION_ACTIVE.reset(token)
    if safe is None:
        safe = _minimal_validation_error(model_cls)
    summary = validation_summary(safe, codes_only=True)
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
    config: TomlJsonObject = {}
    name: str | None = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, name: str | None) -> str | None:
        """Keep explicit component names printable, nonempty and outside human review."""
        if name is not None and (not name or not name.isprintable() or name.startswith("human:")):
            raise ValueError("invalid component name")
        return name

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
        >>> from swarm_arena.executor import LocalExecutor
        >>> Ref.of(LocalExecutor).path
        'swarm_arena.executor:LocalExecutor'
        """
        if isinstance(component, type):
            return cls(path=import_path(component))
        if isinstance(component, Configurable):
            path = import_path(type(component))
            pending: list[Any] = [component.config]
            seen: set[int] = set()
            while pending:
                value = pending.pop()
                if id(value) in seen:
                    continue
                seen.add(id(value))
                if isinstance(value, BaseModel):
                    for name, info in type(value).model_fields.items():
                        child = getattr(value, name)
                        if child is None and info.default is not None:
                            raise ValueError("component config cannot discard explicit null")
                        pending.append(child)
                elif isinstance(value, dict):
                    pending.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    pending.extend(value)
            return cls(
                path=path, config=component.config.model_dump(mode="json", exclude_none=True)
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
            Locations keep schema field names, aliases and list/tuple indices.
            Unknown model keys become ``<field>`` and free-form mapping or Any
            segments become ``<…>``. Only built-in Pydantic error types are shown;
            custom types and other exceptions become ``invalid`` in this summary.

        Examples
        --------
        >>> from swarm_arena import Ref
        >>> from swarm_arena.executor import LocalExecutor
        >>> ref = Ref(path="swarm_arena.executor:LocalExecutor")
        >>> executor = ref.build(LocalExecutor)
        >>> isinstance(executor, LocalExecutor)
        True
        """
        cls, validated = self._resolve_config(base)
        return cls(validated)

    def _resolve_config(self, base: type[T]) -> tuple[type[T], BaseModel]:
        """Resolve, check and validate a component without building or rewrapping imports."""
        cls = self.load()
        if not issubclass(cls, base):
            raise TypeError(f"{self.path} is not a subclass of {base.__name__}")
        return cls, _validate_config(cls.Config, self.config, where=self.path)

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


_REF_PATH_ADAPTER: TypeAdapter[str] = TypeAdapter(Ref.model_fields["path"].rebuild_annotation())


def is_ref_path(value: object) -> bool:
    """
    Recognize a complete Ref path without importing components or doing I/O.

    Parameters
    ----------
    value : object
        Potential component path in ``module:QualName`` form.

    Returns
    -------
    bool
        Whether the value is a string accepted by the ``Ref.path`` field's
        Pydantic constraint, including its Unicode word-character semantics.
        Non-string inputs are rejected without coercion.

    Examples
    --------
    >>> is_ref_path("demo.tools:Outer.Inner")
    True
    >>> is_ref_path("demo:Thing/other")
    False
    """
    if not isinstance(value, str):
        return False
    try:
        _REF_PATH_ADAPTER.validate_python(value)
    except ValidationError:
        return False
    return True


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
        try:
            return Ref.of(value)
        except TypeError as error:
            # Only these fixed importability reasons are allowed out of the boundary.
            reason = str(error)
            if reason not in {
                "cannot reference a lambda",
                "cannot reference a local object",
                "cannot reference an object defined in __main__",
            }:
                reason = "component has no importable class"
            raise PydanticCustomError("ref_not_importable", reason) from None
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
