"""Define user-visible plan, configuration, action, and storage errors."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, TypeVar, cast
from weakref import WeakKeyDictionary


class PlanError(ValueError):
    r"""
    Report all validation errors in a plan.

    Parameters
    ----------
    errors : Sequence[str]
        Individual plan validation errors, copied into a list.

    Examples
    --------
    >>> str(PlanError(["missing goal", "missing team"]))
    'invalid plan:\n- missing goal\n- missing team'
    """

    errors: list[str]

    def __init__(self, errors: Sequence[str]) -> None:
        """Copy the errors and build a readable validation message."""
        self.errors = list(errors)
        super().__init__("invalid plan:\n" + "\n".join(f"- {error}" for error in self.errors))


class ConfigError(ValueError):
    """
    Report an invalid component configuration.

    Examples
    --------
    >>> str(ConfigError("unknown setting"))
    'unknown setting'
    """


class SoftError(ValueError):
    """
    Report a recoverable action error.

    Examples
    --------
    >>> str(SoftError("path is protected"))
    'path is protected'
    """


class StorageError(RuntimeError):
    """
    Report a storage operation failure.

    Examples
    --------
    >>> str(StorageError("storage unavailable"))
    'storage unavailable'
    """


class NotFound(StorageError):
    """
    Report a missing storage key.

    Examples
    --------
    >>> isinstance(NotFound("missing key"), StorageError)
    True
    """


class AlreadyExists(StorageError):
    """
    Report a key that already exists during a conditional create.

    Examples
    --------
    >>> isinstance(AlreadyExists("existing key"), StorageError)
    True
    """


class PreconditionFailed(StorageError):
    """
    Report a storage version or ETag mismatch.

    Examples
    --------
    >>> isinstance(PreconditionFailed("version mismatch"), StorageError)
    True
    """


class WriteOutcomeUnknown(StorageError):
    """
    Report a write whose outcome could not be confirmed.

    Examples
    --------
    >>> isinstance(WriteOutcomeUnknown("response lost"), StorageError)
    True
    """


class ConditionalWritesUnsupported(StorageError):
    """
    Report a backend that cannot guarantee conditional writes.

    Examples
    --------
    >>> isinstance(ConditionalWritesUnsupported("unsupported backend"), StorageError)
    True
    """


class ChainError(StorageError):
    """
    Report an invalid committed journal chain.

    Examples
    --------
    >>> isinstance(ChainError("journal chain is invalid"), StorageError)
    True
    """


class Fenced(RuntimeError):
    """
    Report that the current owner has lost the run.

    Examples
    --------
    >>> isinstance(Fenced("run ownership lost"), RuntimeError)
    True
    """


class LeaseHeld(RuntimeError):
    """
    Report that a live owner already holds the run lease.

    Examples
    --------
    >>> isinstance(LeaseHeld("run lease is held"), RuntimeError)
    True
    """


class _CoreRuntimeError(RuntimeError):
    """Allow core runtime errors to be weakly registered."""


class _CoreValueError(ValueError):
    """Allow core value errors to be weakly registered."""


_CORE_STORAGE_ERRORS: tuple[type[StorageError], ...] = (
    StorageError,
    NotFound,
    AlreadyExists,
    PreconditionFailed,
    WriteOutcomeUnknown,
    ConditionalWritesUnsupported,
    ChainError,
)


class _SignalCategory(Enum):
    """Core-owned authority, independent of an exception's public shape."""

    DIAGNOSTIC = auto()
    LIMIT = auto()
    TERMINAL = auto()
    FENCED = auto()


@dataclass(frozen=True)
class _SignalSnapshot:
    kind: type[Exception]
    args: tuple[str, ...]
    limit: str | None
    errors: tuple[str, ...] | None
    name: str
    category: _SignalCategory


_CORE_SIGNALS: WeakKeyDictionary[Exception, _SignalSnapshot] = WeakKeyDictionary()
_SignalT = TypeVar("_SignalT", bound=Exception)

# Optional transports register their exact types without importing their SDK here.
_CARRIER_TYPES: dict[type[Exception], str] = {}
_CARRIERS: WeakKeyDictionary[Exception, tuple[type[Exception], Exception]] = WeakKeyDictionary()


def _register_carrier_type(kind: type[Exception], name: str) -> None:
    """Declare a transport type and its public diagnostic name at module load."""
    _CARRIER_TYPES[kind] = name


def _carrier_name(error: BaseException) -> str | None:
    """Recognize carrier families for diagnostics only, never for authority."""
    return next(
        (
            _CARRIER_TYPES[cast(type[Exception], kind)]
            for kind in type(error).__mro__
            if kind in _CARRIER_TYPES
        ),
        None,
    )


def _register_carrier(carrier: Exception, core: Exception) -> None:
    """Retain a vetted control outside the transport's transferable attributes."""
    if type(carrier) not in _CARRIER_TYPES or not _is_core_control(core):
        raise TypeError("expected a registered transport and core control")
    _CARRIERS[carrier] = (type(carrier), core)


def _unwrap_control(error: BaseException) -> BaseException:
    """Restore only exact issued carriers, recursively through exception groups."""
    pending = [(error, False)]
    restored: dict[int, BaseException] = {}
    while pending:
        current, visited = pending.pop()
        result = current
        if isinstance(current, BaseExceptionGroup):
            if not visited:
                pending.append((current, True))
                pending.extend((child, False) for child in current.exceptions)
                continue
            children = [restored[id(child)] for child in current.exceptions]
            if any(
                child is not original
                for child, original in zip(children, current.exceptions, strict=True)
            ):
                result = BaseExceptionGroup("SDK outcomes", children)
        elif type(current) in _CARRIER_TYPES:
            saved = _CARRIERS.get(cast(Exception, current))
            if saved is not None and type(current) is saved[0]:
                rebuilt = _rebuild_core_signal(saved[1])
                assert rebuilt is not None
                result = rebuilt
        restored[id(current)] = result
    return restored[id(error)]


def _core_error(
    cls: type[_SignalT], *args: Any, _category: _SignalCategory | None = None
) -> _SignalT:
    """Construct and register one vetted error, including weak runtime errors."""
    if cls is RuntimeError:
        return cast(_SignalT, _core_signal(_CoreRuntimeError(*args), _category=_category))
    if cls is ValueError:
        return cast(_SignalT, _core_signal(_CoreValueError(*args), _category=_category))
    return _core_signal(cls(*args), _category=_category)


def _stop_cleanup_error() -> RuntimeError:
    """Build the shared safe error for exhausted stop-storage settlement."""
    return _core_error(RuntimeError, "stop cleanup timed out", _category=_SignalCategory.TERMINAL)


def _core_signal_kinds() -> tuple[type[Exception], ...]:
    """Return exact core types with a lazy import to avoid model cycles."""
    from swarm_arena._agent.manager import LimitReached

    return (
        SoftError,
        LimitReached,
        PlanError,
        ConfigError,
        LeaseHeld,
        Fenced,
        *_CORE_STORAGE_ERRORS,
        _CoreRuntimeError,
        _CoreValueError,
    )


def _core_signal(error: _SignalT, *, _category: _SignalCategory | None = None) -> _SignalT:
    """Register a freshly core-created signal and snapshot its vetted payload.

    Call only at core construction sites, never on exceptions caught from
    connectors. Weak keys retain provenance without retaining the exception.
    The payload is stored outside its transferable, mutable attributes.
    LimitReached is imported lazily to avoid the model/error import cycle.
    """
    from swarm_arena._agent.manager import LimitReached

    if not any(type(error) is kind for kind in _core_signal_kinds()):
        raise TypeError("expected a core error")
    if error not in _CORE_SIGNALS:
        category = _category
        if category is None:
            category = (
                _SignalCategory.FENCED
                if type(error) is Fenced
                else _SignalCategory.LIMIT
                if type(error) is LimitReached
                else _SignalCategory.DIAGNOSTIC
            )
        _CORE_SIGNALS[error] = _SignalSnapshot(
            type(error),
            error.args,
            error.limit if isinstance(error, LimitReached) else None,
            tuple(error.errors) if isinstance(error, PlanError) else None,
            "RuntimeError"
            if type(error) is _CoreRuntimeError
            else "ValueError"
            if type(error) is _CoreValueError
            else type(error).__name__,
            category,
        )
    return error


def _exception_type_name(error: BaseException) -> str:
    """Read a preserved originating type without trusting exception attributes."""
    if any(type(error) is kind for kind in _core_signal_kinds()):
        payload = _CORE_SIGNALS.get(cast(Exception, error))
        if payload is not None:
            return payload.name
    if type(error) is _CoreRuntimeError:
        return "RuntimeError"
    if type(error) is _CoreValueError:
        return "ValueError"
    return _carrier_name(error) or type(error).__name__


def _with_exception_type(error: _SignalT, name: str) -> _SignalT:
    """Attach an originating type to a freshly registered, sanitized failure."""
    _CORE_SIGNALS[error] = replace(_CORE_SIGNALS[error], name=name)
    return error


def _is_core_signal(error: BaseException) -> bool:
    """Check exact-type provenance without allocating another signal."""
    return (
        any(type(error) is kind for kind in _core_signal_kinds())
        and cast(Exception, error) in _CORE_SIGNALS
    )


def _core_category(error: BaseException) -> _SignalCategory | None:
    """Read authority only from the exact-type, nontransferable snapshot."""
    if not _is_core_signal(error):
        return None
    return _CORE_SIGNALS[cast(Exception, error)].category


def _is_core_control(error: BaseException) -> bool:
    """Distinguish real controls from safely reconstructible diagnostics."""
    category = _core_category(error)
    return category is not None and category is not _SignalCategory.DIAGNOSTIC


def _rebuild_core_signal(error: BaseException) -> Exception | None:
    """Reconstruct a registered signal from its snapshot, never user changes.

    The storage family preserves its exact core type and vetted message.
    Only the exact supported core exception types may be registry keys: subclasses
    cannot supply custom equality or hashing to impersonate a registered key.
    The fresh result has no notes or exception chain and is registered so
    nested action boundaries preserve the same vetted payload.
    """
    from swarm_arena._agent.manager import LimitReached

    if not any(type(error) is kind for kind in _core_signal_kinds()):
        return None
    payload = _CORE_SIGNALS.get(cast(Exception, error))
    if payload is None:
        return None
    kind, args, limit, errors = payload.kind, payload.args, payload.limit, payload.errors
    if kind is LimitReached:
        assert limit is not None
        rebuilt = _core_error(LimitReached, limit, args[0], _category=payload.category)
    elif kind is PlanError:
        assert errors is not None
        rebuilt = _core_error(PlanError, errors, _category=payload.category)
    else:
        rebuilt = _core_error(kind, *args, _category=payload.category)
    return _with_exception_type(rebuilt, payload.name)
