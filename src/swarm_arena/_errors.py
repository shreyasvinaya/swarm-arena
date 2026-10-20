"""Define user-visible plan, configuration, action, and storage errors."""

from collections.abc import Sequence
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


_CORE_SIGNALS: WeakKeyDictionary[
    Exception, tuple[type[Exception], tuple[str, ...], str | None, tuple[str, ...] | None]
] = WeakKeyDictionary()
_SignalT = TypeVar("_SignalT", bound=Exception)


def _core_error(cls: type[_SignalT], *args: Any) -> _SignalT:
    """Construct and register one vetted error, including weak runtime errors."""
    if cls is RuntimeError:
        return cast(_SignalT, _core_signal(_CoreRuntimeError(*args)))
    if cls is ValueError:
        return cast(_SignalT, _core_signal(_CoreValueError(*args)))
    return _core_signal(cls(*args))


def _core_signal(error: _SignalT) -> _SignalT:
    """Register a freshly core-created signal and snapshot its vetted payload.

    Call only at core construction sites, never on exceptions caught from
    connectors. Weak keys retain provenance without retaining the exception.
    The payload is stored outside its transferable, mutable attributes.
    LimitReached is imported lazily to avoid the model/error import cycle.
    """
    from swarm_arena._agent.manager import LimitReached

    kinds = (
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
    if not any(type(error) is kind for kind in kinds):
        raise TypeError("expected a core error")
    if error not in _CORE_SIGNALS:
        _CORE_SIGNALS[error] = (
            type(error),
            error.args,
            error.limit if isinstance(error, LimitReached) else None,
            tuple(error.errors) if isinstance(error, PlanError) else None,
        )
    return error


def _rebuild_core_signal(error: BaseException) -> Exception | None:
    """Reconstruct a registered signal from its snapshot, never user changes.

    The storage family preserves its exact core type and vetted message.
    Only the exact supported core exception types may be registry keys: subclasses
    cannot supply custom equality or hashing to impersonate a registered key.
    The fresh result has no notes or exception chain and is registered so
    nested action boundaries preserve the same vetted payload.
    """
    from swarm_arena._agent.manager import LimitReached

    kinds = (
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
    if not any(type(error) is kind for kind in kinds):
        return None
    payload = _CORE_SIGNALS.get(cast(Exception, error))
    if payload is None:
        return None
    kind, args, limit, errors = payload
    if kind is LimitReached:
        assert limit is not None
        return _core_error(LimitReached, limit, args[0])
    if kind is PlanError:
        assert errors is not None
        return _core_error(PlanError, errors)
    return _core_error(kind, *args)
