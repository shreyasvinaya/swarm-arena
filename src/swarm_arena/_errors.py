"""Define user-visible plan, configuration, action, and storage errors."""

from collections.abc import Sequence


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
