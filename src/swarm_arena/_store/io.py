"""Shared abandonable storage boundaries for supervised asynchronous owners."""

from collections.abc import Callable
from typing import Any, NoReturn, TypeVar

import anyio

from swarm_arena._errors import StorageError, _exception_type_name, _with_exception_type
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._outcomes import _Outcome
from swarm_arena._store.base import Store

_T = TypeVar("_T")


def _contain_storage(outcome: _Outcome, error: BaseException, site: str) -> NoReturn:
    """Contain a backend outcome while retaining only its type for retry policy."""
    try:
        outcome.contain(error, StorageError, site)
    except StorageError as contained:
        raise _with_exception_type(contained, _exception_type_name(error)) from None


async def _storage_io(operation: Callable[..., _T], *args: Any, site: str = "storage") -> _T:
    """Await normalized worker I/O under the caller's stop, lease and clock scopes."""
    outcome = _Outcome()
    await anyio.lowlevel.checkpoint_if_cancelled()
    try:
        return await run_sync_daemon(operation, *args)
    except BaseException as error:
        _contain_storage(outcome, error, site)


async def _validated_io(operation: Callable[..., _T], *args: Any) -> _T:
    """Preserve trusted validation; its backend reads must contain their own failures."""
    outcome = _Outcome()
    await anyio.lowlevel.checkpoint_if_cancelled()
    try:
        return await run_sync_daemon(operation, *args)
    except BaseException as error:
        outcome.propagate_control(error)
        raise


def _etag(value: object) -> str:
    """Detach a backend version identifier before core code uses it."""
    if not isinstance(value, str):
        raise TypeError("invalid storage version")
    return str.__str__(value)


def _create(store: Store, key: str, data: bytes) -> str:
    """Create one object and normalize the returned version in the worker."""
    return _etag(store.create(key, data))


def _replace(store: Store, key: str, data: bytes, *, if_match: str) -> str:
    """Replace one object and normalize the returned version in the worker."""
    return _etag(store.replace(key, data, if_match=if_match))
