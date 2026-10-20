"""Store write-once arena pointers and validate shared path components."""

import re
from pathlib import PurePosixPath
from typing import ClassVar
from urllib.parse import urlsplit

from pydantic import field_validator

from swarm_arena._errors import AlreadyExists, StorageError, _core_error
from swarm_arena._model.base import StoredModel
from swarm_arena._model.canonical import canonical_json
from swarm_arena._store.base import Store


def _validate_namespace(namespace: str) -> str:
    """Validate the same lowercase namespace slug used by Goal."""
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", namespace) is None:
        raise _core_error(ValueError, "namespace must be a lowercase slug")
    return namespace


def _validate_run_id(run_id: str) -> str:
    """Require a single nonempty, printable POSIX key component."""
    if (
        not run_id
        or run_id in {".", ".."}
        or any(char in "/\\" or not char.isprintable() for char in run_id)
    ):
        raise _core_error(ValueError, "run_id must be a single relative key component")
    return run_id


class RunPointer(StoredModel):
    """
    Locate a run through a write-once arena object.

    Parameters
    ----------
    run_id : str
        Run identifier, a single relative key component.
    run_root : str
        Absolute local path or S3 root reachable by arena readers.
    schema_version : int, default=1
        Stored-format version.

    Examples
    --------
    >>> RunPointer(run_id="r1", run_root="/runs").schema_version
    1
    """

    schema_kind: ClassVar[str] = "run_pointer"
    run_id: str
    run_root: str

    @field_validator("run_id")
    @classmethod
    def _check_run_id(cls, value: str) -> str:
        """Validate before the identifier can enter a pointer key."""
        return _validate_run_id(value)

    @field_validator("run_root")
    @classmethod
    def _check_run_root(cls, value: str) -> str:
        """Accept absolute local roots and credential-free S3 roots."""
        if PurePosixPath(value).is_absolute():
            return value
        try:
            url = urlsplit(value)
            valid = (
                value.startswith("s3://")
                and bool(url.netloc)
                and url.username is None
                and url.password is None
                and not url.query
                and not url.fragment
            )
        except ValueError:
            valid = False
        if not valid:
            raise _core_error(ValueError, "run_root must be absolute or s3://") from None
        return value


def pointer_key(namespace: str, run_id: str) -> str:
    """
    Construct a validated arena-relative run pointer key.

    Parameters
    ----------
    namespace : str
        Lowercase namespace slug.
    run_id : str
        Single relative key component.

    Returns
    -------
    str
        Pointer key below the namespace's runs directory.

    Examples
    --------
    >>> pointer_key("toy", "r1")
    'toy/runs/r1.json'
    """
    return f"{_validate_namespace(namespace)}/runs/{_validate_run_id(run_id)}.json"


def write_pointer(arena: Store, namespace: str, pointer: RunPointer) -> None:
    """
    Create a pointer, accepting an existing byte-identical pointer.

    Parameters
    ----------
    arena : Store
        Arena storage.
    namespace : str
        Lowercase namespace slug.
    pointer : RunPointer
        Run location to publish.

    Raises
    ------
    StorageError
        If the existing pointer has different bytes.

    Examples
    --------
    >>> write_pointer(arena, "toy", RunPointer(run_id="r1", run_root="/runs"))  # doctest: +SKIP
    """
    key = pointer_key(namespace, pointer.run_id)
    data = canonical_json(pointer.model_dump(mode="json"))
    try:
        arena.create(key, data)
    except AlreadyExists:
        if arena.read(key).data != data:
            raise _core_error(StorageError, f"conflicting run pointer: {key}") from None


def list_pointers(arena: Store, namespace: str) -> tuple[list[RunPointer], list[str]]:
    """
    Read valid pointers in run order and report broken neighbors safely.

    Parameters
    ----------
    arena : Store
        Arena storage.
    namespace : str
        Lowercase namespace slug.

    Returns
    -------
    tuple[list[RunPointer], list[str]]
        Pointers sorted by run identifier and key-prefixed warnings. Invalid,
        newer-version, mismatched-location, and unreadable pointers are skipped.

    Examples
    --------
    >>> pointers, warnings = list_pointers(arena, "toy")  # doctest: +SKIP
    """
    prefix = f"{_validate_namespace(namespace)}/runs/"
    pointers: list[RunPointer] = []
    warnings: list[str] = []
    for key in sorted(arena.list(prefix)):
        try:
            pointer = RunPointer.model_validate_json(arena.read(key).data)
            if key != pointer_key(namespace, pointer.run_id):
                raise _core_error(ValueError, "pointer location does not match run_id")
        except (ValueError, StorageError) as error:
            # Exception text can carry malformed JSON, input values, or connector data.
            reason = "unreadable pointer" if isinstance(error, StorageError) else "invalid pointer"
            warnings.append(f"{key}: {reason}")
            continue
        pointers.append(pointer)
    return sorted(pointers, key=lambda pointer: pointer.run_id), warnings
