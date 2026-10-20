"""Validate, persist, and order write-once human review and curation objects."""

import re
import secrets
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import ClassVar, Literal, Self

from pydantic import model_validator

from swarm_arena._board.pointers import _validate_namespace
from swarm_arena._errors import _core_error
from swarm_arena._model.base import StoredModel
from swarm_arena._model.canonical import canonical_json, format_utc
from swarm_arena._model.results import Learning, Score
from swarm_arena._store.base import Store

_Folder = Literal["reviews", "curation"]


def _validate_reviewer(reviewer: str | None) -> str:
    """Require the shared lowercase reviewer slug."""
    if reviewer is None or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", reviewer) is None:
        raise _core_error(ValueError, "reviewer must be a lowercase slug")
    return reviewer


def _validate_basename(basename: str) -> str:
    """Require the reserved UTC-millisecond timestamp and random suffix."""
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}\.[0-9]{3}Z-[0-9a-f]{8}\.jsonl", basename) is None:
        raise _core_error(ValueError, "invalid human object basename")
    try:
        datetime.strptime(basename[:20], "%Y%m%dT%H%M%S.%fZ")
    except ValueError:
        raise _core_error(ValueError, "invalid human object basename timestamp") from None
    return basename


def _validate_human_key(name: str) -> _Folder:
    """Reject malformed reserved locations before any storage access."""
    parts = name.split("/")
    if len(parts) == 3 and parts[1] == "curation":
        expected = human_object_key(parts[0], "curation", parts[2])
        folder: _Folder = "curation"
    elif len(parts) == 4 and parts[1] == "reviews":
        expected = human_object_key(parts[0], "reviews", parts[3], reviewer=parts[2])
        folder = "reviews"
    else:
        raise _core_error(ValueError, "invalid human object location")
    if name != expected:
        raise _core_error(ValueError, "invalid human object location")
    return folder


class HumanObject(StoredModel):
    """
    Store exactly one human operation and its kind-specific payload.

    Parameters
    ----------
    kind : {'pin', 'hide', 'add', 'review'}
        Operation to apply.
    learning_id : str or None, default=None
        Required nonempty target for pin/hide; forbidden for add/review.
    learning : Learning or None, default=None
        Required only for add.
    score : Score or None, default=None
        Required only for review.
    author : str
        Human author identifier.
    schema_version : int, default=1
        Stored-format version.

    Examples
    --------
    >>> HumanObject(kind="pin", learning_id="L-1", author="human:alice").kind
    'pin'
    """

    schema_kind: ClassVar[str] = "human_object"
    kind: Literal["pin", "hide", "add", "review"]
    learning_id: str | None = None
    learning: Learning | None = None
    score: Score | None = None
    author: str

    @model_validator(mode="after")
    def _check_payload(self) -> Self:
        """Require only the payload associated with the operation."""
        if self.kind in {"pin", "hide"}:
            valid = (
                bool(self.learning_id and self.learning_id.strip())
                and self.learning is None
                and self.score is None
            )
        elif self.kind == "add":
            valid = self.learning is not None and self.learning_id is None and self.score is None
        else:
            valid = self.score is not None and self.learning_id is None and self.learning is None
        if not valid:
            raise _core_error(ValueError, "kind requires its matching payload only")
        return self


def human_object_name(now: datetime) -> str:
    """
    Generate a UTC-millisecond basename with a random eight-hex suffix.

    Parameters
    ----------
    now : datetime
        Timezone-aware timestamp, normalized to UTC and truncated to milliseconds.

    Returns
    -------
    str
        Reserved JSONL basename.

    Examples
    --------
    >>> from datetime import UTC
    >>> human_object_name(datetime(2026, 10, 18, 13, tzinfo=UTC))[:21]
    '20261018T130000.000Z-'
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise _core_error(ValueError, "datetime must be timezone-aware")
    stamp = format_utc(now).replace("-", "").replace(":", "")
    return f"{stamp}-{secrets.token_hex(4)}.jsonl"


def human_object_order(name: str) -> tuple[str, str]:
    """
    Return the timestamp and full-name tie breaker for human objects.

    Parameters
    ----------
    name : str
        Arena-relative object name. Synthetic labels can also order in-memory ops.

    Returns
    -------
    tuple[str, str]
        Basename timestamp, then full name.

    Examples
    --------
    >>> human_object_order("toy/curation/20261018T130000.000Z-0a1b2c3d.jsonl")[0]
    '20261018T130000.000Z'
    """
    return name.rsplit("/", 1)[-1].split("-", 1)[0], name


def human_object_key(
    namespace: str, folder: _Folder, basename: str, *, reviewer: str | None = None
) -> str:
    """
    Construct a validated review or curation object location.

    Parameters
    ----------
    namespace : str
        Lowercase namespace slug.
    folder : {'reviews', 'curation'}
        Reserved object folder.
    basename : str
        UTC-millisecond timestamp and eight-hex suffix with JSONL extension.
    reviewer : str or None, default=None
        Required lowercase slug for reviews; forbidden for curation.

    Returns
    -------
    str
        Arena-relative key.

    Examples
    --------
    >>> human_object_key("toy", "curation", "20261018T130000.000Z-0a1b2c3d.jsonl")
    'toy/curation/20261018T130000.000Z-0a1b2c3d.jsonl'
    """
    namespace = _validate_namespace(namespace)
    basename = _validate_basename(basename)
    if folder == "reviews":
        return f"{namespace}/reviews/{_validate_reviewer(reviewer)}/{basename}"
    if folder != "curation":
        raise _core_error(ValueError, "folder must be reviews or curation")
    if reviewer is not None:
        raise _core_error(ValueError, "curation must not have a reviewer")
    return f"{namespace}/curation/{basename}"


def _check_folder_payload(folder: _Folder, objects: Iterable[HumanObject]) -> None:
    """Keep review scores out of curation and curation operations out of reviews."""
    if any((obj.kind == "review") != (folder == "reviews") for obj in objects):
        raise _core_error(ValueError, "human object kind does not match folder")


def write_human_objects(
    arena: Store,
    namespace: str,
    folder: _Folder,
    objects: Sequence[HumanObject],
    *,
    now: datetime,
    reviewer: str | None = None,
    name: str | None = None,
) -> str:
    """
    Publish deterministic JSONL bytes once under a generated or supplied name.

    Parameters
    ----------
    arena : Store
        Arena storage.
    namespace : str
        Lowercase namespace slug.
    folder : {'reviews', 'curation'}
        Object folder; reviews carry review operations only.
    objects : Sequence[HumanObject]
        Validated operations in line order.
    now : datetime
        Aware timestamp used when generating a basename.
    reviewer : str or None, default=None
        Required for reviews; forbidden for curation.
    name : str or None, default=None
        Explicit basename for a caller that must record the key before writing.

    Returns
    -------
    str
        Created arena-relative key. An occupied key raises ``AlreadyExists``.

    Examples
    --------
    >>> key = write_human_objects(arena, "toy", "curation", [], now=now)  # doctest: +SKIP
    """
    key = human_object_key(
        namespace, folder, human_object_name(now) if name is None else name, reviewer=reviewer
    )
    _check_folder_payload(folder, objects)
    data = b"".join(canonical_json(obj.model_dump(mode="json")) + b"\n" for obj in objects)
    arena.create(key, data)
    return key


def list_human_objects(arena: Store, namespaces: Sequence[str]) -> tuple[list[str], list[str]]:
    """
    List valid reserved review and curation keys in fixed application order.

    Parameters
    ----------
    arena : Store
        Arena storage.
    namespaces : Sequence[str]
        Namespace slugs to include, with duplicates ignored.

    Returns
    -------
    tuple[list[str], list[str]]
        Review keys and curation keys, each ordered by timestamp then full name.
        Malformed reserved locations are rejected.

    Examples
    --------
    >>> reviews, curation = list_human_objects(arena, ["toy"])  # doctest: +SKIP
    """
    namespaces = sorted({_validate_namespace(namespace) for namespace in namespaces})
    reviews: set[str] = set()
    curation: set[str] = set()
    for namespace in namespaces:
        for folder, target in (("reviews", reviews), ("curation", curation)):
            for key in arena.list(f"{namespace}/{folder}/"):
                if _validate_human_key(key) != folder or not key.startswith(
                    f"{namespace}/{folder}/"
                ):
                    raise _core_error(ValueError, "invalid human object location")
                target.add(key)
    return sorted(reviews, key=human_object_order), sorted(curation, key=human_object_order)


def read_human_object(arena: Store, name: str) -> list[HumanObject]:
    """
    Load every JSONL operation through stored-version and finite-JSON validation.

    Parameters
    ----------
    arena : Store
        Arena storage.
    name : str
        Validated arena-relative review or curation key.

    Returns
    -------
    list[HumanObject]
        Validated operations in file order. Empty files contain no operations.

    Examples
    --------
    >>> objects = read_human_object(arena, key)  # doctest: +SKIP
    """
    folder = _validate_human_key(name)
    objects = [HumanObject.model_validate_json(line) for line in arena.read(name).data.splitlines()]
    _check_folder_payload(folder, objects)
    return objects


def fold_curation(ops: Iterable[tuple[str, int, HumanObject]]) -> dict[str, tuple[bool, bool]]:
    """
    Fold pins and hides by timestamp, full name, and JSONL line position.

    Parameters
    ----------
    ops : Iterable[tuple[str, int, HumanObject]]
        Object name, line index, and operation triples, in any input order.

    Returns
    -------
    dict[str, tuple[bool, bool]]
        Learning identifier to pinned/hidden flags; later pin or hide wins.
        Add and review operations do not change these flags.

    Examples
    --------
    >>> pin = HumanObject(kind="pin", learning_id="L-1", author="human:alice")
    >>> fold_curation([("n1", 0, pin)])
    {'L-1': (True, False)}
    """
    result: dict[str, tuple[bool, bool]] = {}
    for _, _, obj in sorted(ops, key=lambda op: (human_object_order(op[0]), op[1])):
        if obj.kind in {"pin", "hide"}:
            assert obj.learning_id is not None
            result[obj.learning_id] = (obj.kind == "pin", obj.kind == "hide")
    return result
