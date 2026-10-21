"""Encode source heads and publish views with committed-chain conditional writes."""

import json
import re
import tomllib
from functools import partial
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any, Literal

import anyio
import anyio.lowlevel

from swarm_arena._errors import AlreadyExists, PreconditionFailed, StorageError, _core_error
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.lease import Head
from swarm_arena._journal.segment import parse_segment_name
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.data import finite_json
from swarm_arena._outcomes import _Outcome
from swarm_arena._store.base import Store, _read_blob

_Format = Literal["json", "toml", "md"]
_FORMATS: dict[str, _Format] = {".json": "json", ".toml": "toml", ".md": "md"}
_TEXT_HEAD = r"e([1-9][0-9]*)/([1-9][0-9]*)/(journal/e[1-9][0-9]*-[1-9][0-9]*\.jsonl)"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Refuse ambiguous duplicate JSON fields."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _json_object(text: str) -> dict[str, Any]:
    """Decode one finite JSON object without coercion."""
    value = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("view JSON must be an object")
    finite_json(value)
    return value


def _validated_head(value: object) -> Head:
    """Require strict head fields agreeing with a canonical segment key."""
    head = Head.model_validate(value, strict=True)
    if parse_segment_name(head.segment) != (head.epoch, head.seq):
        raise ValueError("source head does not match segment")
    return head


def embed_source_head(fmt: _Format, content: bytes, source: Head) -> bytes:
    r"""
    Embed a canonical head into a JSON object or a leading text comment.

    JSON is validated as finite data and serialized canonically. TOML and
    Markdown retain their original body bytes after the new first line.
    This helper validates the head; chain membership belongs to ``ViewWriter``.

    Parameters
    ----------
    fmt : {"json", "toml", "md"}
        Exact lowercase format name.
    content : bytes
        UTF-8 document; JSON requires an object and TOML requires valid syntax.
    source : Head
        Source epoch and sequence matching its canonical segment name.

    Returns
    -------
    bytes
        Document containing the source head.

    Raises
    ------
    ValueError
        If the format, document, or head is invalid, without exposing values.

    Examples
    --------
    >>> head = Head(epoch=1, seq=10, segment="journal/e1-10.jsonl")
    >>> embed_source_head("toml", b"a = 1\n", head).splitlines()[0]
    b'# swarm-arena source_head: e1/10/journal/e1-10.jsonl'
    >>> read_source_head("json", embed_source_head("json", b"{}", head)) == head
    True
    """
    try:
        head = _validated_head(source.model_dump())
        text = content.decode("utf-8", errors="strict")
        if fmt == "json":
            value = _json_object(text)
            value["source_head"] = head.model_dump()
            return canonical_json(value)
        marker = f"swarm-arena source_head: e{head.epoch}/{head.seq}/{head.segment}"
        if fmt == "toml":
            tomllib.loads(text)
            prefix = f"# {marker}\n"
        elif fmt == "md":
            prefix = f"<!-- {marker} -->\n"
        else:
            raise ValueError("unsupported view format")
        return prefix.encode("utf-8") + content
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ValueError("invalid view format, content, or source head") from None


def read_source_head(fmt: _Format, content: bytes) -> Head | None:
    """
    Read a strict, canonical source head, returning None for malformed views.

    Validate the complete document's UTF-8 and JSON or TOML syntax. Text
    metadata must occupy the first line exactly. Head fields are never coerced,
    and their epoch and sequence must agree with the canonical segment name.

    Parameters
    ----------
    fmt : {"json", "toml", "md"}
        Exact lowercase format name.
    content : bytes
        Stored view document.

    Returns
    -------
    Head or None
        Valid head, or None for absent metadata, invalid content or format.

    Examples
    --------
    >>> read_source_head("json", b'{"source_head": null}') is None
    True
    >>> head = Head(epoch=1, seq=10, segment="journal/e1-10.jsonl")
    >>> read_source_head("md", embed_source_head("md", b"# Result", head)) == head
    True
    """
    try:
        text = content.decode("utf-8", errors="strict")
        if fmt == "json":
            return _validated_head(_json_object(text).get("source_head"))
        if fmt == "toml":
            tomllib.loads(text)
            pattern = rf"# swarm-arena source_head: {_TEXT_HEAD}"
        elif fmt == "md":
            pattern = rf"<!-- swarm-arena source_head: {_TEXT_HEAD} -->"
        else:
            return None
        first_line = text.partition("\n")[0].removesuffix("\r")
        match = re.fullmatch(pattern, first_line)
        if match is None:
            return None
        epoch, seq, segment = match.groups()
        return _validated_head({"epoch": int(epoch), "seq": int(seq), "segment": segment})
    except (ValueError, TypeError, OverflowError, RecursionError):
        return None


class ViewWriter:
    """
    Publish views only from heads on the journal writer's committed chain.

    Existing views are replaced only when their head is an earlier member of
    the same chain. Equal, unknown, or malformed heads remain untouched.
    Create-only publication preserves any existing view without inspecting it.
    Every Store operation runs in an abandonable daemon worker. The owning
    run supervises the wait; an abandoned read cannot start a later write.

    Parameters
    ----------
    store : Store
        Run-scoped synchronous storage.
    writer : JournalWriter
        Journal authority supplying committed segment names, oldest first.

    Examples
    --------
    >>> views = ViewWriter(store, writer)  # doctest: +SKIP
    >>> await views.write("goals/g/solution.json", b"{}", source=writer.head)  # doctest: +SKIP
    """

    def __init__(self, store: Store, writer: JournalWriter) -> None:
        self._store = store
        self._writer = writer

    async def write(
        self, key: str, content: bytes, *, source: Head, create_only: bool = False
    ) -> None:
        """
        Create a view or conditionally replace one from an earlier committed head.

        After a conditional replacement conflict, reread once and apply the
        same membership and ordering checks. A second conflict leaves the
        competing view intact; a later call can retry publication.

        Parameters
        ----------
        key : str
            Run-relative key ending in exactly ``.json``, ``.toml``, or ``.md``.
        content : bytes
            View body to encode with source metadata.
        source : Head
            Canonical head already present on this writer's committed chain.
        create_only : bool, default=False
            Preserve any existing view on AlreadyExists, without reading or
            replacing it. Used for immutable round pages.

        Raises
        ------
        RuntimeError
            If the source is malformed or not committed.
        ValueError
            If the key suffix or document is invalid.
        StorageError
            If a Store operation fails outside a conditional-write conflict.
        OSError
            If the backend's create operation fails.

        Examples
        --------
        >>> await views.write("run.toml", b"name = 'run'", source=head)  # doctest: +SKIP
        """
        committed = self._writer.committed
        try:
            head = _validated_head(source.model_dump())
            source_index = committed.index(head.segment)
        except (ValueError, TypeError, OverflowError, RecursionError):
            raise _core_error(RuntimeError, "source head is not committed") from None
        fmt = _FORMATS.get(PurePosixPath(key).suffix)
        if fmt is None:
            raise ValueError("view key requires .json, .toml, or .md suffix") from None
        data = embed_source_head(fmt, content, head)
        digest = sha256(data).digest()
        await anyio.lowlevel.checkpoint_if_cancelled()
        try:
            await run_sync_daemon(self._store.create, key, data)
            return
        except AlreadyExists:
            if create_only:
                return
        for _ in range(2):
            await anyio.lowlevel.checkpoint_if_cancelled()
            outcome = _Outcome()
            try:
                existing = await run_sync_daemon(_read_blob, self._store, key)
            except BaseException as error:
                outcome.contain(error, StorageError, "view storage")
            if sha256(existing.data).digest() == digest:
                return
            previous = read_source_head(fmt, existing.data)
            if previous is None or previous.segment not in committed[:source_index]:
                return
            await anyio.lowlevel.checkpoint_if_cancelled()
            try:
                await run_sync_daemon(
                    partial(self._store.replace, key, data, if_match=existing.etag)
                )
                return
            except PreconditionFailed:
                pass
