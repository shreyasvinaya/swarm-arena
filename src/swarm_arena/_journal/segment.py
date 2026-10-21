"""Encode canonical journal segments and validate their stored records."""

import re
from collections.abc import Sequence
from itertools import pairwise
from typing import ClassVar

from swarm_arena._journal.event import Event
from swarm_arena._model.base import StoredModel
from swarm_arena._model.canonical import canonical_json

_SEGMENT_NAME = re.compile(r"journal/e([1-9][0-9]*)-([1-9][0-9]*)\.jsonl")


class SegmentLink(StoredModel):
    """
    Validate the versioned first line of a journal segment.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version, upgraded on loading.
    link : str or None
        Previous committed segment's key, or null for the first segment.

    Examples
    --------
    >>> SegmentLink(link=None).model_dump()
    {'schema_version': 1, 'link': None}
    """

    schema_kind: ClassVar[str] = "segment"
    link: str | None


def segment_name(epoch: int, last_seq: int) -> str:
    """
    Build a canonical key from positive epoch and sequence numbers.

    Parameters
    ----------
    epoch : int
        Positive writer epoch, excluding booleans.
    last_seq : int
        Positive last event sequence number, excluding booleans.

    Returns
    -------
    str
        Relative journal segment key.

    Raises
    ------
    ValueError
        If either number is not a positive integer.

    Examples
    --------
    >>> segment_name(1, 100)
    'journal/e1-100.jsonl'
    """
    if type(epoch) is not int or type(last_seq) is not int or epoch < 1 or last_seq < 1:
        raise ValueError("segment numbers must be positive integers") from None
    return f"journal/e{epoch}-{last_seq}.jsonl"


def parse_segment_name(name: str) -> tuple[int, int]:
    """
    Parse a segment key only when it has the canonical positive-integer form.

    Parameters
    ----------
    name : str
        Relative journal segment key.

    Returns
    -------
    tuple[int, int]
        Writer epoch and last event sequence number.

    Raises
    ------
    ValueError
        If the key is not canonical; the message never includes the key.

    Examples
    --------
    >>> parse_segment_name("journal/e2-150.jsonl")
    (2, 150)
    """
    match = _SEGMENT_NAME.fullmatch(name)
    if match is None:
        raise ValueError("invalid segment name") from None
    try:
        epoch, last_seq = (int(part) for part in match.groups())
        if segment_name(epoch, last_seq) != name:
            raise ValueError("invalid segment name")
    except ValueError:
        raise ValueError("invalid segment name") from None
    return epoch, last_seq


def encode_segment(link: str | None, events: Sequence[Event]) -> bytes:
    r"""
    Encode a link line and event lines as canonical UTF-8 JSON with final newlines.

    Sequence validation belongs to decoding. This pure encoder preserves the
    supplied event order, even when it is empty or nonconsecutive.

    Parameters
    ----------
    link : str or None
        Previous committed segment key, or null for the first segment.
    events : Sequence[Event]
        Event records in the supplied order.

    Returns
    -------
    bytes
        Canonical JSON lines, including a final newline.

    Examples
    --------
    >>> encode_segment(None, [])
    b'{"link":null,"schema_version":1}\n'
    """
    lines = [canonical_json({"schema_version": 1, "link": link})]
    lines.extend(canonical_json(event.model_dump(mode="json")) for event in events)
    return b"\n".join(lines) + b"\n"


def decode_segment(data: bytes) -> tuple[str | None, list[Event]]:
    """
    Load a complete UTF-8 segment with at least one consecutive event.

    Split only on byte newlines so Unicode separators inside JSON strings remain
    intact. Stored models apply migrations and refuse newer schema versions.

    Parameters
    ----------
    data : bytes
        JSON lines with a mandatory terminal newline.

    Returns
    -------
    tuple[str or None, list[Event]]
        Previous segment key and its validated event records.

    Raises
    ------
    ValueError
        If encoding, framing, models or consecutive sequence numbers are invalid.
        Messages contain only fixed reasons, never stored input values.

    Examples
    --------
    >>> from swarm_arena._journal.event import EventType
    >>> from swarm_arena._model.canonical import utc_now
    >>> event = Event(epoch=1, seq=1, ts=utc_now(), run_id="r", context_id="r",
    ...               type=EventType.RUN_STARTED)
    >>> decode_segment(encode_segment(None, [event])) == (None, [event])
    True
    """
    if not data.endswith(b"\n"):
        raise ValueError("segment requires a terminal newline") from None
    lines = data[:-1].split(b"\n")
    try:
        link = SegmentLink.model_validate_json(lines[0].decode("utf-8", errors="strict"))
        events = [
            Event.model_validate_json(line.decode("utf-8", errors="strict")) for line in lines[1:]
        ]
    except ValueError:
        raise ValueError("invalid segment records") from None
    if not events:
        raise ValueError("segment has no events") from None
    if any(current.seq != previous.seq + 1 for previous, current in pairwise(events)):
        raise ValueError("seqs not consecutive") from None
    return link.link, events
