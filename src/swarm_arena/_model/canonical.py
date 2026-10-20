"""Encode canonical JSON, hashes, and millisecond-precision UTC timestamps."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, JsonValue, PlainSerializer


def canonical_json(value: JsonValue) -> bytes:
    """
    Encode a JSON value with sorted keys and compact UTF-8 output.

    Parameters
    ----------
    value : JsonValue
        Value to encode; floating-point values must be finite.

    Returns
    -------
    bytes
        Deterministic JSON encoding without ASCII escaping.

    Examples
    --------
    >>> canonical_json({"b": 1, "a": 2})
    b'{"a":2,"b":1}'
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """
    Compute the lowercase SHA-256 hexadecimal digest of bytes.

    Parameters
    ----------
    data : bytes
        Bytes to hash.

    Returns
    -------
    str
        The 64-character hexadecimal digest.

    Examples
    --------
    >>> sha256_hex(b"")
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    return hashlib.sha256(data).hexdigest()


def _normalize_utc(dt: datetime) -> datetime:
    """Reject naive values and normalize to UTC with truncated milliseconds."""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    normalized = dt.astimezone(UTC)
    return normalized.replace(microsecond=normalized.microsecond // 1000 * 1000)


def utc_now() -> datetime:
    """
    Return the current UTC time truncated to milliseconds.

    Returns
    -------
    datetime
        An aware UTC datetime with millisecond precision.

    Examples
    --------
    >>> now = utc_now()
    >>> now.tzinfo is UTC and now.microsecond % 1000 == 0
    True
    """
    return _normalize_utc(datetime.now(UTC))


def format_utc(dt: datetime) -> str:
    """
    Format an aware datetime as UTC with milliseconds and a trailing Z.

    Parameters
    ----------
    dt : datetime
        A timezone-aware datetime in any timezone.

    Returns
    -------
    str
        Timestamp in ``YYYY-MM-DDTHH:MM:SS.mmmZ`` form.

    Raises
    ------
    ValueError
        If the datetime is naive.

    Examples
    --------
    >>> format_utc(datetime(2026, 10, 18, 13, 52, 1, 123456, tzinfo=UTC))
    '2026-10-18T13:52:01.123Z'
    """
    return _normalize_utc(dt).isoformat(timespec="milliseconds").replace("+00:00", "Z")


UtcDatetime = Annotated[
    datetime,
    AfterValidator(_normalize_utc),
    PlainSerializer(format_utc, return_type=str, when_used="json"),
]
