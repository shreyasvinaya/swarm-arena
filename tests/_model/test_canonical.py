from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from swarm_arena._model.base import FrozenModel
from swarm_arena._model.canonical import (
    UtcDatetime,
    canonical_json,
    format_utc,
    sha256_hex,
    utc_now,
)


def test_canonical_json_sorted_compact() -> None:
    assert canonical_json({"b": 1, "a": [1, "é"]}) == '{"a":[1,"é"],"b":1}'.encode()


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_json({"x": float("nan")})


def test_sha256_hex() -> None:
    assert sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_format_utc() -> None:
    assert (
        format_utc(datetime(2026, 10, 18, 13, 52, 1, 123456, tzinfo=UTC))
        == "2026-10-18T13:52:01.123Z"
    )
    plus2 = timezone(timedelta(hours=2))
    assert format_utc(datetime(2026, 10, 18, 15, 52, 1, tzinfo=plus2)) == "2026-10-18T13:52:01.000Z"
    with pytest.raises(ValueError, match="aware"):
        format_utc(datetime(2026, 10, 18))


def test_utc_now_is_aware_milliseconds() -> None:
    now = utc_now()
    assert now.tzinfo is UTC
    assert now.microsecond % 1000 == 0


def test_utc_datetime_field_roundtrip() -> None:
    class M(FrozenModel):
        t: UtcDatetime

    m = M(t=datetime(2026, 10, 18, 13, 52, 1, 123456, tzinfo=UTC))
    assert m.model_dump(mode="json") == {"t": "2026-10-18T13:52:01.123Z"}
    assert M.model_validate_json(m.model_dump_json()) == m
    assert m.t == datetime(2026, 10, 18, 13, 52, 1, 123000, tzinfo=UTC)
    assert m.t.tzinfo is UTC
    with pytest.raises(ValidationError):
        M(t=datetime(2026, 10, 18))


def test_utc_datetime_normalizes_offset() -> None:
    class M(FrozenModel):
        t: UtcDatetime

    m = M(t=datetime(2026, 10, 18, 15, 52, 1, 999999, tzinfo=timezone(timedelta(hours=2))))
    assert m.t == datetime(2026, 10, 18, 13, 52, 1, 999000, tzinfo=UTC)
    assert m.t.tzinfo is UTC
    assert m.model_dump(mode="python") == {"t": m.t}
