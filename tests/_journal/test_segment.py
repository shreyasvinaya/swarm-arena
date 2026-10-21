import json
from datetime import UTC, datetime

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._journal.event import Event, EventType
from swarm_arena._journal.segment import (
    SegmentLink,
    decode_segment,
    encode_segment,
    parse_segment_name,
    segment_name,
)
from swarm_arena._model.canonical import canonical_json


def _ev(seq: int) -> Event:
    return Event(
        epoch=1,
        seq=seq,
        ts=datetime(2026, 10, 18, 13, 0, tzinfo=UTC),
        run_id="r",
        context_id="r",
        type=EventType.TOOL_END,
    )


def test_segment_name() -> None:
    assert segment_name(1, 100) == "journal/e1-100.jsonl"
    assert parse_segment_name("journal/e2-150.jsonl") == (2, 150)
    assert parse_segment_name(segment_name(1234, 10000)) == (1234, 10000)


@pytest.mark.parametrize(
    ("epoch", "last_seq"), [(0, 1), (1, 0), (-1, 2), (2, -1), (True, 1), (1, False)]
)
def test_segment_name_rejects_nonpositive_or_boolean_numbers(epoch: int, last_seq: int) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        segment_name(epoch, last_seq)


@pytest.mark.parametrize(
    "name",
    [
        "journal/e0-1.jsonl",
        "journal/e1-0.jsonl",
        "journal/e01-1.jsonl",
        "journal/e1-01.jsonl",
        "journal/e-1-2.jsonl",
        "journal/e1-1.jsonl\n",
        "journal/e1-1Xjsonl",
        "/journal/e1-1.jsonl",
        "journal/e1-1.jsonl/private-segment",
        "../private-segment",
        "private-segment",
        "journal/e\u0661-1.jsonl",
        "journal/e1-+1.jsonl",
        "journal/e1-1.jsonl\x00",
    ],
)
def test_parse_segment_name_rejects_noncanonical_keys(name: str) -> None:
    with pytest.raises(ValueError, match="invalid segment name") as caught:
        parse_segment_name(name)
    assert_secret_absent(caught.value, name, sanitized=True)


def test_encode_first_line_is_link() -> None:
    data = encode_segment("journal/e1-100.jsonl", [_ev(101), _ev(102)])
    lines = data.decode().splitlines()
    assert json.loads(lines[0]) == {"schema_version": 1, "link": "journal/e1-100.jsonl"}
    assert [json.loads(x)["seq"] for x in lines[1:]] == [101, 102]
    assert data.endswith(b"\n")
    assert decode_segment(data) == ("journal/e1-100.jsonl", [_ev(101), _ev(102)])


def test_decode_round_trip_and_first_segment_link_none() -> None:
    assert decode_segment(encode_segment(None, [_ev(1)])) == (None, [_ev(1)])


def test_encoding_is_deterministic() -> None:
    assert encode_segment(None, [_ev(1)]) == encode_segment(None, [_ev(1)])
    event = _ev(1).model_copy(update={"payload": {"b": None, "a": 1}})
    assert encode_segment(None, [event]) == (
        b'{"link":null,"schema_version":1}\n'
        + canonical_json(event.model_dump(mode="json"))
        + b"\n"
    )


def test_decode_rejects_unsorted_or_empty() -> None:
    with pytest.raises(ValueError, match="segment has no events"):
        decode_segment(encode_segment(None, []))
    with pytest.raises(ValueError, match="seqs not consecutive"):
        decode_segment(encode_segment(None, [_ev(2), _ev(1)]))
    with pytest.raises(ValueError, match="seqs not consecutive"):
        decode_segment(encode_segment(None, [_ev(1), _ev(1)]))


def test_decode_rejects_gap_inside_segment() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        decode_segment(encode_segment(None, [_ev(1), _ev(3)]))


def test_unicode_separators_null_and_embedded_newlines_round_trip() -> None:
    event = Event.model_validate(
        {
            **_ev(1).model_dump(),
            "payload": {
                "text": "λ\u2028β\u2029λ\nδ\r\nε\x85ζ",
                "nested": [None, {"λ": True}],
            },
        }
    )
    data = encode_segment(None, [event])
    assert data.count(b"\n") == 2
    assert decode_segment(data) == (None, [event])


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\n",
        b"{}\n",
        b"null\n",
        b'{"schema_version":1,"link":null}',
        b'{"schema_version":1,"link":null}\n\n',
        b'{"schema_version":1,"link":null}\nprivate-segment\n',
        b'{"schema_version":1,"link":"private-segment\xff"}\n',
        b'{"schema_version":2,"link":"private-segment"}\n',
        b'{"schema_version":1,"link":123}\n',
        b'{"schema_version":1,"link":null,"private-segment":"private-segment"}\n',
    ],
)
def test_decode_malformed_data_has_fixed_safe_errors(data: bytes) -> None:
    with pytest.raises(ValueError, match=r"segment|consecutive") as caught:
        decode_segment(data)
    assert type(caught.value) is ValueError
    assert_secret_absent(caught.value, "private-segment", sanitized=True)


def test_decode_requires_terminal_newline_and_refuses_trailing_blank_line() -> None:
    data = encode_segment(None, [_ev(1)])
    for malformed in [data[:-1], data + b"\n"]:
        with pytest.raises(ValueError, match=r"segment|consecutive"):
            decode_segment(malformed)


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": 2},
        {"type": "private-segment"},
        {"payload": {"private-segment": float("nan")}},
        {"payload": {"private-segment": "\ud800"}},
        {"private-segment": "private-segment"},
    ],
)
def test_decode_rejects_invalid_event_with_safe_errors(update: dict[str, object]) -> None:
    line = json.dumps({**_ev(1).model_dump(mode="json"), **update}).encode()
    with pytest.raises(ValueError, match=r"segment|consecutive") as caught:
        decode_segment(b'{"schema_version":1,"link":null}\n' + line + b"\n")
    assert type(caught.value) is ValueError
    assert_secret_absent(caught.value, "private-segment", sanitized=True)


def test_decode_rejects_malformed_utf8_in_event() -> None:
    data = encode_segment(None, [_ev(1)])
    data = data.replace(b'"payload":{}', b'"payload":{"private-segment":"\xff"}')
    with pytest.raises(ValueError, match=r"segment|consecutive") as caught:
        decode_segment(data)
    assert type(caught.value) is ValueError
    assert_secret_absent(caught.value, "private-segment", sanitized=True)


def test_segment_link_is_stored_and_refuses_newer_version() -> None:
    link = SegmentLink(link=None)
    assert SegmentLink.schema_kind == "segment"
    assert link.model_dump() == {"schema_version": 1, "link": None}
    assert (
        SegmentLink.model_validate_json('{"link":"journal/e1-2.jsonl"}').link
        == "journal/e1-2.jsonl"
    )
    with pytest.raises(ValueError, match="newer"):
        SegmentLink.model_validate({"schema_version": 2, "link": None})
