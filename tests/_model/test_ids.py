import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

import pytest

from swarm_arena._model.canonical import sha256_hex
from swarm_arena._model.ids import IdGen, human_learning_id, new_run_id, run_tag


def test_run_tag() -> None:
    assert run_tag("2026-10-18T18-30-00Z_ab12cd") == sha256_hex(b"2026-10-18T18-30-00Z_ab12cd")[:16]
    assert len(run_tag("local")) == 16


def test_ids_count_per_prefix_and_continue_across_epochs() -> None:
    ids = IdGen("local", 0)
    tag = run_tag("local")
    assert [ids.next("c"), ids.next("c"), ids.next("s")] == [
        f"c-{tag}-e0-0001",
        f"c-{tag}-e0-0002",
        f"s-{tag}-e0-0001",
    ]
    later = IdGen("local", 2, counters={"c": 192})
    assert later.next("c") == f"c-{tag}-e2-0193"
    assert later.counters() == {"c": 193}
    later.counters()["c"] = 0
    assert later.next("c").endswith("-0194")  # counters() returns a copy
    assert IdGen("r", 1, counters={"x": 9999}).next("x").endswith("-e1-10000")


def test_bad_prefix() -> None:
    with pytest.raises(ValueError, match="prefix"):
        IdGen("r", 0).next(cast(Any, "q"))


def test_human_learning_id() -> None:
    a, b = human_learning_id(), human_learning_id()
    assert re.fullmatch(r"L-h-[0-9a-f]{16}", a)
    assert a != b


def test_new_run_id() -> None:
    assert re.fullmatch(
        r"2026-10-18T18-30-00Z_[0-9a-f]{6}",
        new_run_id(datetime(2026, 10, 18, 18, 30, tzinfo=UTC)),
    )
    with pytest.raises(ValueError, match="aware"):
        new_run_id(datetime(2026, 10, 18))


def test_counters_input_is_copied() -> None:
    counters = {"c": 4}
    ids = IdGen("local", 0, counters)
    counters["c"] = 100
    assert ids.next("c").endswith("-0005")
    assert counters == {"c": 100}


@pytest.mark.parametrize("prefix", ["c", "L", "s", "r", "llm", "x", "k"])
def test_all_prefixes(prefix: Any) -> None:
    assert IdGen("local", 3).next(prefix) == f"{prefix}-{run_tag('local')}-e3-0001"


def test_new_run_id_normalizes_to_utc() -> None:
    now = datetime(2026, 10, 19, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert re.fullmatch(r"2026-10-18T18-30-00Z_[0-9a-f]{6}", new_run_id(now))
