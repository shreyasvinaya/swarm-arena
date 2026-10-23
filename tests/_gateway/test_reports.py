"""Use the same report validation on the producing and replay sides."""

from typing import Any, get_args

import pytest
from _fakes.gateway_events import reconcile
from _fakes.secrecy import assert_secret_absent

from swarm_arena._gateway.reports import ReconcileReport, ReportLabel
from swarm_arena._journal.event import Event
from swarm_arena._journal.replay import RunState
from swarm_arena._model.canonical import utc_now


@pytest.mark.parametrize("label", get_args(ReportLabel))
def test_every_declared_report_label_round_trips(label: str) -> None:
    raw = reconcile(label=label, observed=None if label == "estimated" else 0.0).payload
    report = ReconcileReport.model_validate(raw)
    event = Event(
        seq=1,
        epoch=1,
        run_id="r1",
        context_id="r1",
        ts=utc_now(),
        type="reconcile",
        payload=report.model_dump(mode="json"),
    )
    state = RunState.from_events([event])
    assert state.gateway_reconcile[("c-1", "hash-1")] == report.keys[0]
    assert report.spend_usd() == (1.0 if label == "estimated" else 0.0)


@pytest.mark.parametrize(
    "patch",
    [
        {"reserved_usd": True},
        {"observed_usd": 0.0},
        {"label": "SECRET"},
        {"deleted": "SECRET"},
        {"reasons": [123]},
    ],
)
def test_producer_and_replay_reject_the_same_bad_row(patch: dict[str, Any]) -> None:
    raw = reconcile().payload
    raw["keys"][0].update(patch)
    with pytest.raises(ValueError, match="validation error") as caught:
        ReconcileReport.model_validate(raw)
    assert_secret_absent(caught.value, "SECRET")
    event = Event(
        seq=1, epoch=1, run_id="r1", context_id="r1", ts=utc_now(), type="reconcile", payload=raw
    )
    with pytest.raises(ValueError, match="invalid journal replay payload") as caught:
        RunState.from_events([event])
    assert_secret_absent(caught.value, "SECRET")
