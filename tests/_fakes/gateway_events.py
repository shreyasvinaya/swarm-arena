"""Secret-free gateway records shared by committed-reader regressions."""

from typing import Any

from swarm_arena._journal.event import EventDraft, EventType


def gateway_key(
    candidate_id: str = "c-1", *, key_id: str = "hash-1", cap: str = "soft"
) -> EventDraft:
    """Build the creation payload emitted by the real KeyManager."""
    return EventDraft(
        type=EventType.GATEWAY_KEY,
        payload={
            "action": "created",
            "candidate_id": candidate_id,
            "key_id": key_id,
            "alias": f"cand-{candidate_id}",
            "reservation_id": "res-1",
            "max_budget_usd": 1.0,
            "cap": cap,
            "run_id": "r1",
            "owner_epoch": 1,
            "goal_id": "g1",
            "team": "t1",
            "agent": "a1",
            "round": 1,
            "created_at": "2026-10-18T00:00:00Z",
            "closed_at": None,
        },
    )


def reconcile(
    candidate_id: str = "c-1",
    *,
    key_id: str = "hash-1",
    observed: float | None = None,
    label: str = "estimated",
    run_id: str = "r1",
) -> EventDraft:
    """Build the report payload frozen for post-run reconciliation."""
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "key_id": key_id,
        "alias": f"cand-{candidate_id}",
        "reserved_usd": 1.0,
        "observed_usd": observed,
        "label": label,
        "reasons": ["no maximum request duration is enforced"] if observed is None else [],
        "deleted": False,
    }
    return EventDraft(type=EventType.RECONCILE, payload={"run_id": run_id, "keys": [row]})
