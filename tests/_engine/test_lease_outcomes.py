"""Lease transfer contains backend errors and retries only core CAS races."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena import RunConfig
from swarm_arena._engine.resume import _acquire_lease, _take_over
from swarm_arena._errors import Fenced, PreconditionFailed, StorageError, _is_core_signal
from swarm_arena._journal.lease import Head, Lease
from swarm_arena._store.local import LocalStore


@pytest.mark.parametrize("site", ["acquire", "takeover", "read"])
@pytest.mark.parametrize("kind", ["ordinary", "cancel", "group", "fenced", "cas"])
async def test_lease_outcomes_are_type_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, site: str, kind: str
) -> None:
    secret = "private-lease-backend-marker"
    errors: dict[str, BaseException] = {
        "ordinary": OSError(secret),
        "cancel": asyncio.CancelledError(secret),
        "group": BaseExceptionGroup(secret, [asyncio.CancelledError(secret), ValueError(secret)]),
        "fenced": Fenced(secret),
        "cas": PreconditionFailed(secret),
    }
    store = LocalStore(tmp_path)
    if site != "acquire":
        lease = Lease.acquire_new(store, owner_id="old", host="local", ttl_s=60)
        lease.advance(Head(epoch=1, seq=1, segment="journal/e1-1.jsonl"))
        lease.release()
    calls = 0

    def fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        # A second attempt is an observable error, never an unbounded retry loop.
        if calls > 1:
            raise AssertionError("untrusted failure was retried")
        raise errors[kind]

    if site == "acquire":
        operation = _acquire_lease(fail, settle_timeout_s=1)
    else:
        monkeypatch.setattr(store, "read" if site == "read" else "replace", fail)
        operation = _take_over(RunConfig(), "r", store, force=True)
    with pytest.raises(StorageError) as caught:
        await operation
    assert calls == 1
    assert not isinstance(caught.value, Fenced)
    assert_secret_absent(caught.value, secret)


async def test_takeover_retries_real_local_cas_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalStore(tmp_path)
    old = Lease.acquire_new(store, owner_id="old", host="local", ttl_s=60)
    old.advance(Head(epoch=1, seq=1, segment="journal/e1-1.jsonl"))
    original = store.replace
    raced = False

    def replace(key: str, data: bytes, *, if_match: str) -> str:
        nonlocal raced
        if not raced:
            raced = True
            old.renew()
        return original(key, data, if_match=if_match)

    monkeypatch.setattr(store, "replace", replace)
    new = await _take_over(RunConfig(), "r", store, force=True)
    try:
        assert raced
        assert new.doc.epoch == 2
    finally:
        new.release()


def test_local_cas_failure_has_core_authority(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    store.create("key", b"old")
    with pytest.raises(PreconditionFailed) as caught:
        store.replace("key", b"new", if_match="stale")
    assert _is_core_signal(caught.value)
