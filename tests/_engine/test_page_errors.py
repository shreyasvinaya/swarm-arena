"""Exercise actual page storage from public run and resume boundaries."""

from pathlib import Path
from typing import Any

import anyio
import pytest
from _engine.test_resume import _finished_without_round_page
from _fakes.run import plan
from _fakes.secrecy import assert_secret_absent

from swarm_arena import RunConfig, Swarm
from swarm_arena._errors import StorageError
from swarm_arena._store.local import LocalStore

PAGE = "goals/g/rounds/round-01.md"


@pytest.mark.parametrize("site", ["exists", "create"])
@pytest.mark.parametrize("resumed", [False, True])
async def test_page_storage_errors_are_type_only(
    tmp_path: Path,
    run_config: RunConfig,
    monkeypatch: pytest.MonkeyPatch,
    site: str,
    resumed: bool,
) -> None:
    config = run_config
    if resumed:
        config, _ = await _finished_without_round_page(tmp_path, config)
    original = getattr(LocalStore, site)
    secret = "private-page-storage-contents-4f6ad8"

    def fail(self: LocalStore, key: str, *args: Any) -> Any:
        if key != PAGE:
            return original(self, key, *args)
        error = OSError(secret)
        error.add_note(secret)
        raise error from ValueError(secret)

    monkeypatch.setattr(LocalStore, site, fail)
    operation = (
        Swarm(config).resume("repair", force=True) if resumed else Swarm(config).run(plan(rounds=1))
    )
    with anyio.fail_after(15), pytest.raises((OSError, StorageError, ExceptionGroup)) as caught:
        await operation
    assert_secret_absent(caught.value, secret, sanitized=True)
    leaves: list[BaseException] = [caught.value]
    while any(isinstance(error, BaseExceptionGroup) for error in leaves):
        leaves = [
            child
            for error in leaves
            for child in (error.exceptions if isinstance(error, BaseExceptionGroup) else [error])
        ]
    assert len(leaves) == 1
    assert isinstance(leaves[0], StorageError)
    assert str(leaves[0]) == "round page storage failed: OSError"
