"""Fence authority is registry-backed and survives nested transport."""

import asyncio
import traceback
from pathlib import Path

import pytest

from swarm_arena._errors import Fenced, _core_error, _exception_type_name, _is_core_signal
from swarm_arena._outcomes import _classify_outcome


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("depth", [0, 4])
@pytest.mark.parametrize("cancelled", [False, True])
def test_authority_and_native_cancellation_precedence(
    registered: bool, depth: int, cancelled: bool
) -> None:
    secret = "untrusted-fence-value"
    error: BaseException = (
        _core_error(Fenced, "run ownership lost") if registered else Fenced(secret)
    )
    error.args = (secret,)
    error.add_note(secret)
    error.__cause__ = ValueError(secret)
    for _ in range(depth):
        error = BaseExceptionGroup(secret, [error, asyncio.CancelledError(secret)])
    if cancelled or registered:
        with pytest.raises(asyncio.CancelledError if cancelled else Fenced) as caught:
            _classify_outcome(error, cancelled=cancelled)
        assert caught.value.args == (() if cancelled else ("run ownership lost",))
        assert caught.value.__cause__ is None
        assert secret not in "".join(traceback.format_exception(caught.value))
        if registered and not cancelled:
            assert _is_core_signal(caught.value)
    else:
        result = _classify_outcome(error)
        assert not isinstance(result, Fenced)
        assert _exception_type_name(result) == ("BaseExceptionGroup" if depth else "Fenced")
        assert secret not in "".join(traceback.format_exception(result))


@pytest.mark.parametrize("site", ["policy_constructor", "judge_constructor", "coerce_ref"])
@pytest.mark.parametrize("registered", [False, True])
def test_type_error_group_cannot_bypass_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: str, registered: bool
) -> None:
    from _fakes.pr1_r2 import SECRET
    from _fakes.tools import PyTool

    from swarm_arena._model import ref as ref_module
    from tests._engine import test_pr1_r2_audit as audit

    class TypeErrorGroup(ExceptionGroup, TypeError):
        pass

    def failure(*args: object, **kwargs: object) -> BaseException:
        leaf = _core_error(Fenced, "run ownership lost") if registered else Fenced(SECRET)
        return TypeErrorGroup(SECRET, [leaf])

    if site != "coerce_ref":
        monkeypatch.setattr(audit, "failure", failure)
        audit.test_sync_conversion_boundaries(
            monkeypatch, tmp_path, site, "fenced" if registered else "group"
        )
        return

    def fail(*args: object, **kwargs: object) -> object:
        raise failure()

    monkeypatch.setattr(ref_module.Ref, "of", fail)
    with pytest.raises(Fenced if registered else Exception) as caught:
        ref_module.coerce_ref(PyTool)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    if not registered:
        assert not isinstance(caught.value, Fenced)


@pytest.mark.parametrize("registered", [False, True])
async def test_limit_group_cannot_bypass_offline_manager_classification(
    monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    from _fakes.pr1_r2 import SECRET

    from swarm_arena.manager import LimitReached
    from tests._engine import test_pr1_r2_audit as audit

    class LimitGroup(ExceptionGroup, LimitReached):
        pass

    def failure(*args: object, **kwargs: object) -> BaseException:
        leaf = _core_error(Fenced, "run ownership lost") if registered else Fenced(SECRET)
        return LimitGroup(SECRET, [leaf])

    monkeypatch.setattr(audit, "failure", failure)
    await audit.test_offline_manager_outcomes(monkeypatch, "fenced" if registered else "group")
