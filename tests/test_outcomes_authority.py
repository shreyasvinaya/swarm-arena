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


@pytest.mark.parametrize("kind", ["cleanup", "limit", "diagnostic", "cancel", "group"])
@pytest.mark.parametrize("nested", [False, True])
def test_control_category_survives_mutation_and_nested_groups(kind: str, nested: bool) -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._errors import _is_core_control, _rebuild_core_signal, _stop_cleanup_error
    from swarm_arena.manager import LimitReached

    secret = "private-control-mutation"
    errors = {
        "cleanup": _stop_cleanup_error(),
        "limit": _core_error(LimitReached, "usage_overflow", "budget exhausted by usage overflow"),
        "diagnostic": _core_error(ValueError, "invalid field"),
        "cancel": _classify_outcome(asyncio.CancelledError(secret)),
        "group": _classify_outcome(BaseExceptionGroup(secret, [asyncio.CancelledError(secret)])),
    }
    error = errors[kind]
    expected = _exception_type_name(error)
    error.args = (secret,)
    error.__dict__.update(category="control", limit=secret, original_type=secret)
    error.add_note(secret)
    rebuilt = _rebuild_core_signal(error)
    assert rebuilt is not None
    assert _is_core_control(rebuilt) is (kind in {"cleanup", "limit"})
    assert _exception_type_name(rebuilt) == expected
    assert_secret_absent(rebuilt, secret)
    received: BaseException = (
        BaseExceptionGroup(secret, [asyncio.CancelledError(secret), error]) if nested else error
    )
    result = _classify_outcome(received)
    assert _is_core_control(result) is (kind in {"cleanup", "limit"})
    if kind == "limit":
        assert isinstance(result, LimitReached)
        assert result.limit == "usage_overflow"
    assert_secret_absent(result, secret)
    forged = LimitReached("budget", secret) if kind == "limit" else type(error)(secret)
    forged.__dict__.update(error.__dict__)
    assert not _is_core_control(forged)


async def test_containment_preserves_nested_cleanup_but_not_hook_control_tags() -> None:
    from _fakes.secrecy import assert_secret_absent

    from swarm_arena._errors import StorageError, _is_core_control, _stop_cleanup_error
    from swarm_arena._outcomes import _Outcome

    cleanup = _stop_cleanup_error()
    with pytest.raises(RuntimeError, match="stop cleanup timed out") as caught:
        _Outcome().contain(ExceptionGroup("secret", [cleanup]), StorageError, "storage")
    assert _is_core_control(caught.value)
    forged = RuntimeError("secret")
    forged.__dict__["category"] = "control"
    with pytest.raises(StorageError) as caught:
        _Outcome().contain(forged, StorageError, "storage")
    assert not _is_core_control(caught.value)
    assert_secret_absent(caught.value, "secret")
