import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._errors import (
    AlreadyExists,
    ChainError,
    ConditionalWritesUnsupported,
    ConfigError,
    Fenced,
    LeaseHeld,
    NotFound,
    PlanError,
    PreconditionFailed,
    SoftError,
    StorageError,
    WriteOutcomeUnknown,
)


def test_m1b_errors_are_runtime_errors() -> None:
    assert issubclass(ChainError, StorageError) and issubclass(StorageError, RuntimeError)  # noqa: PT018
    assert issubclass(Fenced, RuntimeError) and issubclass(LeaseHeld, RuntimeError)  # noqa: PT018


def test_hierarchy() -> None:
    assert all(issubclass(e, ValueError) for e in (PlanError, ConfigError, SoftError))
    assert all(
        issubclass(e, StorageError)
        for e in (
            NotFound,
            AlreadyExists,
            PreconditionFailed,
            WriteOutcomeUnknown,
            ConditionalWritesUnsupported,
        )
    )
    assert issubclass(StorageError, RuntimeError)


def test_plan_error_lists_errors() -> None:
    err = PlanError(["a", "b"])
    assert err.errors == ["a", "b"]
    assert str(err) == "invalid plan:\n- a\n- b"


def test_plan_error_copies_sequence() -> None:
    errors = ["a"]
    err = PlanError(errors)
    errors.append("b")
    assert err.errors == ["a"]
    assert PlanError(("a", "b")).errors == ["a", "b"]


def test_core_signal_snapshot_is_weak_and_fresh() -> None:
    import gc
    import weakref

    from swarm_arena._errors import _core_signal, _rebuild_core_signal
    from swarm_arena.manager import LimitReached

    for original in (_core_signal(SoftError("safe")), _core_signal(LimitReached("stop", "safe"))):
        original.args = ("changed",)
        original.add_note("changed")
        original.__cause__ = ValueError("changed")
        if isinstance(original, LimitReached):
            original.limit = "changed"
        rebuilt = _rebuild_core_signal(original)
        assert rebuilt is not None
        assert rebuilt is not original
        assert type(rebuilt) is type(original)
        assert rebuilt.args == ("safe",)
        assert rebuilt.__cause__ is None
        assert rebuilt.__context__ is None
        assert not getattr(rebuilt, "__notes__", ())
        if isinstance(rebuilt, LimitReached):
            assert rebuilt.limit == "stop"
        again = _rebuild_core_signal(rebuilt)
        assert again is not None
        assert again is not rebuilt
        assert again.args == rebuilt.args
        reference = weakref.ref(rebuilt)
        del rebuilt
        gc.collect()
        assert reference() is None
    assert _rebuild_core_signal(SoftError("unregistered")) is None
    assert _rebuild_core_signal(ValueError("unregistered")) is None


def test_core_signal_registry_rejects_user_equality() -> None:
    from typing import Any

    from swarm_arena._errors import _core_signal, _rebuild_core_signal
    from swarm_arena.manager import LimitReached

    original = _core_signal(LimitReached("stop", "safe"))

    class EqualType(type):
        def __eq__(cls, other: object) -> bool:
            return True

        def __hash__(cls) -> int:
            return hash(LimitReached)

    class ForgedLimit(LimitReached, metaclass=EqualType):
        def __eq__(self, other: Any) -> bool:
            return True

        def __hash__(self) -> int:
            return hash(original)

    assert _rebuild_core_signal(ForgedLimit("budget", "forged")) is None


@pytest.mark.parametrize(
    "kind",
    [
        PlanError,
        ConfigError,
        LeaseHeld,
        Fenced,
        RuntimeError,
        ValueError,
        StorageError,
        NotFound,
        AlreadyExists,
        PreconditionFailed,
        WriteOutcomeUnknown,
        ConditionalWritesUnsupported,
        ChainError,
    ],
)
def test_cli_error_snapshots_are_fresh_weak_and_not_transferable(
    kind: type[Exception],
) -> None:
    import gc
    import weakref

    from swarm_arena._errors import _core_error, _core_signal, _rebuild_core_signal

    secret = "MUTATED-CORE-SECRET"
    original = _core_error(
        kind, ["field: invalid", "team: missing"] if kind is PlanError else "safe"
    )
    expected = str(original)
    original.args = (secret,)
    original.add_note(secret)
    original.__cause__ = ValueError(secret)
    if isinstance(original, PlanError):
        original.errors[:] = [secret]
    assert _core_signal(original) is original
    rebuilt = _rebuild_core_signal(original)
    assert rebuilt is not None
    assert rebuilt is not original
    assert isinstance(rebuilt, kind)
    if issubclass(kind, StorageError):
        assert type(rebuilt) is kind
    if kind is ValueError:
        assert not isinstance(rebuilt, ConfigError)
    assert str(rebuilt) == expected
    assert_secret_absent(rebuilt, secret, sanitized=True)
    if isinstance(rebuilt, PlanError):
        assert rebuilt.errors == ["field: invalid", "team: missing"]
    forged = type(original)([secret] if kind is PlanError else secret)
    forged.__dict__.update(original.__dict__)
    assert _rebuild_core_signal(forged) is None
    again = _rebuild_core_signal(rebuilt)
    assert again is not None
    assert str(again) == expected
    reference = weakref.ref(rebuilt)
    del rebuilt
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "kind",
    [
        StorageError,
        NotFound,
        AlreadyExists,
        PreconditionFailed,
        WriteOutcomeUnknown,
        ConditionalWritesUnsupported,
        ChainError,
    ],
)
def test_storage_registry_accepts_only_registered_exact_core_classes(
    kind: type[StorageError],
) -> None:
    from swarm_arena._errors import _core_error, _core_signal, _rebuild_core_signal

    user_kind = type("UserStorageError", (kind,), {})
    for error in (kind("unregistered"), user_kind("impostor")):
        assert _rebuild_core_signal(error) is None
    with pytest.raises(TypeError, match="expected a core error"):
        _core_error(user_kind, "not supported")
    with pytest.raises(TypeError, match="expected a core error"):
        _core_signal(user_kind("not supported"))
    original = _core_error(kind, "journal: unavailable")
    assert type(original) is kind
    rebuilt = _rebuild_core_signal(original)
    assert type(rebuilt) is kind
    assert str(rebuilt) == "journal: unavailable"


def test_storage_subclass_cannot_impersonate_a_registered_error() -> None:
    from swarm_arena._errors import _core_error, _core_signal, _rebuild_core_signal

    original = _core_error(ChainError, "journal: invalid")

    class EqualType(type):
        def __eq__(cls, other: object) -> bool:
            return True

        def __hash__(cls) -> int:
            return hash(ChainError)

    class ForgedChainError(ChainError, metaclass=EqualType):
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return hash(original)

    forged = ForgedChainError("SECRET")
    forged.__dict__.update(original.__dict__)
    assert _rebuild_core_signal(forged) is None
    with pytest.raises(TypeError, match="expected a core error"):
        _core_signal(forged)
