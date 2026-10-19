from swarm_arena._errors import (
    AlreadyExists,
    ConditionalWritesUnsupported,
    ConfigError,
    NotFound,
    PlanError,
    PreconditionFailed,
    SoftError,
    StorageError,
    WriteOutcomeUnknown,
)


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
