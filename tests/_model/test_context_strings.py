import pytest

from swarm_arena._model.context_strings import _context_string


class ClassHook:
    """A non-string whose ``__class__`` property would run user code."""

    touched = False

    @property
    def __class__(self) -> type:
        ClassHook.touched = True
        raise AssertionError("class hook executed")


def test_non_string_class_hook_never_runs() -> None:
    ClassHook.touched = False
    with pytest.raises(TypeError, match=r"^candidate_id must be str$"):
        _context_string(ClassHook(), "candidate_id")
    assert not ClassHook.touched


def test_string_subclass_is_copied_to_an_exact_string() -> None:
    class Sub(str):
        pass

    value = _context_string(Sub("c-1"), "candidate_id")
    assert type(value) is str
    assert value == "c-1"
