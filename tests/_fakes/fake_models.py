"""Fresh registered model factories; importing this helper needs no LLM extra."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from swarm_arena.testing import FakeModel, FakeTurn


@contextmanager
def registered_model(
    turns: Sequence["FakeTurn"], *, name: str = "m", provider: str = "fake"
) -> Generator["FakeModel"]:
    """Build a fresh script and restore all registrations after its run ends."""
    from swarm_arena.llm import _catalog
    from swarm_arena.testing import FakeModel

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_catalog, "_FAKE_MODELS", dict(_catalog._FAKE_MODELS))
        model = FakeModel(turns, provider=provider)
        model.register(name)
        yield model
