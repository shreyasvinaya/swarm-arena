"""Tool identities link to the metered call that emitted them."""

from typing import Any, cast

import pytest
from agents import ModelResponse
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage

from swarm_arena.llm._linking import CallLinks, tool_calls_of


def response_with(*calls: tuple[str, str]) -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseOutputMessage(
                id="msg", type="message", role="assistant", status="completed", content=[]
            ),
            *[
                ResponseFunctionToolCall(
                    type="function_call", call_id=cid, name=name, arguments="{}"
                )
                for cid, name in calls
            ],
        ],
        usage=Usage(),
        response_id=None,
    )


def test_links_map_tool_call_ids_to_call_id() -> None:
    links = CallLinks()
    response = response_with(("c1", "run_tool"), ("c2", "best"))
    links.add("llm-1", response)
    assert tool_calls_of(response) == [("c1", "run_tool"), ("c2", "best")]
    assert (links.get("c1"), links.get("c2"), links.get("zz")) == ("llm-1", "llm-1", None)


def test_link_identifiers_are_normalized_before_hashing() -> None:
    class Identifier(str):
        def __hash__(self) -> int:
            raise AssertionError("custom hash")

        def __str__(self) -> str:
            raise AssertionError("custom str")

    links = CallLinks()
    links.add(Identifier("llm-1"), response_with(("c1", "best")))
    assert links.get(Identifier("c1")) == "llm-1"
    assert type(links.get("c1")) is str
    with pytest.raises(TypeError, match="tool_call_id"):
        links.get(cast(Any, 1))
