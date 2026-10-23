"""Structured parsing and bounded formatting calls."""

import json
from pathlib import Path

import pytest

pytest.importorskip("agents")

from _fakes.llm_run import open_llm_run
from _fakes.secrecy import assert_secret_absent

from swarm_arena import RunConfig
from swarm_arena._budget.llm_scope import llm_scope
from swarm_arena._journal.llm_record import drain
from swarm_arena.config import ModelCatalog, ModelSpec
from swarm_arena.llm._structured import DictOutputSchema, structured_call
from swarm_arena.testing import FakeModel, FakeTurn

SCHEMA = {
    "type": "object",
    "required": ["scores"],
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "value"],
                "properties": {"index": {"type": "integer"}, "value": {"type": "number"}},
            },
        }
    },
}


def test_schema_methods() -> None:
    schema = DictOutputSchema("scores", SCHEMA)
    assert schema.name() == "scores"
    assert schema.json_schema() == SCHEMA
    assert not schema.is_plain_text()
    assert not schema.is_strict_json_schema()
    assert schema.validate_json('{"scores": []}') == {"scores": []}


@pytest.mark.parametrize(
    "text",
    [
        "SECRET",
        '{"scores": NaN}',
        '{"scores": [Infinity]}',
        '{"scores": [{"index": 0, "value": 1e999}]}',
        '{"scores": {}}',
        '{"scores": [4]}',
        '{"scores": [{"index": true, "value": 1}]}',
        '{"scores": [{"index": 0, "value": false}]}',
        '{"scores": [{"index": 0}]}',
    ],
)
def test_invalid_json_raises_value_error(text: str) -> None:
    with pytest.raises(ValueError, match=r"invalid|missing") as caught:
        DictOutputSchema("scores", SCHEMA).validate_json(text)
    assert_secret_absent(caught.value, "SECRET")


def test_schema_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="scores"):
        DictOutputSchema("scores", SCHEMA).validate_json('{"other": 1}')


@pytest.mark.parametrize(
    ("supported", "output_price", "write_price", "use_formatter"),
    [
        (True, 1.0, None, False),
        (False, 1.0, None, True),
        (False, 3.0, None, False),
        (False, 1.0, 100.0, False),
    ],
)
async def test_structured_call_routing_and_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supported: bool,
    output_price: float,
    write_price: float | None,
    use_formatter: bool,
) -> None:
    from swarm_arena.llm import _structured

    a = ModelSpec(
        model="fake/a", roles=["judge"], max_tokens=1000, prices=ModelSpec.Prices(input=1, output=2)
    )
    fmt = ModelSpec(
        model="fake/fmt",
        roles=["judge"],
        max_tokens=100000,
        prices=ModelSpec.Prices(input=0.5, output=output_price, cache_write=write_price),
    )
    models = ModelCatalog({"a": a, "fmt": fmt})
    text = json.dumps({"scores": [{"index": 0, "value": 7.5}]})
    fake_a = FakeModel([FakeTurn(text=text), FakeTurn(text=text)])
    fake_fmt = FakeModel([FakeTurn(text=text)])
    fake_a.register("a")
    fake_fmt.register("fmt")
    monkeypatch.setattr(
        _structured, "supports", lambda spec, capability: supported or spec.model == "fake/fmt"
    )
    async with open_llm_run(tmp_path, config=RunConfig(models=models)) as run:
        scope = run.scope("judge")
        with llm_scope(scope):
            output, ids = await structured_call(
                alias="a",
                role="judge",
                models=models,
                namespace="test",
                system="rubric",
                user="candidate",
                schema=SCHEMA,
                name="scores",
                no_cache_read=True,
                temperature=0,
            )
        await drain(scope, wait=True)
        assert output["scores"][0]["value"] == 7.5
        assert len(ids) == (1 if supported else 2)
        assert set(ids) == {call.id for call in run.calls()}
    assert bool(fake_fmt.calls) is use_formatter
    assert len(fake_a.calls) == (1 if supported or use_formatter else 2)
    if use_formatter:
        assert fake_fmt.calls[0]["model_settings"].max_tokens == 1000
        assert fake_fmt.calls[0]["output_schema"].json_schema() == SCHEMA
    assert (fake_a.calls[0]["output_schema"] is not None) is supported


@pytest.mark.parametrize(("known", "formatter_cap"), [(False, 100000), (True, 500)])
async def test_unknown_prices_and_formatter_own_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, known: bool, formatter_cap: int
) -> None:
    from swarm_arena.llm import _structured

    prices = ModelSpec.Prices(input=1, output=2)
    models = ModelCatalog(
        {
            "a": ModelSpec(
                model="fake/a", roles=["judge"], max_tokens=1000, prices=prices if known else None
            ),
            "fmt": ModelSpec(
                model="fake/fmt", roles=["judge"], max_tokens=formatter_cap, prices=prices
            ),
        }
    )
    a = FakeModel([FakeTurn(text="reasoning"), FakeTurn(text='{"scores": []}')])
    fmt = FakeModel([FakeTurn(text='{"scores": []}')])
    a.register("a")
    fmt.register("fmt")
    monkeypatch.setattr(_structured, "supports", lambda spec, feature: spec.model == "fake/fmt")
    async with open_llm_run(tmp_path, config=RunConfig(models=models)) as run:
        scope = run.scope("judge")
        with llm_scope(scope):
            output, ids = await structured_call(
                alias="a",
                role="judge",
                models=models,
                namespace="n",
                system="r",
                user="c",
                schema=SCHEMA,
                name="scores",
                no_cache_read=True,
                temperature=0,
            )
        await drain(scope, wait=True)
        assert output == {"scores": []}
        assert len(ids) == 2
    if known:
        assert fmt.calls[0]["model_settings"].max_tokens == formatter_cap
    else:
        assert fmt.calls == []
        assert "scores" in a.calls[1]["input"]


@pytest.mark.parametrize("text", ["secret-invalid-json", '{"other": 1}'])
async def test_invalid_call_output_is_value_free(tmp_path: Path, text: str) -> None:
    fake = FakeModel([FakeTurn(text=text)])
    fake.register("a")
    models = ModelCatalog({"a": ModelSpec(model="fake/a", roles=["judge"])})
    async with open_llm_run(tmp_path, config=RunConfig(models=models)) as run:
        scope = run.scope("judge")
        with (
            llm_scope(scope),
            pytest.raises(ValueError, match=r"structured output|scores") as caught,
        ):
            await structured_call(
                alias="a",
                role="judge",
                models=models,
                namespace="n",
                system="r",
                user="c",
                schema=SCHEMA,
                name="scores",
                no_cache_read=True,
                temperature=0,
            )
        await drain(scope, wait=True)
        assert_secret_absent(caught.value, "secret-invalid-json")


@pytest.mark.parametrize("draft", ["", " \n\t"])
async def test_empty_draft_never_starts_formatter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft: str
) -> None:
    from swarm_arena.llm import _structured

    prices = ModelSpec.Prices(input=1, output=2)
    models = ModelCatalog(
        {
            alias: ModelSpec(model=f"fake/{alias}", roles=["judge"], max_tokens=1000, prices=prices)
            for alias in ("a", "fmt")
        }
    )
    first = FakeModel([FakeTurn(text=draft)])
    formatter = FakeModel([FakeTurn(text='{"scores": []}')])
    first.register("a")
    formatter.register("fmt")
    monkeypatch.setattr(_structured, "supports", lambda spec, feature: spec.model == "fake/fmt")
    async with open_llm_run(tmp_path, config=RunConfig(models=models)) as run:
        scope = run.scope("judge")
        with llm_scope(scope), pytest.raises(ValueError, match="empty structured draft"):
            await structured_call(
                alias="a",
                role="judge",
                models=models,
                namespace="n",
                system="r",
                user="c",
                schema=SCHEMA,
                name="scores",
                no_cache_read=True,
                temperature=0,
            )
        await drain(scope, wait=True)
        assert len(first.calls) == 1
        assert formatter.calls == []
        assert len(run.calls()) == 1
        assert scope.tranche.pending() == 0
        assert run.charged(scope).usd == pytest.approx(run.calls()[0].cost_usd)
