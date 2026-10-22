"""Parse finite structured outputs and meter every formatting attempt."""

import json
import math
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from agents import AgentOutputSchemaBase, ModelSettings
from agents.models.interface import ModelTracing

from swarm_arena._board.digest import untrusted
from swarm_arena._budget.llm_scope import current_llm_scope
from swarm_arena._errors import _core_error
from swarm_arena._model.base import LLMRole
from swarm_arena._model.catalog import ModelCatalog, ModelSpec
from swarm_arena._model.data import finite_json
from swarm_arena.llm._catalog import build_model, prices_for, supports
from swarm_arena.llm._hooks import MeteredModel, _prepare
from swarm_arena.llm._langfuse import _call_trace, component_span

_captured_call_ids: ContextVar[list[str] | None] = ContextVar("structured_call_ids", default=None)


def _reject_constant(value: str) -> None:
    raise _core_error(ValueError, "non-finite structured output")


def _validate(value: Any, schema: dict[str, Any], key: str = "output") -> None:
    """Validate recursive object/array shapes without reporting data values."""
    kind = schema.get("type")
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if kind is not None and not any(
        valid.get(k, False) for k in ([kind] if isinstance(kind, str) else kind)
    ):
        raise _core_error(ValueError, f"invalid {key}")
    if isinstance(value, dict):
        for required in schema.get("required", []):
            if required not in value:
                raise _core_error(ValueError, f"missing {required}")
        properties = schema.get("properties", {})
        for name, item in value.items():
            if name in properties:
                _validate(item, properties[name], name)
            elif schema.get("additionalProperties") is False:
                raise _core_error(ValueError, f"invalid {key} keys")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate(item, schema["additionalProperties"], key)
    elif isinstance(value, list):
        for item in value:
            _validate(item, schema.get("items", {}), key)
    elif isinstance(value, float) and not math.isfinite(value):
        raise _core_error(ValueError, f"non-finite {key}")


class DictOutputSchema(AgentOutputSchemaBase):
    """
    Adapt an object schema to the SDK with recursive finite validation.

    Parameters
    ----------
    name : str
        Schema identifier sent to the provider.
    schema : dict[str, Any]
        Object schema with recursive types, properties, required keys and items.

    Examples
    --------
    >>> schema = DictOutputSchema("result", {"type": "object", "required": ["ok"]})
    >>> schema.validate_json('{"ok": true}')
    {'ok': True}
    """

    def __init__(self, name: str, schema: dict[str, Any]) -> None:
        self._name = name
        self._schema = schema

    def is_plain_text(self) -> bool:
        """
        Identify structured rather than plain-text output.

        Returns
        -------
        bool
            Always False.

        Examples
        --------
        >>> DictOutputSchema("result", {}).is_plain_text()
        False
        """
        return False

    def name(self) -> str:
        """
        Return the schema identifier.

        Returns
        -------
        str
            Configured name.

        Examples
        --------
        >>> DictOutputSchema("result", {}).name()
        'result'
        """
        return self._name

    def json_schema(self) -> dict[str, Any]:
        """
        Return the provider-facing schema.

        Returns
        -------
        dict[str, Any]
            Configured JSON schema.

        Examples
        --------
        >>> DictOutputSchema("result", {"type": "object"}).json_schema()
        {'type': 'object'}
        """
        return self._schema

    def is_strict_json_schema(self) -> bool:
        """
        Disable the SDK's strict-schema rewriting requirements.

        Returns
        -------
        bool
            Always False.

        Examples
        --------
        >>> DictOutputSchema("result", {}).is_strict_json_schema()
        False
        """
        return False

    def validate_json(self, json_str: str) -> dict[str, Any]:
        """
        Parse an object and check finite JSON and nested schema shapes.

        Parameters
        ----------
        json_str : str
            Provider output text.

        Returns
        -------
        dict[str, Any]
            Validated finite object. Callers enforce domain-specific rules.

        Raises
        ------
        ValueError
            For malformed JSON, non-finite data or a shape/key mismatch.
            Diagnostics contain only schema identifiers.

        Examples
        --------
        >>> DictOutputSchema("result", {"required": ["ok"]}).validate_json('{"ok": true}')
        {'ok': True}
        """
        try:
            value = json.loads(json_str, parse_constant=_reject_constant)
            finite_json(value)
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise _core_error(ValueError, "invalid structured output") from None
        if not isinstance(value, dict):
            raise _core_error(ValueError, "invalid output object")
        _validate(value, self._schema)
        return value


def _formatter(alias: str, role: LLMRole, models: ModelCatalog) -> str:
    """Select compatible input, output and cache-write tariffs deterministically."""
    first = prices_for(models.resolve(alias, role))
    if first is None:
        return alias
    choices: list[tuple[float, str]] = []
    for name, spec in models.root.items():
        if role not in spec.roles or not supports(spec, "response_schema"):
            continue
        prices = prices_for(spec)
        if (
            prices is not None
            and prices.input <= first.input
            and prices.output <= first.output
            and max(2 * prices.input, prices.cache_write or 0)
            <= max(2 * first.input, first.cache_write or 0)
        ):
            choices.append((prices.output, name))
    return min(choices)[1] if choices else alias


async def structured_call(
    *,
    alias: str,
    role: LLMRole,
    models: ModelCatalog,
    namespace: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    name: str,
    no_cache_read: bool,
    temperature: float | None,
    admission_check: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Return parsed output and every metered call id; the component owns drain.

    Parameters
    ----------
    alias : str
        First model's catalog alias.
    role : LLMRole
        Required catalog permission.
    models : ModelCatalog
        Human-approved model choices.
    namespace : str
        Cache namespace.
    system, user : str
        Stable instructions and changing input.
    schema : dict[str, Any]
        Recursive output shape.
    name : str
        Schema identifier.
    no_cache_read : bool
        Force response-cache bypass on both stages.
    temperature : float or None
        Sampling temperature.

    admission_check : Callable[[], None] or None, optional
        Invocation gate preventing new funded stages after a sibling limit.

    Returns
    -------
    tuple[dict[str, Any], list[str]]
        Parsed object and all successful-stage call identifiers.

    Raises
    ------
    ValueError
        For invalid output or catalog settings.
    BudgetExceeded
        If either full-call envelope cannot be funded before dispatch.

    Examples
    --------
    Scope owners use this helper and drain after all concurrent calls finish.
    The output schema itself can be checked without an active scope:

    >>> DictOutputSchema("result", {"required": ["ok"]}).validate_json('{"ok": true}')
    {'ok': True}
    """
    spec = await _prepare(lambda: models.resolve(alias, role))
    output_schema = DictOutputSchema(name, schema)
    ids: list[str] = []

    async def call(
        alias: str, spec: ModelSpec, prompt: str, output: DictOutputSchema | None
    ) -> str:
        inner = await _prepare(lambda: build_model(spec))
        model = MeteredModel(
            inner,
            alias=alias,
            spec=spec,
            namespace=namespace,
            no_cache_read=no_cache_read,
            temperature=temperature,
            max_tokens=cap,
            admission_check=admission_check,
        )
        try:
            scope = current_llm_scope()
            if scope is None:
                raise _core_error(RuntimeError, "LLM call outside an LLM scope")
            with component_span(scope), _call_trace(scope):
                response = await model.get_response(
                    system,
                    prompt,
                    ModelSettings(),
                    [],
                    output,
                    [],
                    ModelTracing.ENABLED if scope.tracing.langfuse else ModelTracing.DISABLED,
                    previous_response_id=None,
                    conversation_id=None,
                    prompt=None,
                )
        finally:
            ids.extend(model.call_ids)
            captured = _captured_call_ids.get()
            if captured is not None:
                captured.extend(model.call_ids)
        return "".join(
            part.text
            for item in response.output
            if item.type == "message"
            for part in item.content
            if part.type == "output_text"
        )

    cap = spec.max_tokens
    if await _prepare(lambda: supports(spec, "response_schema")):
        text = await call(alias, spec, user, output_schema)
    else:
        draft = await call(alias, spec, user, None)
        if not draft.strip():
            raise _core_error(ValueError, "empty structured draft")
        formatter = await _prepare(lambda: _formatter(alias, role, models))
        formatter_spec = models.resolve(formatter, role)
        native = supports(formatter_spec, "response_schema")
        prompt = (
            "Format the following response as JSON matching this schema: "
            + json.dumps(schema)
            + "\n"
            + untrusted(draft)
        )
        text = await call(formatter, formatter_spec, prompt, output_schema if native else None)
    return await _prepare(lambda: output_schema.validate_json(text)), ids
