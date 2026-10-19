"""Combine judge keys and preserve member scoring identities."""

from collections.abc import Sequence
from typing import ClassVar, cast

from pydantic import Field, JsonValue

from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.goal import Goal
from swarm_arena._model.ref import Ref, RefField, _ConfigInputModel, _validate_config
from swarm_arena._model.results import Candidate, Score
from swarm_arena._scoring.judge import Judge, JudgeContext, is_finite_score


def _panel_hash(hashes: list[str]) -> str:
    """Hash ordered panel identities, preserving the all-empty identity."""
    return sha256_hex(canonical_json(cast(JsonValue, hashes))) if any(hashes) else ""


class Lexicographic(Judge):
    """
    Combine ordered judges into one lexicographic ranking.

    Every member judges the full candidate batch. Only candidates with valid
    scores from every member receive a composite score. Members are referenced
    without imports at construction, then built and cached on first use.
    Each member must return keys of one fixed length within a batch; differing
    lengths fail the entire batch closed, preserving member-first ordering.
    Invalid member items become per-candidate errors. Exceptions raised by a
    member propagate to the runner, including provider failures.

    Parameters
    ----------
    judges : Sequence[Judge or Ref] or Lexicographic.Config
        Ordered member judges, references, or validated configuration.

    Examples
    --------
    >>> from swarm_arena import ProgramJudge
    >>> from swarm_arena.scoring import Lexicographic
    >>> lex = Lexicographic([ProgramJudge('my_judges:quality')])
    >>> lex.config.judges[0].path
    'swarm_arena:ProgramJudge'
    """

    name: ClassVar[str] = "lexicographic"

    class Config(_ConfigInputModel):
        """
        Store ordered judge references.

        Parameters
        ----------
        judges : list[Ref]
            One or more member references, accepting judge instances through coercion.

        Examples
        --------
        >>> from swarm_arena import ProgramJudge
        >>> from swarm_arena.scoring import Lexicographic
        >>> Lexicographic.Config(judges=[ProgramJudge('my_judges:quality')]).judges[0].path
        'swarm_arena:ProgramJudge'
        """

        judges: list[RefField] = Field(min_length=1)

    def __init__(self, judges: Sequence[Judge | Ref] | Config, /) -> None:
        """Validate references without importing or constructing their members."""
        data = judges if isinstance(judges, self.Config) else {"judges": list(judges)}
        self._config = _validate_config(self.Config, data, where="Lexicographic")
        self._members: list[Judge] | None = None

    @property
    def config(self) -> Config:
        """
        Return the ordered member references.

        Returns
        -------
        Lexicographic.Config
            Validated configuration for this composite judge.

        Examples
        --------
        >>> from swarm_arena import ProgramJudge
        >>> from swarm_arena.scoring import Lexicographic
        >>> Lexicographic([ProgramJudge('my_judges:quality')]).config.judges[0].path
        'swarm_arena:ProgramJudge'
        """
        return cast(Lexicographic.Config, self._config)

    def _get_members(self) -> list[Judge]:
        """Build references once, sanitizing arbitrary import or constructor errors."""
        if self._members is None:
            members: list[Judge] = []
            for ref in self.config.judges:
                try:
                    cls = ref.load()
                except TypeError:
                    raise TypeError(f"cannot build judge {ref.path}") from None
                except Exception:
                    raise ValueError(f"cannot import {ref.path}") from None
                if not issubclass(cls, Judge):
                    raise TypeError(f"cannot build judge {ref.path}: expected Judge subclass")
                # The sanitized ConfigError names the member path once, as its prefix.
                validated = _validate_config(
                    cls.Config, ref.config, where=f"cannot build judge {ref.path}"
                )
                try:
                    members.append(cls(validated))
                except TypeError:
                    raise TypeError(f"cannot build judge {ref.path}") from None
                except Exception:
                    raise ValueError(f"cannot build judge {ref.path}") from None
            self._members = members
        return self._members

    def preflight(self) -> None:
        """
        Build the member judges and check each member's readiness.

        Raises
        ------
        ValueError
            If a member cannot be imported, configured, or constructed.
        TypeError
            If a reference does not identify a valid judge class.
        Exception
            If a member raises; provider failures must reach the runner.

        Examples
        --------
        An in-memory module stands in for your own ``my_judges.py``:

        >>> import sys, types
        >>> from swarm_arena import ProgramJudge, Score
        >>> from swarm_arena.scoring import Lexicographic
        >>> def quality(goal, candidate):
        ...     return Score.maximize(1.0)
        >>> module = types.ModuleType('lex_preflight_demo')
        >>> module.quality = quality
        >>> sys.modules['lex_preflight_demo'] = module
        >>> Lexicographic([ProgramJudge('lex_preflight_demo:quality')]).preflight()
        >>> Lexicographic([ProgramJudge('lex_preflight_demo:missing')]).preflight()
        Traceback (most recent call last):
        ...
        ValueError: cannot import lex_preflight_demo:missing
        >>> del sys.modules['lex_preflight_demo']
        """
        for member in self._get_members():
            member.preflight()

    def rubric_hash(self) -> str:
        """
        Hash member names and rubric identities in scoring order.

        Returns
        -------
        str
            SHA-256 digest of ordered name and rubric pairs.

        Raises
        ------
        ValueError
            If a member cannot be imported, configured, or constructed,
            or its rubric cannot be encoded.
        TypeError
            If a reference does not identify a valid judge class.

        Examples
        --------
        >>> from swarm_arena import ProgramJudge
        >>> from swarm_arena.scoring import Lexicographic
        >>> len(Lexicographic([ProgramJudge('my_judges:quality')]).rubric_hash())
        64
        """
        members = self._get_members()
        try:
            return sha256_hex(
                canonical_json(
                    [
                        [ref.name if ref.name is not None else member.name, member.rubric_hash()]
                        for ref, member in zip(self.config.judges, members, strict=True)
                    ]
                )
            )
        except Exception:
            raise ValueError("cannot encode lexicographic rubric") from None

    def panel_hash(self) -> str:
        """
        Combine the default panel identities of all members.

        Returns
        -------
        str
            Empty when all member panels are empty, otherwise their ordered hash.

        Raises
        ------
        ValueError
            If a member cannot be imported, configured, or constructed.
        TypeError
            If a reference does not identify a valid judge class.

        Examples
        --------
        >>> from swarm_arena import ProgramJudge
        >>> from swarm_arena.scoring import Lexicographic
        >>> Lexicographic([ProgramJudge('my_judges:quality')]).panel_hash()
        ''
        """
        return _panel_hash([member.panel_hash() for member in self._get_members()])

    def anchor_count(self) -> int:
        """
        Return the largest member anchor requirement.

        Returns
        -------
        int
            Maximum anchor count required by any member.

        Raises
        ------
        ValueError
            If a member cannot be imported, configured, or constructed.
        TypeError
            If a reference does not identify a valid judge class.

        Examples
        --------
        >>> from swarm_arena import ProgramJudge
        >>> from swarm_arena.scoring import Lexicographic
        >>> Lexicographic([ProgramJudge('my_judges:quality')]).anchor_count()
        0
        """
        return max((member.anchor_count() for member in self._get_members()), default=0)

    def providers(self) -> frozenset[str]:
        """
        Return the union of member provider requirements.

        Returns
        -------
        frozenset[str]
            All providers used by any member.

        Raises
        ------
        ValueError
            If a member cannot be imported, configured, or constructed.
        TypeError
            If a reference does not identify a valid judge class.

        Examples
        --------
        >>> from swarm_arena import ProgramJudge
        >>> from swarm_arena.scoring import Lexicographic
        >>> Lexicographic([ProgramJudge('my_judges:quality')]).providers()
        frozenset()
        """
        return frozenset(
            provider for member in self._get_members() for provider in member.providers()
        )

    def models(self) -> tuple[str, ...]:
        """
        Return member model aliases without duplicates, preserving first use.

        Returns
        -------
        tuple[str, ...]
            Ordered unique model aliases.

        Raises
        ------
        ValueError
            If a member cannot be imported, configured, or constructed.
        TypeError
            If a reference does not identify a valid judge class.

        Examples
        --------
        >>> from swarm_arena import ProgramJudge
        >>> from swarm_arena.scoring import Lexicographic
        >>> Lexicographic([ProgramJudge('my_judges:quality')]).models()
        ()
        """
        return tuple(
            dict.fromkeys(model for member in self._get_members() for model in member.models())
        )

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        """
        Concatenate valid member keys for each fully scored candidate.

        Parameters
        ----------
        goal : Goal
            Goal supplied to each member judge.
        candidates : Sequence[Candidate]
            Full batch supplied to every member in order.
        ctx : JudgeContext
            Shared context retaining each candidate's first error reason.

        Returns
        -------
        list[Score]
            Composite scores in candidate order, with effective panel identities.

        Raises
        ------
        ValueError
            If a member cannot be imported, configured, or constructed.
        TypeError
            If a reference does not identify a valid judge class.
        Exception
            If a member raises; provider failures must reach the runner.

        Examples
        --------
        An in-memory module stands in for your own ``my_judges.py``: quality
        decides first, and lower cost breaks ties.

        >>> import sys, types
        >>> import anyio
        >>> from swarm_arena import Goal, ProgramJudge, Score
        >>> from swarm_arena.board import Candidate
        >>> from swarm_arena.executor import LocalExecutor
        >>> from swarm_arena.scoring import JudgeContext, Lexicographic
        >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
        >>> def quality(goal, candidate):
        ...     return Score.maximize(candidate.output.data['accuracy'])
        >>> def cost(goal, candidate):
        ...     return Score.minimize(candidate.output.data['usd'])
        >>> module = types.ModuleType('lex_judge_demo')
        >>> module.quality, module.cost = quality, cost
        >>> sys.modules['lex_judge_demo'] = module
        >>> goal = Goal(id='g', namespace='demo')
        >>> candidate = Candidate(
        ...     id='c1', run_id='r', goal_id='g', team='t', agent='a', round=1,
        ...     request=ToolRequest(),
        ...     output=ToolOutput(status=RunStatus.OK, data={'accuracy': 0.9, 'usd': 2.0}))
        >>> ctx = JudgeContext(goal=goal, executor=LocalExecutor(), artifact_root='unused')
        >>> lex = Lexicographic([ProgramJudge('lex_judge_demo:quality'),
        ...                      ProgramJudge('lex_judge_demo:cost')])
        >>> anyio.run(lex.judge, goal, [candidate], ctx)[0].key
        (0.9, -2.0)
        >>> del sys.modules['lex_judge_demo']
        """
        members = self._get_members()
        by_member: list[dict[str, Score]] = []
        for ref, member in zip(self.config.judges, members, strict=True):
            name = ref.name if ref.name is not None else member.name
            raw: dict[str, Score] = {}
            invalid_item = False
            for score in await member.judge(goal, candidates, ctx):
                if not isinstance(score, Score) or not isinstance(score.candidate_id, str):
                    invalid_item = True
                    continue
                raw[score.candidate_id] = score
            valid: dict[str, Score] = {}
            errors: dict[str, str] = {}
            for candidate in candidates:
                score = raw.get(candidate.id)
                if score is None:
                    reason = "an invalid score" if invalid_item else "no score"
                    errors[candidate.id] = f"member {name} gave {reason}"
                elif is_finite_score(score):
                    valid[candidate.id] = score
                else:
                    errors[candidate.id] = f"member {name} gave a non-finite key"
            if len({len(score.key) for score in valid.values()}) > 1:
                valid.clear()
                errors = {c.id: f"member {name} gave keys of different lengths" for c in candidates}
            for candidate_id, reason in errors.items():
                ctx.judge_error(candidate_id, reason)
            by_member.append(valid)
        scores: list[Score] = []
        for candidate in candidates:
            if any(candidate.id not in batch for batch in by_member):
                continue
            try:
                parts = [batch[candidate.id] for batch in by_member]
                scores.append(
                    Score(
                        candidate_id=candidate.id,
                        key=tuple(value for part in parts for value in part.key),
                        value=parts[0].value,
                        verdict=next(
                            (part.verdict for part in parts if part.verdict is not None), None
                        ),
                        notes="; ".join(part.notes for part in parts if part.notes),
                        llm_call_ids=[call_id for part in parts for call_id in part.llm_call_ids],
                        panel_hash=_panel_hash(
                            [
                                part.panel_hash or member.panel_hash()
                                for part, member in zip(parts, members, strict=True)
                            ]
                        ),
                    )
                )
            except Exception as error:
                ctx.judge_error(
                    candidate.id, f"{type(error).__name__}: cannot combine member scores"
                )
        return scores
