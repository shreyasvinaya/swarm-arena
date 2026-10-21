from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, ClassVar

import pytest
from _fakes.secrecy import assert_secret_absent
from _planning.test_policy import _state
from pydantic import BaseModel, ValidationError

from swarm_arena import Chain, KeepTopTeams, Patience, Ref, Score, SuccessiveHalving
from swarm_arena._errors import ConfigError
from swarm_arena._model import ref as ref_module
from swarm_arena._model.ref import _ConfigInputModel
from swarm_arena._planning.policy import Policy
from swarm_arena.board import Candidate
from swarm_arena.manager import AgentStatus
from swarm_arena.planning import Decision, RoundState


def _state_by_round(values: dict[str, list[float | None]]) -> RoundState:
    count = max(map(len, values.values()))
    candidates: list[Candidate] = []
    scores: list[Score] = []
    for index in range(count):
        st = _state(
            {team: {"a": history[index]} for team, history in values.items()}, round=index + 1
        )
        for candidate, score in zip(st.candidates, st.scores, strict=True):
            identifier = f"c-{len(candidates) + 1}"
            candidates.append(candidate.model_copy(update={"id": identifier}))
            scores.append(score.model_copy(update={"candidate_id": identifier}))
    return st.model_copy(update={"candidates": candidates, "scores": scores})


def test_successive_halving_keeps_top_half_per_group() -> None:
    st = _state({"a": {"a1": 4, "a2": 1}, "b": {"b1": 3, "b2": 2}})
    d = SuccessiveHalving(keep=0.5, min_per_team=1).after_round(st)
    assert d.prune == {
        "a/a2": "SuccessiveHalving: rank 4 of 4 (best 1.0)",
        "b/b2": "SuccessiveHalving: rank 3 of 4 (best 2.0)",
    }


def test_successive_halving_keeps_min_per_team() -> None:
    st = _state({"a": {"a1": 4, "a2": 3}, "b": {"b1": 1, "b2": 0.5}})
    assert set(SuccessiveHalving(keep=0.5, min_per_team=1).after_round(st).prune) == {"b/b2"}


def test_starved_agent_never_pruned_for_lack_of_results() -> None:
    st = _state({"a": {"a1": 4}, "b": {"b1": 3, "b2": None}}, status={"b/b2": AgentStatus.STARVED})
    assert "b/b2" not in SuccessiveHalving().after_round(st).prune


def test_keep_top_teams_per_group_never_last_team() -> None:
    st = _state({"t1": {"a": 5}, "t2": {"a": 3}}, parts={"p": {"p1": {"a": 1}}})
    assert KeepTopTeams(1).after_round(st).prune == {"t2": "KeepTopTeams: rank 2 of 2 (best 3.0)"}


def test_patience_prunes_team_without_improvement() -> None:
    st = _state_by_round({"t1": [3, 3, 3], "t2": [1, 2, 4]})
    assert Patience(rounds=2).after_round(st).prune == {
        "t1": "Patience: no improvement for 2 rounds (best 3.0)"
    }


def test_chain_merges_in_order() -> None:
    d = Chain(KeepTopTeams(1), Patience(rounds=2)).after_round(
        _state_by_round({"t1": [3, 3, 3], "t2": [1, 2, 4], "t3": [0, 0, 0]})
    )
    assert d.policy == "KeepTopTeams+Patience" and set(d.prune) == {"t1", "t3"}  # noqa: PT018
    assert d.prune["t3"].startswith("KeepTopTeams:")


def test_spec_and_ref_forms_build() -> None:
    built = Ref(path="swarm_arena:SuccessiveHalving", config={"keep": 0.5}).build(Policy)
    assert isinstance(built, SuccessiveHalving)
    assert built.config.keep == 0.5
    assert Chain(Ref(path="swarm_arena:Patience", config={"rounds": 2})).config.policies[
        0
    ].path == ("swarm_arena:Patience")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: KeepTopTeams(0),
        lambda: SuccessiveHalving(0),
        lambda: SuccessiveHalving(1.1),
        lambda: SuccessiveHalving(float("nan")),
        lambda: SuccessiveHalving(min_per_team=-1),
        lambda: Patience(0),
        lambda: Chain(),
    ],
)
def test_invalid_config_is_rejected(factory: Callable[[], Policy]) -> None:
    with pytest.raises(ConfigError):
        factory()


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("KeepTopTeams", "k"),
        ("SuccessiveHalving", "keep"),
        ("SuccessiveHalving", "min_per_team"),
        ("Patience", "rounds"),
        ("Chain", "policies"),
    ],
)
def test_config_errors_hide_secret_values(name: str, field: str) -> None:
    secret = "SENTINEL-private-policy-value"
    with pytest.raises(ConfigError) as exc:
        Ref(path=f"swarm_arena:{name}", config={field: secret}).build(Policy)
    assert_secret_absent(exc.value, secret, sanitized=True)


def test_halving_rounds_up_and_respects_larger_team_minimum() -> None:
    st = _state({"t": {"a": 6, "b": 5, "c": 4}, "u": {"a": 3, "b": 2, "c": 1}})
    assert set(SuccessiveHalving(0.1, min_per_team=2).after_round(st).prune) == {"t/c", "u/c"}
    assert set(SuccessiveHalving(0.51, min_per_team=0).after_round(st).prune) == {"u/b", "u/c"}
    assert SuccessiveHalving(1).after_round(st).prune == {}


def test_halving_collapses_whole_team_prunes_and_ranks_unscored_last() -> None:
    st = _state({"t": {"a": 4, "b": 3}, "u": {"a": None, "b": None}})
    assert SuccessiveHalving(min_per_team=0).after_round(st).prune == {
        "u": "SuccessiveHalving: rank 3 of 4 (best None)"
    }
    assert KeepTopTeams(1).after_round(st).prune == {"u": "KeepTopTeams: rank 2 of 2 (best None)"}


def test_halving_uses_candidate_tiebreaks_and_best_score_per_agent() -> None:
    st = _state({"t": {"a": 1, "b": 1, "c": 1}})
    candidates = [
        candidate.model_copy(update={"id": identifier})
        for candidate, identifier in zip(
            st.candidates, ["c-10000", "c-9999", "c-10001"], strict=True
        )
    ]
    scores = [
        score.model_copy(update={"candidate_id": candidate.id})
        for candidate, score in zip(candidates, st.scores, strict=True)
    ]
    st = st.model_copy(update={"candidates": candidates, "scores": scores})
    assert set(SuccessiveHalving(0.3, 0).after_round(st).prune) == {"t/a", "t/c"}
    duplicate = candidates[0].model_copy(update={"id": "c-10002"})
    improved = scores[0].model_copy(update={"candidate_id": duplicate.id, "key": (2.0,)})
    st = st.model_copy(
        update={"candidates": [*candidates, duplicate], "scores": [*scores, improved]}
    )
    assert set(SuccessiveHalving(0.3, 0).after_round(st).prune) == {"t/b", "t/c"}


@pytest.mark.parametrize(
    "status", [AgentStatus.PRUNED, AgentStatus.STOPPED, AgentStatus.WATCHDOG_STOPPED]
)
def test_policies_skip_inactive_agents(status: AgentStatus) -> None:
    st = _state_by_round({"old": [20, 20, 20], "t": [4, 4, 4], "u": [1, 2, 3]})
    st = st.model_copy(update={"status": {**st.status, "old/a": status}})
    assert KeepTopTeams(1).after_round(st).prune == {"u": "KeepTopTeams: rank 2 of 2 (best 3.0)"}
    assert SuccessiveHalving(min_per_team=0).after_round(st).prune == {
        "u": "SuccessiveHalving: rank 2 of 2 (best 3.0)"
    }
    assert set(Patience().after_round(st).prune) == {"t"}


def test_finished_agents_still_participate() -> None:
    st = _state({"t": {"a": 4}, "u": {"b": 2}}, status={"u/b": AgentStatus.FINISHED})
    assert "u" in KeepTopTeams(1).after_round(st).prune


@pytest.mark.parametrize("change", ["fidelity", "rubric", "panel"])
def test_patience_does_not_compare_different_scales(change: str) -> None:
    st = _state_by_round({"t": [10, 10, 1], "u": [3, 4, 5]})
    if change == "fidelity":
        st = st.model_copy(
            update={
                "candidates": [
                    c.model_copy(update={"fidelity": "f2"}) if c.round == 3 else c
                    for c in st.candidates
                ]
            }
        )
    else:
        field = "rubric_hash" if change == "rubric" else "panel_hash"
        newest_ids = {c.id for c in st.candidates if c.round == 3}
        scores = [
            s.model_copy(update={field: "new"}) if s.candidate_id in newest_ids else s
            for s in st.scores
        ]
        st = st.model_copy(
            update={"scores": scores, "ranking_identity": {None: scores[-1].identity}}
        )
    assert Patience().after_round(st).prune == {}


def test_patience_compares_keys_and_formats_score_value_or_first_key() -> None:
    st = _state_by_round({"t": [1, 2, 3], "u": [0, 0, 0]})
    # The display value stays positive while minimization makes lower values better.
    st = st.model_copy(
        update={"scores": [s.model_copy(update={"key": (-s.key[0],)}) for s in st.scores]}
    )
    assert Patience().after_round(st).prune == {
        "t": "Patience: no improvement for 2 rounds (best 1.0)"
    }
    st = st.model_copy(update={"scores": [s.model_copy(update={"value": None}) for s in st.scores]})
    assert Patience().after_round(st).prune["t"].endswith("(best -1.0)")
    assert KeepTopTeams(1).after_round(st).prune["t"].endswith("(best -1.0)")


def test_policies_rank_parts_independently() -> None:
    st = _state(
        {"m": {"a": 50, "b": 40}},
        parts={"p": {"p1": {"a": 2, "b": 1}}, "q": {"q1": {"a": 4}, "q2": {"a": 3}}},
    )
    assert set(SuccessiveHalving(min_per_team=0).after_round(st).prune) == {"m/b", "p1/b", "q2"}
    assert set(KeepTopTeams(1).after_round(st).prune) == {"q2"}


def test_patience_waits_for_full_interval_and_prunes_unscored_team() -> None:
    assert Patience().after_round(_state_by_round({"t": [1, 1], "u": [2, 2]})).prune == {}
    st = _state_by_round({"t": [None, None, None], "u": [1, 2, 3]})
    assert Patience().after_round(st).prune == {
        "t": "Patience: no improvement for 2 rounds (best None)"
    }


@pytest.mark.parametrize(
    "policy", [KeepTopTeams(1), SuccessiveHalving(), Patience(), Chain(Patience())]
)
def test_every_policy_guards_final_empty_results(policy: Policy) -> None:
    assert policy.after_round(_state({"t": {"a": None}}, round=3, rounds_total=3)).no_result == (
        "no passing candidate"
    )


def test_builtin_config_and_nested_chain_ref_roundtrip() -> None:
    policies = [
        KeepTopTeams(KeepTopTeams.Config(k=2)),
        SuccessiveHalving(SuccessiveHalving.Config(keep=0.75, min_per_team=0)),
        Patience(Patience.Config(rounds=3)),
    ]
    chain = Chain(Chain.Config(policies=[Ref.of(Chain(*policies)), Ref.of(Patience())]))
    ref = Ref.of(chain)
    restored = Ref.model_validate_json(ref.model_dump_json()).build(Policy)
    assert isinstance(restored, Chain)
    assert restored.config == chain.config
    assert restored.after_round(_state({"t": {"a": 1}})) == chain.after_round(
        _state({"t": {"a": 1}})
    )


@pytest.mark.parametrize(
    ("cls", "field"), [(KeepTopTeams, "k"), (SuccessiveHalving, "keep"), (Patience, "rounds")]
)
def test_direct_constructor_validation_is_sanitized(cls: type[Policy], field: str) -> None:
    secret = "SENTINEL-direct-policy-input"
    with pytest.raises(ConfigError) as exc:
        cls(secret)  # ty: ignore[invalid-argument-type]
    assert_secret_absent(exc.value, secret, sanitized=True)
    with pytest.raises(ValidationError) as exc:
        cls.Config.model_validate({field: secret})
    assert_secret_absent(exc.value, secret, sanitized=True)


class _FixedPolicy(Policy):
    class Config(_ConfigInputModel):
        prune: dict[str, str] = {}
        policy: str = ""
        no_result: str | None = None

    builds: ClassVar[int] = 0

    def __init__(self, config: BaseModel | Mapping[str, Any] | None = None, /) -> None:
        super().__init__(config)
        type(self).builds += 1

    def after_round(self, state: RoundState) -> Decision:
        return Decision.model_validate(self.config.model_dump())


class _BrokenPolicy(_FixedPolicy):
    def __init__(self, config: BaseModel | Mapping[str, Any] | None = None, /) -> None:
        raise ValueError("SENTINEL-policy-constructor")


class _WrongConstructorPolicy(_FixedPolicy):
    def __init__(self, config: BaseModel | Mapping[str, Any] | None = None, /) -> None:
        raise TypeError("SENTINEL-policy-constructor")


def test_chain_is_lazy_caches_members_and_guards_merged_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = ref_module.load_object

    def track(path: str) -> object:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(ref_module, "load_object", track)
    monkeypatch.setattr(_FixedPolicy, "builds", 0)
    path = Ref.of(_FixedPolicy).path
    chain = Chain(
        Ref(path=path, config={"prune": {"t": "first"}, "policy": "", "no_result": "first"}),
        Ref(
            path=path,
            config={
                "prune": {"t": "second", "u": "second"},
                "policy": "named",
                "no_result": "second",
            },
        ),
    )
    assert calls == []
    assert _FixedPolicy.builds == 0
    state = _state({"t": {"a": 1}, "u": {"b": 2}})
    result = chain.after_round(state)
    assert result == Decision(prune={"t": "first"}, policy="named", no_result="first")
    assert _FixedPolicy.builds == 2
    assert calls == [path, path]
    assert chain.after_round(state) == result
    assert _FixedPolicy.builds == 2
    assert calls == [path, path]


@pytest.mark.parametrize("cls", [_BrokenPolicy, _WrongConstructorPolicy])
def test_chain_sanitizes_constructor_failures(cls: type[Policy]) -> None:
    chain = Chain(Ref.of(cls))
    with pytest.raises((ValueError, TypeError), match="cannot build policy") as exc:
        chain.after_round(_state({}))
    assert_secret_absent(exc.value, "SENTINEL-policy-constructor", sanitized=True)


def test_chain_leaves_import_error_wrapping_to_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ref_module.importlib.import_module

    def fail(name: str, package: str | None = None) -> ModuleType:
        if name == "policy_import_failure":
            raise RuntimeError("SENTINEL-policy-import")
        return original(name, package)

    monkeypatch.setattr(ref_module.importlib, "import_module", fail)
    chain = Chain(Ref(path="policy_import_failure:Policy"))
    with pytest.raises(ValueError, match=r"^cannot import policy_import_failure:Policy$") as exc:
        chain.after_round(_state({}))
    assert_secret_absent(exc.value, "SENTINEL-policy-import", sanitized=True)


def test_chain_rejects_nonpolicy_members_lazily() -> None:
    chain = Chain(Ref(path="swarm_arena:FunctionTool", config={"function": "unused:fn"}))
    with pytest.raises(TypeError, match="not a subclass of Policy"):
        chain.after_round(_state({}))
