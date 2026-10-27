# Roadmap

The development version runs manual TOML and Python plans with scripted managers,
local tools, checks, judges, pruning policies, round snapshots, shared budgets,
and durable stop/resume. It includes an offline conformance kit and runnable
Rosenbrock and bring-your-own-script examples. Start with
[Getting started](getting-started.md) or [Planning modes](planning-modes.md).

The [shared arena](arena.md), human review and curation are available, alongside
model-driven managers, judges and planning with the `llm` extra. The `ray` extra
adds [cluster execution and Ray Job submission](running-on-ray.md). An optional
LLM gateway gives tool runs virtual keys and supports post-run reconciliation;
the pinned LiteLLM profile provides
[soft caps and estimated spend only](limits-and-safety.md#llm-gateway-hard-and-soft-caps).

Use the [installation guidance](getting-started.md#install), including its
source fallback until 0.1.0 is on PyPI. The
[offline kit](own-files.md#try-it-offline) remains available without model or
cluster integrations.

## Reserved for v2 and later

These are future features. The reserved fields and interfaces let them grow
without putting their implementations into the first release.

| Future feature | What v1 already reserves |
| --- | --- |
| Direct and broadcast messages (v2) | A2A v1.0.1 data types, including the input-required task state; `Plan.flows` as an allow-list; stable agent addresses; `context_id` and the agent-round `task_id` on journal events; and `LLMCall.parent_call_id`. |
| Live sharing with no barrier | `Plan.sharing` can be widened from `"round"`. |
| Outside agents joining | The board API is the arena API; a future server can expose it. |
| Forum (threads, replies, votes) | `Learning.tags` and lineage; votes can be `Score`s. |
| SSH, container, E2B, or Ray-sandbox executors | The `Executor` ABC. |
| Pareto ranking | `Score.key` is already a tuple, and ranking is a named judge. |
| A database arena backend (for example Postgres), for when outside agents join | The `Board` ABC; run journals and review and curation objects can be imported. |
| Nested teams | `schema_version` migration. |
| Mid-run release of unused gateway reservations | The private gateway adapter's `terminal_accounting` flag. The LiteLLM adapter sets it to false; v1 keeps full reservations charged. |

The messaging design uses an asynchronous mailbox: `send()` returns immediately
and a recipient reads `inbox()` on its own turns. A `send_message` action would
restrict recipients to the allowed flows, with one open request and one thread
per sender/recipient pair. It would add per-round message counts, byte caps, and
a hop limit; replies would use the recipient's budget share. A Runner would
never run inside another Runner. Messages would be marked untrusted and would
never count as evidence; pruning would remove the agent's flows with a reason.
The future `swarm-arena[a2a]` extra would map the data types to the wire. This
extra and these messaging actions are not available in v1.

## Follow-up projects

The **DeepRetro connector** belongs in its own repository and needs a separate
design. It would wrap `AutoSolver` and `run_batch.py` as tools, add integrity
checks, and define an LLM-judge rubric. Open decisions include whether the
manager chooses only settings and replicates or also continuations, grafting,
and learnings passed into DeepRetro prompts; ranking could use an LLM alone or
a deterministic audit key followed by an LLM through `Lexicographic`. The design
records uncommitted `audit_az_route`, continuation, and grafting code on DeepRetro
`dev` as a prerequisite to resolve before that work.

For **physika**, start with `ScriptTool`: supply `.phyk` files and a Python
runner, then score them with a `ScriptJudge` using hidden instances. A native
connector is a later option if needed for parsing, type checking, unit checks,
execution, and typed manager feedback. The first real goal remains to be chosen.

## Compatibility and 1.0

Release 1.0 after both connectors have run real campaigns. Until then, the
[versioning policy](versioning.md) follows SemVer: breaking changes in 0.x bump
the minor version and appear in `CHANGELOG.md`; removals first receive one minor
release with a `DeprecationWarning`. The reserved features above are a design
direction, not a release schedule or a promise that they all precede 1.0.
