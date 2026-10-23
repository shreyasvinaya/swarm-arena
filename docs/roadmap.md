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

Release packaging is still ahead. Keep using the documented GitHub installation
until the first release. The [offline kit](own-files.md#try-it-offline) remains
available without model or cluster integrations.
