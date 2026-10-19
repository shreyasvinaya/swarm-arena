# swarm-arena: design

- **Status:** **APPROVED (r8)**. The owner's rule was: "if Codex gives LGTM, consider it approved". Codex (gpt-6.1-sol) gave LGTM in review round 4 on 2026-10-18 (§17). r8 applies its 2 non-blocking notes. Next step: the implementation plan.
- **Date:** 2026-10-16 (r3: 2026-10-18, r4: 2026-10-18, r5: 2026-10-18, r6: 2026-10-18, r7: 2026-10-18, r8: 2026-10-18)
- **Owner:** Shreyas Vinaya
- **Package:** `swarm-arena` on PyPI, `import swarm_arena`, MIT license

A tag on a line shows who made that decision:

| Tag | Meaning |
|---|---|
| **[owner]** | The owner decided it. |
| **[call]** | The designer decided it while answering the builder check. The owner should check it. |
| (no tag) | It follows directly from an [owner] decision. |

## 1. Intent

`swarm-arena` is an open-source Python library. It runs a **swarm** of agents on one or more hard goals, each on its own goal → teams → agents tree, and returns the best answers it can prove.

```
                 ┌── Team A ──┬─ agent ─┬─ agent ──┐
                 │            ├─ agent ✗           ├──┐
  Goal ──────────┼── Team B ──┼─ agent ─┴─ agent ──┘  │
                 │            └─ agent ✗              ├──► Solution
                 └── Team C ✗ (pruned)                │
                                        (survivors) ──┘
```

1. A **goal** fans out to several **teams**.
2. Each team has several **agents**, and each agent does one dedicated task.
3. Weak branches get **pruned**.
4. The good results **converge** into one solution.

### Why

The owner works on hard chemistry and physics problems at Deep Forest Sciences:
- **DeepRetro:** LLM-driven retrosynthesis.
- **physika:** a physics programming language.

Today the owner does swarm-like work by hand. For example, DeepRetro's local `run_comparison.py` runs molecules × models with time, memory, and credit stops, and `terminal_continuation.py` sends unsolved parts out as new jobs. These scripts are not committed yet.

A web survey looked at about 15 frameworks: CrewAI, LangGraph, AutoGen/MAF, Google ADK, Agno, Strands, the OpenAI Agents SDK, the Claude Agent SDK, PydanticAI, Bernstein, mco, Einstein Arena, and others. None of them does all four of these:
1. Makes the goal → teams → agents → solution tree the main concept.
2. Treats pruning as a first-class step with a recorded reason.
3. Lets any program, not only an LLM loop, be the thing an agent drives.
4. Keeps a small, stable API.

The research is in `tasks/research/`. That folder is not shipped.

### Who it serves

- **Now [owner]:** one research group, running its own agents on its own machines: a laptop, a Ray cluster, and some VMs.
- **Later:** outside agents may join the arena. The design leaves a clean place for that. v1 does not build it.

## 2. Goals, non-goals, anti-goals

### Goals (v1)

- **G1. Generic core.** The core has no chemistry, no physics, and no project names. Each project adds a **connector** on the core's abstract base classes. The connector lives in that project's own repo. **[owner]**
- **G2. Agent = tool + manager LLM.** **[owner]**
  - A **tool** is one running copy of a program, in its own environment.
  - The **manager LLM** drives the tool, reads results, writes learnings, and reads other agents' learnings.
- **G3. Four planning modes.** These are manual, LLM, together, and autonomous. In autonomous mode, the LLM re-plans between rounds from scores and learnings. **[owner]**
- **G4. Layered scoring.** All three layers can be swapped. **[owner]**
  - **Checks:** pass or fail, decided by a program.
  - **Judges:** a score from a program or an LLM. LLM judges can vote as a panel.
  - **Human review:** happens after the run.
- **G5. Pruning and convergence between rounds.** Every cut has a reason.
- **G6. Shared learnings in v1.** Agents use a shared **board**. Every learning must cite a real result as evidence. **[owner]**
- **G7. Same code on a laptop and on a Ray cluster.** **[owner]**
- **G8. A private arena.** It is a long-lived store of problems, results, scores, learnings, and lineage, so later runs can build on earlier ones. **[owner]**
- **G9. Safety rails.** The core enforces them. They are never left to the LLM. **[owner]**
- **G10. Open-source quality.** **[owner]**
  - A small public API.
  - numpydoc docstrings with examples.
  - Sphinx docs on Read the Docs.
  - Tests.
  - A written backward-compatibility policy.

### Non-goals (v1)

- A hosted website, user accounts, or outside agents joining over the network.
- Direct agent-to-agent or team-to-team messages. They come in **v2**. v1 reserves the types for them. **[owner]**
- Chat-style "org chart" teams where LLMs delegate to each other (D25). **[owner]**
- Owning any tool's internal search. For example, DeepRetro's tree search stays inside DeepRetro.
- Domain logic in the core.
- SSH or container executors.
- A GUI.
- Windows support.

### Anti-goals (failure even if it "works")

- **Unruly agents** that act outside their allowed actions, loop, flood the board, or spend past the limits. **[owner]**
- **The wrong number of agents.** Too many wastes money on duplicate work. Too few gives poor coverage. **[owner]**
- **Results you cannot trust.** For example, the swarm optimizes a weak proxy, such as a DeepRetro route that is "solved" but adds carbons from nowhere.
- **API churn** that breaks connectors in other repos.
- **Bloat:** heavy dependencies, framework lock-in, or too many concepts.

## 3. Decisions made with the owner

| # | Decision |
|---|---|
| D1 | The system is private, for the owner's own agents on the owner's own machines. A clean place for outside agents is kept for later. |
| D2 | The compute is a Ray cluster plus some VMs. Executors are `local` and `ray`. |
| D3 | An agent is a program instance plus a manager LLM. The manager drives the program, keeps learnings, and shares them. |
| D4 | The manager does not grade its own work. Checks and judges do. |
| D5 | Every learning links to proof. |
| D6 | v1 shares through a board. v2 adds direct messages. |
| D7 | Managers are provider-agnostic. |
| D8 | Teams and agents can come from a human, an LLM, both together, or an autonomous LLM that re-plans from learnings. |
| D9 | Autonomous re-planning happens only between rounds and inside hard limits. Every change has a reason. |
| D10 | DeepRetro: an LLM judge filters routes, then a chemist reviews the shortlist after the run. The chemist never interrupts a run. |
| D11 | physika: a deterministic program scores, or at most an LLM judge. No human. |
| D12 | The first manager add-on uses the OpenAI Agents SDK. The core does not import it. |
| D13 | Connectors are not in this repo. The core ships the abstract base classes. |
| D14 | The name is `swarm-arena`. **Reserved on PyPI on 2026-10-18** with a placeholder `0.0.0a1` (no code). `agent-swarm` was refused by PyPI ("too similar to an existing project": `agentswarm`). |
| D15 | Guardrails and agent-count rules as in §8. |
| D16 | A run can hold **one or many goals**. Many goals share one `RunConfig` and one budget (§4.10). |
| D17 | v1 supports splitting a goal into **parts** (sub-goals), which are solved separately and then joined (§4.11). |
| D18 | The cluster has **S3**, and **more than one swarm can run at the same time**. Shared state lives on S3 (§7). |
| D19 | A user can bring **a set of files plus a script that uses them**, with no connector code. For example: some `.phyk` files plus a Python file that runs them or calls an LLM. The swarm uses those files (§4.12). |
| D20 | The learnings board is **human-readable** (§6.3.1). |
| D21 | **Every LLM call is logged**, through Langfuse and LiteLLM, so every result can be traced to the exact LLM calls behind it (§7.3). |
| D22 | The board must not bloat agent context, and must not be one ever-growing file. Agents get a small digest and search on demand. Humans get one page per round plus a searchable viewer (§6.3, §6.3.1). |
| D23 | **Multi-LLM support through LiteLLM** for every LLM call made by swarm-arena's built-in LLM components (in the optional `llm` extra), with a human-only model catalog of allowed models (§4.13). |
| D24 | **Cache writes and reads are on:** provider prompt caching by default, plus optional LiteLLM response caching with separate read and write switches per role (§4.13). |
| D25 | Lessons from **Agency Swarm** (agency-swarm.ai, studied 2026-10-18) are applied. It is a complement, not a competitor. Its message-flow ideas shape v2. Its known problems (no turn or cost limits, blocking nested calls, races on shared tool state, exact SDK pins) are designed out (§4.5, §8.1, §10, §11). Chat-style "org chart" teams, where LLMs delegate to each other, are a non-goal (§2). Notes: `tasks/research/agency-swarm.md`. |

## 4. Concepts

The system has a small set of plug-in points (abstract base classes) and plain data types.

| Concept | Kind | One line |
|---|---|---|
| `Goal` | data | What to solve: text for the LLMs, plus structured inputs. |
| `RunConfig` | data | **Human-only** settings: hard limits, the model catalog, the planner catalog, executor, storage, and secrets policy. An LLM never writes it. **[call]** |
| `Plan` | data | Teams, agents, judges, checks, policy, and rounds. An LLM may write it, but only from the planner catalog (§4.1). It is versioned. |
| `Tool` | ABC | How to run one copy of a program and read its output. |
| `Check` | ABC | Pass or fail on one output, decided by a program. |
| `Judge` | ABC | Scores for a batch of outputs that passed their checks. |
| `Manager` | ABC | The brain of one agent. It acts only through `AgentContext`. |
| `Planner` | ABC | Makes or changes a `Plan`. |
| `Policy` | ABC | Between rounds, decides who survives and why. |
| `Executor` | ABC | Where tool runs happen (local or Ray). |
| `Board` | ABC | Shared learnings, results, scores, and lineage. This is the arena store. |
| `Curator` | ABC | Optional. At round end, merges near-duplicate learnings (§6.3). **[call]** |
| `Swarm` | class | Runs one or more `Plan`s (one per goal) under one `RunConfig`. This is the one entry point. |

### 4.1 Components are referenced by import path [call]

Every component in a `Plan` (tool, manager, check, judge, policy) and in a `RunConfig` (executor, board, curator) is a **`Ref`**:
- `path`, for example `"deepretro_swarm:AutoSolveTool"`.
- `config`, a dict.
- `name: str | None`. The default is the class's `name`. Names must be unique within `Plan.checks` and within `Plan.judges`, so two `ScriptJudge`s can be told apart. `Plan.ranking` and `Score.judge` use this name. **[call]**

Each component class declares a nested pydantic `Config` model and is built as `cls(cls.Config.model_validate(config))`. This has four effects:
1. Plans are fully TOML/JSON-serializable, so they can be resumed.
2. Ray workers can build tools from the plan.
3. `validate_plan` and `LLMPlanner` can check an LLM's tool config **before any money is spent**.
4. The Python API also accepts component instances, as long as their class is importable. A lambda or local function is rejected with a clear error.

`FunctionTool` therefore needs a module-level function. It runs in a subprocess with the current interpreter, via `python -m swarm_arena.run_function`, so cancel and kill behave the same as for any other tool. Its `Config` has `deterministic: bool = True` and `providers: list[str] = []`, like `ScriptTool` (§4.3). **[call]**

**The planning boundary** **[call]**. One rule covers all executable configuration that an LLM writes.
- `RunConfig.planner_catalog: PlannerCatalog` is human-only. It lists every component that an LLM-written plan may use: tools, managers, checks, judges, policies, and the converge team's tools and managers.
- Each `CatalogEntry` has:
  - `name`: what the planner LLM picks.
  - `ref`: the import path.
  - `description`: shown to the planner LLM.
  - `config`: the fixed config, set by the human.
  - `settable`: a JSON schema of the **only** config fields an LLM may set. The default is none. Every field that is not in `settable` is fixed by the human.
- **Script components are human-only.** For `ScriptTool`, `ScriptCheck`, and `ScriptJudge`, these fields can **never** be in `settable`: `command`, `python` (the interpreter), `files`, `env` and secret access, and `editable`. Loading a catalog with such an entry fails. Only a human writes them.
- **Which plans it applies to.** In a plan whose `origin` is not `human` (that is `llm`, `together`, or `autonomous`, §4.6), every `Ref` that the revision adds or changes must resolve to a catalog entry: the same `path` as the entry's `ref`, and the entry's fixed value for every config field outside `settable`. Settable values must match the `settable` schema. `validate_plan` rejects anything else.
- **Carried-over human parts keep their provenance** **[call]**. Refs carried over unchanged from a human-origin parent revision (plan-level checks, judges, ranking, and unchanged teams, agents, and parts) keep their human provenance, through every later revision while they stay unchanged. The catalog check applies to every `Ref` that a non-human revision adds or changes. The core finds what changed by comparing each revision with its parent: agents by stable address (§4.2), and checks, judges, and parts by name. So an autonomous replanner can keep a human r0, for example one with the `ScriptJudge` of §4.12, without copying its components into the catalog.
- **Overrides follow the same rule** **[call]**. A `RoundSpec` or `ConvergeSpec.round` carried over unchanged from a human revision keeps its human provenance, and the fields its `tool_overrides` set count as human-set. An override that a non-human revision adds or changes may set only settable fields. A carried-over tool that is not in the catalog has no settable fields, so such an override may not change its config.
- **Where it applies.** Everywhere a `Ref` or config can appear: `Plan.teams`, `Plan.checks`, `Plan.judges`, `Plan.policy`, `PartSpec` checks and judges, nested refs inside a config (for example the judges of `Lexicographic` or the policies of `Chain`), `RoundSpec.tool_overrides`, and `ConvergeSpec.round` and `ConvergeSpec.team`. An added or changed override may set only settable fields.
- A human-written plan (`origin = "human"`) may use any importable ref.

### 4.2 Data types

All data types are pydantic v2 models, frozen where possible. Every stored format carries `schema_version`.

**Inputs and settings**

- **`Goal`**: `id`, `namespace`, `title`, `description`, `inputs: dict`, `files: FileSet`, `metadata`.
  - `files` holds the user's files for this problem (D19, §4.12).
  - In TOML they are written as globs, relative to the plan file.
  - When the plan is validated, they are **snapshotted**: each file is stored by sha256 in `artifact_root`, and the plan revision records `path → sha256`. So a run is reproducible even if the files on disk change later. Ray workers fetch the files from S3.
  - `namespace` is the problem family, for example `"deepretro"`. Learnings are shared by namespace (§6.3).
  - The identity of a goal across runs (`GoalKey`) is `(namespace, id, sha256(inputs), sha256(files manifest))`. The files manifest is the sorted `path → sha256` map recorded at validation. So results for changed files never mix with old ones. **[call]**
- **`RunConfig`** **[call]**: the safety rails live here, apart from the plan, so a replanner cannot loosen them.
  - `limits: Limits` (§8.1)
  - `reserve_fraction` (default 0.15)
  - `goal_weights` (optional, D16)
  - `executor: Ref`
  - `board: Ref | None` (default `ArenaBoard` on `arena_root`, §4.9) **[call]**
  - `curator: Ref | None` (default none, §6.3) **[call]**
  - `run_root`, `arena_root`, `artifact_root`: each one is a local path or `s3://…` (§7)
  - `secrets: SecretsPolicy`: `file` and `allow`, the secret names a tool may get (§11) **[call]**
  - `schedule_timeout_s`
  - `sync_interval_s` (default 60, §7.1)
  - `lease_ttl_s` (default 60): the run-ownership lease (§7.1). It sets only when a takeover may happen (liveness). Correctness does not depend on it, or on any clock (§7.1). **[call]**
  - `settle_timeout_s` (default 60, §8.1) **[call]**
  - `tracing: TracingConfig` (§7.3) and `llm_gateway: GatewayConfig | None` (default none, §7.3). `llm_gateway.require_hard_cap` defaults to true when `max_usd` is set (§7.3). **[call]**
  - `llm_gateway.request_timeout_s`: the longest time one gateway request may run. The strict gateway profile must enforce a limit no longer than this, and preflight checks it. The final reconcile after the run waits this long after a key is closed (§7.3). **[call]**
  - `models: ModelCatalog`: the model catalog, with prices (§4.13)
  - `planner_catalog: PlannerCatalog`: the components an LLM-written plan may use (§4.1) **[call]**
  - `cache: CachePolicy`: the response-cache policy (§4.13)
- **`Plan`**:
  - `schema_version`, `goal`, `parts: list[PartSpec]` (empty by default), `teams: list[TeamSpec]`
  - `checks: list[Ref]`, `judges: list[Ref]`, `ranking: str` (the `Ref.name` of the ranking judge, §4.1)
  - `policy: Ref`, `rounds: list[RoundSpec]`, `converge: ConvergeSpec`
  - `sharing: Literal["round"]`
  - `flows: list[FlowSpec] = []`. **Reserved for v2 messaging (§10).** v1 rejects a list that is not empty, so the TOML shape and the planner prompt do not change in v2. **[call]**
  - `rationale`
  - `origin: "human" | "llm" | "together" | "autonomous"`, `revision`, `parent_revision`. The core sets these. A planner's output never does (§4.6). **[call]**
- **`FlowSpec`** (v2, reserved): `sender`, `recipient` (an agent, a team, or `"*"` for broadcast), `rationale`. This is an allow-list: no edge means no messages.
- **Stable addresses** **[call]**: the address `goal/team/agent` is unique and is **never reused** across plan revisions. Renaming an agent creates a new agent. Lineage, budgets, and v2 message threads all depend on this.
- **Ids are unique across epochs** **[call]**. Candidate, learning, and score ids are unique across epochs. They include the writer's epoch (§7.1), for example `c-e1-0193`, or a random part. So a new epoch never reuses an id that an old owner may have used in work it never committed, for example in a log file name or a gateway key alias (§7.1, §7.3). Reservation ids follow the same rule, because settlement is idempotent by reservation id (§8.1).
- **`PartSpec`** (D17): `name`, `goal: Goal` (the sub-goal), `checks: list[Ref] | None`, `judges: list[Ref] | None`, `ranking: str | None`. A field left as `None` uses the plan-level value.
- **`TeamSpec`**: `name`, `rationale` (required), `part: str | None` (`None` means the team works on the main goal), `agents: list[AgentSpec]`.
- **`AgentSpec`**:
  - `name`, `rationale` (required)
  - `tool: Ref`
  - `manager: Ref`, for example model, instructions, and seed inputs
  - `replicates: int = 1`
  - `resources: Resources` (`cpus`, `gpus`, `memory_gib`, `custom: dict[str, float]`)
- **`RoundSpec`** (optional, one per round) **[call]**:
  - `tool_overrides: dict`, merged into every agent's tool config for that round, for example `{"epochs": 100}`.
  - `tool_timeout_s` (optional, §6.1).
  - `fidelity: str | None`: a label for the effort level. The default is a hash of `tool_overrides`. `Candidate.fidelity` is copied from its round (§4.4).
  - This is how "more effort for the survivors each round" works, for example physika epochs 10, then 100, then 400.
  - **Number of rounds:** `len(plan.rounds)` if it is not empty, else `limits.max_rounds`. `validate_plan` rejects `len(plan.rounds) > limits.max_rounds`. "Rounds that are left" (§8.1) uses the current revision.
- **`ConvergeSpec`** **[call]**:
  - `mode: "select" | "team"`, `top_n: int = 2`, `distinct: bool = True` (distinct by `Tool.fingerprint`), `allow_partial: bool = False` (§4.3).
  - When `plan.parts` is not empty, `mode` must be `"team"`. `validate_plan` rejects `"select"`.
  - `min_improvement: float | None = None` (§4.9).
  - `team: TeamSpec | None` and `round: RoundSpec | None`, used when `mode="team"`. The round settings allow, for example, a re-run at full fidelity. With no `round`, the converge team uses the last round's settings and fidelity. The converge team is not in `Plan.teams`, does not count toward `max_teams`, runs one round, and is paid from the goal's final slice (§8.1).
  - `parents_per_part: int = 1` and `max_join_attempts: int = 1` (§4.11).

**Tool runs and results**

- **`ToolRequest`**:
  - `inputs: dict`, validated against `Tool.input_model`.
  - `parents: list[ParentRef]`: the candidates this run builds on. **[call]** For each parent, the executor puts its `data` and its artifacts, read-only, in `workspace/parents/<candidate-id>/` before `command()` runs, locally or on a Ray worker. Only a base parent that passes the overlay rule also changes the workspace root (§4.12). Lineage is declared, never guessed.
  - `replicate: int` (§6.2).
  - `round_overrides: dict`.
  - `edits: dict[str, str]`: new contents for files the manager may change (§4.12). Edited files are stored by sha256 and recorded in the candidate's `overlay`, so other agents can build on them.
- **`ToolOutput`**:
  - `status: RunStatus`
  - `partial: bool` **[call]**
  - `data: dict`
  - `files: list[FileRef]` (name, path, sha256, size). It is named `FileRef` to avoid a clash with A2A `Artifact`. **[call]**
  - `usage: Usage`
  - `logs_ref`, `started_at`, `ended_at`
- **`Candidate`**:
  - `id`, `run_id`, `goal_id`, `team`, `agent`, `round`, `fidelity` (copied from its `RoundSpec`)
  - `request`, `output`, `checks: list[CheckResult]`
  - `gated_out`: derived, never stored. It is true when any check failed (§4.4).
  - `parent_ids`
  - `overlay: dict[str, str]`: the full effective file overlay, `path → sha256`. It is the base parent's overlay, if the overlay rule let it onto the workspace root, plus this request's edits (§4.12). **[call]**
  - `read_ids`: the learnings and results the manager read before this run (recorded automatically)
  - `requested_by_call`: the manager LLM call that asked for this run (§7.3)
  - `llm_call_ids`: the LLM calls made **inside** the tool run (§7.3)
  - `llm_calls_source: "gateway" | "reported" | "langfuse" | "none" | "unknown"`: where `llm_call_ids` came from (§7.3) **[call]**
  - `origin: "agent" | "human"`
- **`RunStatus`**: `ok | error | timeout | oom | budget_exceeded | cancelled | provider_failed | lost`.
  - It covers execution only. Whether a result is good is a separate question.
  - `lost` means the driver died while the run was in flight, or the run's completion event was not committed (§7.1). Its reservation stays charged (§8.1). **[call]**
- **`AgentStatus`** (one per agent per round): `active | finished | starved | pruned | stopped | watchdog_stopped`. **[call]**

**Scores and learnings**

- **`CheckResult`**: `check`, `passed`, `reasons: list[str]`, `details`.
- **`Score`**:
  - `id`, `candidate_id`, `judge`, `rubric_hash`
  - `panel_hash`: a hash of the panel members that made the score. It is empty for a judge that is not a panel. **[call]**
  - The **judge identity** is `judge` + `rubric_hash` + `panel_hash` (§4.4).
  - `key: tuple[float, ...]`: larger is better, compared in order. Every value must be finite (§4.4).
  - `verdict`, `notes`, `batch_id`
  - `llm_call_ids`: for LLM judges, the calls that produced this score (§7.3)
  - Helpers: `Score.minimize(x)` and `Score.maximize(x)`.
- **`LLMCall`** (D21, §7.3): `id`, `run_id`, `goal_id`, `team`, `agent`, `round`, `role` (`manager | judge | planner | curator | tool`), `candidate_id` (for tool-internal calls), `model_alias`, `model`, `provider`, `request_ref`/`response_ref` (sha256 of the full payloads), `tool_calls`, `usage` (including `cache_read_tokens`, `cache_write_tokens`, and the 5m/1h split), `cache_hit`, `cached_from_call_id` (for a response-cache hit, the original call, §4.13), `reservation_id` (§8.1), `cost_usd`, `latency_ms`, `status`, `error`, `trace_url`, `parent_call_id` (the call that led to this one, for example a judge call made for a manager's run, or a v2 reply).
- **`Event`**: one journal line. Fields: `schema_version`, `epoch`, `seq`, `ts`, `run_id`, `context_id` (= run, A2A), `task_id` (the agent-round or tool-run task, A2A), `goal_id`, `team`, `agent`, `round`, `type`, `payload`. Every event carries its A2A ids, so v2 replies and threads can link to v1 history with no migration. **[call]**
  - There is **one journal per run** (§7.1). `goal_id` tells which goal an event belongs to. It is empty for run-level events. Per-goal views are derived from it.
  - The journal is also the run's **arena record**. Other runs read it directly (§7.2). There are no separate arena files for a run's results. **[call]**
  - `seq` is one counter for the whole run. A new epoch continues it from the committed head (§7.1). `epoch` is the writer's ownership epoch (§7.1). **[call]**
- **`Learning`**:
  - `id`, `namespace`, `run_id`, `goal_id`, `round`
  - `author`: team/agent, or `human:<name>`
  - `title`: one plain sentence, at most 120 characters (D20)
  - `text`: short Markdown, at most 1,500 characters (D20)
  - `evidence: list[str]`: at least one candidate id. A human learning may have none, and is then marked "from a human".
  - `tags`
  - `verified: bool`: true when a complete (not partial) evidence candidate passed its checks (§4.3)
  - `pinned` / `hidden`: set only by a human, through the CLI
  - `refines: list[str]`, `supersedes: list[str]`: links to older learnings (§6.3)
  - `posted_by_call`: the LLM call that wrote it (§7.3)

**A2A-shaped protocol types** (in `swarm_arena.protocol`): `Message`, `Part`, `Artifact`, `Task`, `TaskStatus`, `TaskState`, `Role`.
- They are frozen dataclasses that mirror **A2A v1.0.1** (`lf.a2a.v1`) field for field. Enum values are the proto names, for example `"TASK_STATE_COMPLETED"`.
- How v1 uses them **[call]**:

| swarm-arena thing | A2A record |
|---|---|
| One run | One `context_id` |
| **One tool run (candidate)** | One `Task`, with `reference_task_ids` = its parents |
| **One agent-round** | One manager `Task`. Its `Artifact`s are the learnings (JSON `data` part plus a text summary) |

- `RunStatus` maps to `TaskState` with no new states:
  - `cancelled` maps to `CANCELED`.
  - `budget_exceeded`, `timeout`, and `lost` map to `FAILED` with `metadata["swarm_arena.reason"]`.
- v1 writes these records to the journal and sends nothing. v2 adds sending (§10).

### 4.3 `Tool` (the connector's main class)

A tool says **what** to run. The executor decides **where** it runs.

```python
class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]               # shown to the manager LLM
    input_model: ClassVar[type[BaseModel]]   # what the manager may set
    Config: ClassVar[type[BaseModel]]        # plan-level config (§4.1)

    def preflight(self) -> None: ...                                # optional
    def is_deterministic(self, request: ToolRequest) -> bool: ...   # optional, default True; False → replicates allowed (§6.2)
    def editable(self) -> Sequence[str]: ...                        # optional, default (): globs of files the manager may edit (§4.12)
    def estimate(self, request: ToolRequest) -> Estimate | None: ...  # optional: seconds, tokens, usd
    def providers(self) -> frozenset[str]: ...                      # optional: LLM providers it calls (§6.4)
    def fingerprint(self, output: ToolOutput) -> str: ...           # optional: identity for "distinct"
    def extra_actions(self) -> Sequence[Action]: ...                # optional: typed manager actions
    @abstractmethod
    def command(self, request: ToolRequest, workspace: Path) -> Command: ...
    @abstractmethod
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput: ...
```

- **`Command`** fields:
  - `argv`, `cwd`
  - `env`: an **allowlist**
  - `timeout_s`
  - `python`: the tool's own venv interpreter
  - `grace_s`
- `command()` and `collect()` run **next to the tool**, on the worker. Only the `ToolOutput` and its files travel back.
- `is_deterministic()` is a method, not a class constant, so one generic tool can be either. `ScriptTool` and `FunctionTool` read `deterministic` and `providers` from their `Config`. **[call]**
- After a timeout or a cancel, `collect()` **still runs** once the grace period ends. The output is kept with `partial=True`. **[call]**
  - Partial candidates are checked, judged, and can be parents, but they are flagged.
  - Unless `ConvergeSpec.allow_partial` is set, they are not selected for the solution, and policies, `best()`, `min_improvement`, and anchors rank only complete candidates.
  - Why: research found DeepRetro runs that were killed after using up to about 1M tokens, with nothing saved.
- The tool program is **never imported** by the core or the driver.

### 4.4 `Check` and `Judge`

```python
class Check(ABC):
    name: ClassVar[str]
    feedback: Literal["pass_fail", "reasons"] = "reasons"   # what managers see; ScriptCheck sets it from its config
    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult: ...

class Judge(ABC):
    name: ClassVar[str]
    async def judge(self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext) -> list[Score]: ...
```

**Checks**
- They run right after each tool run, so the manager sees the result at once.
- A failed check marks the candidate `gated_out`. The candidate is kept, but it is never judged or selected.
- `CheckContext` and `JudgeContext` both have `execute(command)`. It runs a command through the executor, for example a check that needs RDKit in the tool's venv. `ScriptCheck` and `ScriptJudge` always use it. **[call]**

**Judges**
- `Plan.judges` lists every judge to run. `Plan.ranking` names the one that drives pruning and selection. **[call]** The other scores are stored for re-ranking later.
- **One scale across rounds** **[call]**:
  - LLM judges score each candidate on its own, against a fixed rubric scale with anchors. They batch only to save cost.
  - Each batch also re-scores up to `k` current leaders as **anchors**, so any drift between batches is visible.
  - **Judge identity** **[call]** is the judge name + `rubric_hash` + the panel membership hash (`Score.panel_hash`). If the panel's models change, that is a new judge identity, so old and new scores are never compared.
  - Policies compare only scores with the same judge identity and the same `fidelity`. They rank within the newest fidelity in each group (§4.11). Select and the final judge use only the newest fidelity reached in the group.
  - Anchor re-scores never read the response cache (§4.13).
- **Bad scores** **[call]**. A score key with a non-finite value (NaN or inf) is rejected. It is recorded as a judge error for that candidate, and the candidate stays unjudged by that judge.
- **Final judge** **[call]**: after converge, the ranking judge re-scores all finalists **together in one batch**. That batch sets the final order of the solution, and nothing else changes it (`min_improvement` does not, §4.9). Its cost comes from the goal's final slice of the reserve (§8.1).
- **Built-in judges:**
  - `ProgramJudge(ref)`
  - `Lexicographic([judge_a, judge_b])`: a deterministic key first, then the LLM breaks ties. **[call]**
  - `LLMJudge(rubric, models, aggregate)`, in the `llm` extra. A panel can mix providers, for example one Claude, one GPT, and one Gemini judge, so the judges are less likely to make the same mistake together.
  - `HumanReview`: export and import after the run (§6.6)
- **Judge-private data** **[call]**, for example physika's hidden test instances:
  - It sits only in the judge's or check's config, under `RunConfig` secrets, or at a private path.
  - It is never in `Goal`, never shown to managers, and never in a tool's workspace or env.
  - Checks and judges that use it run in a **separate process after** the tool run.
  - Managers see only the score and what each check exposes. Each check sets `feedback`: `pass_fail` shows only passed or failed; `reasons` also shows `CheckResult.reasons`.
  - Real isolation (containers) comes later.

### 4.5 `Manager` and the action set

```python
class Manager(ABC):
    Config: ClassVar[type[BaseModel]]
    async def run_round(self, ctx: AgentContext) -> RoundReport: ...
```

`AgentContext` is the **only** way a manager can act. Every action checks the limits first and raises `LimitReached` if a limit would be broken.

| Action | What it does |
|---|---|
| `ctx.goal`, `ctx.spec`, `ctx.round`, `ctx.round_spec` | Read-only facts |
| `await ctx.run_tool(inputs, *, parents=(), replicate=None, edits=None) -> Candidate` | Validates the request, commits a budget reservation before dispatch (§7.1, §8.1), applies the repeat rules (§6.2), runs the tool, runs the checks, and records everything. `edits` is refused unless every path matches `Tool.editable()` (§4.12). **[call]** |
| `ctx.my_candidates()` | This agent's results, with their checks and scores |
| `ctx.digest` | The small board digest for this round (§6.3). It is also put in the manager's first prompt. |
| `ctx.search_learnings(...)`, `ctx.read_learning(id)`, `ctx.get_candidate(id)`, `ctx.best(k)` | On-demand board reads from the agent's view (§6.3), capped by `max_board_reads_per_round` |
| `await ctx.post_learning(title, text, evidence, *, tags=(), refines=(), duplicate_reason=None)` | Rate-limited. Evidence is required. Checks for near-duplicates first (§6.3). |
| `ctx.notes` / `await ctx.save_notes(text)` | The agent's own private memory, carried across rounds and capped in size **[call]** |
| `ctx.budget` | What is left for this agent |

- There is no shell, no file access, and no network.
- **Soft errors come back as tool results** **[call]**. The LLM can read them and fix its call. Examples: a bad input, a refused repeat, a duplicate learning, a rate limit, an unknown candidate id, a parent that was not overlaid (§4.12). Only `LimitReached` and `BudgetExceeded` end the round.
- **Actions are safe to call in parallel** **[call]**. An action keeps per-call state only in local variables or in the per-agent `deps`, never on a shared tool object. The conformance kit tests this with `parallel_tool_calls=True`. Agency Swarm has a race bug of exactly this kind.
- A connector may add typed, named actions through `Tool.extra_actions()`. They go through the same executor and the same limits.
- **Managers are stateless across rounds** **[call]**:
  - Each `run_round` starts a fresh LLM conversation.
  - Continuity comes from `ctx.notes`, the agent's own candidates, and the board.
  - So the journal never stores full transcripts, and resume stays simple.

**Built-in managers**
- **`ScriptedManager(requests=[...])`**: no LLM. Used for baselines, tests, and old configs that must keep running as they are.
- **`AgentsSDKManager`** (extra `swarm-arena[llm]`, pinned to `openai-agents[litellm]>=0.24,<0.25`; facts checked against the 0.24.0 source):
  - **Actions:**
    - Each action is a `@function_tool`.
    - Per-agent state is passed with `Runner.run(context=deps)`, which never reaches the LLM.
    - The round ends when the manager calls a `finish_round(summary)` action. It never ends through `output_type`, because mixing structured output with tools breaks on some providers (SDK issue #3709, and LiteLLM's forced JSON tool for Claude).
  - **Turns:** `max_turns=limits.max_turns_per_round`.
  - **Tokens:**
    - A `RunHooks` hook **pre-checks** the reservation in `on_llm_start`, against the agent's committed LLM tranche (§8.1).
    - The actual charge comes from swarm-arena's LiteLLM logger (§4.13), not from the SDK's `Usage`. The SDK drops `cache_hit`, the response cost, and the 5m/1h cache-write split, and it counts a response-cache hit as real spend.
    - When the budget is gone, the hook raises `BudgetExceeded(AgentsException)`.
    - `ModelSettings(max_tokens=..., include_usage=True)` limits each call.
  - **Wall time:** an anyio `fail_after` wraps `Runner.run`.
  - **Model:** always an alias from the run's model catalog (§4.13), run in-process through `LitellmModel`. No hosted tools and no server-side state are used.
  - **Tool timeouts** **[call]**: `run_tool` has no SDK timeout (`timeout=None`). The core clocks govern it (§6.1). Other actions use an SDK timeout with `timeout_behavior="error_as_result"`, so a timeout comes back as a soft error.
  - **Prompt safety:** board content is marked as **untrusted input** from other agents.
  - **Containment:** if the SDK changes, only this adapter changes.

### 4.6 `Planner`

```python
class Planner(ABC):
    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan: ...
```

| Mode | How |
|---|---|
| Manual | `Plan.load("plan.toml")`, or build it in Python. |
| LLM | `LLMPlanner(model)`. It uses the human-only `RunConfig.planner_catalog` (§4.1). The LLM sees each entry's name, description, and `settable` schema. It picks entries and fills **only their settable fields** (for a manager entry, typically `instructions` and the model alias), plus team and agent rationales. **[call]** |
| Together | `CollaborativePlanner(LLMPlanner(...), reviewer: Reviewer)`. `Reviewer` is a protocol: `async def review(self, plan: Plan) -> Approve \| Feedback \| Edited`, that is `Approve`, `Feedback(text)`, or `Edited(plan)`. **[call]** The CLI gives a terminal reviewer that loops until you approve. The catalog rules still apply to an edited plan. A human who needs a component outside the catalog adds it to the catalog, or writes the plan by hand. |
| Autonomous | `Swarm(..., replanner=LLMPlanner(...))`. Between rounds, it gets a `History` (scores, learnings, prune log, budget left) and returns a new revision. |

**Provenance is set by the core** **[call]**:
- The planner LLM's output schema has no `origin`, `revision`, or `parent_revision`. The core sets them from the planner type: `human` (manual), `llm`, `together`, or `autonomous`.
- A plan file that says `origin = "human"` is trusted only when it comes in through the human CLI or Python API path (`swarm-arena run`, `Plan.load`, `Swarm.run`).
- On resume, each revision's `origin` comes from the run journal, never from the `plan.rN.toml` file.

**`validate_plan(plan, config)`** runs on every plan, from any planner. It:
- rejects anything over the limits,
- in a plan whose `origin` is not `human`, rejects every `Ref`, at any depth, that the revision adds or changes and that does not resolve to a `RunConfig.planner_catalog` entry (§4.1). Refs carried over unchanged from a human-origin parent revision (plan-level checks, judges, ranking, and unchanged teams, agents, and parts) keep their human provenance (§4.1), **[call]**
- validates every `Config`,
- requires a rationale for every team and agent,
- rejects a tool `env` name that is not in `RunConfig.secrets.allow` (§11), **[call]**
- applies the catalog check and the `RunConfig.secrets.allow` check to the effective tool config, after `RoundSpec.tool_overrides` and `ConvergeSpec.round` are merged. In a plan whose `origin` is not `human`, an override that the revision adds or changes may set only settable fields. A carried-over human override keeps its human provenance (§4.1), **[call]**
- rejects a revision in which a part has no team (§4.11), or that would push the run over `max_agents_total` (§4.10), **[call]**
- rejects any revision whose main `GoalKey` (namespace, id, inputs hash, files manifest, §4.2) differs from r0's. The main goal is frozen for the whole run. **[call]**
- rejects a revision with `origin` other than human that removes or changes any plan-level check, judge, or ranking that has **ever** been in the plan: those of r0, and every check added by a later revision. So every plan-level check is kept for the whole run. Such a revision may add checks, but only from the catalog, and it may never remove one. It may remove a whole part, with its teams, to re-partition (§4.11). Parts it keeps may not change their checks, judges, or ranking. New parts must leave `checks`, `judges`, and `ranking` as `None`. The join output is still scored against the main goal, so a replanner cannot loosen scoring. **[call]**
- applies the duplicate rule. **[call]** An agent's fingerprint is the tool ref, the tool config, and the manager model. Instructions are not part of it. At most `limits.max_agents_per_fingerprint` agents (default 2) may share one fingerprint. So agents that differ by one word of instructions cannot fill the swarm.

A replanner can add, remove, or change agents, and add, remove, or re-partition parts, using only catalog components. It may keep the human components of earlier revisions unchanged (§4.1). It can never touch the `RunConfig` or the main goal. It can never remove or change a plan-level check, judge, or ranking that has ever been in the plan, whether from r0 or added later. It may only add checks from the catalog.

### 4.7 `Policy` (pruning between rounds)

```python
class Policy(ABC):
    def after_round(self, state: RoundState) -> Decision: ...
```

- **Built-ins:**
  - `KeepTopTeams(k)`
  - `SuccessiveHalving(keep=0.5, min_per_team=1)`: never prunes a team's only passing candidate.
  - `Patience(rounds=2)`
  - `Chain(*policies)`
- **Starved agents are never pruned for lack of results.** A `starved` agent is one whose runs never started because no compute was free. **[call]**
- **Order between rounds:** the deterministic policy runs first, then the replanner.
- **No silent empty result.** If nothing is left, the policy writes a `NO_RESULT` record.
- **Mid-round stops are not policy decisions.** They are fixed core rules (§6.4).

### 4.8 `Executor`

```python
class Executor(ABC):
    async def submit(self, job: Job) -> ToolOutput: ...   # cancelling the await stops the job (with grace + partial collect)
    async def capacity(self) -> Resources | None: ...     # for plan-time warnings
    async def aclose(self) -> None: ...
```

**`LocalExecutor`** (POSIX) uses one internal `ProcessSupervisor`:
- It starts the tool through a launcher (below), with `start_new_session=True`, so the tool gets its own process group.
- On stop, inside a shielded `finally`, it:
  1. sends SIGTERM to the group,
  2. waits `grace_s`,
  3. runs `collect()` for the partial output,
  4. sends SIGKILL to the group.
- stdout and stderr go to files, not pipes.
- It calls the venv's Python directly. It never uses `uv run`, because uv cannot forward SIGKILL.
- It sets rlimits where the OS supports them.
- A resource pool caps how many runs happen at once.
- **Start-up handshake** **[call]**. The tool starts through a tiny launcher, `python -m swarm_arena._launch`, run with the current interpreter. The launcher stays inert, blocked on a pipe from the driver.
  1. The driver commits the tool-run start event: host id, pid, process group id, and process start time (§7.1).
  2. Only then does it send "go" on the pipe. The launcher then execs the tool's command. An exec keeps the pid, the process group, and the process start time, so the recorded values stay true for the tool.
  - If the commit fails, the driver kills the launcher before the tool runs.
  - If the pipe closes before "go", for example because the driver died, the launcher exits and the tool never runs.
  - So every tool that runs has a committed start event, and resume can find every orphaned group.
- Pids can be reused. So resume kills an orphaned group only after it checks that it runs on the same host and that the process start time is the same (§6.5).

**`RayExecutor`** (extra `swarm-arena[ray]`): see §11.

**The design rule:** managers, judges, planners, and the board all run **in the driver process**, because they only wait on the network. Only tool runs, and checks and judges that use `ctx.execute`, go to the executor. `ScriptCheck` and `ScriptJudge` always do, each in its own workspace. The core uses the anyio API **on the asyncio backend only**, because the Agents SDK and Ray awaitables are asyncio-native.

### 4.9 `Board` (the arena store)

```python
class Board(ABC):
    def put_candidate(...); put_score(...); put_learning(...); put_decision(...)
    def frontier(self) -> Frontier                    # the current arena frontier, recorded at round start (§7.2)
    def snapshot(self, goal: GoalKey, frontier: Frontier) -> BoardView    # one goal's round-start view (§7.2)
    def best(self, goal: GoalKey, k: int, *, judge: str, run_id: str | None = None,
             part: str | None = None, fidelity: str | None = None) -> list[Candidate]
    def learnings(self, namespace: str, *, tags=(), goal: GoalKey | None = None, limit: int = 50) -> list[Learning]
    def search_learnings(self, namespace: str, query: str, *, tags=(), goal: GoalKey | None = None, limit: int = 10) -> list[Learning]
    def get_learning(self, id: str) -> Learning
    def get_candidate(self, id: str) -> Candidate
    def lineage(self, candidate_id: str) -> LineageGraph
    def leaderboard(self, goal: GoalKey, *, judge: str) -> list[Row]
```

- A `BoardView` has the same read methods, limited to its snapshot. All `ctx` reads go through it, merged with the agent's own items from this round (the agent's view, §6.3). **[call]**
- A `Frontier` names exactly which arena records a snapshot holds (§7.2): for each run read, a `(run, run_root, head)` mark, with the committed head as read from that run's `owner.json` (§7.1), plus the names of the review and curation objects included. The same frontier gives the same snapshot on any machine, even after a run pointer is deleted. **[call]**
- **One judge identity per ranking** **[call]**. `judge` names the judge (its `Ref.name`, §4.1). `best` and `leaderboard` rank only scores of one judge identity (§4.4): the run's current `rubric_hash` and `panel_hash` for that name. Scores of other identities, for example from a panel that dropped a member (§6.4) or from a past run with another rubric, are never ranked together with them. Each `Row` records the identity it was ranked under. With no current run, for example `swarm-arena leaderboard <namespace>`, the newest identity for that name is used. This covers every reader of these methods: the digest top `k` (§6.3), `ctx.best`, round pages (§6.3.1), and `min_improvement`'s first place.
- The default is `ArenaBoard(arena_root)`. `RunConfig.board` can pick another one. `arena_root` is a local path or `s3://…`. Its storage model is in §7: each run's own journal is its arena record, `arena_root` holds only run pointers and write-once review and curation objects, and readers keep a local SQLite cache. `ArenaBoard.put_*` appends to the run's own journal. So many swarms can run at once with no locks between them and no database server (D18). The only lock is each run's own ownership lease (§7.1).
- **Scope** **[call]**:

| Reads cover the whole namespace, past runs included | Decisions use this run only |
|---|---|
| Manager snapshot reads (`best`, `learnings`), with each item labelled with its `run_id`. | Policies, `min_improvement`, converge, and `solution.json`. |

- A past candidate can be used **as a parent**, with explicit lineage. It is never selected as this run's solution.
- `ConvergeSpec.min_improvement` is optional. **[call]** It applies **only to first place on the leaderboard**: who the arena shows as this run's leader (`swarm-arena leaderboard <run>` and round pages). A new leader must beat the old one by this margin, and ties go to the earliest. It does not change policies or the final solution. The final solution order is always the final judge batch's order (§4.4).
- Gated-out and rejected candidates are **kept**. Einstein Arena deletes them. This design keeps them.

### 4.10 One or many goals per run (D16)

- **What a run is.** A run is one `RunConfig` plus **a list of plans**, one plan per goal. A single-goal run is a list of one.
  - Python: `await Swarm(config).run([plan_a, plan_b, ...])`.
  - CLI: `swarm-arena run run.toml plan_a.toml plan_b.toml …` or `swarm-arena run run.toml plans/`.
- **Goals run independently and at the same time.**
  - Each goal has its own rounds, its own barrier, its own policy, and its own `solution.json`.
  - A slow goal never holds back a fast one.
- **Shared budget.**
  - The run's budget pool is split across goals by `goal_weights`. The default is equal shares.
  - When a goal ends, early or not, its unused budget goes back to the run pool. It is split by `goal_weights` among the goals that are running or queued. **[call]**
  - The reserve (`reserve_fraction`) is held at run level (§8.1).
- **Limits at two levels.**
  - **Per goal:** `max_teams`, `max_agents_per_team`, `max_agents_per_goal`.
  - **Per run:** `max_concurrent_goals` and `max_agents_total`. `max_agents_total` caps the agents working at the same moment, across all goals.
  - Goals wait in a queue. A goal leaves the queue only when a goal slot is free and all its active agents fit under `max_agents_total`. So an agent never waits for an agent slot in the middle of a round. **[call]**
- **Learnings across goals.** Goals in the same namespace share learnings through the arena, at round boundaries, like any other runs (§6.3). So molecule 3 can learn from molecule 1 within the same run.

### 4.11 Parts: split, solve, join (D17)

Some goals are solved in pieces. Two examples:
- DeepRetro2 cuts a huge molecule into fragments.
- DeepRetro continues work on the unsolved leaves of a route.

**How parts work**

- **Declaring parts.** `Plan.parts` lists the parts. Each part has its own sub-goal, and can have its own checks and judges.
- **Assigning teams.** A team with `part="A"` works only on part A's sub-goal.
- **Ranking groups.** Teams are ranked **only within their group**. A group is either one part, or the main goal (`part=None`).
  - So several teams can still compete on one part.
  - Parts never compete with each other.
- **Policies keep every part alive.**
  - `SuccessiveHalving`, `KeepTopTeams` and similar policies work **per group**.
  - A policy may never prune a part's last team. If it tried, it would leave a part with no candidate. `validate_plan` applies the same rule to a replanner (§4.6).
  - If mid-round stops (§6.4) leave a part with no active agent, the part keeps its best earlier candidate for the join, and an event is logged. **[call]**
- **Joining.**
  - The converge team does the join. Its agents get, as `parents`, the top `ConvergeSpec.parents_per_part` candidates of every part.
  - The join tool comes from the connector, for example DeepRetro grafting.
  - Its output is checked and judged against the **main** goal.
- **If the join fails.** The candidate fails its checks. **[call]**
  - In autonomous mode, if the goal has run fewer than `limits.max_rounds` rounds and its pool has budget left, the core calls the replanner once with the failure reasons. It may return a new revision with **different parts**, for example a new partition, within the limits. The new revision may add RoundSpecs up to `max_rounds`. It runs for the rounds that are left, then converges again. This happens at most `ConvergeSpec.max_join_attempts` times.
  - Otherwise the goal writes `NO_RESULT` with the join's failure reasons.
- **Who decides the parts.** Any planner can produce them:
  - a human (for example, chemist-chosen fragments; the plan comes in through the human path, so the core sets `origin="human"`, §4.6),
  - an LLM, or
  - a connector-specific planner (for example, DeepRetro2's automatic partitioning). The core only provides the mechanics.

### 4.12 Bring your own files and script: `ScriptTool` (D19)

A user does not have to write a connector. They can give:
- **a set of files**, for example `ns_solver.phyk`, `helpers.phyk`, and `obs.npz`, and
- **a script that uses them**, for example `run.py`, which runs the `.phyk` files, calls an LLM, or both.

The built-in `ScriptTool` then turns that into an agent's tool. The whole setup is TOML, with no Python.

```toml
[goal]
id = "ns-viscosity"
namespace = "physika"
description = "Recover rho and nu from the observed flow field."
files = ["problem/*.phyk", "problem/obs.npz", "problem/run.py"]   # snapshotted at plan time

[[teams]]
name = "fd-solver"
rationale = "Finite-difference forward model, tune the inverse step."

  [[teams.agents]]
  name = "tuner-1"
  rationale = "Explore learning rate and step size."
  manager = { path = "swarm_arena.llm:AgentsSDKManager", config = { model = "opus" } }   # alias from run.toml [models]

    [teams.agents.tool]
    path = "swarm_arena:ScriptTool"
    [teams.agents.tool.config]
    command  = ["python", "problem/run.py"]    # cwd = the workspace; goal files keep their paths; "python" = config.python or the current interpreter
    python   = "/opt/envs/physika/bin/python"  # the script's own environment (optional)
    editable = ["problem/*.phyk"]              # which files the manager may change or add (Tool.editable())
    params   = { lr = { type = "number", minimum = 1e-5, maximum = 1.0 }, epochs = { type = "integer", maximum = 400 } }
    timeout_s = 1200
    env = ["ANTHROPIC_API_KEY"]                # allowlist: for a script that calls an LLM itself

[[checks]]
path = "swarm_arena:ScriptCheck"
config = { command = ["python", "checks/check.py"], files = ["checks/check.py"] }

[[judges]]
name = "rel-error"
path = "swarm_arena:ScriptJudge"
config = { command = ["python", "judge/score.py"], files = ["judge/score.py", "judge/hidden/*.npz"], direction = "min" }
```

**The workspace contract.** This is what the script sees, and it is documented as a stable format.

```
<workspace>/                    # the script's cwd
  <goal files, same relative paths>   # e.g. problem/run.py, problem/ns_solver.phyk
  <the base parent's overlay, only if the overlay rule allows it; then this request's edits>
  parents/<candidate-id>/…      # every parent, read-only: its data.json and its full overlay
  swarm/request.json            # {"params": {...}, "round": n, "round_overrides": {...}, "replicate": k}
  swarm/result.json             # the SCRIPT writes this (path also in env SWARM_ARENA_RESULT)
```

- **Staging order** **[call]**: goal files, then the base parent's overlay (only if the overlay rule below allows it), then this request's edits. A later layer replaces an earlier one only at a path that the receiving tool's own `editable` globs allow.
- **The overlay rule** **[call]**. The base parent's overlay is laid on the workspace root only when both of these hold:
  1. The parent was made by a tool with the **same tool identity** as the receiving tool: the same ref and the same fixed config. The fixed config is the tool config as the plan writes it, without round overrides and without the fields an LLM may set (§4.1). For `ScriptTool`, it always holds `command`, `python`, `files`, and `editable`. (Tool identity is not `Tool.fingerprint`, which compares outputs.)
  2. Every path in the overlay passes the receiving tool's own `editable` globs.
- **Protected paths are never overlaid**: the tool's script, its interpreter, any check or judge file, and the reserved names (below).
- **Any other parent** is staged read-only under `parents/<id>/` only, and it never replaces a file at the workspace root. This covers every parent after the first, a base parent from another tool, and a base parent with any path that fails the rule. When a base parent is not overlaid, the run still goes ahead, and the `run_tool` result carries a soft error that names the parent and the paths that were not overlaid (§4.5).
- **Path rules** **[call]**. These apply to goal files, edits, and output files alike:
  - Paths are relative only. Absolute paths and `..` are rejected.
  - Symlinks are rejected.
  - The top-level names `swarm/`, `parents/`, and `candidate/` are reserved. A goal file, edit, or output that uses one is an error. So staging can never collide with the core's own files.

- **Script output.** The script writes `swarm/result.json`:
  - `{"data": {...}}` is required.
  - `"usage": {"input_tokens": …, "output_tokens": …, "usd": …}` is optional. A script that calls an LLM should report it, so the budget ledger counts it when there is no gateway (§8.1).
  - `"files": ["out/plot.png", …]` is optional. These files are kept as outputs.
- **Exit code.** Exit code 0 plus a valid `result.json` means `ok`. Anything else, except after a timeout or cancel, means `error`, and the logs are kept.
- **Timeout or cancel** **[call]**. The core sends SIGTERM and waits `grace_s`. The status is `timeout` or `cancelled`. If a valid `result.json` exists then, its data is kept with `partial=true` (§4.3). If not, the data is empty. A script that wants partials writes `result.json` atomically (temp file, then rename) as it goes, or on SIGTERM.
- **Parents.** The **first** parent is the **base**. When the overlay rule allows it, its full overlay is laid on the goal files at the workspace root. So "take the best program and improve it" works with no extra steps for parents made by the same tool, and the changes add up along a chain. Every parent is also available, read-only, under `parents/<id>/`.
- **Manager actions** **[call]**:
  - `ScriptTool` adds only `list_files(*, candidate=None)` and `read_file(path, *, candidate=None)`, through `Tool.extra_actions()`. They are read-only. With no `candidate`, they show the goal files. `candidate` may be any candidate in the agent's view (§6.3), and then they show that candidate's effective files. Each call counts toward `max_board_reads_per_round`. This is how an LLM manager reads the `.phyk` files, and a parent's version before it edits it.
  - Runs use the core `ctx.run_tool(inputs, parents=..., edits={"problem/ns_solver.phyk": "<new content>"})`. When `Tool.editable()` is not empty, the `run_tool` schema shown to the LLM gets an `edits` property. `ScriptTool` writes `inputs` to `request.json` as `params`.
- **Edit rules.**
  - An edit must match an `editable` glob. New files are allowed only if they match too.
  - The size of each edit is capped by `limits.max_edit_bytes` (§8.1).
  - The script itself, checks, and judges are **never** editable. So an agent cannot change how it is scored.
- **Only a human writes a script component** **[call]**. The `command`, `python`, `files`, `env` and secret access, and `editable` fields of `ScriptTool`, `ScriptCheck`, and `ScriptJudge` are never settable by an LLM (§4.1).
- **`ScriptCheck` and `ScriptJudge`** follow the same contract:
  - They run through the executor (`ctx.execute`, §4.4), in a separate workspace that holds **their own** files, for example the hidden test data, plus the candidate:
    ```
    <check-or-judge workspace>/       # cwd
      <its own files, same relative paths>
      candidate/request.json          # the candidate's swarm/request.json, plus "parents": [ids]
      candidate/output.json           # {"status", "partial", "data", "usage", "files": [...]}
      candidate/files/…               # the candidate's effective files (goal files + overlay) and its output files
      swarm/result.json               # the check or judge writes this
    ```
    `candidate/files/` exists even when the candidate produced no files. **[call]** `ScriptJudge` runs once per candidate, each time in its own workspace, so there is always exactly one `candidate/`.
  - The same path rules apply to their own files.
  - They write `{"passed": bool, "reasons": [...]}` (check) or `{"key": [...]} / {"value": x}` (judge) to `swarm/result.json`.
  - Their files are never staged into a tool's workspace (§4.4, judge-private data).
- **Where it runs.** The executor runs `ScriptTool`, so it works the same locally and on Ray. On Ray, workers download the snapshotted files from S3 by sha256.
- **Connectors stay optional.** When a project needs more, for example typed outputs, `Tool.fingerprint`, or native integrity checks, it writes a connector by subclassing `Tool`. `ScriptTool` is itself a `Tool` subclass, so a connector can also subclass `ScriptTool`.

### 4.13 Many LLMs, prompt caching, and response caching (D23, D24)

Facts below were checked against openai-agents 0.24.0 and LiteLLM 1.105.1 source, the low end of the supported range (§11).

**Many LLMs, through LiteLLM**

Every LLM call made by swarm-arena's built-in LLM components in `swarm_arena.llm` (managers, judges, the planner, and the curator) goes through the Agents SDK's in-process **`LitellmModel`**. So any LiteLLM provider works: Anthropic, OpenAI, Gemini, DeepSeek, Fireworks/Kimi, OpenRouter, Bedrock, Vertex, and others. No extra server is needed.

**The model catalog** (in `RunConfig`, human-only, D23):

```toml
[models.opus]
model = "anthropic/claude-opus-5-5"     # LiteLLM "provider/model" name
max_tokens = 8000
roles = ["manager", "judge", "planner"] # what it may be used for
prompt_cache = { ttl = "5m" }           # optional override

[models.gpt]
model = "openai/gpt-6"
roles = ["manager", "judge"]

[models.flash]
model = "gemini/gemini-3-flash"
roles = ["judge", "curator"]
prices = { input = 0.3, output = 2.5 }  # optional USD per 1M tokens; overrides LiteLLM's price map
```

- **Aliases only.** Plans refer to models **by alias only**. The planner can pick only models in the catalog, and only for the roles they allow. It cannot choose an expensive model you did not list.
- **Validation.** `validate_plan` checks each model's capabilities with LiteLLM's helpers:
  - a manager model needs function calling;
  - a judge or planner model needs structured output, or the two-stage fallback (a free-text answer, then a cheap formatting call).
- **Mixed providers** are encouraged in teams and judge panels. Research showed that identical agents make correlated mistakes.
- **Repeatable costs.**
  - Prices come from LiteLLM's price map, pinned with `LITELLM_LOCAL_MODEL_COST_MAP=True`, or from the catalog's `prices`.
  - A model with no known price fails `validate_plan` when `max_usd` is set. Otherwise it would cost 0 and never trip the budget.

**Prompt caching (provider-side cache writes and reads)**

This is **on by default**.

- **Prompt layout.** The built-in LLM components build every prompt so that the large, stable part comes first:
  1. system instructions
  2. tool descriptions
  3. the goal
  4. the goal's reference files, for example physika's language docs
  5. then the parts that change: the round digest, results, and the latest turns.
- **Per provider:**

| Provider | What `swarm_arena.llm` does |
|---|---|
| Claude (anthropic, bedrock, vertex) | Sends `extra_args={"cache_control_injection_points": [{"location": "message", "role": "system"}, {"location": "message", "index": -1}]}`. The SDK removes hand-placed `cache_control`, and LiteLLM re-adds these points. There is a limit of 4 breakpoints. The TTL is `5m` by default, or `1h` per model. |
| OpenAI, DeepSeek, Fireworks/Kimi | Caching is automatic on the stable prefix. No markers are needed. |
| Gemini | Implicit caching. Explicit context caching is a later option. |

- **Counting.** Every `LLMCall` records `cache_read_tokens`, `cache_write_tokens`, and the 5m/1h write split.
- **Budget.** The ledger prices cache tokens correctly, using LiteLLM's cache read and write prices. A cache write costs more than a normal input token, and a cache read costs much less.
- **No silent failure.** A prompt shorter than the provider's minimum is not cached, and the provider gives no error. The board viewer shows cache hit rates per agent, so this cannot go unnoticed.

**Response caching (exact-match, LiteLLM cache)**

This is **optional** (extra `swarm-arena[cache]`). It is configured in `RunConfig.cache`:

```toml
[cache]
backend = "s3"                       # "disk" | "s3" | "redis"
s3_path = "s3://bucket/swarm-arena/llm-cache"
# namespace = "physika"               # optional override; default: each call uses its goal's Goal.namespace
ttl_s = 2592000                       # 30 days

[cache.roles]                         # separate read and write switches per role
judge   = { read = true,  write = true  }   # repeat judge requests (for example after resume) cost nothing; anchors and the final judge never read (see below)
planner = { read = false, write = true  }
manager = { read = false, write = false }   # managers should explore, not replay
curator = { read = true,  write = true  }
```

- **How it is wired.** `swarm_arena.llm` sets `litellm.cache = Cache(type=..., mode="default_off", ...)`. Each call then passes `cache={"use-cache": True, "no-cache": not read, "no-store": not write, "namespace": ...}` through `extra_args`.
- **The cache key** is a hash of the model plus every request parameter: messages, tools, temperature, response_format, and stream. `metadata` is not in the key, so the tracking ids never break cache hits.
- **Judge settings.** Judges use temperature 0 and a dict JSON schema. A pydantic class would be keyed by its name only, so changing the schema could return stale answers.
- **A hit is a hit.** `swarm_arena.llm` reads LiteLLM's `cache_hit` flag:
  - A hit costs **0** in the ledger.
  - It is logged as an `LLMCall` with `cache_hit=true`, pointing to the original call through `cached_from_call_id`.
- **Anchors and the final judge never read the cache** **[call]**. They always send `no-cache: true`, whatever `cache.roles.judge.read` says, so the drift check and the final order are real. Their results are still written to the cache.
- **Streamed and thinking calls are not response-cached.** LiteLLM's replay of those loses data.

**How every call is seen**

The Agents SDK hooks do not see everything. They lose `cache_hit`, the response cost, the 5m/1h cache-write split, and the response id. So `swarm_arena.llm` also registers one LiteLLM `CustomLogger`.
- Every call carries `metadata = {run_id, goal_id, team, agent, round, role, reservation_id}`. This is passed through `ModelSettings.metadata`, which is not part of the cache key. `reservation_id` lets parallel calls with the same other fields settle the right reservation (§8.1). **[call]**
- The logger turns each success or failure into an `LLMCall` (§7.3), with the raw usage, cost, cache fields, and LiteLLM call id, and settles that reservation.
- The SDK hooks are used only for the budget pre-check and for linking a `tool_call_id` to its call.

**Tools that call LLMs themselves.** Through the optional gateway (§7.3), the same caching works for tool-internal calls. The gateway's config turns on `cache`, and prompt-cache markers pass through it. Without a gateway, a tool's own caching is up to the tool.

## 5. One run, step by step

```
plan ─► validate ─► estimate ─► preflight ─► [ round 1 … N ] ─► converge ─► final judge ─► solution + review export
                                                 │
                ┌────────────────────────────────┴───────────────────────────────┐
                │ snapshot board → all active managers run in parallel           │
                │   each: run_tool → checks → (repeat within its budget share)   │
                │         read snapshot, post learnings, save notes              │
                │ barrier → ranking + other judges on new passing candidates     │
                │ policy.after_round → prune with reasons                        │
                │ (autonomous) replanner → new plan revision (validated)         │
                └────────────────────────────────────────────────────────────────┘
```

1. **Plan, then validate** (§4.6).
2. **Estimate.**
   - The worst case is Σ goals Σ rounds Σ agents [`max_tool_runs_per_round` × `Tool.estimate()` + `max_turns_per_round` × the manager model's `max_tokens` × its price], plus the reserve. Replicates count toward `max_tool_runs_per_round`. **[call]**
   - Manual and together modes show it for approval. Autonomous mode logs it.
   - If the estimate is over the limits, or the reserve cannot hold every goal's final slice (§8.1), the run does not start.
   - A warning appears if peak resource demand is more than `Executor.capacity()`.
3. **Preflight.** It runs once for each distinct tool config. With an LLM gateway, it also checks the strict gateway profile (§7.3). It also checks that other readers can reach `run_root` (§7.2).
4. **Rounds.** All active managers run at the same time. A round ends when every manager returns, or when the round timeout fires.
5. **Between rounds:**
   1. The judges score.
   2. The policy prunes.
   3. The replanner changes the plan (autonomous mode only).
   4. The goal closes the round with a journal commit (§7.1). Then other runs can see the round's candidates, scores, learnings, and decisions (§7.2).
   5. A new snapshot is taken. Its frontier is committed in the run journal first (§7.2).
6. **Converge** (`ConvergeSpec`, §4.2).
   - **Select:** take the top `top_n` distinct candidates (by `Tool.fingerprint`), using the ranking judge.
   - **Or a converge team:** an ordinary team whose agents get the survivors as `parents`. Examples: a DeepRetro grafting tool, a physika program merger, or a re-run at full fidelity.
   - Converge-team outputs go through the same checks and judges.
7. **Final judge.** It re-scores the finalists in one batch (§4.4).
8. **Solution.**
   - `solution.json` holds the top `top_n` candidates, their scores, lineage, and plan revisions.
   - For tools that are not deterministic, it reports replicate statistics (mean and [min, max] per config) separately from the best-of-N.
   - It includes the run manifest.
   - It is a view of the committed journal, like the plan files and the round pages. It is written only after the commit that covers it, with a conditional create (§7.1).
   - The review export is written too (§6.6).

## 6. Behaviour details

### 6.1 Clocks [call]

| Clock | When it starts |
|---|---|
| `tool_timeout_s` | When the process starts. Queue time does not count. |
| `round_timeout_s` | When the round starts. Queue time counts. |
| `run_timeout_s` | Counts only the time the driver is running. Time while the run is stopped does not count. |

- A Ray request that no node can meet fails after `schedule_timeout_s`.
- **Effective tool timeout:** `RoundSpec.tool_timeout_s` if set, else the `Command.timeout_s` from the tool's config, else `limits.tool_timeout_s`. It is always capped at `limits.tool_timeout_s`, and the core writes it into `Command.timeout_s`.

### 6.2 Repeats and replicates [call]

- **Repeat key:** `GoalKey` (§4.2) + tool ref + tool config + round overrides + canonical inputs + edits (`path → sha256`) + parent ids. The `GoalKey` is that of the agent's goal, or of its part's sub-goal. So results of another goal or part are never linked. For tools that are not deterministic, the replicate index is added.
- **Which earlier results count.** Only one with status `ok` or `error` and `partial=false`. After `timeout`, `oom`, `provider_failed`, `cancelled`, `budget_exceeded`, `lost`, or a partial result, the same request runs again as a new run and is never linked.
- **Deterministic tools:**
  - The same agent repeating a request is refused, and the refusal counts toward the watchdog.
  - On resume, a restarted agent's identical request returns its own existing candidate, with no refusal and no watchdog count.
  - Another agent's identical request gets the result that **finished in an earlier round of this run** (from the snapshot). It is linked, not copied, and it does not use the agent's run count.
  - A match from a past run is only a hint: the manager gets its id and score, and the request still runs (§4.9).
  - Identical requests in the same round both run. So no agent sees another agent's result in the middle of a round.
- **Non-deterministic tools** (`is_deterministic()` returns False):
  - Repeats are allowed up to `AgentSpec.replicates`.
  - Each repeat gets the next replicate index.

### 6.3 Shared learnings (v1): pull, not push (D22)

The board can grow to thousands of learnings over many runs. So agents never get "the board" in their prompt. They get a **small digest**, and they **search** for more only when they need it.

**What the digest is.**
- It is built at the start of each round, from the round-start snapshot.
- It covers the whole namespace: past runs and other goals in the same family. **[call]**
- It has a **hard token cap**, `limits.digest_tokens`, default 1,500. It holds:
  1. This goal's leaderboard top `k`, one line each: candidate id, team/agent, score, and what it was built on.
  2. The **titles only** (with ids) of the pinned learnings and of the top learnings by relevance. Relevance means: this goal first, then the namespace, then verified, then matching tags, then newest.
  3. Counts, for example "412 more learnings, 37 from this round. Use `search_learnings`."
- The digest size does **not** grow with the board.

**The agent's view** **[call]** is the goal's round-start snapshot (§7.2), fixed by its recorded frontier, plus this agent's own items from this round. Reads, `parents`, `evidence`, `read_file`, and the duplicate check all use this view. Other agents' items from this round stay hidden until the barrier. The curator merges same-round duplicates at round end.

**Read actions** (these count toward the agent's turns and budget):

| Action | Returns |
|---|---|
| `search_learnings(query, *, tags=(), scope="goal" or "namespace", limit=10)` | Titles, ids, authors, and markers. Ranked by SQLite FTS5 full-text match (bm25), then by the relevance order above. |
| `read_learning(id)` | Full text, plus the evidence with each candidate's score or error. |
| `get_candidate(id)` | A summary: request, status, checks, scores, parents, and changed file names. |
| `best(k, *, part=None)` | Leaderboard rows. |

- **Limits.** `max_board_reads_per_round` caps how often an agent can search and read, so agents cannot dump the board into their context.

**Keeping the board small.**
- **Duplicate check.** `post_learning` searches the board for close matches first. When it finds them, the agent is told so and must either cite the old learning (`refines=[id]`) or post anyway with `duplicate_reason`.
- **Curator (optional, `RunConfig.curator`).** At round end, a `Curator` merges groups of near-duplicate learnings into one **summary learning**, with `supersedes=[ids]` and the union of their evidence. Old learnings stay searchable but rank lower.
  - Interface **[call]**:
    ```python
    class Curator(ABC):
        Config: ClassVar[type[BaseModel]]
        async def curate(self, namespace: str, new: Sequence[Learning], view: BoardView) -> list[Learning]: ...
    ```
    `new` is the goal's learnings from this round. It returns the summary learnings, and the core stores them.
  - `LLMCurator` is in the `llm` extra and is paid from the reserve.
  - A human can undo a merge with `learnings hide`.
- **Hidden** learnings are never shown to agents.

**Rules for posting.**
- Each learning needs evidence and is rate-limited by `max_learnings_per_agent_round`.
- Content from the board is shown inside "from other agents, unverified" markers.

**Human control.** **[call]**
- `swarm-arena learnings pin|hide <id>` stops a wrong learning from spreading.
- `swarm-arena learnings add --title … --text …` lets a human, for example a chemist, post a learning.
- Each of these commands writes one new object to `curation/`, with a unique name, and never changes an old one (§7.2). Readers see it from their next round start. Every reader applies these objects in one fixed order, so a later pin or hide of the same learning wins on every machine (§7.2).

### 6.3.1 A board that humans can read (D20, D22)

The board is for people too. A single growing Markdown file would not scale, so humans get the same "small summary plus search" design as agents.

- **Learnings are readable by design.**
  - Each learning has a one-sentence `title` (at most 120 characters) and a short Markdown `text` (at most 1,500 characters).
  - Managers are told: "write for a colleague: what you tried, what happened, and the number."
  - `post_learning` rejects a learning that is over the limits or has an empty title.
- **One small file per round.** `goals/<goal-id>/rounds/round-NN.md` is written once at the end of round NN and **never grows afterwards**. It is a view of the committed journal, written only after the round-close commit succeeds, with a conditional create (§7.1). Its size is bounded. It holds:
  1. The leaderboard (top 10).
  2. The learnings that are new **in that round**.
  3. What was pruned, and why.
  4. Links (ids) to everything else.

  It reads like a lab notebook page: one page per round.
- **Full board, searchable.** `swarm-arena board serve <run-or-namespace>` starts a small **local web viewer**. It uses the stdlib HTTP server on the local SQLite cache, so it adds no new dependencies. It has:
  - search (FTS5),
  - filters by goal, part, team, agent, round, tag, and verified / human / pinned / hidden,
  - pages, so it stays fast for any board size,
  - a page per learning with its evidence, and a page per candidate with its lineage and its LLM calls (§7.3).

  It is read-only, except for pin, hide, and add, which write new curation objects, as the CLI does (§7.2).
- **Terminal.** `swarm-arena board search "<query>" [filters]` and `swarm-arena board show <id>` give the same data in the terminal.
- **One writer per file still holds.** Each run writes only its own round pages and its own journal. The viewer and the CLI read every run's journal, found through the pointers in `runs/`, plus the review and curation objects (§7.2).

Example round page (shortened):

```markdown
# physika / ns-viscosity · run 2026-10-18T13-00-00Z_ab12 · round 2 of 3
Leaderboard (rel-error, lower is better)
| # | Candidate | Team / agent        | Score  | Built on  |
|---|-----------|---------------------|--------|-----------|
| 1 | c-e1-0193 | fd-solver / tuner-1 | 0.0041 | c-e1-0112 |
| 2 | c-e1-0171 | spectral / author-b | 0.0090 | —         |

New learnings (3)
- L-e1-0088 **Fitting log(nu) instead of nu converges about 3× faster.** fd-solver/tuner-2 · verified
  Same lr and epochs. Error 0.005 at epoch 120 instead of 380. Evidence: c-e1-0190 (0.0052) · c-e1-0166 (0.0161)
- …

Pruned: team `surrogate` (SuccessiveHalving: best 0.31, round median 0.04).
Full board: `swarm-arena board serve <run>` · 412 learnings in namespace `physika`.
```

### 6.4 Mid-round stops: fixed core rules only [call]

Only these events stop work in the middle of a round:

| Event | What stops |
|---|---|
| `stop` command | Everything (§6.5) |
| `run_timeout_s` reached | Everything, as for `stop --now` (§6.5) |
| Run-level budget used up | Everything. New actions are blocked, and in-flight runs drain until their grace period ends. Then judge, converge, and final judge run from the reserve, as for `stop`. |
| Goal pool used up | That goal. In-flight runs drain, then the goal goes to converge. |
| `round_timeout_s` reached | That goal's round. Its managers and in-flight runs are cancelled, with grace and partial collect. A late output belongs to the round that started it. |
| `provider_failed` (credits used up, or provider unavailable) | Every agent whose providers include that one. An agent's providers are its manager model's provider plus `Tool.providers()`. |
| Watchdog | That agent only (§8.2) |

All other pruning is done by `Policy`, between rounds.

**When a provider fails outside an agent:**
- **Ranking judge:** judging pauses. Candidates stay unjudged, and the policy does not prune on missing scores. If the provider is still failed at converge, the goal writes `NO_RESULT` with reason `provider_failed`.
- **Judge panel:** the panel drops the failed member and records this. The smaller panel has a new `panel_hash`, so it is a new judge identity, and its scores are not compared with the full panel's (§4.4). **[call]**
- **Planner or curator:** that step is skipped, with an event.
- Agents stopped for `provider_failed` become active again at the next round start, if a probe call succeeds.

### 6.5 Stop, resume, failure [call]

**Stop modes**

| Command | In-flight runs | Then |
|---|---|---|
| `swarm-arena stop <run>` | No new agent runs start. In-flight runs finish. | Judge, converge, and final judge, paid from the reserve. A converge team still runs. |
| `swarm-arena stop --now <run>` | Cancelled, with grace and partial collect. | Judge the unjudged candidates, then select-only converge (no converge team). For a goal with parts, `solution.json` lists the best candidate of each part, marked `unjoined`. |

- In both modes, goals still in the queue do not start. Each one writes `NO_RESULT` with reason `stopped`.

**Resume** (`swarm-arena resume <run> [--config run.toml] [--force]`)
- **Ownership first.** It takes over the run lease only if the lease has expired, that is `expires_at` has passed by its own clock (§7.1), or with `--force`. `--force` is a human decision, and it is logged. **[call]** In both cases the takeover fences the old owner: its next conditional write fails, so it can neither commit nor dispatch new work. Until its next conditional write fails, it can only continue work it committed before the takeover. Then it cancels that work and exits. It can commit none of the results, so those runs become `lost`. Their reservations are already charged in full (§7.1, §8.1). A second `resume` at the same time fails cleanly on the conditional write.
- The takeover is **one** conditional write to `control/owner.json`. It bumps the epoch and keeps the committed head (§7.1). Then resume replays the one run journal, up to that head.
- **Budget is conservative** **[call]**. Every reservation with no committed settle event counts as **fully spent**. Settlement is idempotent by reservation id. A tool run with a gateway key stays fully charged for the whole run, after a resume too (§7.3, §8.1).
- Tool runs whose completion event was committed are never repeated. A run whose completion event was not committed is `lost`: its reservation stays charged, and running it again needs new budget.
- Counters for the current round are kept.
- The managers of the unfinished round restart, and each one is told which runs it already made.
- An unfinished round resumes against its **recorded snapshot frontier** (§7.2), plus each agent's own items recovered from the committed run journal.
- Orphaned local process groups from the journal are killed, but only after resume checks that it runs on the recorded host and that the recorded process start time matches (§4.8). Pids can be reused, so a pid alone is never enough. If the check fails or cannot be done, nothing is killed. **[call]**
- In-flight Ray runs from a dead driver become `lost`, as above. Resume finds them by the job id in their committed dispatch-intent event, which is also the Ray task name (§11). **[call]**
- A stopped run, or a run that hit a limit, can resume with a **new `RunConfig` from a human** that raises the limits. This is journaled as `limits_changed`.

**Failures**
- A tool failure is a candidate with an error status. It never crashes the swarm.
- Run directories are never overwritten. Journal segments and view files use conditional writes, and a view file holds only committed content (§7.1).

### 6.6 Human review (after the run)

- **Export:** `swarm-arena review export <run> [--goal <goal-id>] --top 20` writes, for the top 20 of each goal:
  - `review.csv` (id, team, agent, scores, verdict, notes),
  - one JSON file per candidate,
  - a static `index.html`.
- **Import:** `swarm-arena review import <run> review.csv --reviewer alice` stores `Score(judge="human:alice")` in the run and in the arena. The arena copy is one new object in `reviews/`, with a unique name, never changed (§7.2). For the run copy, it takes the run lease (§7.1). If the run still has a live owner, it fails and writes nothing. Run it again after the run ends. **[call]**
- **Re-rank:** `swarm-arena leaderboard <run-or-namespace> [--goal <goal-id>] --by human:alice` re-ranks.
- Chemist verdicts feed later runs: they appear as scores and verified signals in the namespace.

## 7. Storage and observability

All three roots (`run_root`, `arena_root`, `artifact_root`) can be a local path or `s3://…`. The core reaches storage through `fsspec`. S3 support comes from the extra `swarm-arena[s3]` (`s3fs`), and GCS can be added the same way. On a laptop, everything is local folders. On the cluster, everything is on S3 (D18).

### 7.1 Run folders

```
<run_root>/<run-id>/                  # run-id = 2026-10-18T18-30-00Z_<short-hash>
  run.toml                            # RunConfig (secret names only, never values)
  manifest.json                       # versions, tool commits, hashes, host profile
  control/owner.json                  # the run lease and its SINGLE commit point: {epoch, owner_id, host, expires_at, head}
  control/stop | control/stop-now     # written by `swarm-arena stop` from anywhere
  journal/e<epoch>-<last-seq>.jsonl … # THE run journal and the run's arena record: run-level and goal events (each with goal_id),
                                      #   write-once segments; only the chain reached back from owner.json's head counts
  goals/<goal-id>/
    plan.r0.toml, plan.r1.toml …      # every plan revision (a view of the committed journal, below)
    logs/<candidate-id>.{out,err}
    rounds/round-01.md, round-02.md …  # one human-readable page per round, never grows afterwards (§6.3.1; a view)
    solution.json                     # a view of the committed journal (below)
<artifact_root>/<sha256>               # content-addressed files, shared by all runs
```

- **One journal per run** **[call]**. Goals do not have their own journals. Each event carries its `goal_id`, and per-goal views are derived. The journal's `seq` is the single checkpoint for both run-level and goal-level ledger state, so there is no consistency problem between journals.
- **Writing the journal.**
  - The driver appends events to a local file.
  - It commits the new events (below):
    - at every round boundary,
    - every `RunConfig.sync_interval_s` (default 60 s),
    - at every write-through event (below), and
    - when the run ends.
  - The sync interval governs **ordinary events** only.
  - **A commit has two steps** **[call]**:
    1. The driver uploads the new events to `run_root` (S3 on the cluster) as one immutable **segment** object, `journal/e<epoch>-<last-seq>.jsonl`. It uses a conditional create, so the object is never replaced. The segment holds **every** event after the current committed head. Its first line is its **link**: the name of that head's segment, or none for the run's first segment.
    2. It advances `head` in `control/owner.json` to the new segment, with a conditional replace (`If-Match` on the ETag it last wrote or read).
    - The segment is **committed** only when step 2 succeeds.
    - **Membership.** A segment is part of the journal only if it is on the **chain** that ends at the committed head: the segments reached by following links back from `head.segment`. Comparing names, epochs, or seqs never decides membership. For example, after a takeover at head `e1-100`, a fenced owner's segment `e1-105` is not on the chain of a later head `e2-150`, so it is not in the journal.
  - **One writer task, one commit at a time** **[call]**. One writer task in the owner runs the **whole** two-step commit. It also makes every other conditional write to `owner.json`, such as lease renewals. It runs one commit at a time: a commit starts only after the one before it has succeeded or failed. Events that arrive during a commit wait for the next one. Each head advance also renews the lease, so the task always holds the ETag of the last write.
    - **The link rule.** A new segment holds every event after the current committed head and links to that head. The head may advance only to a segment whose link is the head that the advance replaces. So no commit can skip a committed event, and every earlier head stays on every later chain.
    - **Unknown results.** If the result of a write is unknown, for example after a timeout, the task reads `owner.json` again. It builds the next segment only after this re-read shows which head is current. If the head did not advance, the next segment holds those events again, from the current head. If it did advance, the next segment starts after it. If the re-read shows a new epoch, this owner has lost the run (below). A segment that the head never named is garbage, and no reader reaches it.
    - A retry can need a segment name that an earlier, unknown attempt already used. That object holds the same events and the same link, because the head did not move. So the task checks its sha256 and uses it in step 2.
  - **Durable means committed** **[call]**. A write-through event is durable only after the head advance that covers it succeeds. Work that depends on it starts only after that. For example, a tool run is dispatched only after the commit that covers its `reserve` event, and an LLM tranche is spent only after the commit that covers its reservation (§8.1). An ordinary event, for example a plan revision with its `origin` (§4.6), is durable when any later commit is.
  - **Write-through events** **[call]**: budget reservations and settlements, tool-run starts (host id, pid, process group id, and process start time, §4.8; on Ray, a dispatch-intent event with a pre-generated job id, §11), tool-run completions, snapshot frontiers (§7.2), and round closes at each goal's barrier (§7.2). Write-through events made within about 200 ms share one commit (group commit), to keep latency low. Group commit only puts more events in one commit. Commits still run one at a time (above). A tool-run start is committed before the tool runs (§4.8, §11), so resume can find and check every started run (§6.5, §7.3).
  - S3 objects are written whole and never edited, so each segment is a new object. Each segment name holds the writer's epoch, so two owners can never write the same key.
- **Files made from the journal** **[call]**. `plan.rN.toml`, `rounds/round-NN.md`, and `solution.json` are **views** made from the committed journal. They are never a source of truth.
  - The owner writes one only after the commit that covers its content succeeds. It uses a conditional create.
  - A view is a pure function of the committed events it covers, and of the frontiers those events record (§7.2). Each view records its **source head**, `(epoch, seq)` of the commit it was made from. **[call]**
  - Two owners that write a view from the same head write the same bytes, and the second one finds a match by sha256 and goes on.
  - A view can change after a resume, for example `solution.json` after more rounds. A view replaces an existing one only if its source head is **later on the committed chain**. An older snapshot never replaces a newer one. The replace is conditional on the existing object's ETag. **[call]**
  - So a fenced old owner cannot write a view of state it never committed: the commit that would cover that state fails, so it never writes that view.
- **The journal is the source of truth.**
  - If the driver dies, `resume` reads the committed journal from S3.
  - Ordinary events after the last commit are lost. Committed events are never lost, and the driver acts on a write-through event only after it is committed.
  - A tool run whose completion event was not committed becomes `lost`. Its reservation stays charged, and running it again needs new budget (§6.5, §8.1).
- **Control from a laptop.**
  - `swarm-arena stop` writes `control/stop` (or `control/stop-now`). The driver checks for it every few seconds.
  - `status` reads `control/owner.json`, then the committed journal up to its head (below).
  - So every command works from a laptop against a run on the cluster, with no open connection to the driver.
- **Atomic writes.** Local writes use a temp file, then rename. Every file has a sha256.

**Run ownership: one commit point** **[call]**
- A run has **exactly one live owner**: the driver, or a one-shot command that writes run state (below). Ownership is a lease object, `control/owner.json`, holding `{epoch, owner_id, host, expires_at, head}`.
- **`owner.json` is the run's single commit point and its lease.** `head` is `{epoch, seq, segment}`: the name of the newest committed segment, the epoch that wrote it, and its last seq. It is empty before the first commit. No other file says which events count.
- **Acquire** with an atomic conditional create: on S3, a conditional PUT with `If-None-Match`; on a local filesystem, an `O_EXCL` create.
- **Conditional replace** on the current version: on S3, `If-Match` with the current ETag; on a local filesystem, a lock file plus an atomic rename, with a version number in the file. Head advances, renewals, takeovers, and releases all use it.
- The exact S3 API support (boto3 and s3fs conditional writes) is checked at implementation. If the storage cannot do conditional writes, swarm-arena refuses to run on it.
- **Renewal.** The driver renews every `lease_ttl_s / 3` (`RunConfig.lease_ttl_s`, default 60), unless a head advance renewed it sooner. If any conditional write fails because `owner.json` has changed (an ETag mismatch), the owner has lost the run. It stops dispatching at once, cancels its in-flight work, and exits. It commits nothing more. If it cannot renew before its own `expires_at`, for example because storage cannot be reached, it does the same, because a new owner may take over. That rule is for liveness only.
- **Clocks and safety** **[call]**. Correctness does not depend on clocks or on lease timing. Clocks decide only when a lease counts as expired, that is, when a takeover may happen. If clock skew lets a takeover happen too early, the old owner's next conditional write fails, and it stops. Nothing it did not commit counts.
- **A takeover is one write.**
  1. The new owner reads `owner.json` and keeps its version (the ETag).
  2. One conditional replace (`If-Match`) sets `epoch + 1` and the new `owner_id`, `host`, and `expires_at`. It keeps `head` unchanged.
  - This one atomic object write is the whole takeover. So there is no half-done state. A takeover that fails before the write changes nothing. If the old owner commits between steps 1 and 2, step 2 fails, and the new owner starts again at step 1. So the kept head covers every commit the old owner made.
  - After the takeover, the old owner's next conditional write fails (an ETag mismatch). So it can neither commit a segment nor start new work that needs a commit. It must stop. Its uncommitted segment objects are garbage, and every reader ignores them.
  - The new epoch's `seq` starts at `head.seq + 1`, and its first segment names `head.segment` as the one before it. So the committed journal is one unbroken run of `seq`. The new owner then logs the takeover (and `--force`, if used) as an ordinary event of its epoch.
- **What readers read** **[call]**. Every reader (`resume`, `status`, `reindex`, and `ArenaBoard` when it builds a new frontier, §7.2) reads the run's `owner.json` once. This is the **authority read**. Then it reads only the chain that ends at the head that this read names. It starts at `head.segment` and follows each segment's link to the one before it, until it reaches a segment it already has, or the first one. It never finds segments by listing, and it never compares names or seqs to decide membership. So two readers that read the same head read exactly the same events: every committed event with `seq` up to `head.seq`.
  - **Chain check.** The reader checks that `head.seq` is the last seq of `head.segment`, and that each segment's first seq is the previous segment's last seq + 1. If not, it rejects the chain with an error and reads none of it. The link rule never makes such a gap, so it can only come from a bug or a damaged object.
- So a superseded owner can never publish accepted state, with or without `--force`. Readers read only the committed chain, and the view files hold only committed content (above). What an owner did not commit is never read.
- **`--force`** **[call]** only skips the wait for `expires_at`. The takeover is the same one write, and it fences the old owner in the same way. After it, the old owner can spend only through work it committed before the takeover: tool runs it had dispatched and LLM tranches it had reserved. Those reservations are in the kept head, so they are already charged in full (§8.1).
- **Resume** takes over only if the lease has expired (`expires_at` has passed, by its own clock), or with `--force` (§6.5). A second resume at the same time fails cleanly on the conditional write.
- **Other commands that write run state** **[call]**. `swarm-arena reconcile` and the run copy of `review import` (§6.6) take the run lease like `resume`. If a live owner holds the lease, the command fails before it writes anything. For `reconcile`, that owner's own reconcile at run end covers the work (§7.3). For `review import`, run it again after the run ends. Otherwise the command takes over with a new epoch (the one write above), commits its events, and then releases the lease.
- **Release** **[call]** is a conditional replace that keeps the epoch and the head, and sets `expires_at` in the past. The owner releases only after its last commit. So the next owner does not have to wait. It takes over with `epoch + 1` and the same head, and epochs are never reused. The driver releases the lease the same way when the run ends.
- The lease is per run only. Independent runs use different run folders and stay lock-free with respect to each other.

### 7.2 The arena: many writers, no locks (D18)

Two drivers cannot share one SQLite file, and SQLite is not safe on NFS. So the arena has **no file that two writers change**. **The run journal is the arena record** **[call]**: each run's candidates, scores, learnings, decisions, and lineage are events in its own journal (§7.1), and other runs read that journal directly. The arena root holds only small write-once objects:

```
<arena_root>/<namespace>/
  runs/<run-id>.json                                 # pointer: where the run's run_root is; written once at run start
  reviews/<reviewer>/<timestamp>-<random-id>.jsonl   # written once by `review import`, never changed
  curation/<timestamp>-<random-id>.jsonl             # written once by `learnings pin|hide|add` or the board viewer, never changed
```

- **Write** **[call]**.
  - At run start, the driver writes `runs/<run-id>.json` in each namespace of its goals, with a conditional create (§7.1). It holds `{schema_version, run_id, run_root}`. It is never changed.
  - Results go only to the run's own journal. `ArenaBoard.put_*` appends journal events. A run writes nothing else to `arena_root` except its pointers.
  - Human commands write one new object each, with a unique name (a timestamp plus a random id). They never change an object.
  - So no two writers ever write the same key, and no locks and no database server are needed.
  - **Round barrier.** Each goal closes its round with a journal commit (§5, §7.1). So a round's items are visible to other runs by the end of that goal's barrier at the latest. Some are visible sooner, because every commit covers every earlier event.
  - **Reachable run roots.** Other machines can read a run's journal only if they can reach its `run_root`. So preflight refuses a local `run_root` with an `s3://` `arena_root`.
- **Read.**
  - `ArenaBoard` keeps an incremental **local SQLite cache** of the namespace, keyed by `(run, epoch, seq)`.
  - At each round start, it lists the pointers in `runs/`. For each run, it reads `control/owner.json` once, the authority read (§7.1). Then it downloads only the committed segments it does not have yet: it walks back from that head to the newest segment it already has (§7.1). It never lists journal segments.
  - It keeps only the events of goals in this namespace. A goal's namespace is in its plan, which is in the journal (§4.2).
  - It also lists the new `reviews/` and `curation/` objects.
  - **One order for human objects** **[call]**. Curation and review objects are applied in one fixed order: by the timestamp in their name, then by the object name. When a pin and a hide of the same learning conflict, the later one in this order wins. Every machine uses the same order, so the same frontier gives the same result everywhere.
  - A `review import` writes its scores to both the run journal and `reviews/` (§6.6). Arena reads take human scores from `reviews/` only, so each one counts once.
  - **Per-goal snapshots by frontier** **[call]**. Goals in one run share one cache but have their own rounds. At each goal-round start, the core commits the snapshot **frontier** in the run journal:
    - for each run read (this run included): `(run, run_root, head)`, with the head as read from that run's `owner.json` at that moment (the authority read, §7.1). `run_root` is copied from the run's pointer, so a rebuild never needs the pointer again. **[call]**
    - the exact names of the `reviews/` and `curation/` objects included.
  - A head names one exact chain of committed segments (§7.1), so these marks name exactly which events are in. The goal's snapshot (`Board.snapshot`) is **exactly** the records inside that frontier, whatever order the local cache got them in. All of that goal's reads filter on it. There is no `ingest_seq` or `snapshot_seq`.
  - This run's own items are in the arena as soon as their segment is committed, and by the writing goal's round barrier at the latest. So another goal of the same run sees them from its next round start after that barrier, at the latest.
  - Many swarms on many machines therefore see each other's learnings with a delay of at most one round.
- **Rebuild.** `swarm-arena reindex` rebuilds the cache from the run journals, each read up to the head of one authority read (§7.1), and from the review and curation objects. `reindex` and `resume` rebuild the same snapshot membership from a recorded frontier, on any machine. They use the heads as recorded, not the runs' current heads: they walk each run's chain back from its recorded head (§7.1). They find each run's journal through the `run_root` in its marks, not through the run's pointer. **[call]**
- **Rules.** Journal segments, run pointers, and review and curation objects are written once, never changed, and carry `schema_version`. Deleting a run's pointer removes that run from later snapshots only. Recorded frontiers that name the run still rebuild the same, through their recorded `run_root`, while the run's journal exists.
- **Secrets** are never written anywhere by the core: not in journals, arena objects, logs, or LLM payloads.

### 7.3 Every LLM call is logged and linked (D21)

Goal: for any result, you can see **every LLM call behind it**: the planner call that created the agent, the manager call that asked for the run, the calls made inside the tool, and the judge calls that scored it.

**1. Calls made by swarm-arena's built-in LLM components** (`swarm_arena.llm`: managers, judges, the planner, the curator). These are always logged.
- Every call becomes an `LLMCall` event in the journal: model, provider, usage, cost, latency, status, and the full request and response.
- The full payloads are stored once by sha256 in `artifact_root`. Repeated system prompts are therefore stored only once.
- `RunConfig.tracing.store_payloads` (default `true`) can turn payload storage off.
- **How calls are captured.** A LiteLLM `CustomLogger` records each call with its raw usage, cost, cache fields, and call id (§4.13). The Agents SDK hooks add the budget pre-check and the `tool_call_id` link. The ledger is charged from the same records, so logging and budget cannot disagree.
- **How a run is linked to its call.** When a manager calls `run_tool`, `swarm_arena.llm` matches the SDK's `tool_call_id` to the LLM response that contains it. That call's id becomes the candidate's `requested_by_call`.
- In the same way:
  - Each learning records `posted_by_call`.
  - Each LLM score records `llm_call_ids`.
  - Each plan revision records its planner calls.

**2. Calls made inside tools** (for example DeepRetro's own LiteLLM calls, or a `ScriptTool` script that calls an LLM). There are three levels, from strongest to weakest:

- **a. LLM gateway (optional; recommended on the cluster).** A LiteLLM Proxy, set as `RunConfig.llm_gateway`. It needs Postgres and a master key, and it is run by the owner, not by swarm-arena. Facts were checked against LiteLLM 1.105.1 source (§4.13, §11).
  1. **Before each tool run that may call an LLM**, the core creates a **virtual key** with `/key/generate`. A tool run gets a key only if its tool declares `Tool.providers()` (not empty) or allowlists a provider key that the gateway replaces (step 2). Other tool runs get no key and no gateway variables. They settle from reported usage, as without a gateway (§8.1). **[call]** The key has:
     - `key_alias = "cand-<id>"`, a `duration`, and `max_budget` = the run's budget reservation;
     - `metadata = {candidate_id, run_id, goal_id, team, agent, tags, spend_logs_metadata: {candidate_id, …}}`;
     - `default_estimated_output_tokens`.
  2. **The core points the tool at the gateway** (for a tool run with a key). It sets `OPENAI_BASE_URL` (`…/v1`), `ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` to the gateway and the virtual key. LiteLLM-based tools follow these variables too. **The tool never gets the real provider keys.** The core drops provider key names from the tool's `env` allowlist and sets the gateway values instead. `validate_plan` warns when a tool allowlists a provider key that the gateway does not replace, for example `GEMINI_API_KEY`. **[call]**
  3. **After the tool run**, spend is **read back** for reports and tracing **[call]**:
     - The core blocks the key, so no new calls can use it.
     - It waits up to `settle_timeout_s` (§8.1) for a first read, because the gateway writes its logs in batches. It reads `/spend/logs/v2` for that key and turns each row into an `LLMCall` with `role="tool"` and the candidate id. Later reads at round end, at run end, and on resume add late rows idempotently (by LiteLLM call id).
     - **No mid-run release** **[call]**. These rows are for reports and tracing only. During the run, they never lower the tool run's reservation. The run ledger keeps the **full reservation** charged, because the core cannot prove that the spend logs are complete (§8.1). Reading the rows twice, or matching the key's request count, is only a consistency check, never proof.
     - **Final reconcile, after the run** **[call]**. After the run has ended, so no more work can use the money, `swarm-arena reconcile <run>` records each key's **observed spend** for reports. Unless the adapter has terminal accounting (below), the report labels it `observed (completeness unverified)`, because a log batch can still be late. Reconcile keeps the key ids and the candidate ids, so a later `reconcile` can update the numbers. The driver also runs it at run end, best-effort. It changes reports, never the budget of a live run, and a resumed run still counts the full reservations. It records a key's observed spend only when **all** of these hold:
       1. The tool is confirmed dead (§4.8, §11). With `LocalExecutor`, its process group is confirmed dead (§4.8). On Ray, condition 1 holds when the task returned after `p.wait()` reaped the tool and the `max_calls=1` worker has exited, or when Ray reports the worker dead after `ray.cancel(force=True)` (§11). A run that is lost (its driver died) meets condition 1 only after resume or reconcile makes that check from the run's committed start event (§7.1): the recorded process group on its host (§6.5), or the Ray task named by its dispatch-intent event (§11). **[call]**
       2. The gateway API confirms that the key is blocked or deleted. The time of that confirmation is the key's **closure time** T.
       3. At least `llm_gateway.request_timeout_s` has passed since T. So every request the gateway accepted before T has finished or timed out. The final read is made at least one log batch-write interval after that. Preflight reads that interval from the gateway's config.
     - If any condition cannot be shown, the report keeps the full reservation as that run's spend, marked `estimated`, until a later `reconcile` succeeds. For example, condition 3 needs the gateway's request-duration limit, which preflight verifies as part of the strict profile (step 5).
     - `reconcile` takes the run lease, and it fails while a live owner holds it (§7.1). The key is deleted once its observed spend is recorded. Its id stays in the journal for later reconciles.
     - **Extension point: terminal accounting** **[call]**. The gateway client is an adapter with a `terminal_accounting` flag. An adapter may set it to true only if its gateway gives authoritative evidence that a closed key has no outstanding requests **and** that the key's final charges are durably recorded, for example a terminal total or a persistence watermark. Only then may the core release unused reservation during a run, once conditions 1 to 3 hold. The LiteLLM adapter sets it to false in v1. In v1 the adapter interface is private (§10).
  4. **Langfuse logging.** The gateway logs to Langfuse with its `langfuse` callback. Its key tags and metadata carry the candidate id.
  5. **Hard caps, only with the strict gateway profile** **[call]**. The gateway can give **hard USD caps inside black-box tools**, but only when it runs with the **strict profile**:
     - budget reservations are on (not disabled);
     - enforcement fails closed when spend counters are stale or unavailable;
     - only priced models, on supported routes;
     - max output tokens are bounded per key (`default_estimated_output_tokens` and `max_tokens`);
     - the gateway enforces a maximum request duration of at most `llm_gateway.request_timeout_s` **[call]**;
     - virtual keys are required.

     **Preflight** queries the gateway (health, plus its config and model-info endpoints) and checks the profile:
     - If the profile is verified, the per-tool-run cap (each key's `max_budget`) is **hard**.
     - If not, and `RunConfig.llm_gateway.require_hard_cap` is true (the default when `max_usd` is set), the run fails preflight.
     - If not, and `require_hard_cap` is false, the run goes on, and the cap is labelled **soft** in the manifest and on the board.

     The exact LiteLLM setting names are checked at implementation against the pinned LiteLLM range. The design requirement is the profile, not the key names.
  - Known limits:
    - Caps are in USD only. The gateway has no total-token cap.
    - Pin the gateway's model cost map. The strict profile allows only priced models, because a model with no known price costs 0 and never trips its budget.
    - A tool that hardcodes a provider URL bypasses the gateway. Blocking outbound traffic from workers to provider hosts closes that gap. That is the owner's network setting.
    - A tool that calls an LLM but declares no providers and allowlists no provider key gets no key (step 1). Its spend is known only from reported usage.
- **b. Tools that cooperate.** Every tool run gets the environment variables `SWARM_ARENA_RUN_ID`, `…_GOAL_ID`, `…_CANDIDATE_ID`, `…_TEAM`, `…_AGENT`, and `…_ROUND`. A tool can then do either of these:
  - write `swarm/llm_calls.jsonl` in its workspace (a documented format; the core collects it), or
  - log to Langfuse with `session_id = SWARM_ARENA_RUN_ID` and `metadata.candidate_id`. DeepRetro already has Langfuse tracing, so its connector only has to pass the ids through.
- **c. Neither.** The candidate is marked `llm_calls_source = "unknown"`. If a tool declares `providers()` and there is no gateway, `validate_plan` warns about this.

**3. Langfuse export of the built-in LLM components' calls** (optional, extra `swarm-arena[langfuse]`):
- It uses Langfuse SDK v5 (built on OpenTelemetry) and `OpenAIAgentsInstrumentor` (openinference).
- Each manager, judge, and planner run is wrapped in `propagate_attributes(session_id=run_id, metadata={goal, team, agent, round, candidate_id}, tags=[…])`.
- So one Langfuse session shows a whole run, the built-in components' calls and the tools' calls together.
- `LLMCall.trace_url` links to the Langfuse trace.
- **Default:** nothing goes to any vendor until the owner configures Langfuse (self-hosted or cloud). The local journal always has everything.

**4. Seeing the chain.**
- `swarm-arena trace <candidate-id>` prints the chain: plan revision and planner calls, then the manager call that asked for the run, then the tool run and its inside calls, then the checks, then the judge calls, then the learnings that cite the candidate.
- Each step shows its model, tokens, cost, and Langfuse link.
- The board viewer's candidate page (§6.3.1) shows the same chain.

## 8. Safety rails (approved, D15)

### 8.1 Hard limits and budget (in `RunConfig`, human-only)

| Limit | Default (the builder may tune these) |
|---|---|
| `max_teams` (per goal; parts count) | 4 |
| `max_agents_per_team` | 4 |
| `max_agents_per_goal` | 12 |
| `max_concurrent_goals` (run) | 4 |
| `max_agents_total` (run, working at the same moment) | 32 |
| `max_agents_per_fingerprint` | 2 |
| `max_rounds` | 3 |
| `max_turns_per_round` (per agent) | 20 |
| `max_tool_runs_per_round` (per agent) | 3 |
| `max_tokens` (run) | Required when any LLM component (manager, judge, planner, or curator) is used |
| `max_usd` (run) | Optional. Every model used must have a known price (LiteLLM price map or the catalog's `prices`, §4.13). |
| `default_run_reservation` (tokens / usd) | Used when a tool has no estimate |
| `llm_tranche_usd` / `llm_tranche_tokens` (§8.1 ledger) **[call]** | Small: 0.25 USD / 25,000 tokens |
| `tool_timeout_s` | 3600 |
| `round_timeout_s` | 4 × `tool_timeout_s` |
| `run_timeout_s` | 24 h |
| `max_learnings_per_agent_round` | 3 |
| `max_board_reads_per_round` (per agent) | 10 |
| `digest_tokens` | 1,500 |
| `max_edit_bytes` (per edit, §4.12) **[call]** | 256 KiB |
| `max_consecutive_failures` | 3 |
| `max_refused_repeats` (§8.2) **[call]** | 3 |
| `watchdog_idle_rounds` (§8.2) **[call]** | 2 |

**How the budget is split** **[call]**:
- `reserve_fraction` (default 15%) is held back for the judges, the curator, the replanner, converge, and the final judge.
- **Final slice.** At run start, the core holds a final slice from the reserve for every goal, queued goals included. It is the estimate for 1 + `ConvergeSpec.max_join_attempts` converges plus the final judge. If the reserve cannot cover all the slices, the estimate step fails (§5). Per-round judges, the replanner, and the curator cannot use it.
- If the shared reserve cannot pay for a round's judging, the cost comes from that goal's pool before the next round's split. If both are empty, the goal skips to select-only converge and logs `reserve_exhausted`.
- Each goal gets a share of the run's pool, set by `goal_weights` (§4.10).
- At the start of each round, that round gets the goal's pool that is left, divided by the rounds that are left (§4.2 `RoundSpec`).
- Each active agent gets an equal share of the round's pool. An agent's unused share stays in the goal pool for later rounds, not for other agents in the same round.
- So when weak agents are pruned, the survivors get bigger shares. This is how pruning moves budget to strong teams.

**Reservation ledger** **[call]**:
- **Tool runs.** Before every tool run, the core reserves the estimate. For a tool with no estimate, it reserves `default_run_reservation`. The `reserve` event is **committed** to the run journal (§7.1) before the run is dispatched. Reservations made within about 200 ms share one commit (§7.1).
- **LLM calls by built-in components** (managers, judges, the planner, the curator). The core reserves a bounded **tranche** for each agent (and for each goal's judges, planner, and curator) with a committed `reserve` event: `limits.llm_tranche_usd` and `limits.llm_tranche_tokens`. Calls reserve and spend locally against the tranche, but only after that commit succeeds. When it is used up, a new tranche is reserved and committed the same way. A call bigger than one tranche gets its own committed reservation. The unused part of a tranche is returned with a committed settle event.
- It refuses the action if the reservation does not fit.
- After each tool run or LLM call, it settles the reservation against the reported usage, except for gateway tool runs (below). Each reservation has an id, sent as `metadata.reservation_id` (§4.13), and the record with that id settles it. **Settlement is idempotent by reservation id**, and settle events are written through.
- If no record arrives within `settle_timeout_s`, the reservation stays charged in full and is marked `estimated`.
- **Gateway tool runs** **[call]**. Only a tool run that may call an LLM gets a gateway key: its tool declares `Tool.providers()`, or it allowlists a provider key that the gateway replaces (§7.3). Other tool runs get no key, and they settle from reported usage (above). During a run, a tool run with a gateway key keeps its **full reservation** charged in the run ledger. Unused reservation is never returned to the run's pools in the middle of a run, because the core cannot prove that the gateway's spend logs are complete (§7.3). The actual gateway spend is still recorded, as `LLMCall` rows per candidate, for reports and tracing. After the run has ended, the final reconcile records the actual spend for reports only (§7.3).
  - **The trade-off.** With a gateway, every keyed tool run holds its full reservation until the run ends, even if it made few or no LLM calls. A tool with no estimate holds `default_run_reservation`. So when estimates are high, less of the budget can be used during a run. A tighter `Tool.estimate()` helps, and so does a correct `Tool.providers()`: a tool that declares none and allowlists no provider key gets no key.
  - The one exception is a gateway adapter with `terminal_accounting` (§7.3). The LiteLLM adapter has none in v1.
- **Crash recovery is conservative.** On resume, every reservation with no committed settle event counts as **fully spent**. A gateway tool run stays fully charged in any case (above).
- **The bound.** Total spend can never go over the run cap because of a crash. A crash costs at most the unsettled reservations: the tool runs in flight, plus one open tranche for each tranche holder (each agent, and each goal's judges, planner, and curator).
- **A takeover adds nothing** **[call]**, with or without `--force`. After the takeover, the old owner cannot commit, so it can neither dispatch a tool run nor reserve a new tranche (§7.1). It can spend only through work it committed before the takeover: tool runs it had dispatched and tranches it had reserved. Those reservations are in the committed head, so the new owner counts them as fully spent (above).
- Without a hard cap, the worst-case overspend is the error in the estimates of runs in flight. The docs state this, and the journal records every overspend.

**Tool-internal LLM spend** (for example, DeepRetro calling LLMs):
- **With the LLM gateway** (§7.3), each tool run that may call an LLM gets a virtual key with `max_budget` equal to its reservation. This is a **hard USD cap only when preflight verified the strict gateway profile** (§7.3). Otherwise the cap is soft, and it is labelled so in the manifest and on the board. During the run, the ledger keeps the full reservation (above). Spend is read back for reports and tracing, and the final reconcile records it after the run (§7.3). In reports, the gateway's spend logs are the only source of actual spend. `result.json` usage and `llm_calls.jsonl` rows are kept for information only, and rows are deduplicated by LiteLLM call id. **[call]**
- **Without the gateway**, the connector or script must report spend in `ToolOutput.usage` (or `result.json`), and pass the tool's own caps to the tool, for example DeepRetro `max_api_calls`.

### 8.2 Watchdog

The watchdog stops an agent with status `watchdog_stopped` and writes the reason, when:
- it repeats a refused request more than `max_refused_repeats` times,
- it makes no new passing candidate for `watchdog_idle_rounds` rounds in which it ran at least one tool run (starved rounds do not count), or
- it hits `max_consecutive_failures`. `error`, `oom`, and `timeout` count as failures. `provider_failed`, `lost`, `cancelled`, and `budget_exceeded` do not. **[call]**

### 8.3 The right number of agents

- Defaults are small, and every agent has a rationale.
- Duplicate fingerprints are capped.
- Budget shares grow for survivors.
- The cost estimate is shown before the run.
- The autonomous replanner may add agents to strong teams, within the limits.

### 8.4 Tool sandboxing in v1

- Each tool run gets its own workspace, an environment allowlist, and rlimits.
- **This is not strong isolation**, and the docs say so.
- Containers come in a later executor.
- Judge-private data is kept away from tool runs (§4.4).

## 9. Public API and compatibility promise

**Top-level `swarm_arena.__all__`** **[call]**. It is kept small. It holds only what an end user needs to write a plan and run it:

```
Goal, Plan, RunConfig, PartSpec, TeamSpec, AgentSpec, RoundSpec, ConvergeSpec, Limits, Resources, Ref,
Swarm, validate_plan,
ScriptTool, ScriptCheck, ScriptJudge, FunctionTool, ProgramJudge, Score, ScriptedManager,
KeepTopTeams, SuccessiveHalving, Patience, Chain
```

**Public submodules** **[call]**. Connector and backend types live in explicit submodules:

| Submodule | `__all__` |
|---|---|
| `swarm_arena.tool` | `Tool`, `Command`, `CompletedRun`, `ToolRequest`, `ParentRef`, `ToolOutput`, `FileRef`, `FileSet`, `Usage`, `Estimate`, `Action`, `RunStatus` |
| `swarm_arena.scoring` | `Check`, `CheckContext`, `CheckResult`, `Judge`, `JudgeContext`, `Lexicographic`, `HumanReview` |
| `swarm_arena.manager` | `Manager`, `AgentContext`, `RoundReport`, `LimitReached`, `AgentStatus` |
| `swarm_arena.planning` | `Planner`, `History`, `Policy`, `RoundState`, `Decision`, `FlowSpec` |
| `swarm_arena.board` | `Board`, `BoardView`, `ArenaBoard`, `Frontier`, `Learning`, `Candidate`, `LLMCall`, `GoalKey`, `Row`, `LineageGraph`, `Curator` |
| `swarm_arena.executor` | `Executor`, `LocalExecutor`, `Job` |
| `swarm_arena.config` | `ModelCatalog`, `ModelSpec`, `CachePolicy`, `PlannerCatalog`, `CatalogEntry`, `SecretsPolicy`, `TracingConfig`, `GatewayConfig` |
| `swarm_arena.llm` | `AgentsSDKManager`, `LLMPlanner`, `CollaborativePlanner`, `Reviewer`, `Approve`, `Feedback`, `Edited`, `LLMJudge`, `LLMCurator`, `BudgetExceeded`. Only the LiteLLM and SDK wiring lives here. |
| `swarm_arena.ray` | `RayExecutor` |
| `swarm_arena.protocol` | The A2A-shaped types |
| `swarm_arena.testing` | The conformance kit |

- `swarm_arena.config` holds the parts of `RunConfig` that are not at the top level (`Limits` and `Ref` are). They are plain pydantic models, so `RunConfig` needs no extra. `PlannerCatalog` and `CatalogEntry` are here, not in `swarm_arena.llm`, because the catalog is part of `RunConfig` (§4.1). `SecretsPolicy`, `TracingConfig`, and `GatewayConfig` are here for the same reason (§4.2). **[call]**
- **The rule.** Everything in the `__all__` of the top level and of the public submodules above is public. Each of these modules has its own `__all__` snapshot test. In 0.x, public names can still change, but only after one minor release of deprecation (rule 1).

Everything else is private.

**Rules:**
1. **Versioning.**
   - Follow SemVer. In 0.x, breaking changes bump the minor version and are listed in `CHANGELOG.md`.
   - Before removing anything, ship one minor release with a `DeprecationWarning`. Use `typing_extensions.deprecated`.
   - Release 1.0 after both connectors have run real campaigns.
2. **Stored formats.** Every stored format has `schema_version`. Older versions load through migration functions. Newer versions are refused with a clear error.
3. **ABCs grow only by optional methods with defaults.** A component's `__init__` does no I/O. Side effects such as uploads, network checks, or model loading go in `preflight()`.
4. **Conformance kit.** `swarm_arena.testing` has `check_tool`, `check_check`, `check_judge`, and `check_manager`, plus a **scripted fake model**. With the fake model, managers and judges can be tested offline with no API key. Connectors run these checks in their CI.
5. **Extras keep vendor churn out of the core.**
6. **Snapshot tests.** Each public module's `__all__` (the top level and every public submodule above) has its own snapshot test. Together they guard the public surface.

## 10. v2 and later: reserved now, built later

| Future feature | What v1 already reserves |
|---|---|
| Direct and broadcast messages (v2) | **Reserved now:**<ul><li>The A2A v1.0.1 types, including `INPUT_REQUIRED`.</li><li>`Plan.flows`: an allow-list of who may message whom.</li><li>Stable agent addresses.</li><li>`context_id` and the agent-round `task_id` on every journal event.</li><li>`LLMCall.parent_call_id`.</li></ul>**v2 design, taking lessons from Agency Swarm:**<ul><li>A `send_message` action whose `recipient` is a JSON-schema enum of the allowed recipients only.</li><li>`send()` is a **mailbox**: it returns at once, and the recipient reads `inbox()` on its own turns. A Runner never runs inside another Runner, so A→B→A cycles cannot deadlock or nest costs.</li><li>One open request per sender→recipient pair, and one thread per pair.</li><li>New limits: `max_messages_per_agent_round`, `max_message_bytes`, and a hop limit. A reply is charged to the recipient's budget share.</li><li>A message is never evidence, and it is marked untrusted. When the policy prunes an agent, its flows are dropped, with a reason.</li></ul>The optional `swarm-arena[a2a]` extra maps the types to the wire. The core never imports it. |
| Live sharing with no barrier | `Plan.sharing` will be widened from `"round"`. |
| Outside agents joining | The board API is the arena API. A v2 server will expose it. |
| Forum (threads, replies, votes) | `Learning.tags` and lineage. Votes are `Score`s. |
| SSH, container, E2B, or Ray-sandbox executors | The `Executor` ABC. |
| Pareto ranking | `Score.key` is already a tuple, and ranking is a named judge. |
| A database arena backend (for example Postgres), for when outside agents join | The `Board` ABC. The run journals and the review and curation objects can be imported. |
| Nested teams | `schema_version` migration. |
| Mid-run release of unused gateway reservations | The gateway adapter's `terminal_accounting` flag (§7.3). In v1 the adapter interface is private, and the LiteLLM adapter sets the flag to false. **[call]** |

## 11. Dependencies, project setup, and Ray

### Dependencies

- **Python:** 3.11 or newer, POSIX.
- **Core:** `anyio>=4.15,<5`, `pydantic>=2`, `typing-extensions`, `fsspec`.
- **Extras:**
  - `llm`: `openai-agents[litellm]>=0.24,<0.25` and `litellm>=1.105.1,<1.106`, with `!=` pins for releases found bad. A weekly CI job tests against the newest releases of both, so a breaking upstream change is found early. An upper bound is raised only after that job passes on the newer release. **[call]** Exact pins, as Agency Swarm uses, block users from installing alongside other packages.
  - `cache`: `litellm[caching,proxy]`. LiteLLM imports proxy modules on a response-cache hit, so a hit crashes on a plain install. That makes this extra required for response caching.
  - `ray`: `ray[default]>=2.59,<3`
  - `s3`: `s3fs`
  - `langfuse`: `langfuse>=5,<6`, `openinference-instrumentation-openai-agents`
  - The LLM gateway client uses only the stdlib (`urllib`). The gateway itself (LiteLLM Proxy plus Postgres) is run by the owner.
  - `a2a`: v2
- **Docs:** sphinx, sphinx-rtd-theme, numpydoc, myst-parser.

### Project setup

- **Tooling:**
  - uv: `uv init --lib`, src layout, `py.typed`.
  - ruff with numpy docstring rules.
  - ty, pinned to an exact version (it is still beta).
  - pytest 9 with anyio's plugin, `anyio_mode = "auto"`.
  - `-W error::DeprecationWarning`.
  - A snapshot test of each public `__all__` (§9).
- **CI:** GitHub Actions runs ruff, ty, and pytest on 3.11 to 3.13, and builds the docs. Read the Docs builds with uv. Releases go to PyPI with trusted publishing.
- **Repo files:** `README.md` (pitch, picture, offline quickstart), `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`, `.readthedocs.yaml`.
- **Docs pages:**
  1. Getting started
  2. Concepts
  3. Use your own files and script (`ScriptTool`)
  4. Writing a connector
  5. Planning modes
  6. Scoring and review
  7. Limits and safety
  8. Running on Ray
  9. The arena
  10. Versioning policy
  11. Roadmap
  12. API reference

### Secrets [call]

- Secrets come from each node's own environment, or from a secrets file at `RunConfig.secrets.file`, a path that exists on each node.
- A tool names the keys it needs in its `Command.env` allowlist. Every name must also be in the human-only `RunConfig.secrets.allow`, or `validate_plan` rejects the plan.
- Secrets are **never** put in Ray `runtime_env`, because Ray stores and shows that data.
- If only some nodes have keys, the tools that need keys request a custom Ray resource, for example `{"llm_keys": 1}`.

### Ray mechanics (checked against Ray 2.59.0)

- **Driver:**
  - The driver runs inside the cluster as a Ray Job, with `ray.init(address="auto")`.
  - `swarm-arena submit` wraps `JobSubmissionClient.submit_job`.
  - Ray Client (`ray://`) is not used for long runs.
  - `ray.init` is called through `anyio.to_thread.run_sync`.
  - Token auth setup is in the docs.
- **Tool environments:**
  - The tool's venv runs as a subprocess of a generic Ray task. `runtime_env.py_executable` is not used, because it needs the cluster's Ray and Python versions.
  - The worker environment needs `swarm-arena` plus the connector. Both come from the cluster image or `runtime_env` (`pip` / `working_dir`).
- **The task:**
  - **Dispatch intent first** **[call]**. Before it submits, the driver commits a dispatch-intent event with a pre-generated job id (§7.1). The job id is also the Ray task name (`.options(name=...)`). So after a driver crash, resume or reconcile finds every task the driver may have submitted by that name, through Ray's state API.
  - It uses `max_retries=0`, because a silent retry would break the budget accounting.
  - It uses `max_calls=1`.
  - The subprocess stays in the **worker's** process group. There is no `setsid`, so Ray's `killpg` still reaches it.
  - It waits with `p.wait(timeout=0.2)` in a loop, so a soft cancel (`KeyboardInterrupt`) can get in.
  - On a soft cancel, it sends SIGTERM to the tool, waits `grace_s`, runs `collect()`, and returns a partial `ToolOutput`.
- **Cancel:** cancelling an `await ref` does **not** cancel the task. So the executor does this, in a shielded scope:
  1. `ray.cancel(ref)`
  2. wait `grace_s` plus a margin for `collect()`
  3. `ray.cancel(ref, force=True)`
- **Confirmed dead** **[call]** (reconcile condition 1, §7.3). A Ray tool run is confirmed dead when the task returned after `p.wait()` reaped the tool and the `max_calls=1` worker has exited, or when Ray reports the worker dead after `ray.cancel(ref, force=True)`. After a driver crash, resume or reconcile finds the task by the name in its committed dispatch-intent event and asks Ray's state API about that task's worker. Until Ray confirms it, the final reconcile cannot record that run's spend, and its report keeps the full reservation (§7.3).
- **Files:**
  - With `artifact_root` on S3 (the cluster default), workers upload output files to `<artifact_root>/<sha256>` and download parent files from there. Only hashes travel through Ray.
  - With a local `artifact_root`, files travel through the Ray object store instead, with a cap (default 256 MiB per run).
- **GPUs:** Ray's `CUDA_VISIBLE_DEVICES` is kept. Ray resources are logical only, so the rlimits still apply.
- **VMs:** VMs join the Ray cluster as worker nodes.

## 12. Toy example (in the repo, runs offline)

The example minimises the Rosenbrock function.

- **Tools:** two `FunctionTool`s, `gradient_descent(lr, steps, x0, y0)` and `random_search(n, radius, seed)`. `random_search` is declared `deterministic=False` in its `FunctionTool` config, so replicates are allowed. `seed` is an optional input.
- **Check:** the result is finite and inside the bounds.
- **Judge:** a `ProgramJudge` that gives `Score.minimize(f)`.
- **Plan:** 2 teams with `ScriptedManager` agents, `SuccessiveHalving`, and a select-top-2 converge.
- **With an API key:** change one line to use `AgentsSDKManager`.
- **Where it is used:** it is the docs' Getting Started page, and an end-to-end test.

A second example, **"bring your own script"** (D19), uses no Python API at all:
- **Files:** a folder with two small data-and-formula files plus `run.py`, which reads them and prints a fit error into `swarm/result.json`.
- **Plan:** a `plan.toml` with `ScriptTool`, `ScriptCheck`, and `ScriptJudge`. The judge has its own hidden data file.
- **What it shows:** agents edit the editable files, build on each other's best version through `parents`, and the judge's hidden data never reaches them.
- **Where it is used:** the docs' "Use your own files" page, and an end-to-end test.

## 13. Follow-up projects (in other repos)

1. **DeepRetro connector.**
   - It has `Tool`s for `AutoSolver` and `run_batch.py`, integrity `Check`s, and an LLM-judge rubric.
   - It needs its own design. Two questions are open for that design:
     - What does the DeepRetro manager decide? Only settings and replicates? Also building on other agents' routes (continuations)? Also passing learnings into DeepRetro's own prompts?
     - Is the ranking judge an LLM, or a deterministic audit key followed by an LLM (`Lexicographic`)?
   - **Known blocker:** `audit_az_route` and the continuation and grafting code are not committed in DeepRetro `dev`.
2. **physika.**
   - **Start with `ScriptTool` (D19), with no connector code.** You give the `.phyk` files plus a Python script that runs them, and a `ScriptJudge` with hidden instances.
   - **Write a connector later**, only if it is needed. It would parse, type-check, unit-check, and run `.phyk` programs natively, and give typed error feedback to managers.
   - The first real goal is still to be chosen.

## 14. Left to the builder

- Internal module layout, private names, and the SQL schema. Public names (§9) and stored formats are fixed.
- Exact default numbers in §8.1.
- The look of the review `index.html`.
- CLI flags, beyond these commands: `plan`, `run`, `submit`, `resume`, `status`, `stop [--now]`, `review export|import`, `leaderboard`, `board serve|search|show`, `learnings pin|hide|add`, `trace`, `reindex`, `reconcile`.

## 15. Builder check log

Three builders reviewed draft r1. Their lenses were: a library engineer, a distributed-systems engineer, and an applied-AI engineer. Their questions are in `tasks/research/builder-check.md`.

Every question was either answered by a **[call]** above, deferred to the connector designs (§13), or listed in §16.

A consistency check of r3 found 71 issues (`tasks/research/consistency-issues.json`). r4 fixes them, except where a fix would change an owner decision. The new rules are tagged **[call]**.

## 16. Owner answers (2026-10-18)

| Question | Answer | Where it is in the design |
|---|---|---|
| Q1. Many goals in one run? | **"It can be one or many."** | D16, §4.10 |
| Q2. Parts in v1? | **Yes.** | D17, §4.11 |
| Q3. Shared storage? | **S3. More than one swarm can run at once.** | D18, §7 |

## 17. External review log

**Codex (gpt-6.1-sol), round 1** (2026-10-18): **CHANGES REQUESTED**, with 5 blocking and 3 non-blocking items. All are addressed in r5. The new rules are tagged **[call]**.

| # | Item | Fixed in |
|---|---|---|
| B1 | A crash could erase spend. Now: one journal per run whose durable part is always a prefix of `seq`, durable reservations and arena segment records before use, LLM tranches, and conservative recovery. | §4.2 `Event`, §6.5, §7.1, §8.1 |
| B2 | Two drivers could own one run. Now: a run lease with epochs, fences in every namespace the run writes, full charge for an old owner's late reservations, and the lease for every command that writes run state. | §6.5, §6.6, §7.1, §7.2, §7.3, §8.1 |
| B3 | LLM plans could reach executable config outside the catalog. Now: one planning boundary (`RunConfig.planner_catalog`) for every ref a non-human revision adds or changes, provenance set by the core and kept for unchanged human components, and a frozen main goal. | §4.1, §4.2, §4.6, §4.12 |
| B4 | The gateway "hard cap" was not proven. Now: a strict gateway profile checked at preflight, and spend reconciliation. | §7.3, §8.1 |
| B5 | Snapshots depended on local cache order. Now: a durable snapshot frontier in the run journal. | §4.9, §6.3, §6.5, §7.2 |
| N1 | The top-level API was too large. Now: a small facade plus public submodules, each with a snapshot test. Every part of `RunConfig` has a named public type. | §4.2, §9, §11 |
| N2 | The script protocol was incomplete. Now: the check and judge workspace, path rules, reserved names, and the staging order. | §4.12 |
| N3 | Scoring edge cases. Now: judge identity includes the panel, non-finite keys are rejected, `best` and `leaderboard` rank one judge identity only, and `min_improvement` covers only leaderboard first place. | §4.2 `Score`, §4.4, §4.9, §6.4 |

A follow-up check of r5 found 9 gaps in these fixes. They are closed in r5 too, and the rows above include them.

r6 replaces the arena segments, segment records, and fences of B1 and B2 with the round-2 design below. The rows above stay as a record of r5.

**Codex (gpt-6.1-sol), round 2** (2026-10-18): **CHANGES REQUESTED**. B1 is resolved. There are 4 blocking items (R2-1 to R2-4) and 3 non-blocking items (N-a to N-c). All are addressed in r6. The new rules are tagged **[call]**.

| # | Item | Fixed in |
|---|---|---|
| R2-1 | Arena segments, segment records, and per-namespace writer prefixes were a second record next to the journal, so membership was not exact. Now: the run journal is the arena record, `arena_root` holds only run pointers and write-once review and curation objects, segments are published in order, readers stop at the first gap, and a frontier is a `(run, run_root, epoch, last accepted seq)` mark for each epoch of each run read, plus the names of the review and curation objects. Ids are unique across epochs. | §4.2 `Event` and ids, §4.9, §5, §6.3, §6.3.1, §6.6, §7.1, §7.2, §10 |
| R2-2 | A base parent's overlay could replace files of a different tool. Now: only a parent of the same tool identity is overlaid, only on the receiving tool's editable paths, never on protected paths. Other parents are read-only under `parents/<id>/`, with a soft error. | §4.2 `ToolRequest`, `Candidate`, §4.5, §4.12 |
| R2-3 | Gateway spend was released on a row-count match while requests could still run. Now: `llm_gateway.request_timeout_s`, a gateway duration limit in the strict profile, and a four-condition release rule (tool confirmed dead, locally or on Ray; key closed at T; timeout passed since T; two stable log reads made after the timeout). | §4.2 `RunConfig`, §6.5, §7.3, §8.1, §11 |
| R2-4 | A takeover wrote fences in many places, so it could stop half done. Now: `control/owner.json` is the only acceptance authority, one conditional write changes the epoch and records the old epoch's cutoff, and a lease margin M (`lease_margin_s`) makes the cutoff after an expired lease cover every old-owner upload. | §4.2 `RunConfig`, §6.5, §7.1, §7.2, §8.1 |
| N-a | The §4.12 example commands did not match the staged paths. Now they use the staged paths (`problem/run.py`, `checks/check.py`, `judge/score.py`). | §4.12 |
| N-b | A replanner could remove a check that an earlier revision added. Now: every plan-level check that was ever in the plan is kept. | §4.6 |
| N-c | Orphan cleanup could kill a reused pid. Now: a write-through tool-run start event records host, pid, pgid, and process start time, and resume checks host and start time first. | §4.8, §6.5, §7.1 |

A follow-up check of r6 found 8 gaps in these fixes: spend-log reads made too early, tool death on Ray, a lease margin for slow uploads and clock skew, which readers ignore late segments, `run_root` in frontier marks, ids reused across epochs, tool-run starts that were not written through, and the frontier wording in row R2-1. They are closed in r6 too, and the rows above include them.

r7 replaces these parts of the round-2 design with the round-3 design below: the contiguous publication, the stop-at-first-gap rule, the cutoffs, and the lease margin of R2-1 and R2-4, and the four-condition release rule of R2-3. The rows above stay as a record of r6.

**Codex (gpt-6.1-sol), round 3** (2026-10-18): **CHANGES REQUESTED**. B3, B5, and R2-2 are resolved. There are 2 blocking items (R3-1, R3-2) and 2 non-blocking items (N3-1, N3-2). All are addressed in r7. The new rules are tagged **[call]**.

| # | Item | Fixed in |
|---|---|---|
| R3-1 | Fencing depended on clocks and a lease margin, and readers found segments by gap scanning. Now: `owner.json` holds the committed journal head, and a segment counts only if it is on the chain reached back from that head. One writer task runs one commit at a time, a head advance needs a segment that links to the head it replaces, and readers check that seqs have no gap. Plan files, round pages, and `solution.json` are views, written only after their commit. A takeover keeps the head and bumps the epoch, so the old owner's next conditional write fails. | §4.2, §4.5, §4.9, §5, §6.3.1, §6.5, §7.1, §7.2, §8.1 |
| R3-2 | Gateway reservations were released mid-run on spend logs that cannot be proven complete. Now: only a tool run that may call an LLM gets a gateway key, its full reservation stays charged for the whole run, a post-run `reconcile` changes reports only, and a `terminal_accounting` adapter flag is the only way to release mid-run (false for LiteLLM in v1). | §4.2, §6.5, §7.3, §8.1, §10, §11 |
| N3-1 | A tool could start before its start event was committed. Now: a launcher waits for "go" until the start event is committed, and on Ray a dispatch-intent event with the job id (also the task name) is committed before submission. | §4.8, §6.5, §7.1, §7.3, §11 |
| N3-2 | Curation and review objects had no fixed order. Now: they apply by name timestamp, then object name, and the later pin or hide wins, on every machine. | §6.3, §7.2 |

A follow-up check of r7 found 6 gaps in these fixes: overlapping commits that could skip a segment, a membership rule that could be read as a name or seq compare, view files written from state that was not committed, gateway keys for tool runs that call no LLM, the old owner's in-flight work after a takeover, and reconcile's Ray lookup. They are closed in r7 too, and the rows above include them.

**Codex (gpt-6.1-sol), round 4** (2026-10-18): **LGTM**. B2, B4, R2-1, R2-3, R2-4, R3-1, and R3-2 are all resolved. There are no blocking items. Its 2 non-blocking notes are applied in r8:

| # | Item | Fixed in |
|---|---|---|
| N4-1 | Post-run reconcile called its totals "final actual spend". Now they are "observed spend", labelled `completeness unverified` unless the adapter has terminal accounting. Key ids are kept for later reconciles. | §7.3 |
| N4-2 | After a resume, an older view could replace a newer one, for example `solution.json`. Now each view records its source head, and a view replaces another only if its head is later on the committed chain. | §7.1 |

Under the owner's rule, this LGTM approves the design.
