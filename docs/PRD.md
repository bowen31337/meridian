# Meridian — Product Requirements Document

**Codename:** Meridian
**Version:** 0.4 (draft)
**Author:** Bowen Li
**Date:** 2026-05-13
**Status:** Draft — pending review
**Supersedes:** v0.3 (2026-05-13)

> **What's new in v0.4.** **Language decision flipped to Python backend
> + TypeScript UI** (polyglot monorepo, OpenAPI codegen bridge). The
> trigger: `claude-agent-sdk-python` exists at parity with the TS SDK,
> AND it allows users to authenticate via OAuth (Claude Pro/Max
> subscription) instead of API key — a major economic win for the
> dev-loop persona. AnthropicProvider gains a **two-mode plugin
> architecture** (`api_key` mode via raw SDK; `oauth` mode via
> claude-agent-sdk → Claude Code CLI). Python wins also because Skill
> Forge / Hermes interop / agentskills.io / Honcho / ACP reference
> impls are all Python-native. New: **all LLM providers configurable
> via `~/.meridian/config.yml`**, with `secret_ref://` indirection
> through Vaults, hot reload via SIGHUP or HTTP. Tooling locked:
> **`uv`** is the only Python entrypoint (workspace + lockfiles +
> installs); **Biome** replaces eslint + prettier for TypeScript
> (single tool for lint + format, Rust-backed, 10–100× faster). PRD
> §10 R2 drops "TypeScript-first" as a differentiator (it was thin);
> architectural differentiators carry the load.
>
> **What's new in v0.3.** Delivery strategy locked: **all user-facing features
> ship in one v1 release** (the engineering sequence is internal — no
> public v0.x → v1.0 march). **Infrastructure features** (channel
> backends, environment backends, model providers, persistence backends,
> auth backends, deployment topology) phase across v1.x → v2. **Every
> feature listed in §5 is load-bearing for SOTA — deferring any of them
> forfeits the SOTA claim, so none are deferred.** Open questions from
> v0.2 §9 resolved in §9 (Decisions). NG3 flipped: thin web UI + basic
> Live Canvas now in v1. Skill Forge precision target reframed.

**Frame.** Meridian is a **self-hosted control plane for AI agents** that runs
equally as a *coding/dev agent* and as a *personal-assistant gateway* — one
binary, one data model, two front-doors. Architecturally it borrows the
**Session / Harness / Sandbox** abstractions from Anthropic's Managed Agents,
the **multi-channel gateway** from OpenClaw, and the **self-improving skill
forge + model/environment abstractions** from Nous Research's
hermes-agent. The product is Meridian; the gateway is its control plane.

> **What's new in v0.2.** Primitive renames to match Anthropic's actual beta
> SDK (`Session` not `Thread`; `Vault`, `MemoryStore`, `UserProfile`,
> `Environment`, `Webhook` as first-class). Added a multi-channel `Gateway`
> surface from OpenClaw. Added the `Skill Forge` learning loop, `Model
> Router`, `Environment Manager`, `ACP Adapter`, and `agentskills.io`
> compatibility from hermes-agent. Replaced the v0.1 top-level `Run` state
> machine with **session phases**. Replaced "tool registry" framing with
> Anthropic's **sandbox/`execute(name, input)`** model so containers, MCP
> servers, HTTP tools, and in-process handlers share one dispatch surface.

---

## 1. Executive Summary

### 1.1 The thesis

A SOTA agent platform is not a coding CLI, not a chatbot, not a Slackbot —
it is a **persistent, versioned, capability-scoped control plane** that the
user interacts with through whichever surface fits the moment: a CLI when
coding, a phone DM when away from the keyboard, a webhook when something
needs to wake a sleeping agent.

Anthropic's Managed Agents beta validates the architectural shape:
**Session** (durable append-only event log), **Harness** (stateless,
replaceable inference loop), **Sandbox** (unified `execute(name, input)`
that abstracts containers, MCP servers, custom tools). OpenClaw validates
the *channel surface* — the most useful agent answers you on the channels
you already use. Hermes Agent validates the *operational layer* — model
agnosticism, multi-backend sandboxes, and a learning loop that compounds
skills from real trajectories.

Meridian fuses these. The same Meridian daemon that drives a coding loop
through a CLI also routes a Telegram message through the gateway,
executes a tool inside a Modal container, summarizes the day to your
MemoryStore, and offers the result back through a Live Canvas in your
browser. One Session is the truth; everything else is a viewer.

### 1.2 What makes this SOTA, in one paragraph

**Local-first** (runs on your laptop with no internet aside from model
calls); **API-compatible** with Anthropic's managed agents beta where
semantics align (so application code is portable cloud ↔ local);
**multi-channel** at the front door (CLI, Telegram, Slack, Discord,
WhatsApp, iMessage, web, webhook); **capability-sandboxed** by
intersection (no upward escalation, ever); **deterministic replay** of
any historical run; **self-improving** via a skill forge that distills
durable skills from agent trajectories using the agentskills.io standard;
**model-agnostic** via a router over 200+ providers; **multi-environment**
across 7 sandbox backends (local, Docker, SSH, Modal, Daytona, Vercel,
Singularity); **observable** out of the box via OpenTelemetry +
Prometheus; **multi-agent** via typed handoffs, parallel fan-out, and an
ACP adapter for cross-system agent communication.

### 1.3 The three-source synthesis at a glance

| Source | What Meridian inherits |
|--------|------------------------|
| **Anthropic Managed Agents** | Session/Harness/Sandbox abstractions; versioned Agents and Skills; first-class Vault, MemoryStore, UserProfile, Environment, Webhook, File, Message; stateless harness with `wake(sessionId)` recovery; credentials never enter sandboxes. |
| **OpenClaw** | Multi-channel inbox (WhatsApp/Telegram/Slack/Discord/Signal/iMessage/Matrix/IRC/Teams …); Live Canvas; voice (wake-word + continuous); companion apps; browser/canvas/cron as first-class tools; *"Gateway is the control plane, the assistant is the product."* |
| **hermes-agent** | Self-improving Skill Forge; agentskills.io standard; Model Router across 200+ providers; 7 environment backends; Honcho-style dialectic user modeling; FTS5 session search + LLM summarization; trajectory compression; ACP adapter & registry; MCP bridge; `hermes claw migrate`-style import. |
| **Meridian-original** | Capability-by-intersection enforcement; deterministic replay with fixtures; mid-run checkpoints; hooks-as-verdict; hybrid retrieval KB; per-Agent / per-Session / per-Run budget aggregation; full local-first deployment without external infra dependencies. |

---

## 2. Problem Statement

### 2.1 The four ceilings

Current agent platforms hit one of four ceilings, and each ceiling forces
a workaround that erodes the system:

1. **Anthropic Managed Agents** ships the right *abstractions* (Session,
   Harness, Sandbox, Vault, MemoryStore, Skill versioning) — but is
   **vendor-managed**, charges by token, and can't run on a developer's
   laptop. You can't `meridian` your way through a flight.
2. **OpenClaw** ships the right *surface* (every messaging channel, voice,
   canvas) — but its agent core is a single Gateway control plane without
   the architectural rigor Anthropic ships (no versioned skills, no
   formal capability scoping, no deterministic replay).
3. **Hermes Agent** ships the right *operational layer* (model
   agnosticism, environment backends, learning loop) — but is
   Python-first, opinionated, and not API-compatible with Anthropic's
   managed agents (which means code doesn't round-trip).
4. **Claude Code / Cursor / Cline / Aider** ship great *coding UX* — but
   are coding-only, ephemeral-by-default, and have no first-class
   primitives for multi-agent, multi-channel, or self-improvement.

### 2.2 The gap Meridian fills

No system today:

- Is **API-compatible** with Anthropic Managed Agents AND
- Runs **locally with no managed dependency** AND
- Has a **multi-channel gateway** for personal-assistant use AND
- Has a **self-improving skill loop** with the agentskills.io standard AND
- Has a **capability sandbox** that enforces "no upward escalation" between
  parent and child agents AND
- Can **deterministically replay** any historical run.

That's the Meridian shaped hole.

### 2.3 Why now (2026)

- **Anthropic published the managed-agents beta SDK** in late Q1 2026.
  The resource shape is now public and stable enough to mirror.
- **The agentskills.io standard exists** (Hermes ships it). Skill
  portability across systems is achievable for the first time.
- **MCP is widely adopted.** Tool integration no longer requires
  bespoke per-vendor work.
- **OpenTelemetry tracing for agents** is mainstream. Observability is a
  near-free win.
- **Local model quality** (via Ollama/llama.cpp/vLLM at the laptop tier)
  is good enough that "no internet" is a real workflow, not a stunt.

### 2.4 What "SOTA" means for v1

Meridian is SOTA when **every** condition holds:

1. Mirrors Anthropic beta SDK resource shape (`/v1/agents`,
   `/v1/sessions`, `/v1/skills`, `/v1/environments`, `/v1/memory_stores`,
   `/v1/vaults`, `/v1/user_profiles`, `/v1/files`, `/v1/webhooks`,
   `/v1/messages`) so code written against Meridian works against
   Anthropic cloud, modulo Meridian-only extensions.
2. Implements the OpenClaw gateway surface for at least Telegram, Slack,
   Discord, and webhook channels in v1; iMessage / WhatsApp / Signal in
   v1.1.
3. Implements at least the local, Docker, and SSH environment backends
   from Hermes in v1; Modal/Daytona/Vercel/Singularity in v1.1.
4. Implements a Model Router over at least Anthropic, OpenAI,
   OpenRouter, and a local provider (Ollama) in v1.
5. Implements the Skill Forge learning loop, producing
   agentskills.io-compatible skills from trajectories, in v1.
6. Has the Meridian extensions (capability-by-intersection enforcement,
   deterministic replay, mid-run checkpoint, hooks-with-verdict, hybrid
   retrieval KB, run/session/agent budgets) in v1.
7. Has observability that lets a developer answer "why did the agent do
   X?" weeks later in under a minute using Meridian artifacts alone.

---

## 3. Goals and Non-Goals

### 3.1 Goals

**G1. Anthropic-compatible primitives.** Sessions, Agents, Skills,
Environments, MemoryStores, Vaults, UserProfiles, Files, Messages,
Webhooks match Anthropic beta SDK semantics. Extensions live under
`/v1/x/...`.

**G2. Stateless harness, durable session.** The harness can crash and a
fresh harness recovers via `wake(session_id)` from the session's event
log. State lives in the session, never in the harness.

**G3. Unified sandbox dispatch.** Every executable action — built-in
tools, MCP servers, HTTP tools, custom containers — exposes one shape:
`execute(name, input) → result`. The harness doesn't know or care which
backend.

**G4. Multi-channel gateway.** Users reach their agent through any
configured channel (CLI / Telegram / Slack / Discord / webhook in v1;
WhatsApp / iMessage / Signal / Matrix / IRC / Teams in v1.1). The
Session is identical across channels.

**G5. Model-agnostic routing.** Configurable router selects the right
model per task across ≥4 providers in v1, ≥200 in v1.1. Routing is
declarative (per agent, per skill, per request).

**G6. Multi-backend environments.** Sandboxes run locally, in Docker, on
SSH targets, in Modal, in Daytona, on Vercel, on Singularity — picked
per-environment, hot-swappable per-tool.

**G7. Self-improving skill forge.** A background process distills
durable skills from session trajectories. Output is agentskills.io
schema-compatible. Skills are versioned content-addressed objects.

**G8. Capability-by-intersection.** Agents declare grants, tools declare
requirements, dispatch enforces the intersection. Subagents inherit a
*subset*. No upward escalation, ever.

**G9. Deterministic replay.** Every Session is an append-only event
log. Given canned model + sandbox responses, replay produces an
identical agent trajectory. Used for CI regression of prompt changes.

**G10. Multi-agent orchestration.** Typed handoffs, parallel fan-out,
parent-budget aggregation, propagation of cancellation and error. Cross-system
agent comms via ACP adapter.

**G11. Observability + budgets.** OpenTelemetry traces, Prometheus
metrics, hard/soft token & dollar & wall-clock budgets per Agent /
Session / Run, with the orchestrator stopping or asking before
overshooting.

**G12. Hooks as verdicts.** Lifecycle hooks (pre/post tool, pre/post
message, on_stop, on_compact, on_handoff, on_checkpoint, on_error) can
continue / mutate / veto / fail.

**G13. Hybrid retrieval KB.** Glob + BM25 + vector, fused via RRF;
scoped to global / project / agent / session.

**G14. Migration paths.** `meridian import openclaw $PATH` and
`meridian import hermes $PATH` produce a working Meridian configuration
from existing OpenClaw or Hermes installs.

### 3.2 Non-Goals (v1)

- **NG1.** Multi-tenant SaaS hosting. v1 is single-tenant per Meridian
  install.
- **NG2.** Inference hosting. Meridian *calls* models; it does not host
  inference. (We do support local providers via Ollama / llama.cpp;
  Meridian doesn't run the kernels, it talks to them.)
- **NG3.** *(removed in v0.3 — web UI flipped into v1.)* A thin web UI
  (read-only viewer + basic chat) and a basic Live Canvas (text +
  markdown + simple form widgets) ship in v1. Deeper A2UI integration
  is infrastructure-tier and lands in v1.2.
- **NG4.** Cross-language tool authoring in v1. In-process handlers
  are **Python only** (v0.4 language flip). Out-of-process / MCP / HTTP
  handlers can be any language from day one.
- **NG5.** Reinforcement learning fine-tuning of base models. The Skill
  Forge produces *skills* (instructions + tools + tests), not LoRAs or
  weights. RL on top of Meridian trajectories is a v2 conversation; we
  preserve trajectory storage so it remains possible.
- **NG6.** Replacing IDEs or messaging apps. Meridian integrates with
  them, not against them.

### 3.3 Explicit Tradeoffs

| Decision | Why |
|----------|-----|
| **Python 3.11+ (daemon) + TypeScript (UI), polyglot monorepo with OpenAPI codegen bridge** | Python wins where the gravity is: `claude-agent-sdk-python` parity (subscription OAuth path for Anthropic users), Skill Forge ML-native, Hermes/Honcho/agentskills.io/ACP reference impls Python-first, embedding/vector libs native. TypeScript wins at the UI surface only (React, Live Canvas, channel-frontend SDKs). Bridge: FastAPI emits OpenAPI; `openapi-typescript` generates TS types; CI fails on drift. Cost: ~5% engineering tax on schema sync, paid once and amortized. |
| **`uv` for all Python tooling** | Single contributor entrypoint for venvs, lockfiles, monorepo workspaces, package installs, script runs. Replaces pip/poetry/pipenv/pdm/conda — none of which compose well across a daemon + plugins + SDKs + tests workspace. Reproducible installs in seconds. |
| **`Biome` for all TypeScript lint + format** | Single Rust-backed tool replaces eslint + prettier + their plugin sprawl. One config file (`biome.json`), one command (`biome check --write`), 10–100× faster than the eslint stack. Combined with `tsc --noEmit` for type checking, that's the whole TS toolchain. |
| **SQLite (local) → Postgres (cloud)** through one repository interface | Local-first is non-negotiable; Postgres path preserved without code branching. |
| **NDJSON event log on disk + projection tables** | Anthropic's session = event log model. Append is cheap, replay is `cat \| jq`, projections accelerate listing. |
| **SSE for streaming, HTTP POST for tool-result submission** | Simpler intermediaries than WebSocket; sufficient for unidirectional server-push; the established pattern in Anthropic's SDK. |
| **JSON Schema everywhere** | Matches Anthropic's tool format; language-agnostic; works for tools, hook verdicts, handoff schemas, and agentskills.io. |
| **agentskills.io as the skill format** | Industry standard. Inter-system skill portability. Hermes already ships it. |
| **Three categories of channels in v1 (CLI / chat-net / webhook), more in v1.1** | Cuts surface area without hard-coding assumptions that block expansion. |
| **No top-level Run resource** | Anthropic doesn't expose one; the session event log is the source of truth for "what happened". A "run" is a *span* over the log, not an object. |

---

## 4. Users and Use Cases

### 4.1 Personas

**P1 — The dev-loop user (coding agent).**
Wants Meridian to plan + execute work alongside them in a workspace. Cares
about tool safety, replay, multi-agent supervision, branch/PR workflows.
*Lives in:* CLI, editor plugin.

**P2 — The on-the-go user (PA gateway).**
Wants the agent reachable on Telegram or Slack — *"what's my todo for
today?"*, *"book a haircut Saturday"*, *"summarize the Foo project
meeting"*. Cares about memory continuity, channel availability, voice.
*Lives in:* Telegram / Slack / iMessage / Discord / wake-word voice.

**P3 — The platform / infra engineer.**
Self-hosts Meridian on a team server. Cares about deployment, sandboxing
isolation, secret management, multi-user separation, audit logs, cost
controls.
*Lives in:* the daemon's config, observability surfaces, the audit log.

**P4 — The tool/skill author.**
Publishes reusable tools (capability-scoped, JSON-schema'd, MCP-compatible
or HTTP) or skills (agentskills.io-shaped). Cares about distribution,
versioning, capability docs.
*Lives in:* the skill registry CLI, tool spec docs.

### 4.2 Headline use cases

| # | Persona | Scenario | What v1 must deliver |
|---|---------|----------|----------------------|
| U1 | P1 | "Spawn a planner that fan-outs 3 worker agents in parallel, collects their PRs." | Typed multi-agent fan-out under one supervisor session; budget aggregation; child capability subsetting. |
| U2 | P1 | "Replay last week's failed run with verbose tool logs and find the bad tool call." | Per-session event log + replay endpoint with fixture-driven model/tool responses. |
| U3 | P3 | "Agent can read/edit files in `$WORKSPACE` only; cannot fetch from the open internet." | Capability declarations enforced at sandbox dispatch; FS jail; per-agent network allowlist. |
| U4 | P1 | "Pause this 4-hour migration, edit something by hand, then resume from the same point." | Mid-session checkpoint; resumable from event log even after daemon restart. |
| U5 | P1 | "Same prompt change, run against last week's golden trajectories in CI." | FakeModelAdapter + FakeSandboxAdapter + recorded fixtures; deterministic replay; divergence error pinpoints the first changed event. |
| U6 | P2 | "Ask the agent on Telegram what's on the calendar; have it follow up via webhook when the meeting moves." | Multi-channel gateway; channel-agnostic Session; webhook outbound. |
| U7 | P2 | "Remember my brother's birthday and remind me 3 days before, every year." | MemoryStore (persistent) + cron + channel push; not a session memory (which decays). |
| U8 | P1 | "Find auth-rotation code by intent, not filename." | Hybrid (glob + BM25 + vector) retrieval, scoped to project. |
| U9 | P3 | "Audit-log every `exec` call before it runs; block on a regex denylist." | Lifecycle hook with veto verdict; sandboxed hook execution. |
| U10 | P1 | "Cost broken down by Agent and tool call for this week." | Usage events on every model/tool call; per-resource budget reports. |
| U11 | P1 | "Have the agent learn from yesterday's session and turn the recurring pattern into a reusable skill." | Skill Forge background loop; agentskills.io-compatible output; user approval gate before promotion. |
| U12 | P3 | "Swap the model from Claude Opus to a local Qwen for cheap drafts, fall back to Claude on hard tasks." | Model Router with declarative routing rules; per-agent and per-skill model selection. |
| U13 | P1 | "Run the same agent inside a Modal container when WSL is slow." | Environment Manager with hot-swappable backends; same Sandbox dispatch interface. |
| U14 | P4 | "Publish my custom `gh-pr` skill; have other agents import it by name+version." | agentskills.io packaging; content-addressed versioning; registry CRUD. |
| U15 | P3 | "Migrate my existing OpenClaw install (channels, sessions, MEMORY.md) without losing history." | `meridian import openclaw $PATH` (and likewise hermes) — schema-mapped, lossless where possible, lossy with audit log otherwise. |

---

## 5. Functional Requirements

Wording per RFC 2119 (MUST / SHOULD / MAY).

### 5.1 Control plane primitives

**F-CP-1.** The system MUST expose CRUD over the following resources via
HTTP API, mirroring Anthropic beta SDK shapes where applicable:
`Agent` (and `Agent.Version`), `Session` (and `Session.Thread`,
`Session.Event`), `Skill` (and `Skill.Version`), `Environment`,
`MemoryStore`, `Vault`, `UserProfile`, `Channel` *(Meridian extension)*,
`File`, `Message`, `Webhook`, `Model` *(listing only)*.

**F-CP-2.** Every `Agent` and `Skill` MUST be content-addressed and
versioned. Sessions MUST pin to a specific `Agent.Version` at creation;
replays MUST use the pinned version.

**F-CP-3.** Sessions MUST persist their **append-only event log** as the
source of truth. Relational projections MAY exist for query
acceleration but MUST NOT be authoritative.

**F-CP-4.** Sessions MUST expose a **phase** projection:
`idle / waiting_for_model / waiting_for_tool / waiting_for_user /
paused / terminated`. Transitions MUST be derivable from the event log.

**F-CP-5.** A `wake(session_id)` operation MUST allow any harness instance
to resume a session by reading its event log and projections. The harness
MUST NOT carry session state between requests.

### 5.2 Sandbox (Environment + Tool) dispatch

**F-SB-1.** Every executable action MUST resolve to one of the dispatch
kinds: `in_process` (TS), `subprocess` (any language), `mcp` (MCP server),
`http` (POST endpoint), `container` (Docker/Modal/Daytona/Vercel/SSH/
Singularity). All kinds present the same `execute(name, input) → result`
interface to the harness.

**F-SB-2.** Tool definitions MUST carry: name, description, input JSON
schema, output JSON schema, declared capabilities (§5.4), required
environment (e.g. `requires: env=docker`), timeout, memory cap.

**F-SB-3.** Validation MUST run on input args (pre-dispatch) and on
output (post-dispatch). Schema failure MUST surface as a `tool_result`
with `is_error: true` returned to the model, **not** as an orchestrator
crash.

**F-SB-4.** MCP servers MUST be importable as tools without code changes
beyond registering the server URL + manifest in the Environment.

### 5.3 Environments

**F-EN-1.** Meridian MUST support these environment backends in v1:
`local`, `docker`, `ssh`. v1.1 MUST add `modal`, `daytona`, `vercel`,
`singularity` (matching hermes-agent's set).

**F-EN-2.** An `Environment` MUST declare its backend, image/template,
mounted workspace path, env-var passthrough policy, network policy
(default-deny + allowlist), and cap envelope (CPU/mem/disk).

**F-EN-3.** Switching environments per-tool MUST be a configuration
change, not a code change. The Sandbox dispatch surface is identical
across backends.

### 5.4 Capabilities

**F-CA-1.** Capabilities are dotted strings with optional parameters:
`fs.read[glob]`, `fs.write[glob]`, `net.fetch[host]`, `exec.shell`,
`exec.sudo`, `kb.read[scope]`, `kb.write[scope]`, `agent.spawn[ids]`,
`secret.read[name]`, `channel.send[channel_id]`, `memory.write[scope]`,
plus implementation-defined extensions.

**F-CA-2.** Tools MUST declare required capabilities; Agents MUST
declare granted capabilities. Dispatch MUST verify the *intersection*
before invocation. No upward escalation: a parent cannot grant a child
a capability the parent does not itself hold.

**F-CA-3.** Capability denial MUST produce a synthetic `tool_result`
with `is_error: true` back to the model — never a silent failure or
orchestrator crash.

### 5.5 Vaults

**F-VL-1.** Secrets MUST be stored in a `Vault` resource separately from
agent/session/skill bodies. The default backend is the OS keychain; an
encrypted-file backend MUST also be supported. Pluggable backends (AWS
KMS, HCP Vault) SHOULD be expressible.

**F-VL-2.** Tool args may reference secrets via `secret_ref://vault/name`.
Substitution MUST happen at the harness boundary *after* the args have
been written to the event log in their `secret_ref://` form. Plaintext
secrets MUST NOT appear in the event log, hook stdin, or trace
attributes.

**F-VL-3.** Hooks declaring `secret.read[name]` MAY observe substituted
values; otherwise they observe the ref form.

**F-VL-4.** Per Anthropic's principle: **credentials never enter
sandboxed tool execution unless the sandbox proves it needs them**.
Network tools that need an OAuth token SHOULD use an in-harness proxy
that injects the token at outbound-request time.

### 5.6 Memory stores

**F-MS-1.** `MemoryStore` is a distinct primitive from Session. Sessions
hold recent events; MemoryStores hold long-lived facts ("birthdays",
"how user prefers PR descriptions").

**F-MS-2.** MemoryStores MUST support hybrid retrieval (BM25 + vector +
filter). Scopes: `global`, `user`, `agent`, `project`.

**F-MS-3.** MemoryStores SHOULD support **dialectic user modeling**
(Honcho-style): write-time integration where new facts are reconciled
with existing ones, conflicting facts produce a reconciliation event,
and the user can audit/edit.

**F-MS-4.** Memories MUST be referenceable from prompts via templated
fields (e.g. `{{ memory.user.preferences.commit_style }}`) that the
harness expands at run time.

### 5.7 Channels (gateway)

**F-CH-1.** A `Channel` MUST represent a configured front-door:
`{ kind: 'cli' | 'telegram' | 'slack' | 'discord' | 'webhook' | …,
config: { token_vault_ref, …}, default_agent_id, default_user_profile_id,
pairing: { … } }`.

**F-CH-2.** Inbound messages on any channel MUST resolve to a Session
attached to the right Agent + UserProfile, using a deterministic
pairing rule (channel + remote_id → user_profile).

**F-CH-3.** Outbound messages from a Session MUST route to all attached
channels whose `egress` policy permits. The Session is **channel-agnostic** —
the same Session is reachable from CLI, Telegram, and webhook concurrently.

**F-CH-4.** v1 MUST ship: CLI, Telegram, Slack, Discord, generic
webhook. v1.1 SHOULD add: WhatsApp, iMessage (via macOS bridge), Signal,
Matrix, IRC, Teams.

**F-CH-5.** Channel access MUST default to **untrusted-inbound** (per
OpenClaw): non-paired senders are quarantined; pairing happens via an
out-of-band token. The primary user has full host access; other
authenticated users default to sandboxed sessions.

### 5.8 Skills + Skill Forge

**F-SK-1.** A `Skill` MUST conform to the **agentskills.io** schema.
Required fields: `name`, `description`, `instructions`, `tools[]`,
`tests[]` (optional but recommended), `metadata`. Skills MUST be
content-addressed and versioned.

**F-SK-2.** Skills MAY be installed from: local directory, npm package,
git URL, agentskills.io registry. A skill registry endpoint SHOULD list
installed skills with version + provenance.

**F-SK-3.** The **Skill Forge** is a background process that consumes
session trajectories and produces candidate skills. Forge proposals
MUST be quarantined until a human approves promotion to active.

**F-SK-4.** Forge proposals MUST include: the proposed skill body, the
trajectory(ies) it was derived from, an optional reproducer test, an
A/B comparison (optional) of the trajectory with and without the skill.

**F-SK-5.** Skill activation per Agent MUST be explicit. An installed
skill does not automatically apply to every agent.

### 5.9 Models + Model Router

**F-MD-1.** A `Model Router` MUST select the provider/model for each
model call based on a declarative rule set: per Agent, per Skill, per
Session metadata, per estimated token cost, per available context.

**F-MD-2.** v1 MUST support these providers: Anthropic, OpenAI,
OpenRouter, Ollama (local). v1.1 SHOULD broaden to ≥10 (Google,
Together, NVIDIA NIM, Hugging Face, vLLM-self-hosted, llama.cpp-self-hosted,
…). Hermes's coverage of 200+ is the long-term ceiling.

**F-MD-3.** Routing decisions MUST be observable: every model call
records the rule that fired and the chosen provider/model in the event
log.

**F-MD-4.** Failover MUST be configurable: on rate limit / 5xx / timeout,
the router MAY retry against a fallback model. Failovers MUST be logged.

**F-MD-5. Anthropic provider has two modes.** The Anthropic provider
MUST support both `api_key` mode (raw `anthropic` SDK; per-token
billing; full feature surface) and `oauth` mode (claude-agent-sdk →
Claude Code CLI; subscription billing; feature surface bounded by what
the CLI exposes). Mode is per-provider-instance configuration.
Per-Agent routing rules MAY pin a mode (e.g. "prefer oauth for drafts,
api_key for high-context"). See Architecture §13.4 for the bridging
model.

**F-MD-6. YAML provider configuration.** All providers (and their
credentials, routing rules, and per-Agent overrides) MUST be
configurable via a single YAML config file (default
`~/.meridian/config.yml`; system-wide `/etc/meridian/config.yml`;
overridable via `$MERIDIAN_CONFIG`). The config file MUST support:
- `secret_ref://vault/{vault_id}/{key}` indirection for credentials.
- Inline plaintext credentials (allowed for dev; daemon MUST log a
  `config.plaintext_secret` warning per provider with a plaintext
  credential on every startup).
- Per-provider `kind`, `mode`, `auth`, `base_url`, and plugin-specific
  fields.
- Top-level `routing` rules + `fallbacks` matching the §13.2
  routing-policy shape (or per-Agent overrides under `agents`).
- Daemon-level settings (`bind`, `workspace_root`, `log_level`),
  storage backends (`storage.database`, `storage.event_log`,
  `storage.blob_store`), and Vault declarations.

**F-MD-7. Hot reload.** The config file MUST be reloadable without
daemon restart via `POST /v1/x/config/reload` and `SIGHUP`. Reload
MUST be **validate-then-atomic-swap**: invalid new config keeps the
old config in effect, errors logged, no service interruption.

**F-MD-8. Schema versioning + IDE support.** The config file MUST
declare `version: 1` at the top. Forward-only migrations via
`meridian config migrate`. JSON Schema for editor autocomplete MUST be
emittable via `meridian config schema > schema.json`, generated from
the Pydantic config model.

### 5.10 Multi-agent + ACP

**F-MA-1.** Sessions MAY spawn **child sessions** via an `agent.spawn`
capability. Child sessions MUST inherit `parent_session_id`,
cancellation propagation, and a configurable subset of parent
capabilities.

**F-MA-2.** Handoff between agents MUST be typed: parent declares an
output schema, child must produce a value matching the schema before
its session reaches the `completed` phase.

**F-MA-3.** Parallel fan-out (parent spawns N children, awaits all)
MUST be a built-in tool (`parallel_runs`). Budgets aggregate at parent.

**F-MA-4.** An **ACP adapter** MUST permit Meridian agents to call
agents in *other* systems (hermes, openclaw, foreign Meridian
installs) via the Agent Communication Protocol. ACP exchanges are
recorded as `acp.outbound` / `acp.inbound` events in the session log.

### 5.11 Cron + scheduling

**F-CR-1.** A `Cron` resource MUST schedule recurring or one-shot
session invocations. Triggers MAY be timestamp/interval, channel event,
file change, webhook, or "memory anniversary" (e.g. fires N days
before a `birthday` memory).

**F-CR-2.** Cron-triggered sessions MUST inherit the capabilities of
their declared agent, never escalate.

### 5.12 Webhooks

**F-WB-1.** A `Webhook` resource MUST allow outbound push of session
events to external URLs, filterable by event type. Use cases:
"page me when this session needs review", "post completed PR url to
GitHub Actions".

**F-WB-2.** Webhooks MUST be retryable with exponential backoff and a
configurable max retry count; failures land in a dead-letter queue.

### 5.13 State, checkpoint, resume

**F-ST-1.** A session MUST be checkpointable on demand by a hook or by
external API call. Checkpoint serializes the phase + pending tool
calls + event-log seq.

**F-ST-2.** A paused session MUST be resumable from the same
checkpoint, possibly via a different harness instance.

**F-ST-3.** Session state MUST be fully reconstructable from the event
log; checkpoints are an optimization, not a source of truth.

### 5.14 Hooks

**F-HK-1.** Lifecycle hooks MUST be available for:
`session_start`, `session_end`, `pre_message`, `post_message`,
`pre_tool_call`, `post_tool_call`, `on_stop`, `on_compact`,
`on_handoff`, `on_checkpoint`, `on_error`, `on_channel_inbound`,
`on_channel_outbound`, `on_model_call`.

**F-HK-2.** Hook verdicts: `continue` (no-op), `continue` with
mutation, `veto` (only for `pre_*`), `fail`. The harness MUST honor
the verdict.

**F-HK-3.** Hooks MUST be sandboxed (same dispatch surface as tools);
timeouts, capability scoping, isolation apply.

### 5.15 Observability

**F-OB-1.** Every session MUST emit an OpenTelemetry trace with one
span per phase transition, model call, tool call, hook call, and child
session spawn.

**F-OB-2.** Prometheus metrics MUST cover: sessions per phase, tool
calls (count, latency, result), model token usage (input/output/cache,
by provider/model), hook latencies/verdicts, channel inbound/outbound
counts, queue depth, harness wakeups.

**F-OB-3.** `GET /v1/sessions/{id}/events?stream=true` MUST stream the
live event log as SSE with resumption via `Last-Event-ID`.

### 5.16 Budgets

**F-BG-1.** Hard and soft budgets MUST be settable per Agent, per
Session, per Run-span (a user-message → next idle window). Dimensions:
input tokens, output tokens, cache tokens, dollars, wall-clock seconds.

**F-BG-2.** Soft budget exceeded → `pre_message` hook is given a
`budget_warning` and the user/agent is asked to approve continuation.
Hard budget exceeded → session phase transitions to `terminated` with
reason `budget_exceeded`.

### 5.17 CLI + TUI

**F-CL-1.** A `meridian` CLI MUST cover CRUD over every resource +
session interaction.

**F-CL-2.** `meridian run` MUST stream events to the terminal with a
human-friendly TTY renderer (token streaming, tool-call inlining,
collapsed thinking blocks).

**F-CL-3.** A minimal TUI gateway (`meridian tui`) SHOULD provide
multi-session navigation, channel view, and live event tail (matching
hermes's `tui_gateway`).

### 5.18 Migration

**F-MG-1.** `meridian import openclaw $PATH` MUST import an OpenClaw
install's channels, sessions, and MEMORY.md into the Meridian schema.

**F-MG-2.** `meridian import hermes $PATH` MUST import a hermes-agent
install's skills, environments, model config, and session histories
into the Meridian schema.

**F-MG-3.** Imports MUST be transactional (all-or-nothing) and produce
an audit log of lossy mappings.

---

## 6. Non-Functional Requirements

### 6.1 Performance

- Harness wake (no model call): **< 100 ms p99**.
- Sandbox dispatch overhead (validation + capability check + spawn):
  **< 20 ms p99** in-process; **< 200 ms p99** subprocess;
  **< 500 ms p99** container backend (warm pool).
- KB query (50k-file workspace): **< 250 ms p95** hybrid.
- Channel inbound → harness wake: **< 1 s p95** end-to-end.
- Event log fan-out to SSE subscribers: **< 10 ms p99**.

### 6.2 Reliability

- A single-node Meridian MUST survive `SIGKILL` mid-session and a
  fresh harness MUST recover via `wake(session_id)` to the last event.
- Tool / hook / skill-forge crashes MUST NOT crash the harness or the
  daemon.
- Harness restart with active sessions MUST auto-resume them.

### 6.3 Security

- All sandboxed code runs in an isolated worker (subprocess on local,
  container on docker/modal/…, remote shell on ssh).
- Default FS jail at `$WORKSPACE`; outside-jail symlinks rejected.
- Default-deny network; allowlist per agent.
- Credentials never enter sandboxed code paths (Vault § 5.5).
- Channel inbound treats non-paired senders as untrusted (§ 5.7).
- Audit log is append-only and signed (Ed25519, optional in v1, MUST
  in v1.1 for shared deployments).

### 6.4 Local-first

- Single binary + SQLite + local FS works **with zero external
  dependencies** apart from outbound model API.
- Local-mode model provider (Ollama / llama.cpp) MUST be supported so
  the entire stack runs offline.

### 6.5 Compatibility

- HTTP API SHOULD mirror Anthropic beta SDK paths where semantics
  match.
- Extensions live under `/v1/x/...`.
- Skill format MUST be agentskills.io-compatible.
- Inbound: MUST accept the major messaging-channel webhook formats
  natively.

---

## 7. Success Metrics

### 7.1 Adoption (90 days post-v1)

- ≥ 1 internal team using Meridian daily.
- ≥ 5 published agentskills.io skills produced by users (not
  hand-authored by us).
- ≥ 1 multi-agent workflow + ≥ 1 multi-channel PA workflow in real use.
- ≥ 1 successful OpenClaw or hermes migration to Meridian.

### 7.2 Engineering quality

- Replay fidelity: ≥ 99% of recorded sessions replay identically given
  canned model + sandbox responses.
- Mean time to root cause for a failed session: < 5 min using only
  Meridian artifacts.
- Skill Forge precision (proposed skills that survive human review):
  ≥ 50% at v1 release, climbing to ≥ 75% within 90 days post-v1 as the
  trajectory corpus grows. (The Skill Forge *feature* ships in v1 — the
  precision *grows* with corpus, not with versions.)

### 7.3 Cost discipline

- Average overrun vs. configured soft budget: < 5%.
- 100% of hard-budget transitions tagged with correct reason code.

### 7.4 Harness operability

- p99 harness wake latency within § 6.1 target.
- Successful resume rate after `SIGKILL`-induced restart: > 99% over
  10k synthetic crashes.

---

## 8. Delivery Plan

Meridian ships on **two tracks**. The Feature Track is one big-bang v1
release containing **every functional requirement in §5** — none are
optional, because each is load-bearing for the SOTA claim. The
Infrastructure Track adds backends, deployment modes, and scale options
*after* v1, behind the same feature surface.

### 8.1 Feature Track — v1 (single release, every feature)

All of §5 lands at v1. There is no public preview / beta sequence; the
table below is the **internal** engineering milestone sequence. Users
get the full feature set on day one.

| Milestone | Engineering scope | Duration |
|-----------|------------------|----------|
| **E0 — Skeleton** | Monorepo, types, HTTP API skeleton, SQLite schema, Session event log + replay primitives, Agent/Session CRUD + versioning, Anthropic provider adapter, in-process TS tool handlers, SSE streaming. | 2 wk |
| **E1 — Sandbox + Capabilities** | Sandbox dispatch (in_process / subprocess / http / mcp / container), JSON schema validation on input *and* output, capability-by-intersection enforcement, built-in tools (`exec` / `read` / `write` / `grep` / `kb_search` / `spawn` / `parallel_runs`). | 2 wk |
| **E2 — Environments + Model Router** | Environment Manager (local + docker + ssh backends), Model Router (Anthropic + OpenAI + OpenRouter + Ollama), declarative routing rules, per-call routing logging, failover. | 2 wk |
| **E3 — Multi-agent + Hooks + ACP** | `agent.spawn` with typed handoff schemas, parallel fan-out, parent ↔ child budget aggregation + cancellation propagation, full lifecycle hooks with veto verdict, ACP adapter inbound + outbound (Hermes-compatible). | 2 wk |
| **E4 — Memory + KB + Vaults + UserProfiles** | MemoryStore primitive with hybrid (BM25 + vector + filter) retrieval and Honcho-style dialectic write, hybrid KB tool, Vault backends (OS keychain + encrypted file) with credential proxy, UserProfile pairing. | 2 wk |
| **E5 — Gateway + Channels** | Channel resource + driver SDK; CLI + Telegram + Slack + Discord + Webhook drivers; pairing tokens; untrusted-inbound quarantine; cross-channel session fan-out. | 2 wk |
| **E6 — Skill Forge** | agentskills.io-compatible skill packaging, content-addressed versioning, registry CRUD, forge pipeline (cluster → distill → propose → quarantine → approve → activate), per-agent activation, auto-suggest opt-in. | 2 wk |
| **E7 — Checkpoints + Replay + Observability + Budgets + Web UI + TUI** | Mid-session checkpoint API, deterministic replay endpoint with fixture mode, OTel traces, Prometheus metrics, hard/soft budgets per Agent/Session/Run-span, thin web UI viewer, basic Live Canvas, TTY/TUI renderer with token streaming and tool inlining. | 3 wk |
| **E8 — Migration + Hardening** | `meridian import openclaw` + `meridian import hermes` with audit log, crash-recovery soak (10k synthetic SIGKILLs), integration tests, environment & channel conformance suites, docs, internal dogfooding. | 3 wk |

**Total: ~20 weeks engineering to v1.**

Acceptance gate for v1 release: every functional requirement in §5
passes its acceptance tests; success metrics targets in §7 met on a
dogfooded corpus; environment + channel conformance suites green for
every shipped backend; ≥ 1 internal team running v1 daily for ≥ 2
weeks.

### 8.2 Infrastructure Track — v1.x → v2 (backends, scale, deployment)

After v1, the *feature* surface is stable. What evolves is the
**infrastructure surface** — number of channel backends, number of
environment backends, number of model providers, persistence backends,
auth backends, deployment topology. None of these add new user-facing
capability; they expand the operational envelope.

| Version | Infra scope | ETA |
|---------|-------------|-----|
| **v1.1** | + WhatsApp, iMessage (macOS bridge), Signal channels. + Modal, Daytona environment backends. Model Router providers expanded to ≥ 10 (add Google, Together, NVIDIA NIM, HuggingFace, …). + OIDC auth backend. Signed audit log mandatory for multi-user installs. agentskills.io self-hosted registry mirror. + AWS KMS Vault backend. | ~3 mo post-v1 |
| **v1.2** | + Matrix, IRC, Teams channels. + Vercel, Singularity environment backends. Model Router via Provider SDK toward the 200+ long tail. **Postgres + S3 backend as supported tier** (Repository pattern means zero application-code change). Deeper A2UI integration in Live Canvas. + HCP Vault backend. | ~6 mo post-v1 |
| **v1.3** | Horizontal **harness pool** (stateless harnesses already; this is a config flip). + NATS / Redis Streams option for the event bus. Per-tenant capability ceilings. Multi-harness session routing. Cross-host checkpoint resume. | ~9 mo post-v1 |
| **v2** | Multi-tenant cloud-SaaS option (NG1 lifted only here). Distributed orchestrator. Ephemeral container workers per call. Cross-tenant ACP. | when a real user needs it |

**The contract.** Application code written against v1's HTTP API works
unchanged against v1.1, v1.2, v1.3, and v2. Capability declarations,
skill manifests, session event logs, and migration importers are
forward-compatible across the infrastructure track.

---

## 9. Decisions (resolved open questions from v0.2)

Each decision below answers a v0.2 §9 open question. Decisions are
classified as **F (Feature → ships in v1)** or **I (Infrastructure →
phases over v1.x → v2 per §8.2)**.

**D1 — ACP spec lock-in. [F]**
Target Hermes's ACP **exactly**. We are downstream of Hermes; defining
a Meridian flavor would fragment the agent-to-agent ecosystem. CI runs
Hermes's reference compliance suite where one exists; any deliberate
deviation is documented in `docs/acp-deviations.md` with a rationale.
*Why F:* the ACP feature itself ships in v1 (it is load-bearing for
multi-agent cross-system delegation).

**D2 — Skill activation UX. [F]**
Both **manual activation** and **auto-suggest with user confirmation**
ship in v1. Manual is the default. Auto-suggest is opt-in per Agent
(`agent.skill_activation_mode = 'auto_suggest'`); when active, the
harness emits a `skill_suggestion` event that the user must approve
before the skill is bound to the agent. **No skill ever auto-activates
without an audit-logged human approval.**
*Why F:* user-facing UX is a feature; neither mode is SOTA-defeating
to ship.

**D3 — Daemon topology. [I]**
**Single daemon in v1**, with the harness already stateless per
Principle 2 (Architecture §1). Horizontal harness pool lands in
**v1.3**. The seam is preserved by design: harnesses are addressable
via the Session Service and reconstitute state via `wake(session_id)`
— so v1.3 is a config flip, not a refactor.
*Why I:* deployment topology, not a user-visible feature.

**D4 — Session retention. [F]**
Indefinite retention by default. **Auto-compaction** kicks in at 30
days idle (event log tail → compressed summary; full log archived to
blob store). Explicit `meridian sessions archive <id>` and
`meridian sessions restore <id>` always available. Compaction and
archive are v1 features; retention *policy* is configurable per
install.
*Why F:* compaction and archive are user-visible and load-bearing for
the local-first claim (disks fill).

**D5 — Channel auth. [F + I]**
**Pairing tokens** are the v1 mechanism — out-of-band single-use
tokens bind a remote channel identity to a UserProfile, matching the
OpenClaw model. This is the **feature** that ships in v1. **OIDC** is
the infrastructure-tier auth backend that ships in **v1.1** for
team-shared installs. Both coexist (one install may use both).
*Why mixed:* pairing UX is a feature; OIDC is an auth backend
substitution.

**D6 — Web UI + Live Canvas. [F]**
**Thin web UI** (read-only session viewer + chat composer) and
**basic Live Canvas** (agent-driven text + markdown + simple form
widgets) ship in v1. They are pure clients over the existing HTTP+SSE
API — no new server logic. **Deep A2UI integration** lands in v1.2
once the A2UI surface stabilizes.
*Why F:* visibility into agent behavior is SOTA-critical for the
P1 (dev) and P2 (PA) personas; a v1 without a viewer makes
observability §13 effectively inaccessible to non-developers.

**D7 — agentskills.io registry. [F + I]**
**Public registry + local cache** in v1 (feature). **Self-hosted
mirror** in v1.1 (infrastructure) for air-gapped / enterprise
deployments. Skills installed from any source carry provenance
metadata recorded on the SkillVersion.

**D8 — Database backend. [I]**
**SQLite WAL** is the v1 supported backend. **Postgres** lands in
**v1.2** as a fully supported alternative. The Repository pattern
(Architecture §7) means application code is unchanged across
backends.
*Why I:* persistence backend, not a user feature.

**D9 — Vault backends. [F + I]**
**OS keychain + encrypted file** ship in v1 (both are features — the
encrypted file matters for headless servers). **AWS KMS** lands in
v1.1; **HCP Vault** in v1.2 (both infrastructure backends).

**D10 — Channel set in v1. [F + I]**
Five channels in v1 (feature surface): **CLI, Telegram, Slack,
Discord, generic Webhook**. They cover both frames (dev-agent and
PA-gateway) and exercise every gateway primitive (pairing,
quarantine, inbound/outbound, multi-channel session). The
**Channel Driver SDK** is itself a v1 feature — third parties can
add channels without forking Meridian. Additional first-party
channels (WhatsApp / iMessage / Signal in v1.1; Matrix / IRC / Teams
in v1.2) are infrastructure-tier.

**D11 — Environment set in v1. [F + I]**
Three backends in v1 (feature surface): **local, docker, ssh**. They
cover laptop, reproducible, and remote-dev — the three primary
developer modalities. The **Environment backend SDK** is itself a v1
feature. Modal / Daytona land in v1.1; Vercel / Singularity in v1.2.

**D12 — Model provider set in v1. [F + I]**
Four providers in v1 (feature surface): **Anthropic, OpenAI,
OpenRouter, Ollama**. They cover frontier (Anthropic / OpenAI),
aggregator-gateway (OpenRouter unlocks many on one adapter), and
local (Ollama). The **Provider SDK** is itself a v1 feature. The
remainder grow toward the Hermes 200+ ceiling via v1.1 → v1.2
(infrastructure).

### 9.1 What we explicitly chose NOT to defer

For the record — these were considered for deferral and **rejected**
because each is load-bearing for SOTA:

- Capability-by-intersection enforcement (without it, "self-hosted
  agent with safe multi-agent" loses the safety claim).
- Deterministic replay (without it, the differentiator vs. Hermes
  disappears).
- Skill Forge (without it, the differentiator vs. Anthropic disappears).
- Multi-channel Gateway (without it, the differentiator vs. Claude
  Code / coding-only agents disappears).
- Mid-session checkpoints (without them, long-running multi-agent
  workflows are unrecoverable).
- ACP adapter (without it, Meridian is an island).
- Hooks with veto (without them, capability sandboxing has no escape
  hatch for site-specific policy).
- Hybrid retrieval KB (without it, agents are blind to the workspace).
- Live Canvas (without a viewer, the PA persona has no surface).
- Migration importers (without them, the bridge from OpenClaw / Hermes
  is missing, and the "successor" framing is hollow).

If timeline pressure forces a trim, the answer is **slip v1, not
shrink v1**. The SOTA claim is the product.

---

## 10. Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Anthropic beta SDK shape continues to evolve; v0.1 of this PRD already missed `Session/Vault/MemoryStore/UserProfile/Environment/Webhook`. | Generate the Meridian API client from Anthropic's OpenAPI spec; CI check for drift; pin to a specific beta SDK version per Meridian release. |
| R2 | Hermes is *already* mature and competes in the same space. | Differentiate on three **architectural** axes (the language axis is gone as of v0.4 — we are now also Python at the backend): (a) Anthropic-compatibility (Hermes is not), (b) capability-by-intersection sandboxing (Hermes is permissive), (c) deterministic replay with fixtures (Hermes doesn't ship it). Plus: (d) two-mode AnthropicProvider unlocking subscription-tier OAuth, which Hermes doesn't have as a first-class pattern. Offer first-class import via `meridian import hermes`. |
| R3 | OpenClaw channel surface is enormous; v1 cannot match 20+ channels. | Ship 5 (CLI / Telegram / Slack / Discord / Webhook) and a clean ChannelDriver SDK; community / v1.1 add the rest. |
| R4 | Capability system over-restrictive, blocks practical workflows. | Sensible defaults; one-click capability grant in CLI; per-grant expiry. |
| R5 | Multi-backend Environments multiply test surface. | Contract tests against the Sandbox interface; environment-conformance suite each backend must pass. |
| R6 | Skill Forge produces bad/dangerous skills. | Quarantine until approval; require tests to promote; activation explicit per agent. |
| R7 | Vault leaks via hook stdin / event-log debug toggles / log messages. | Redaction at the harness boundary, audited; secret never reaches a sandbox; CI test that no plaintext from `vault://*` keys appears in any log file in a soak run. |
| R8 | Sessions grow unbounded; event logs exhaust disk. | Compression policy after N days; archive command; KB-extracted summaries replace tails. |
| R9 | Determinism is harder than it sounds (nondet tools, clocks, sandbox state). | Scope determinism to "same recorded model + sandbox responses ⇒ same trajectory"; don't promise wall-clock; document explicitly. |
| R10 | OpenClaw migration is lossy (channel-token formats, pairing semantics, MEMORY.md). | Audit log per migration; manual review checklist; convert MEMORY.md to MemoryStore entries with provenance tag. |
| R11 | Shipping every feature in one v1 (per §8.1) extends time-to-first-user-feedback to ~20 weeks, which is enough time for the landscape to move under us. | Mandatory **internal dogfooding from E5 onwards** (gateway available) on real coding + PA workflows; weekly demo gate; if a feature is mis-shaped, fix it before v1 rather than ship-then-fix. Per D-section: slip v1 rather than shrink v1 — SOTA is the product, and a partial v1 is not SOTA. |
| R12 | One of the four originally-targeted v1 providers (Anthropic / OpenAI / OpenRouter / Ollama) breaks API or pricing during the 20-week build. | Provider SDK is itself a v1 feature (D12); we can hot-swap a replacement without changing the feature surface. Pin SDK versions per provider; CI smoke-tests each weekly. |
| R13 | Sandbox / Skill Forge / Replay individually pass their tests but their *interactions* surface bugs only under real load (e.g. forge produces a skill that fails replay). | E8 hardening phase budgets explicit time for **integration soak tests**: a multi-day, multi-agent, multi-channel dogfood with crashes injected, plus regression replay of every recorded session from E5 onwards. |

---

## 11. Resource Map — Meridian vs. Anthropic vs. OpenClaw vs. Hermes

| Concept | Anthropic Beta | OpenClaw | hermes-agent | Meridian |
|---------|----------------|----------|--------------|----------|
| Agent identity, versioned | `agents/agents.py + agents/versions.py` | implicit | `agent/` dir, implicit | `/v1/agents` + `/v1/agents/{id}/versions` |
| Conversation state | `sessions/sessions.py + sessions/events.py + sessions/threads/` | sessions | sessions | `/v1/sessions` (event log) + `/v1/sessions/{id}/threads` |
| Skill, versioned, portable | `skills/skills.py + skills/versions.py` | skills | `skills/` (agentskills.io) | `/v1/skills` + `/v1/skills/{id}/versions` (agentskills.io) |
| Sandbox / Environment | `environments.py` | implicit (browser/canvas/nodes) | `environments/` (7 backends) | `/v1/environments` + ≥3 backends in v1 |
| Long-term memory | `memory_stores/` | MEMORY.md | FTS5 + Honcho | `/v1/memory_stores` + hybrid retrieval |
| Credentials | `vaults/` | `.env` | `.env` / OS keychain | `/v1/vaults` + credential proxy |
| Multi-user | `user_profiles.py` | pairing | per-user partition | `/v1/user_profiles` |
| Outbound push | `webhooks.py` | channels | gateway push | `/v1/webhooks` |
| Files | `files.py` | files | files | `/v1/files` |
| Messages (single-shot) | `messages/` | messages | messages | `/v1/messages` |
| Model listing | `models.py` | n/a | provider abstraction (200+) | `/v1/models` + Model Router |
| Multi-channel inbox | (n/a) | core | core | `/v1/channels` + drivers |
| Self-improving skill loop | (n/a) | (n/a) | Skill Forge | `/v1/x/skill_forge` |
| Agent-to-agent comms | (n/a) | (n/a) | `acp_adapter/ + acp_registry/` | `/v1/x/acp` |
| Cron / scheduling | (n/a) | cron | `cron/` | `/v1/x/cron` |
| Capability sandboxing (intersection) | (n/a) | (n/a) | (n/a) | `/v1/x/capabilities` (Meridian-original) |
| Deterministic replay with fixtures | (n/a) | (n/a) | (n/a) | `/v1/x/sessions/{id}/replay` (Meridian-original) |
| Mid-session checkpoint | (n/a) | (n/a) | (n/a) | `/v1/x/sessions/{id}/checkpoint` (Meridian-original) |
| Hooks with verdicts | (n/a) | (n/a) | partial | `/v1/x/hooks` (Meridian-original) |
| Hybrid retrieval KB | (limited) | (limited) | FTS5 | `/v1/x/kb` (Meridian-original) |

---

## 12. What "Meridian" means here

We considered renaming (Skein, Loom, Ledger, Substrate) but decided to
keep **Meridian** for v0.2. The name's neutrality is an asset: Meridian is
both a coding agent AND a personal assistant gateway, and a name that
forced one metaphor (Loom → threads → coding; Hermes → messenger → PA)
would bias the framing. *Meridian* — a reference line you align by — fits
both: when coding, the agent is your reference; when away, it relays
between channels. Final.

---

**End of PRD v0.2.**

Cross-references:
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) for technical realization.
- [`ANTHROPIC_MANAGED_AGENTS_ANALYSIS.md`](./ANTHROPIC_MANAGED_AGENTS_ANALYSIS.md)
  for source-of-truth analysis of the upstream SDK we mirror.
- Upstream sources: https://www.anthropic.com/engineering/managed-agents ,
  https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/resources/beta ,
  https://github.com/openclaw/openclaw ,
  https://github.com/nousresearch/hermes-agent .
