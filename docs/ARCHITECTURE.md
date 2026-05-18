# Meridian — Technical Architecture Guideline

**Version:** 0.4 (draft)
**Author:** Bowen Li
**Date:** 2026-05-13
**Supersedes:** v0.3 (2026-05-13)
**Companion:** [`PRD.md`](./PRD.md), [`ANTHROPIC_MANAGED_AGENTS_ANALYSIS.md`](./ANTHROPIC_MANAGED_AGENTS_ANALYSIS.md)

This document describes the technical realization of the requirements in
`PRD.md`. Where the PRD says "what and why", this doc says "how".

**What changed in v0.4.** **Daemon language flipped from TypeScript to
Python 3.11+; UI stays TypeScript; OpenAPI codegen bridges the two.**
Driven by PRD v0.4: `claude-agent-sdk-python` parity + OAuth
(subscription-tier) economics + Skill Forge / Hermes / agentskills.io
/ Honcho / ACP Python-native gravity. Specifically:
(1) §3 entity types are now Pydantic v2 models (still presented in TS
syntax inline for readability — both languages have generated bindings
from the OpenAPI source-of-truth); (2) §13.1 `ModelProvider` interface
is Python; (3) **§13.4 completely rewritten** for the **two-mode
AnthropicProvider plugin** (`api_key` via raw `anthropic` SDK; `oauth`
via `claude-agent-sdk-python` → Claude Code CLI subprocess with
MCP-tool bridging back to Meridian's Sandbox); (4) new §13.6
**YAML configuration** (`~/.meridian/config.yml`) with `secret_ref://`
indirection and validate-then-atomic-swap hot reload; (5) §15.5
plugin SDKs become two-language (Python SDKs for daemon-side plugins:
provider/environment/channel; TypeScript SDK for UI plugins); (6) §23
repo layout rebuilt for the polyglot monorepo (`apps/meridiand`
Python; `apps/meridian-ui` TS; `packages/schemas` is the OpenAPI
source-of-truth; `packages/sdk-py` + `packages/sdk-ts` are generated);
(7) §27 lint rules ported (**ruff + pyright + importlinter** Python
side via **uv**-managed workspace; **Biome + tsc** TS side — Biome
unifies linting and formatting, replacing the prior eslint + prettier
split); (8) §28 adds the OpenAPI codegen pipeline as a v1 infra
concern.

**What changed in v0.3.** Aligned with PRD v0.3's **two-track delivery
model**: every feature in PRD §5 lands in one v1 release; backends and
deployment topology phase across v1.1 → v2 per PRD §8.2. Specifically:
(1) backend version markers in §7 (storage), §12 (environments), §13
(providers), §15 (channels), §18 (vaults) updated to the new
v1 → v1.1 → v1.2 → v1.3 → v2 progression; (2) §25 deployment topology
rewritten to that progression with the horizontal harness pool moved
from v2 → v1.3; (3) new §15.5 codifies the **Channel Driver SDK**,
**Environment Backend SDK**, and **Model Provider SDK** as v1 features
so backend phasing happens behind stable plugin contracts; (4) new §29
adds the **Web UI + Live Canvas** thin-client architecture (per PRD
v0.3 §3.2 NG3 flip).

**What changed in v0.2.** Primitives renamed to match Anthropic's real beta
SDK (`Thread → Session`; no top-level `Run` — replaced by **session
phases** and run-spans over the session event log). New components
introduced: **Gateway** (multi-channel front door, from OpenClaw),
**Environment Manager** (multi-backend sandbox, from hermes-agent),
**Model Router** (provider-agnostic dispatch, from hermes-agent), **Skill
Forge** (self-improvement loop, from hermes-agent + agentskills.io
standard), **ACP Adapter** (agent-to-agent communication, from
hermes-agent), **Vault** (credential proxy, from Anthropic), **MemoryStore**
(persistent facts, from Anthropic + Honcho-style dialectic modeling),
**UserProfile** (multi-user pairing, from Anthropic), **Channel Drivers**
(per-platform messaging adapters). The harness is **stateless and
replaceable** per Anthropic's *"harness as cattle"* principle, recovering
via `wake(session_id)`.

---

## 1. Design Principles

1. **The session is the truth.** Everything reconstructable from a
   session's append-only event log is canonical. Relational tables and
   checkpoints are derived projections.
2. **The harness is cattle, not pets.** No session state in the harness.
   Any harness can `wake(session_id)` and resume.
3. **One Session, many surfaces.** A single Session is reachable from
   CLI, channels, webhooks, and the API simultaneously. Surface ≠ state.
4. **Sandbox is one shape.** `execute(name, input) → result` unifies
   built-in tools, MCP servers, HTTP tools, containers, SSH, and remote
   environments. The harness doesn't branch on backend.
5. **Capabilities by intersection.** Tools declare required caps; agents
   declare grants; dispatch enforces the intersection. No upward
   escalation, ever.
6. **Credentials never enter sandboxes.** Vaults inject secrets at the
   harness boundary, not the executor.
7. **Local-first, cloud-portable.** Same binary speaks SQLite or
   Postgres, local FS or S3, picked by config — not by code path.
8. **Streaming first.** SSE for server → client; HTTP POST for client
   tool-result submission. No polling.
9. **Deterministic replay is narrow.** Same recorded model + sandbox
   responses ⇒ same trajectory. We do not promise wall-clock determinism.
10. **No half-finished implementations.** A feature ships with tests,
    docs, metrics, and a migration story — or it does not exist.

---

## 2. System Overview

### 2.1 Component diagram

```
                      ┌────────────────────────────────────────────────┐
                      │                  Front doors                    │
                      │  CLI │ TUI │ Telegram │ Slack │ Discord │ ...   │
                      │  webhook │ Live Canvas (v1.1) │ web UI (v1.1)   │
                      └──────────────────┬──────────────────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────────────────┐
                      │                    Gateway                        │
                      │   Channel drivers, pairing, untrusted-inbound     │
                      │   quarantine, channel egress fan-out              │
                      └──────────────────┬───────────────────────────────┘
                                         │
                                         ▼
            ┌──────────────────────────────────────────────────────────────┐
            │                  Meridian HTTP API                            │
            │  /v1/agents  /v1/sessions  /v1/skills  /v1/environments       │
            │  /v1/memory_stores  /v1/vaults  /v1/user_profiles             │
            │  /v1/channels  /v1/files  /v1/webhooks  /v1/messages          │
            │  /v1/models   /v1/x/{capabilities,replay,checkpoint,acp,...}  │
            └─────────────────────┬─────────────────────────────────────────┘
                                  │
            ┌─────────────────────┴────────────────────────────┐
            ▼                                                   ▼
    ┌──────────────────┐                          ┌─────────────────────────┐
    │ Harness Pool     │  ◀── wake(session_id) ──│ Session Service          │
    │ (stateless,      │                          │ (event log, projections, │
    │  inference loop) │                          │  phases, checkpoints)    │
    └────────┬─────────┘                          └─────────────────────────┘
             │
             ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                          Event Bus (in-proc)                        │
   └────────────────────────────────────────────────────────────────────┘
        │           │              │              │              │
        ▼           ▼              ▼              ▼              ▼
  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
  │  Model   │ │  Sandbox │ │   Skill      │ │  Hook    │ │ Observability│
  │  Router  │ │  Dispatch│ │   Forge      │ │ Executor │ │ (OTel + Prom)│
  └─────┬────┘ └─────┬────┘ │  (background)│ │   Pool   │ └──────────────┘
        │            │      └──────────────┘ └──────────┘
        ▼            ▼
  ┌──────────┐  ┌─────────────────────────────────────────┐
  │ Provider │  │             Environment Manager          │
  │ adapters │  │  local │ docker │ ssh │ modal │ daytona │
  │ (Claude, │  │  vercel │ singularity │ mcp │ http     │
  │  OpenAI, │  └────────────────────┬─────────────────────┘
  │  OpenR.,│                        │
  │  Ollama,│                        ▼
  │   …200+)│             ┌─────────────────────┐
  └──────────┘             │  Sandboxed workers  │
                           │  (capability-scoped │
                           │   per call)         │
                           └─────────────────────┘

  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │     Vaults       │    │   Memory Stores  │    │  ACP Adapter +   │
  │ (OS keychain or  │    │ (FTS5 + vec +    │    │  Registry        │
  │  encrypted file) │    │  Honcho dialect) │    │  (agent ↔ agent) │
  └──────────────────┘    └──────────────────┘    └──────────────────┘

  ┌──────────────────────────────────────────────────────────────────┐
  │                       Storage Plane                                │
  │  SQLite (default) ─ or ─ Postgres ─ + ─ blob (local FS or S3)     │
  │  Session event log: NDJSON files (append-only) per session         │
  │  KB index: SQLite FTS5 + sqlite-vec (or Postgres + pgvector)       │
  └──────────────────────────────────────────────────────────────────┘
```

### 2.2 Process model

Default deployment:
- **One daemon** (`meridiand`) hosts: HTTP API, Session Service, Harness
  Pool (1+ in-process harness workers), Gateway (in-process channel
  drivers), Event Bus, Model Router, Environment Manager, Vault, KB,
  observability exports.
- **Sandboxed workers** spawn per tool/hook call. Local-backend workers
  are warm-pooled subprocesses; container backends call out to Docker /
  Modal / Daytona / Vercel / Singularity / SSH.
- **Skill Forge runs as a separate background worker** with its own
  process and rate-limited model budget.

We pick one daemon for v1 because a laptop is not a cluster, *and* we
preserve the seam: harnesses are already stateless and addressable via
the Session Service, so splitting into a harness pool across hosts is a
config flip, not a refactor.

### 2.3 Plane separation

| Plane | Responsibility | Components |
|-------|----------------|------------|
| Surface | Inbound user contact | Gateway, CLI, channel drivers, webhooks-in |
| Control | CRUD, auth, routing, schema validation | HTTP API, Agent/Session/Skill/... services |
| Orchestration | Phase transitions, sandbox dispatch, hook fan-out, multi-agent, budgets | Harness Pool, Event Bus, Model Router, ACP Adapter |
| Data | Execution, model calls, memory I/O | Sandboxed workers, Environment Manager, model adapters, Memory Service |
| Knowledge | Long-lived state | MemoryStores, KB, Vaults, Files |
| Storage | Durable backing | DB (SQLite/Postgres), event log files, blob store |

Crossing planes goes through narrow, versioned interfaces.

---

## 3. Core Domain Model

### 3.1 IDs

ULIDs throughout: sortable, monotonic per-process, URL-safe.

```ts
type AgentId        = `agent_${string}`;
type AgentVersion   = `agentver_${string}`;     // content hash
type SessionId      = `sess_${string}`;
type ThreadId       = `thr_${string}`;          // sub-resource of Session
type EventSeq       = number;                    // 0-indexed per session
type MessageId      = `msg_${string}`;
type ToolCallId     = `tc_${string}`;
type SkillId        = `skill_${string}`;
type SkillVersion   = `skillver_${string}`;
type EnvironmentId  = `env_${string}`;
type MemoryStoreId  = `mem_${string}`;
type VaultId        = `vault_${string}`;
type UserProfileId  = `usr_${string}`;
type ChannelId      = `chan_${string}`;
type FileId         = `file_${string}`;
type WebhookId      = `wh_${string}`;
```

### 3.2 Entities

```ts
interface Agent {
  id: AgentId;
  current_version: AgentVersion;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
}

interface AgentVersionRecord {
  id: AgentVersion;
  agent_id: AgentId;
  instructions: string;
  model_routing: ModelRoutingPolicy;      // §13
  skills: SkillId[];                       // activated skills
  tools: string[];                         // tool names (resolve via Environment)
  default_environment_id: EnvironmentId;
  capabilities: Capability[];              // §6
  hooks: HookBinding[];                    // §9
  budgets: BudgetConfig;
  memory_store_refs: MemoryStoreId[];
  metadata: Record<string, unknown>;
  created_at: string;
}

interface Session {
  id: SessionId;
  agent_id: AgentId;
  agent_version: AgentVersion;             // pinned
  user_profile_id?: UserProfileId;
  channel_id?: ChannelId;                  // origin channel (if any)
  parent_session_id?: SessionId;           // multi-agent: child of supervisor
  phase: SessionPhase;
  phase_reason?: string;
  metadata: Record<string, unknown>;
  usage: SessionUsage;
  created_at: string;
  updated_at: string;
}

type SessionPhase =
  | 'idle'
  | 'waiting_for_model'
  | 'waiting_for_tool'
  | 'waiting_for_user'
  | 'paused'
  | 'terminated';

interface Thread {
  id: ThreadId;
  session_id: SessionId;
  branch_of_event_seq?: EventSeq;          // forked from this point
  title?: string;
  created_at: string;
}

interface Message {
  id: MessageId;
  session_id: SessionId;
  thread_id: ThreadId;
  role: 'user' | 'assistant' | 'tool' | 'system';
  content: ContentBlock[];
  parent_id?: MessageId;
  created_at: string;
}

interface ToolCall {
  id: ToolCallId;
  session_id: SessionId;
  message_id: MessageId;
  tool_name: string;
  environment_id: EnvironmentId;
  arguments: unknown;
  result?: unknown;
  error?: { code: string; message: string };
  started_at: string;
  ended_at?: string;
}

interface Skill {
  id: SkillId;
  current_version: SkillVersion;
  name: string;
  source: 'authored' | 'forge';            // forge-produced skills tagged
  created_at: string;
}

interface SkillVersionRecord {
  id: SkillVersion;
  skill_id: SkillId;
  // agentskills.io schema
  instructions: string;
  tools: ToolRef[];
  tests: SkillTest[];
  metadata: Record<string, unknown>;
  derived_from_session_ids?: SessionId[];  // forge provenance
  approved_by?: UserProfileId;
  created_at: string;
}

interface Environment {
  id: EnvironmentId;
  name: string;
  backend: 'local' | 'docker' | 'ssh' | 'modal' | 'daytona' | 'vercel' | 'singularity' | 'mcp' | 'http';
  config: Record<string, unknown>;          // backend-specific
  workspace_path?: string;                  // for backends that mount
  env_passthrough: string[];                 // env-var allowlist
  network_policy: NetworkPolicy;
  caps_envelope: ResourceCaps;               // cpu/mem/disk
  default_timeout_ms: number;
}

interface MemoryStore {
  id: MemoryStoreId;
  name: string;
  scope: 'global' | 'user' | 'agent' | 'project';
  backend: 'sqlite-vec' | 'pgvector' | 'http';
  embedder_id?: string;
  config: Record<string, unknown>;
}

interface Vault {
  id: VaultId;
  name: string;
  backend: 'os_keychain' | 'encrypted_file' | 'aws_kms' | 'hcp_vault';
  config: Record<string, unknown>;
}

interface UserProfile {
  id: UserProfileId;
  display_name: string;
  is_primary: boolean;                       // primary user has full host access
  memories: MemoryStoreId[];
  capabilities: Capability[];                // per-user grant ceiling
  channel_pairings: Array<{ channel_id: ChannelId; remote_id: string }>;
}

interface Channel {
  id: ChannelId;
  kind: 'cli' | 'telegram' | 'slack' | 'discord' | 'webhook'
       | 'whatsapp' | 'imessage' | 'signal' | 'matrix' | 'irc' | 'teams';
  config: Record<string, unknown>;          // token_vault_ref etc.
  default_agent_id: AgentId;
  default_user_profile_id?: UserProfileId;
  inbound_policy: 'open' | 'paired_only' | 'quarantine';
  egress_policy: 'enabled' | 'disabled';
}

interface Webhook {
  id: WebhookId;
  name: string;
  url: string;
  secret_ref: string;                        // vault ref for HMAC signing
  event_filter: { types: string[]; session_id?: SessionId };
  max_retries: number;
  backoff: 'exponential' | 'linear';
}
```

### 3.3 Content blocks

```ts
type ContentBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: ToolCallId; name: string; input: unknown }
  | { type: 'tool_result'; tool_use_id: ToolCallId; content: ContentBlock[]; is_error?: boolean }
  | { type: 'image'; source: ImageSource }
  | { type: 'thinking'; signature: string; thinking: string };
```

Wire format matches Anthropic's so the model adapter can forward
unchanged.

### 3.4 Why content-addressed Agents and Skills

A new `AgentVersionRecord` is created any time any field of an agent's
body changes. The ID is the SHA-256 of the canonical JSON body, prefixed
`agentver_`. Sessions and child runs always reference an exact version.

Skills follow the same rule. This is what makes deterministic replay
possible weeks later: the *exact* instructions + tools + skills used
during a session never silently change underneath you.

---

## 4. Session Phases (replaces Run state machine)

The session is the central state object. Its **phase** is a projection
over the event log; phase transitions are derivable, not stored as
truth.

### 4.1 Phase diagram

```
                    create
                      │
                      ▼
                  ┌────────┐
                  │  idle  │ ◀──────────────────────────────────────┐
                  └───┬────┘                                         │
       user message   │                                              │
                      ▼                                              │
            ┌─────────────────────┐                                  │
            │ waiting_for_model   │  ── model_call_complete ─▶       │
            └─────────┬───────────┘   - end_turn       ▶ idle ───────┤
                      │                - tool_use      │              │
                      ▼                                ▼              │
            ┌─────────────────────┐         ┌───────────────────┐     │
            │  waiting_for_tool   │ ──────▶│ waiting_for_model │ ────┘
            └─────────┬───────────┘  tool_  └───────────────────┘
                      │              result
                      │ checkpoint, awaiting user
                      ▼
            ┌─────────────────────┐
            │     paused          │ ── resume ──▶ waiting_for_model / tool
            └─────────┬───────────┘
                      │ terminate (cancel, expire, budget)
                      ▼
            ┌─────────────────────┐
            │    terminated       │  (final; immutable)
            └─────────────────────┘
```

Notes:
- A *run-span* is a contiguous slice of the event log from a user
  message until phase returns to `idle`. Runs are conceptual — there
  is no `Run` table.
- Transitions are issued by the **harness only**.
- Every transition writes a single `session.phase_change` event with
  before/after, timestamp, reason.
- `paused` is reachable from any non-terminated phase. Paused sessions
  are durable; harness restarts auto-resume.
- `terminated` is final and immutable. Reasons:
  `cancelled | failed | expired | budget_exceeded | ended_normally`.

### 4.2 The harness loop

The harness, given a `session_id`:

```
1. wake(session_id):
     - load session record + agent version + active skills
     - tail event log to determine current phase
     - rebuild model context from messages in the most recent thread
2. while phase ∉ {idle, paused, terminated}:
     a. if phase == waiting_for_tool:
          - dispatch pending tool calls via Sandbox + Environment
          - on result(s), append events, transition → waiting_for_model
     b. if phase == waiting_for_model:
          - pre_message hooks (may mutate, veto)
          - Model Router selects provider/model
          - stream completion; emit message.delta events
          - on stop_reason:
              * end_turn → append final message; phase → idle
              * tool_use → for each tool_use: schema-validate args,
                           cap-intersect, pre_tool_call hooks,
                           record tool_call.requested events; phase → waiting_for_tool
              * max_tokens → emit partial; loop if continue allowed
              * error → on_error hooks; phase → terminated unless recoverable
     c. budget check each iteration: hard breach → terminated;
        soft breach → user-facing question, phase → waiting_for_user
3. session is now idle/paused/terminated; harness releases the session
   (any harness can re-wake)
```

The loop is flat. Multi-agent fan-out spawns *child sessions* with their
own loops.

---

## 5. The Event Log (per Session)

### 5.1 Layout

```
$STORAGE_ROOT/events/<YYYY>/<MM>/<DD>/<session_id>.ndjson
```

One event per line. Append-only. `fsync` configurable: every N events
or T milliseconds; defaults 100 events / 100 ms.

```ts
interface SessionEvent {
  seq: EventSeq;                  // monotonic, 0-indexed per session
  ts: string;                     // ISO 8601 ms precision
  thread_id?: ThreadId;
  type: EventType;
  data: Record<string, unknown>;
}

type EventType =
  | 'session.created'
  | 'session.phase_change'
  | 'message.added'
  | 'message.delta'
  | 'tool_call.requested'
  | 'tool_call.dispatched'
  | 'tool_call.result'
  | 'tool_call.error'
  | 'model_call.started'
  | 'model_call.completed'
  | 'hook.invoked'
  | 'hook.verdict'
  | 'usage.delta'
  | 'budget.warning'
  | 'budget.exceeded'
  | 'checkpoint.created'
  | 'child_session.spawned'
  | 'child_session.completed'
  | 'channel.inbound'
  | 'channel.outbound'
  | 'acp.outbound'
  | 'acp.inbound'
  | 'memory.read'
  | 'memory.write'
  | 'error';
```

### 5.2 Why NDJSON per session

Same reasoning as v0.1: cheap appends, trivial streaming, crash-safe
via `O_APPEND` + fsync, replay is `cat | jq`. Difference from v0.1 is
*granularity* — log is per **Session**, not per **Run** (because no
top-level Run exists). This matches Anthropic's beta SDK shape exactly.

### 5.3 Projections

Background indexer reads new events and writes summary rows:

- `sessions(id, phase, last_event_seq, ...)` for listing/filtering.
- `tool_calls(...)` for tool-usage analytics.
- `usage_rollups(session_id, hour, input_tokens, output_tokens, ...)`.
- `message_index(session_id, message_id, role, created_at)`.

Projections trail the log by milliseconds. They never contradict it.

### 5.4 Replay

```
POST /v1/x/sessions/{id}/replay
{
  "model_responses": "fixture:./fixtures/sess_xyz.model.ndjson",
  "tool_responses":  "fixture:./fixtures/sess_xyz.tools.ndjson",
  "into_session_id": "sess_new"
}
```

The harness runs normally; calls to the Model Router and Sandbox return
canned responses from the fixture. Divergence (agent picks a different
tool / output than recorded) aborts the replay with a `divergence` error
pinned to the first deviating event.

---

## 6. Capability System

Unchanged from v0.1 in mechanics; capability list now includes
channel/memory/acp grants.

### 6.1 Base set

```
fs.read[/glob]            fs.write[/glob]           fs.delete[/glob]
net.fetch[host]           net.listen
exec.shell                exec.sudo                 exec.pty
kb.read[scope]            kb.write[scope]
memory.read[memstore_id]  memory.write[memstore_id]
agent.spawn[agent_id]     agent.cancel
secret.read[vault_id/name]
hook.invoke[name]
channel.send[channel_id]  channel.receive[channel_id]
acp.outbound[target]      acp.inbound[target]
```

### 6.2 Intersection check at dispatch

```ts
function authorize(agentCaps: Capability[], required: Capability[], args: unknown): Result {
  for (const req of required) {
    if (!agentCaps.some(g => grantSatisfies(g, req, args))) {
      return err(`missing capability: ${req}`);
    }
  }
  return ok();
}
```

Glob-parameterized grants narrow as expected.

### 6.3 Sub-session capability subsetting

Spawned child sessions inherit a **subset** of parent capabilities,
declared at spawn time. Passing a capability the parent does not hold
fails the spawn. No upward escalation.

---

## 7. Storage and Repository Pattern

### 7.1 Stores

| Data | v1 backend | Added in |
|------|-----------|----------|
| Relational (agents, sessions, threads, messages, skills, environments, memory_stores, vaults, user_profiles, channels, webhooks, tool_calls, projections) | SQLite WAL | Postgres 14+ supported tier in **v1.2** |
| Session event log | NDJSON files (local FS) | NDJSON in S3 multipart in **v1.2** |
| KB / memory vectors | SQLite FTS5 + sqlite-vec | Postgres + pgvector in **v1.2** |
| Blobs (Files, attachments, fixtures) | Local FS | S3-compatible in **v1.2** |
| Vault backing | OS keychain + encrypted file | AWS KMS in **v1.1**; HCP Vault in **v1.2** |

The Repository pattern (§7.2) keeps application code unchanged across
backends. Postgres + S3 lands in **v1.2** per PRD §8.2; the v1 single-
daemon laptop install needs no external services.

### 7.2 Repository interface

Every service talks to storage via a `Repository<T>` interface. SQLite
and Postgres implementations share migration SQL written in the
lowest-common dialect.

```ts
interface SessionRepository {
  create(input: NewSession): Promise<Session>;
  get(id: SessionId): Promise<Session | null>;
  list(filter?: SessionFilter): AsyncIterable<Session>;
  updatePhase(id: SessionId, phase: SessionPhase, reason?: string): Promise<void>;
  appendEvent(id: SessionId, ev: Omit<SessionEvent, 'seq' | 'ts'>): Promise<EventSeq>;
  readEvents(id: SessionId, since?: EventSeq): AsyncIterable<SessionEvent>;
}
```

### 7.3 Migrations

`db/migrations/NNNN_name.sql`. Forward-only. Daemon refuses to start if
schema version on disk exceeds binary's supported version.

---

## 8. HTTP API

### 8.1 Conventions

- Base path `/v1`; extensions under `/v1/x`.
- JSON in/out; SSE for streams.
- Cursor pagination; `Idempotency-Key` on POST.
- Errors:
  ```json
  { "error": { "code": "string", "message": "string", "details": {} } }
  ```

### 8.2 Endpoint inventory

**Agents** (mirror Anthropic):
- `POST /v1/agents` — create (body = first version)
- `GET /v1/agents/{id}` — current view
- `GET /v1/agents/{id}/versions` — list
- `GET /v1/agents/{id}/versions/{ver}` — specific
- `POST /v1/agents/{id}/versions` — new version
- `GET|DELETE /v1/agents/{id}`

**Sessions** (mirror Anthropic):
- `POST /v1/sessions` — create
- `GET /v1/sessions/{id}` — get
- `GET /v1/sessions/{id}/events` — log; `?stream=true` for SSE
- `GET /v1/sessions/{id}/threads` — list threads in session
- `POST /v1/sessions/{id}/threads` — fork a thread at an event seq
- `GET /v1/sessions/{id}/messages` — list (cursor)
- `POST /v1/sessions/{id}/messages` — append (user/system only)
- `POST /v1/sessions/{id}/cancel`
- `POST /v1/sessions/{id}/submit_tool_results` — resume `waiting_for_tool`
- `POST /v1/sessions/{id}/wake` — explicit harness wake (usually implicit)

**Skills** (mirror Anthropic, agentskills.io-compatible):
- `POST /v1/skills` `GET /v1/skills/{id}` `GET /v1/skills/{id}/versions`
  `POST /v1/skills/{id}/versions` `GET /v1/skills/{id}/versions/{ver}`

**Environments**:
- `POST|GET|PATCH|DELETE /v1/environments`

**Memory stores**:
- `POST|GET|DELETE /v1/memory_stores`
- `POST /v1/memory_stores/{id}/write` `POST /v1/memory_stores/{id}/query`

**Vaults**:
- `POST|GET|DELETE /v1/vaults`
- `POST /v1/vaults/{id}/secrets` `GET /v1/vaults/{id}/secrets/{name}/meta`

**User profiles**:
- `POST|GET|PATCH|DELETE /v1/user_profiles`

**Channels** (Meridian extension):
- `POST|GET|PATCH|DELETE /v1/channels`
- `POST /v1/channels/{id}/pair` — pair an external identity
- `POST /v1/channels/{id}/inbound` — channel driver pushes inbound event

**Files / Messages / Webhooks / Models** mirror Anthropic.

**Meridian extensions** under `/v1/x`:
- `POST /v1/x/sessions/{id}/checkpoint` `POST /v1/x/sessions/{id}/resume`
- `POST /v1/x/sessions/{id}/replay`
- `POST /v1/x/sessions/{id}/spawn` — explicit child spawn
- `GET|POST /v1/x/capabilities` — list/define capabilities
- `GET|POST /v1/x/hooks` — register hooks
- `GET|POST /v1/x/acp` — ACP exchanges
- `POST /v1/x/skill_forge/proposals` `POST /v1/x/skill_forge/proposals/{id}/approve`
- `GET /v1/x/kb` `POST /v1/x/kb/query` `POST /v1/x/kb/index`

### 8.3 Auth (v1)

Loopback + Unix socket by default; optional `Authorization: Bearer`
token in config. OIDC/JWT in v1.1.

---

## 9. Hooks

### 9.1 Bindings

```ts
interface HookBinding {
  event: HookEvent;
  name: string;
  handler: HookHandler;             // same kinds as tools (in_process / subprocess / mcp / http / container)
  match?: { tool_name?: string; channel_kind?: string };
  timeoutMs: number;
  failure_mode: 'block' | 'log';
}

type HookEvent =
  | 'session_start' | 'session_end'
  | 'pre_message' | 'post_message'
  | 'pre_tool_call' | 'post_tool_call'
  | 'on_stop' | 'on_compact'
  | 'on_handoff' | 'on_checkpoint'
  | 'on_error'
  | 'on_channel_inbound' | 'on_channel_outbound'
  | 'on_model_call';
```

### 9.2 Verdict

```ts
type HookVerdict =
  | { action: 'continue' }
  | { action: 'continue'; mutate?: { args?: unknown; messages?: Message[] } }
  | { action: 'veto'; reason: string }        // pre_* only
  | { action: 'fail'; reason: string };
```

Same isolation model as tools. Same dispatch surface (`execute(name, input)
→ result`).

---

## 10. Multi-Agent + ACP

### 10.1 Patterns

- **Fan-out.** Parent calls `parallel_runs` built-in; harness spawns N
  child sessions, awaits all. Each child has its own log, own budget
  slice, own capability subset.
- **Handoff.** Parent calls `spawn_and_await` with `output_schema`;
  child's terminal message must validate or the child is given one
  retry (`waiting_for_user`).
- **Streamed conversation.** Parent and child share a session/thread;
  the child appends messages and the parent reads them as they arrive.

### 10.2 ACP Adapter

For cross-system agent communication (Meridian ↔ hermes ↔ openclaw ↔ other
Meridian installs), Meridian implements the ACP protocol via:

```ts
interface AcpAdapter {
  outbound(target: AcpTarget, message: AcpMessage): Promise<AcpResponse>;
  // inbound handled at /v1/x/acp/inbound; registered targets are authoritative.
}
```

Outbound calls record `acp.outbound` events; inbound calls record
`acp.inbound` events. Capabilities `acp.outbound[target]` and
`acp.inbound[target]` gate which targets are reachable.

### 10.3 Budget aggregation

Parent's hard budget overflow cancels all descendants synchronously.
Cost accounting reads from `usage.delta` events; price book is a JSON
per-provider config.

---

## 11. Sandbox Dispatch (unified)

The Sandbox is the **single dispatch surface** for every executable
action. Anthropic's `execute(name, input) → result` shape, generalized
to any backend.

### 11.1 Handler kinds

```ts
type ToolHandler =
  | { kind: 'in_process'; module: string }
  | { kind: 'subprocess'; path: string }
  | { kind: 'mcp'; server_url: string; tool_name: string }
  | { kind: 'http'; url: string; auth?: AuthConfig }
  | { kind: 'container'; environment_id: EnvironmentId; entrypoint: string }
  | { kind: 'wasm'; module: string };           // future
```

Each tool definition picks a kind. The harness doesn't branch — it asks
the Sandbox to `execute(name, input)` and the Sandbox routes to the
right backend.

### 11.2 Subprocess / container protocol

```
stdin  → { "args": ..., "context": { "workspace": "...", "session_id": "...", "thread_id": "...", "scratch_dir": "...", ... } }
stdout ← { "result": ... } | { "error": { "code": "...", "message": "..." } }
stderr → captured, truncated to 64KB, attached to tool_call.result event
```

### 11.3 MCP integration

MCP servers register as a tool source. The Sandbox proxies tool calls
into the MCP server using the MCP protocol; results are normalized to
the Sandbox's `execute` return shape. Capability scoping still applies:
an MCP tool that needs `net.fetch` still goes through the intersection
check.

### 11.4 Failure handling

- Schema/cap failure → synthetic `tool_result` with `is_error: true`
  back to the model.
- Subprocess timeout → SIGTERM, SIGKILL after 2s grace.
- Subprocess crash → error result with stderr tail.
- Container backend transient errors → router-style retry.

In **none** of these cases does the session phase transition to
terminated. The agent decides what to do.

### 11.5 Tool-author contract

Every tool implementation **must satisfy** the following contract.

#### Idempotency on retry

If the caller supplies an `idempotency_key` in `ToolContext`, the SDK
guarantees that a second invocation with the same `(tool_name,
idempotency_key)` pair returns the cached first result **without
re-executing the handler**:

```python
ctx = ToolContext(
    workspace="/workspace",
    session_id="sess_abc",
    idempotency_key="order-42-ship",   # stable key chosen by caller
)
result1 = await ship_order.execute(args, ctx)
result2 = await ship_order.execute(args, ctx)  # handler does NOT run again
assert result1 == result2
```

Key semantics:

- The cache is keyed by `(tool_name, idempotency_key)`.  Two different
  tools may share the same key without conflict.
- Both **success and failure** results are cached; a retry after a
  handler crash replays the error rather than re-executing.
- **Input-validation failures are not cached** — they are caller-side
  errors (wrong payload) and should be fixed before retrying.
- The cache is in-process and lives for the lifetime of the Sandbox
  worker.  Long-lived retries across process restarts rely on the
  session event-log replay path (§5.4).

#### Error surface

All failures — schema violations, capability denials, handler
exceptions — are returned as `ToolResult(is_error=True,
error=ToolError(code, message, details))`.  Handlers **must never
raise** — catch every exception, translate it into a `ToolResult.err`,
and let the agent decide whether to retry or escalate.

Every failure is also written to the audit log (`~/.meridian/audit.ndjson`
or the path in `MERIDIAN_AUDIT_LOG`) so operators can inspect what went
wrong without live OTel infrastructure (§22.4).  The audit record
includes `idempotency_key` when present so retries can be correlated.

#### Subprocess / HTTP handler contract

Out-of-process handlers must honour the same contract via the
stdin/stdout JSON protocol (§11.2):

```
stdout ← { "result": ... }           # success
        | { "error": { "code": "...", "message": "..." } }  # failure
```

The Sandbox normalises both shapes into `ToolResult` before returning
to the execution pipeline.

---

## 12. Environment Manager

### 12.1 Backend matrix

| Backend | Ships in | Use case |
|---------|----------|----------|
| `local` | **v1** | Default; fast; warm subprocess pool |
| `docker` | **v1** | Reproducible; image-pinned |
| `ssh` | **v1** | Remote dev hosts |
| `mcp` | **v1** | Any MCP server |
| `http` | **v1** | Arbitrary HTTP tool service |
| `modal` | **v1.1** | Serverless burst |
| `daytona` | **v1.1** | Cloud dev environments |
| `vercel` | **v1.2** | Edge functions |
| `singularity` | **v1.2** | HPC clusters |

The **Environment Backend SDK** itself is a v1 feature (see §15.5);
adding a backend in v1.1 or v1.2 means writing a plugin against the
SDK, not forking Meridian.

### 12.2 Lifecycle

The Environment Manager provisions on first use, warms a pool, and
reclaims idle workers. Container/serverless backends provision **on
demand at tool-call time** (Anthropic's pattern; ~60% p50 / 90% p95
TTFT improvement vs. cold-create-per-session).

### 12.3 Conformance suite

Every backend MUST pass a contract test suite: schema-valid round-trip
for a reference set of inputs, capability enforcement, timeout
behavior, scratch directory isolation, env-var scoping, network policy
honored.

---

## 13. Model Router

### 13.1 Provider adapters

A `ModelProvider` adapter is a Python class implementing:

```python
from typing import AsyncIterator, Protocol
from pydantic import BaseModel

class ModelProvider(Protocol):
    name: str                                       # 'anthropic-oauth', 'anthropic-api', 'openai', 'openrouter', 'ollama', …
    kind: str                                       # 'anthropic', 'openai', 'openrouter', 'ollama', …
    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]: ...
    async def count_tokens(self, req: ModelCountReq) -> TokenCount: ...
    async def close(self) -> None: ...
```

Providers are instantiated from the YAML config (§13.6); a `name` is a
config-level identifier (multiple instances of the same `kind` are
permitted — e.g. `anthropic-oauth` and `anthropic-api` are two
instances of `kind: anthropic`).

**v1** ships four provider *kinds* (with five typical config-level
instances): Anthropic in both `api_key` and `oauth` modes (§13.4),
OpenAI, OpenRouter (which by itself unlocks many models on one
adapter), Ollama (local). **v1.1** broadens to ≥ 10 commonly-used
kinds (Google, Together, NVIDIA NIM, HuggingFace, …). **v1.2** opens
the **Model Provider SDK** (§15.5) as a plugin surface for the long
tail toward Hermes's 200+. The Model Router *feature* (declarative
rules, failover, per-call logging) ships in full at v1; what scales
over v1.x is the count of supported provider kinds.

### 13.2 Routing policy

```ts
interface ModelRoutingPolicy {
  rules: ModelRoutingRule[];
  fallbacks?: Array<{ on: 'rate_limit' | 'timeout' | '5xx' | 'any'; model: ModelRef }>;
}

interface ModelRoutingRule {
  when?: {
    skill_id?: SkillId;
    estimated_input_tokens?: { gt?: number; lte?: number };
    metadata_match?: Record<string, unknown>;
    role?: 'planner' | 'worker' | 'reviewer' | string;
  };
  model: ModelRef;                        // 'anthropic:claude-opus-4-7', 'ollama:qwen2-coder', …
}
```

### 13.3 Logging

Every model call records the routing rule that fired and the chosen
provider/model in the event log (`model_call.started` event). This is
auditable and replay-stable.

### 13.4 The Anthropic provider — two-mode plugin architecture

The Anthropic provider ships as **two separate plugin packages**, both
implementing `ModelProvider` (§13.1). Users enable either or both via
`~/.meridian/config.yml` (§13.6). The trigger for this two-mode design
is economic: OAuth (Claude Pro/Max subscription) billing is dramatically
cheaper than API-key billing at dev-loop usage levels — flat ~$100–200
per month vs. per-token fees that compound to $50–200 per day. Forcing
API-key-only would price Meridian out of the P1 persona's workflow.

**Mode 1 — `api_key`** (package `meridian-provider-anthropic-apikey`):
uses the raw `anthropic` Python SDK. Auth via Vault'd API key. Direct
HTTPS to `api.anthropic.com`. Per-token billing. Full feature surface
(all models, all tiers, all rate-limit ceilings).

```python
class AnthropicApiKeyProvider:
    name = "anthropic-api"; kind = "anthropic"; mode = "api_key"
    def __init__(self, auth: SecretRef, default_model: str | None = None):
        self._client = anthropic.AsyncAnthropic(api_key=resolve(auth))
    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        async with self._client.messages.stream(
            model=opts.model, messages=opts.messages, tools=opts.tools,
            system=opts.system, extra_headers=opts.cache_headers,
        ) as stream:
            async for ev in stream:
                yield translate_to_meridian_event(ev)
```

**Mode 2 — `oauth`** (package `meridian-provider-anthropic-oauth`):
uses `claude-agent-sdk-python`, which drives the **Claude Code CLI as
a subprocess**. Auth via OAuth token managed by `claude login` (stored
in the user's OS keychain by Claude Code itself; Meridian never sees
the raw token). Subscription billing. Feature surface = what the CLI
exposes at the current subscription tier.

The architectural cost of `oauth` mode is that **Claude Code is itself
a full agent runtime** (sessions, tools, hooks, MCP, prompt cache,
thinking blocks). We do not want two harnesses competing. The plugin
bridges Claude Code's inner loop back into Meridian's plane via four
explicit contracts:

```python
class AnthropicOAuthProvider:
    name = "anthropic-oauth"; kind = "anthropic"; mode = "oauth"
    def __init__(self, cli_path: str | None = None, sandbox: Sandbox = ...):
        self._sandbox = sandbox
        self._options = ClaudeAgentOptions(
            cli_path=cli_path,
            disallowed_tools=ALL_CLAUDE_CODE_BUILTIN_TOOLS,           # (1)
            mcp_servers=[self._meridian_tool_bridge_server()],         # (2)
            hooks={
                "PreToolUse": [HookMatcher(matcher="*",
                                           hooks=[self._meridian_cap_check])],  # (3)
            },
        )
```

**Contract 1 — Disable Claude Code's built-in tools.** Claude Code's
own `Read` / `Write` / `Bash` / `Edit` are placed in
`disallowed_tools`. The only tools the inner loop can call are those
Meridian exposes via the MCP bridge below. This is what preserves
Meridian's capability boundary (§6) across the bridge — there is no
path for the inner runtime to bypass Meridian's Sandbox.

**Contract 2 — Bridge Meridian tools as in-process MCP tools.** Every
tool the Agent has access to is exposed to Claude Code via
`create_sdk_mcp_server(tools=[...])`. Each `@tool`-decorated function
forwards the call back into Meridian's Sandbox dispatcher (§11.1):

```python
def _meridian_tool_bridge_server(self):
    @tool("meridian_tool_proxy", "Forwards to Meridian Sandbox",
          {"name": str, "input": dict})
    async def proxy(args):
        # Routes back through Meridian's Sandbox: cap check, schema
        # validation, hooks, audit, capability intersection — all of it.
        result = await self._sandbox.execute(args["name"], args["input"], self._ctx)
        return {"content": result.to_mcp_content_blocks()}
    return create_sdk_mcp_server(name="meridian", version="1.0.0", tools=[proxy])
```

**Contract 3 — Map Claude Code SDK hooks to Meridian lifecycle
hooks.** `PreToolUse` → Meridian's `pre_tool_call` (§9). The verdict
protocol (§9.2: `continue` / `mutate` / `veto`) is translated across
the bridge.

**Contract 4 — Translate Claude Code events into Meridian's session
event log.** Token deltas, tool-use blocks, thinking blocks, message
completions emitted by Claude Code are translated into Meridian
`SessionEvent` rows (§5.1) on the way out. Claude Code's own inner
session is ephemeral from Meridian's view; Meridian's Session is the
source of truth.

#### 13.4.1 What this preserves

Despite the bridged-runtime shape, all load-bearing properties hold:

| Property | How `oauth` mode preserves it |
|----------|-------------------------------|
| Capability sandboxing (§6) | Contract 1 (disallow built-ins) + Contract 2 (Sandbox proxy) |
| Event-log-as-truth (§5) | Contract 4 (translate to Meridian events) |
| Hooks with veto (§9) | Contract 3 (SDK → Meridian hook mapping) |
| Credential boundary (§18) | OAuth token lives in Claude Code's process; Meridian never reads it |
| Deterministic replay (§5.4) | Contract 4 events are replayable; Claude Code's internal state is regenerable from them |
| Model-Router polymorphism (§13.1) | Both modes implement `ModelProvider`; harness unchanged |

#### 13.4.2 What this costs

- **CLI version pin.** Each Meridian release pins to a Claude Code CLI
  version (recorded in `meridian.lock`). Upgrades are infra-track.
- **Subprocess management.** The plugin owns Claude Code's process
  lifecycle (spawn, health-check, restart on hang, kill on cancel).
- **Feature ceiling.** Bounded by subscription-tier rate limits and
  model selection. Heavy workflows fall through Router fallback rules
  to `api_key` mode automatically.
- **Bridge engineering.** ~1.5 weeks one-time + ongoing maintenance as
  `claude-agent-sdk-python` evolves.

#### 13.4.3 Both modes coexist; the user picks per call

The YAML config (§13.6) declares both providers; the Model Router
picks per call:

```yaml
providers:
  - name: anthropic-oauth
    kind: anthropic
    mode: oauth
  - name: anthropic-api
    kind: anthropic
    mode: api_key
    auth: secret_ref://vault/default/anthropic_api_key

routing:
  default:
    rules:
      - when: { estimated_input_tokens: { gt: 100000 } }
        model: anthropic-api:claude-opus-4-7   # API for high context
      - when: { role: "worker" }
        model: anthropic-oauth:claude-opus-4-7 # subscription covers it
    fallbacks:
      - on: rate_limit
        model: anthropic-api:claude-opus-4-7   # fall back on cap hit
```

#### 13.4.4 Patterns explicitly rejected (carried forward from v0.3)

**Rejected: claude-agent-sdk as the global harness.** Meridian's
harness (§4.2) is provider-polymorphic, capability-enforcing,
event-log-emitting, phase-driven, session-resumable. claude-agent-sdk
drives a single CLI subprocess. Making it the global harness forfeits
model agnosticism, ACP, capability intersection, paused/wake
semantics, and cross-provider budget aggregation.

**Rejected: Anthropic's managed-agents API as Meridian's backing
store.** Sessions, Skills, Vaults, MemoryStores live locally. Meridian
**mirrors** the managed-agents API shape (PRD §11) for portability; it
**does not delegate** to it. Local-first (PRD §6.4), capability
sandboxing (PRD G8), deterministic replay (PRD G9), and data residency
all depend on this.

### 13.5 Provider isolation contract

Because of §13.4, every provider adapter MUST be isolatable behind the
`ModelProvider` interface (§13.1). Specifically:

- No provider adapter calls into the Session Service, the event log
  writer, the Sandbox, or the Vault directly. It receives a
  `ModelCallOpts` and yields `ModelEvent` values. The one exception
  is the Sandbox-proxy MCP bridge in `oauth` mode (§13.4 Contract 2),
  which crosses *into* the Sandbox (the correct direction) — it does
  not reach across to peers.
- Cross-cutting concerns (cache, retries, budgets, logging, capability
  intersection on the harness side) live in the **harness wrapper
  around** the adapter, not inside the adapter.
- An adapter can use any client library it likes internally
  (`anthropic`, `claude-agent-sdk-python`, `openai`, `ollama`, …)
  without leaking that choice past its `ModelProvider` interface.

This is what makes the two-mode design tenable: each plugin is welded
shut around its choice of underlying SDK. Enforcement: §27.2 Rule L1.

### 13.6 YAML configuration (`~/.meridian/config.yml`)

The single declarative source of truth for providers, routing,
storage, vaults, and daemon settings. Validated by a Pydantic
`MeridianConfig` model at load and on every reload.

#### 13.6.1 Locations (first match wins)

1. `$MERIDIAN_CONFIG` if set.
2. `~/.meridian/config.yml` (single-user default).
3. `/etc/meridian/config.yml` (system-wide).

#### 13.6.2 Shape

```yaml
version: 1

providers:
  - name: anthropic-oauth
    kind: anthropic
    mode: oauth
    cli_path: null                        # auto-detect from PATH / bundled

  - name: anthropic-api
    kind: anthropic
    mode: api_key
    auth: secret_ref://vault/default/anthropic_api_key

  - name: openai
    kind: openai
    auth: secret_ref://vault/default/openai_api_key

  - name: openrouter
    kind: openrouter
    auth: secret_ref://vault/default/openrouter_api_key

  - name: local-ollama
    kind: ollama
    base_url: http://localhost:11434

routing:
  default:
    rules:
      - when: { role: planner }
        model: anthropic-oauth:claude-opus-4-7
      - when: { role: worker, estimated_input_tokens: { lt: 8000 } }
        model: local-ollama:qwen2.5-coder
      - when: { estimated_input_tokens: { gt: 100000 } }
        model: anthropic-api:claude-opus-4-7
    fallbacks:
      - on: rate_limit
        model: openrouter:claude-opus-4-7

vaults:
  - id: default
    backend: os_keychain

daemon:
  bind: unix:///Users/me/.meridian/meridiand.sock
  workspace_root: /Users/me/code
  log_level: info

storage:
  database: sqlite:///Users/me/.meridian/meridian.db
  event_log: file:///Users/me/.meridian/events/
  blob_store: file:///Users/me/.meridian/blobs/
```

#### 13.6.3 Credential indirection

`auth: secret_ref://vault/{vault_id}/{key}` is resolved at first use,
not at config load — so secrets can be rotated in the Vault without a
daemon restart.

#### 13.6.4 Plaintext credentials

Allowed (`auth: sk-ant-…`) but the daemon logs a
`config.plaintext_secret` warning per occurrence on every startup.
Rationale: dev ergonomics on day 1; security-by-default by day 30. CI
templates lint for plaintext.

#### 13.6.5 Hot reload

`POST /v1/x/config/reload` and `SIGHUP` both trigger
**validate-then-atomic-swap**:

```
1. Read config from disk.
2. Parse + Pydantic-validate. On error: log, do not swap, return error.
3. Diff old vs new. For each changed provider:
     - construct the new instance (this can fail per-provider).
     - if any new instance fails: log, do not swap, return error.
4. Atomic swap: replace the provider-registry pointer in one move.
5. Drain old provider instances (in-flight calls complete on old
   instance; new calls hit new instance).
6. close() old instances after drain.
```

There is no partial reload. Either the whole new config takes effect
or the whole old config remains. This is what makes config evolution
boring.

#### 13.6.6 Schema versioning + editor support

`version: 1` is required. Forward-only migrations via
`meridian config migrate`. Each version bump ships with an idempotent
migration script.

`meridian config schema > schema.json` emits a JSON Schema generated
from the `MeridianConfig` Pydantic model. Drop into
`.vscode/settings.json` or your YAML language server for autocomplete
+ validation while editing.

#### 13.6.7 What is NOT in the YAML

- **Agents, Skills, Sessions, MemoryStores** — managed via HTTP API
  with their own lifecycle. The config file is for the daemon's
  configuration, not user data.
- **Capability declarations on agents** — live on
  `AgentVersionRecord` (§3.2), versioned independently.
- **Tool definitions** — owned by their hosting Environment / MCP
  server.

The boundary: anything the daemon *boots with* is in the YAML;
anything the daemon *runs against* is in the database + event log.

---

## 14. Skill Forge

### 14.1 What it does

Background process; consumes session trajectories; produces **proposed
skills** for human approval.

### 14.2 Pipeline

```
1. Watch terminated sessions; pull the event log + tool-call summary.
2. Cluster trajectories by structural similarity (tool call sequence,
   pre/post conditions, file scopes touched).
3. For each cluster: ask a model to extract:
     - skill name + description
     - distilled instructions (general, not session-specific)
     - tool list + capability requirements
     - candidate test cases (replayable)
4. Build an agentskills.io-shaped SkillVersionRecord; mark
   source='forge', derived_from_session_ids=[...].
5. Store as PROPOSAL. Notify the primary user.
6. User reviews → approve | reject | request edits.
7. On approve: promote to active; recompute content hash; available to
   agents that opt-in.
```

### 14.3 Safety gates

- Forge runs with **its own** budget; can be throttled or disabled.
- Forge model calls go through Model Router same as anything else.
- Proposals are **quarantined**; no skill auto-activates.
- Skill activation per Agent is explicit.
- All forge proposals are reviewable and reversible.

### 14.4 Quality metrics

- **Forge precision**: proportion of proposals a user approves.
- **Skill efficacy**: A/B trajectory metric (with vs. without skill) on
  proposal test cases.
- Target: 50% precision in v1, 75% in v1.1.

---

## 15. Channels and Gateway

### 15.1 Channel Drivers

Each channel kind has a driver implementing:

```ts
interface ChannelDriver {
  kind: ChannelKind;
  start(channel: Channel): Promise<void>;          // bind to platform (long poll, webhook URL, etc.)
  send(channel: Channel, session: Session, content: ContentBlock[]): Promise<void>;
  stop(channel: Channel): Promise<void>;
}
```

**v1** ships drivers: `cli`, `telegram`, `slack`, `discord`, `webhook`.
**v1.1** adds: `whatsapp`, `imessage` (macOS bridge), `signal`.
**v1.2** adds: `matrix`, `irc`, `teams`.

The **Channel Driver SDK** (§15.5) is itself a v1 feature, so third
parties can land a channel in v1 without waiting on the first-party
roadmap. The five v1 channels are the minimum set that exercises every
gateway primitive: synchronous TTY (CLI), bot-token chat (Telegram,
Discord), workspace chat with rich blocks (Slack), and inbound HTTP
(webhook).

### 15.2 Pairing

Inbound from a new remote_id on a channel:
1. Check `channel.inbound_policy`:
   - `open` → resolve/create UserProfile by remote_id; create Session.
   - `paired_only` → require existing pairing; otherwise reject.
   - `quarantine` → drop into a quarantine UserProfile with minimal
     capabilities; primary user receives an audit event.
2. Pairing tokens (v1) bind a remote_id to a UserProfile out-of-band.

### 15.3 Cross-channel session

A Session may be reached from multiple channels concurrently (CLI +
Telegram on the same Session). Outbound: the harness fans out to all
attached channels whose `egress_policy = enabled`.

### 15.4 Untrusted inbound

Per OpenClaw's security default: inbound DMs from non-paired senders
are **untrusted**. Untrusted sessions:
- Run in a quarantine Environment by default.
- Cannot grant capabilities beyond a minimal envelope (`fs.read` on a
  dedicated sandbox dir, no `exec.*`, no `net.fetch`).
- Auto-expire after N minutes of silence.

### 15.5 Plugin SDKs (v1 feature)

Plugin SDKs ship in v1 and are load-bearing for the PRD v0.3 §8.2
infrastructure track: the *count* of backends grows across v1.1 /
v1.2, but the *contracts* land in v1 so third parties can ship
backends in parallel with first-party additions.

**Daemon-side plugins are Python.** Channel Drivers, Environment
Backends, and Model Providers run inside or alongside the daemon —
they're Python packages (PyPI distribution or local pip install).

**UI-side plugins are TypeScript.** Live Canvas widgets, Web UI
extensions are TS packages (npm distribution).

| SDK | Package | Language | Purpose |
|-----|---------|----------|---------|
| Channel Driver SDK | `meridian-sdk-channel` | Python | Implement `ChannelDriver` (§15.1), declare manifest, register via `meridian channel install ./pkg`. Driver runs in-daemon (trusted) or out-of-process (sandboxed) per manifest. |
| Environment Backend SDK | `meridian-sdk-environment` | Python | Implement the Environment contract (provision, execute, reclaim, network policy, capability envelope). MUST pass the environment conformance suite (§12.3). |
| Model Provider SDK | `meridian-sdk-provider` | Python | Implement `ModelProvider` (§13.1) + optional `count_tokens`, `cache_control`, `streaming`, `thinking` capability hints. Provider declares its feature flags; Router honors them. |
| Tool Plugin SDK | `meridian-sdk-tool` | Python or any | In-process Python tools or out-of-process subprocess/HTTP/MCP tools. Schema-validated per §11. |
| UI Widget SDK | `@meridian/sdk-widget` | TypeScript | Live Canvas widgets, Web UI extensions. Renders content_block.canvas_op events. |

All daemon-side SDKs share the same plugin-manifest shape, install
flow, and capability-by-intersection enforcement (so a Channel Driver
can't escalate, an Environment Backend can't peek at vault secrets it
isn't granted).

---

## 16. Memory Stores

### 16.1 Distinction from Sessions

- **Session event log**: recent, append-only, decays via compaction/archive.
- **MemoryStore**: persistent facts ("user prefers PR descriptions with
  test plans", "spouse name", "preferred timezone"). Survives session
  archival.

### 16.2 Dialectic write (Honcho-style)

When a write arrives:
1. Retrieve top-K existing memories most similar to the new fact.
2. Run a small model to **reconcile**: is the new fact a duplicate, a
   refinement, a contradiction, or net-new?
3. Apply outcome: dedupe / merge / supersede (with provenance edge) /
   insert.
4. Reconciliations are recorded as `memory.write` events with the
   action taken.

### 16.3 Hybrid retrieval

BM25 (FTS5) + vector + scope filter, fused via reciprocal-rank-fusion.
Tunable weights per query. Same retrieval shape as KB §17.

### 16.4 Prompt expansion

Agent instructions and skills can reference memories via templated
fields:

```
{{ memory.user.preferences.commit_style }}
```

The harness expands these at run start (or per turn for short-TTL
memories).

---

## 17. Knowledge Base

Same as v0.1: workspace indexer (chokidar/fsevents), tree-sitter for
code, BM25 + vector + glob hybrid, scoped queries. Now exposed as a
`kb_search` tool with capability `kb.read[scope]`. Distinction from
MemoryStore: KB indexes the workspace and arbitrary documents; Memory
holds curated facts.

---

## 18. Vaults and Credential Proxy

### 18.1 Vault backends

| Backend | Ships in | Use case |
|---------|----------|----------|
| `os_keychain` | **v1** | Default on workstations (macOS Keychain / Windows Credential Manager / libsecret) |
| `encrypted_file` | **v1** | Headless servers; age/sops-encrypted file unlocked at daemon start |
| `aws_kms` | **v1.1** | Team-shared cloud deployments |
| `hcp_vault` | **v1.2** | Enterprise; rotation, leasing, audit native |

The Vault **interface** is a v1 feature; backend backends phase per
PRD v0.3 §8.2 and D9.

### 18.2 The credential boundary (load-bearing)

**Sandboxed tool code MUST NEVER receive raw secrets.** Two patterns:

- **Inline ref substitution.** Tool args carry `secret_ref://vault/name`.
  At dispatch, the harness substitutes the value *into the args
  payload sent to the worker*. The event log retains the ref form. This
  is acceptable for local-only tools where stdin is private.
- **In-harness proxy.** For network tools needing OAuth: the worker
  calls a proxy URL on the harness loopback. The harness injects the
  token at the outbound HTTP request. The worker never sees the token
  even on stdin.

For high-sensitivity secrets, the proxy pattern is mandatory. For
low-sensitivity, ref substitution is acceptable; the capability system
gates `secret.read[name]` per agent.

### 18.3 Audit

Every secret access (read or substitute) emits an `audit.secret_access`
event with vault_id, name, requester agent_id, requester tool_call_id.

---

## 19. Streaming

### 19.1 SSE protocol

```
GET /v1/sessions/{id}/events?stream=true
Accept: text/event-stream
```

Events:
```
event: <type>
id: <seq>
data: <json>
```

Resumption via `Last-Event-ID: <seq>`.

### 19.2 Backpressure

Bounded in-process channels (default 1024 events per session).
Overflow drops *live SSE subscribers* with a `subscriber_lagged`
event; the disk event log is the durable record. Harness never blocks.

---

## 20. Checkpoints and Resume

### 20.1 Snapshot

```ts
interface SessionCheckpoint {
  session_id: SessionId;
  seq: EventSeq;
  phase: SessionPhase;
  pending_tool_calls: ToolCallId[];
  message_tail: Message[];                // recent N messages for warm replay
  usage: SessionUsage;
  taken_at: string;
}
```

Stored as `$STORAGE_ROOT/checkpoints/<session_id>/<seq>.json`. Latest
pointed by `latest.json` (atomic rename).

### 20.2 Resume

`POST /v1/x/sessions/{id}/resume`:
1. Load latest checkpoint (fallback: replay log).
2. Re-dispatch any tool calls whose result is missing.
3. Transition phase appropriately.
4. Wake harness; continue.

Tool authors are guided to make tool implementations idempotent on
retry; the contract is documented.

---

## 21. Observability

### 21.1 Tracing

One OTel trace per session. Spans:
- `session` (root)
- `harness.wake`
- `session.run_span` (between user message and idle)
- `model.call`
- `tool.call`
- `hook.call`
- `skill_forge.proposal`
- `acp.outbound` / `acp.inbound`
- `child_session` (span link)

### 21.2 Metrics

`/metrics` endpoint exposes Prometheus metrics:

- `meridian_sessions_total{phase}`
- `meridian_session_duration_seconds{result}` (histogram)
- `meridian_tool_calls_total{tool, backend, result}`
- `meridian_tool_call_duration_seconds{tool, backend}` (histogram)
- `meridian_model_tokens_total{provider, model, kind}`
- `meridian_model_call_duration_seconds{provider, model}` (histogram)
- `meridian_hook_invocations_total{event, verdict}`
- `meridian_channel_inbound_total{channel_kind}`
- `meridian_channel_outbound_total{channel_kind}`
- `meridian_active_sessions{phase}`
- `meridian_harness_wakes_total`
- `meridian_skill_forge_proposals_total{outcome}`
- `meridian_vault_accesses_total{vault_id}`

### 21.3 Logs

JSON to stderr; fields include `ts`, `level`, `component`, `session_id?`,
`agent_id?`, `tool_name?`, `provider?`, `msg`. Application logs are
distinct from the event log; events are domain truth, logs are
diagnostic.

---

## 22. Security Model

### 22.1 Threat model

Protect against:
- Agent escapes (cannot break out of sandbox).
- Tool misuse (cannot exceed declared capabilities).
- Secret leakage (Vault § 18).
- Channel-borne prompt injection (untrusted inbound; § 15.4).
- Crash → corruption (event log append-only; checkpoint atomic).

Do not protect against:
- Malicious tool installation (requires user consent + capability
  allowlist).
- API key theft from compromised host (use OS keychain).
- Cross-tenant attacks (single-tenant in v1).

### 22.2 Filesystem jail

`fs.*` capabilities take glob parameters. Workspace root `$WORKSPACE`
configured at daemon start; tool-args paths canonicalized and matched
against the glob. Symlinks outside the jail rejected. Container/SSH
backends mount only the workspace.

### 22.3 Network policy

Default-deny per Environment. Allowlists per agent. Proxy enforces
host allowlist + outbound logging.

### 22.4 Audit log

Append-only `audit.ndjson` stream covering capability decisions, vault
accesses, channel pairings, skill promotions, environment changes.
Signed via Ed25519 (optional v1, mandatory v1.1 for multi-user).

---

## 23. Repository Layout

Polyglot monorepo: Python daemon, TypeScript UI, OpenAPI codegen
bridging. Orchestrated by `uv` for Python workspaces and `pnpm` (or
`bun`) for TypeScript workspaces. Top-level `Makefile` (or `just`)
wraps the cross-language tasks.

```
meridian/
├── apps/
│   ├── meridiand/                       # daemon (Python 3.11+)
│   │   ├── src/meridiand/
│   │   │   ├── __init__.py
│   │   │   ├── __main__.py              # `python -m meridiand`
│   │   │   ├── http/                    # FastAPI app, routes
│   │   │   ├── gateway/                 # Channel drivers (in-daemon glue)
│   │   │   ├── harness/                 # inference loop (stateless)
│   │   │   ├── sessions/                # service + event log writer
│   │   │   ├── sandbox/                 # dispatch + worker pool
│   │   │   ├── environments/            # local/docker/ssh impls
│   │   │   ├── model_router/
│   │   │   ├── providers/               # built-in provider plugins
│   │   │   │   ├── anthropic_apikey/    # mode=api_key (§13.4)
│   │   │   │   ├── anthropic_oauth/     # mode=oauth (§13.4)
│   │   │   │   ├── openai/
│   │   │   │   ├── openrouter/
│   │   │   │   └── ollama/
│   │   │   ├── skill_forge/             # in-process forge pipeline
│   │   │   ├── memory/                  # MemoryStores + KB
│   │   │   ├── vault/
│   │   │   ├── hooks/
│   │   │   ├── acp/                     # ACP adapter (Hermes-compatible)
│   │   │   ├── webhooks/
│   │   │   ├── storage/                 # repositories + migrations
│   │   │   ├── events/                  # bus + log writer + projections
│   │   │   ├── streaming/               # SSE
│   │   │   ├── observability/           # OTel + Prom exporters
│   │   │   └── config/                  # MeridianConfig (Pydantic) + YAML loader
│   │   ├── tests/
│   │   ├── pyproject.toml               # uv-managed
│   │   └── README.md
│   ├── meridian-cli/                    # CLI client (Python, thin)
│   │   ├── src/meridian_cli/
│   │   └── pyproject.toml
│   └── meridian-ui/                     # Web UI + Live Canvas (TypeScript)
│       ├── src/
│       │   ├── main.tsx
│       │   ├── routes/
│       │   ├── components/
│       │   ├── live-canvas/
│       │   └── api/                     # generated client (see packages/sdk-ts)
│       ├── package.json
│       └── vite.config.ts
├── packages/
│   ├── schemas/                         # OpenAPI source of truth (YAML)
│   │   ├── openapi.yaml                 # generated by FastAPI app export
│   │   ├── agentskills.schema.json
│   │   ├── meridian-config.schema.json  # emitted by `meridian config schema`
│   │   └── README.md
│   ├── sdk-py/                          # Python client SDK
│   │   ├── src/meridian_sdk/            # generated from packages/schemas/openapi.yaml
│   │   └── pyproject.toml
│   ├── sdk-ts/                          # TypeScript client SDK + types
│   │   ├── src/                         # generated by openapi-typescript
│   │   └── package.json
│   ├── sdk-channel/                     # Channel Driver SDK (Python)
│   ├── sdk-environment/                 # Environment Backend SDK (Python)
│   ├── sdk-provider/                    # Model Provider SDK (Python)
│   ├── sdk-tool/                        # Tool Plugin SDK (Python)
│   └── sdk-widget/                      # Live Canvas Widget SDK (TypeScript)
├── plugins/                             # First-party plugin packages
│   ├── channel-telegram/                # Python
│   ├── channel-slack/                   # Python
│   ├── channel-discord/                 # Python
│   ├── channel-webhook/                 # Python
│   ├── env-docker/                      # Python
│   ├── env-ssh/                         # Python
│   └── widget-form/                     # TypeScript
├── db/
│   └── migrations/                      # Alembic-compatible SQL
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── ANTHROPIC_MANAGED_AGENTS_ANALYSIS.md
│   └── codegen.md                       # how the OpenAPI bridge works
├── tests/
│   ├── integration/                     # pytest, real daemon
│   ├── replay/                          # pytest, fixture-driven
│   └── environments/                    # backend conformance (pytest)
├── fixtures/                            # recorded model + tool responses
├── scripts/
│   ├── codegen.sh                       # OpenAPI → sdk-ts + sdk-py
│   └── lint.sh                          # uv run ruff + pyright + lint-imports + pnpm biome + tsc
├── Makefile                             # top-level task wrapper
├── pyproject.toml                       # uv workspace root
├── uv.lock                              # uv lockfile (committed)
├── importlinter.ini                     # Python cross-package layering (§27.2)
├── biome.json                           # TS lint + format unified (§27.1)
├── pnpm-workspace.yaml                  # pnpm workspace root
├── pnpm-lock.yaml                       # pnpm lockfile (committed)
└── README.md
```

**Workspace orchestration.**
- Python packages share `pyproject.toml` lockfiles via `uv workspace`.
- TypeScript packages share via `pnpm workspace`.
- `make dev` / `just dev` runs both side-by-side for local development.
- CI runs `make ci` which is `make codegen && make lint && make test`.

**Codegen pipeline** (see §28 and `docs/codegen.md`):
1. FastAPI app in `apps/meridiand` exports OpenAPI YAML to
   `packages/schemas/openapi.yaml` (CI step on every PR).
2. `openapi-typescript` generates `packages/sdk-ts/src/`.
3. `datamodel-code-generator` generates `packages/sdk-py/src/` (for
   Python clients that aren't the daemon itself).
4. CI fails if the committed `openapi.yaml` is out of date or if
   `sdk-ts` / `sdk-py` are out of date with the YAML.

---

## 24. Testing Strategy

| Layer | Style | Examples |
|-------|-------|----------|
| Unit | Pure functions | Capability intersection, JSON schema validation, phase transition derivation |
| Component | Single service, stubbed deps | Harness with stubbed model + stubbed sandbox |
| Integration | Real daemon, in-memory SQLite, FakeModel + FakeSandbox | Full POST /sessions flow |
| Replay | Real daemon, recorded fixtures | Re-run last week's session for regression |
| Environment conformance | Real daemon, each backend | Contract suite per backend |
| Channel conformance | Real daemon, each driver | Pairing, inbound/outbound round-trip |
| Skill Forge soak | Real daemon, real fixtures | Run forge on a corpus; measure proposal precision |
| Multi-agent | Real daemon, multi-session | Fan-out + budget aggregation + cancellation propagation |
| Soak | Nightly real-model run | Hour-long multi-channel session with crashes injected |

Determinism: `FakeModelAdapter` and `FakeSandboxAdapter` read canned
responses from fixtures. Test runs reproduce identically across hosts.

Property tests for the phase machine: every `(phase, event_type)` pair
asserts a derivation outcome.

---

## 25. Deployment Topology

Aligned with PRD v0.3 §8.2. The *feature surface* (every primitive in
§3, every endpoint in §8.2) is constant from v1 onwards; what evolves
is the operational envelope.

### 25.1 Single developer — v1

- `meridiand` as launchd/systemd user service.
- Listens on `~/.meridian/meridiand.sock` (UDS) + optional loopback.
- All state under `~/.meridian/`.
- SQLite WAL; local FS event log; OS keychain Vault.
- Single harness in-process. Sessions persist across SIGKILL.
- Zero external services required (model API calls aside).

### 25.2 Team-shared — v1.1

- `meridiand` on a shared host.
- Bearer-token auth in v1; **OIDC** lands here in v1.1.
- TLS reverse proxy.
- Per-user workspace partitioning enforced by capabilities + `$WORKSPACE`.
- **Signed audit log mandatory** (Ed25519, per PRD v0.3 D5/D9).
- Vault upgraded to AWS KMS if cloud KMS is preferred over OS keychain
  on a server.
- Still SQLite for the relational tier; still single harness.

### 25.3 Team-shared at scale — v1.2

- **Postgres 14+** as the supported relational backend (Repository
  pattern means zero application change).
- **S3-compatible blob store** for event logs, fixtures, files.
- pgvector for KB / MemoryStore vectors.
- HCP Vault available for enterprise credential management.
- Suitable for a team of 10s sharing one Meridian deployment.

### 25.4 Horizontal harness pool — v1.3

- Multiple harness workers, all stateless, sharing the Session Service.
- Sessions routed to harness instances by hash; any harness can
  `wake(session_id)` and resume.
- Event bus moves from in-process channel to **NATS** or **Redis Streams**.
- Per-tenant (per-UserProfile-group) capability ceilings.
- Cross-host checkpoint resume.
- Single-tenant still — multi-tenant is v2.

### 25.5 Multi-tenant SaaS — v2

- Multi-tenant cloud-SaaS option (PRD §3.2 NG1 lifted **only here**).
- Distributed orchestrator; sessions partitioned by tenant.
- Workers as ephemeral containers per tool call.
- **Cross-tenant ACP** with explicit allowlists (a tenant's agent can
  call another tenant's agent only if both opt in).
- Triggered by a real user with a concrete need, not a roadmap entry.

The contract across all five topologies: **application code written
against v1 HTTP API works unchanged through v2** (PRD v0.3 §8.2). The
Repository pattern, the stateless harness, the event-log-as-truth
model, and the capability boundary are what make this true.

---

## 26. Migration

### 26.1 OpenClaw

`meridian import openclaw $PATH`:
1. Read channel configs → `/v1/channels` records.
2. Read sessions → `/v1/sessions` + replay messages into the event log.
3. Read MEMORY.md → `/v1/memory_stores` entries (scope=agent, tagged
   `from:openclaw`).
4. Read tool definitions → tool registry + capability inference (with
   conservative defaults).
5. Audit log per record with lossy-mapping notes.

### 26.2 Hermes

`meridian import hermes $PATH`:
1. Read skills (already agentskills.io) → `/v1/skills`.
2. Read environments → `/v1/environments` (backend mapping preserved).
3. Read providers → Model Router config.
4. Read session histories → `/v1/sessions` + event-log replay.
5. Read Honcho user profiles → `/v1/user_profiles` + memory imports.
6. Map cron jobs → `/v1/x/cron`.
7. ACP registry entries → ACP adapter config.

### 26.3 Migration audit

Every import produces `meridian-import-<timestamp>.audit.ndjson` listing
each record translated and any lossy field mapping. Manual review
checklist generated.

---

## 26A. Web UI and Live Canvas (v1 feature)

Per PRD v0.3 NG3 flip and D6, both ship in v1. The architectural
constraint: they are **pure clients** over the existing HTTP + SSE
API — no new server logic, no new persistence, no privileged backdoor.

### 26A.1 Web UI

- Thin SPA (React or Solid; final pick at E7).
- Talks to the same `/v1/*` endpoints CLI uses.
- Connects to `/v1/sessions/{id}/events?stream=true` for live tail.
- Read-only session viewer + chat composer + agent picker + channel
  config + skill / vault / memory inspector.
- No authentication of its own; rides on whatever the daemon enforces
  (loopback in v1; bearer + OIDC in v1.1+).
- Bundled in the daemon binary, served at `/ui` when enabled in config.

### 26A.2 Live Canvas

- A panel the agent can write to via a built-in `canvas` tool.
- v1 widget set: text, markdown, simple form (text input, select,
  checkbox, button), code block, image embed, table, progress bar.
- Canvas state lives as a special message kind on the session
  (`content_block.canvas_op`); replay/rebuild is automatic.
- User interactions (form submit, button click) produce events that
  the harness surfaces as new `user` messages with structured payload.
- Deeper A2UI integration (richer widgets, declarative layout,
  bidirectional binding) lands in **v1.2** (infrastructure tier; the
  v1 canvas already covers the agent → user surface).

### 26A.3 Why thin-client

The Web UI / Live Canvas avoid becoming a parallel surface area. They
have no opinion about what an agent *is*; they render whatever the
session event log says. If we ship a new feature on the API
(streamed tool result diffs, citation highlighting, etc.), the UI
inherits it through the event types it already knows how to render.

---

## 27. Style Guide (Brief)

### 27.1 Code conventions

**Python (daemon, plugins, SDKs):**
- Python 3.11+; type-annotated everywhere; **pyright strict** in CI.
- **`uv`** for all package management, virtualenvs, lockfiles, and
  monorepo orchestration. `uv workspace` ties the daemon, CLI, SDKs,
  and plugins together. No `pip` / `poetry` / `pipenv` / `pdm` in
  contributor workflow; `uv` is the only entrypoint.
- `ruff` for formatting + lint (replaces black + isort + flake8).
- Pydantic v2 for all DTOs / config / event payloads.
- `from __future__ import annotations` everywhere.
- Async-first; `anyio` for compatibility with both asyncio and trio.
- `Result[T, E]`-style (`returns` library or hand-rolled) for expected
  errors; raise only on invariant violation.
- Repository pattern always; SQL only in `meridiand.storage`.
- Errors carry codes (`class MeridianError(Exception): code: str`).
- Tests next to source via `tests/` subdir per package for unit;
  `/tests/integration/` at repo root for integration; pytest +
  pytest-anyio.
- Comments explain *why*; names explain *what*.

**TypeScript (UI, UI SDK, UI widgets):**
- TypeScript strict + `noUncheckedIndexedAccess`.
- Named exports only.
- **Biome** is the single tool for both **lint and format** — replaces
  the eslint + prettier split. One config file (`biome.json`), one
  command (`biome check --write`), much faster than the eslint stack
  (Rust-backed). Combined with `tsc --noEmit` for type checking, that
  is the entire TS toolchain.
- Generated types from `packages/sdk-ts` are read-only; never edit by
  hand (enforced by L6 pre-commit hook, §27.2).
- React function components + hooks; no class components.
- Comments explain *why*; names explain *what*.

### 27.2 CI-enforced architectural lint rules

The architectural boundaries in this doc are only as load-bearing as
the lint rules that defend them. The following lint rules MUST be
enforced in CI; a PR that adds or modifies imports against any of
them MUST be rejected at the lint stage, not at review.

Lint enforcement is split across two toolchains. The Python side uses
**ruff** for code style + lint and **import-linter** (`importlinter` /
`lint-imports`) for cross-package layering, all orchestrated via
**`uv`**. The TypeScript side uses **Biome** (`biome check`) as the
single tool for both lint and format, plus **`tsc --noEmit`** for type
checking. Biome's `noRestrictedImports` rule covers the equivalent of
ESLint's `no-restricted-imports`. The *intent* of each rule below is
identical across both toolchains; the syntax differs.

#### Rule L1 — `claude-agent-sdk-python` is scoped to the OAuth provider

Operationalizes §13.4 and §13.5. The `claude_agent_sdk` package may
only be imported from inside
`apps/meridiand/src/meridiand/providers/anthropic_oauth/`.

`importlinter.ini` (repo root):

```ini
[importlinter]
root_packages = meridiand
include_external_packages = True

[importlinter:contract:claude-agent-sdk-scope]
name = claude-agent-sdk is scoped to AnthropicOAuthProvider
type = forbidden
source_modules =
    meridiand
forbidden_modules =
    claude_agent_sdk
ignore_imports =
    meridiand.providers.anthropic_oauth.** -> claude_agent_sdk
    meridiand.providers.anthropic_oauth.** -> claude_agent_sdk.**
```

Error message in CI output points at `docs/ARCHITECTURE.md §13.4` so
the contributor learns the *why* from the lint output itself.

#### Rule L2 — Other provider SDKs follow the same shape

The same isolation contract (§13.5) applies to every provider-specific
SDK. One approved import location per package; imports elsewhere are
a lint error:

| Package | Allowed import location |
|---------|-------------------------|
| `claude_agent_sdk` | `meridiand/providers/anthropic_oauth/**` |
| `anthropic` (low-level) | `meridiand/providers/anthropic_apikey/**` |
| `openai` | `meridiand/providers/openai/**` |
| `ollama` (or HTTP client) | `meridiand/providers/ollama/**` |
| OpenRouter client | `meridiand/providers/openrouter/**` |

One `importlinter` `forbidden` contract per provider SDK, with one
`ignore_imports` entry per allowed location. Keeps each provider's
transitive surface area inside its adapter.

#### Rule L3 — No SQL outside `storage/`

Operationalizes Principle 4 (Repository pattern, §1). `sqlite3`,
`aiosqlite`, `psycopg`, `asyncpg`, `sqlalchemy`, and any other SQL
client may only be imported from `meridiand/storage/**`. Services
talk to repositories; repositories talk to SQL.

```ini
[importlinter:contract:sql-in-storage-only]
name = SQL clients live in storage/
type = forbidden
source_modules = meridiand
forbidden_modules = sqlite3, aiosqlite, psycopg, asyncpg, sqlalchemy
ignore_imports = meridiand.storage.** -> *
```

#### Rule L4 — No SDK / runtime deps in shared schemas

Packages under `packages/schemas/**` and the shared portion of
`packages/sdk-py/**` MUST NOT import any provider SDK, channel driver,
environment backend, or other runtime-specific dependency. These are
the lingua franca shared between the daemon and clients; pulling a
runtime dep here forces every consumer to also pull it in.

Enforced via `importlinter` `forbidden` contract scoped to the schema
packages. The same applies on the TS side for `packages/sdk-ts/**`
via Biome's `noRestrictedImports` rule in `biome.json`:

```json
{
  "linter": {
    "rules": {
      "style": {
        "noRestrictedImports": {
          "level": "error",
          "options": {
            "paths": {
              "@anthropic-ai/claude-agent-sdk": "TS daemon code is the wrong layer — claude-agent-sdk is Python-only in Meridian. See ARCHITECTURE.md §13.4.",
              "anthropic": "Provider SDKs live in apps/meridiand (Python). See ARCHITECTURE.md §13.5.",
              "openai": "Provider SDKs live in apps/meridiand (Python). See ARCHITECTURE.md §13.5."
            }
          }
        }
      }
    }
  }
}
```

Generated `packages/sdk-ts/**` is excluded via `biome.json`'s
`"files.ignore"` so the codegen output isn't linted against these
rules.

#### Rule L5 — Service ↔ service direct calls are banned

Services in `meridiand/` MUST communicate either through HTTP handlers
(cross-service) or through the in-process Event Bus (asynchronous).
Direct cross-service function imports are a lint error.

```ini
[importlinter:contract:service-layering]
name = Services communicate via HTTP or Event Bus, not direct imports
type = layers
layers =
    meridiand.http
    meridiand.harness | meridiand.gateway | meridiand.skill_forge
    meridiand.sessions | meridiand.sandbox | meridiand.memory | meridiand.vault | meridiand.acp | meridiand.webhooks
    meridiand.model_router | meridiand.environments | meridiand.providers.*
    meridiand.events | meridiand.storage | meridiand.observability | meridiand.config
```

Each row imports only from rows below; same-row services do not
import each other. Cross-row up-imports are forbidden.

#### Rule L6 — Generated code is read-only

`packages/sdk-ts/src/generated/` and `packages/sdk-py/src/meridian_sdk/generated/`
are produced by the codegen pipeline (§28). Hand-edits are rejected by
a pre-commit hook (`scripts/check-generated.sh` compares working tree
against codegen output). The hook also runs in CI.

#### Where the lint rules live

- Python: `importlinter.ini` (repo root), `pyproject.toml` (ruff +
  pyright config), `uv.lock` for reproducible installs.
- TypeScript: `biome.json` (repo root, lint + format together),
  `tsconfig.json` per workspace.
- Each rule has an inline comment pointing to the architecture-doc
  section it operationalizes — so a contributor reading the config
  learns the *why*, not just the *what*.
- CI runs `make lint`, which wraps:
  ```
  uv run ruff check .
  uv run pyright
  uv run lint-imports
  pnpm exec biome check .
  pnpm exec tsc --noEmit
  ```
  PRs that fail cannot merge.

#### Migration / exemption policy

If a future provider's SDK genuinely needs to be used outside its
adapter — for example, a shared retry helper identical across SDKs —
the fix is to **extract the helper into a Meridian-owned utility**
that does not re-export the SDK type, not to weaken the lint rule.
Adding to the `ignore_imports` list requires an architecture-doc
update with rationale; do not add silent exemptions.

---

## 28. What This Architecture Buys You

| PRD requirement | Primitive that delivers it |
|-----------------|----------------------------|
| Anthropic API compat (G1) | Endpoint shape mirroring + `/v1/x` extensions |
| Stateless harness (G2) | `wake(session_id)` + session event log as truth |
| Unified sandbox (G3) | One `execute(name, input)` dispatch surface |
| Multi-channel gateway (G4) | Channel Drivers + Session-as-channel-agnostic |
| Model routing (G5) | Provider adapters + routing policy + logged decisions |
| Multi-environment (G6) | Environment Manager + conformance suite |
| Skill Forge (G7) | Background pipeline + agentskills.io schema + approval gate |
| Capability sandboxing (G8) | Capability sets + intersection check + subset propagation |
| Deterministic replay (G9) | Event log + fixtures + divergence error |
| Multi-agent + ACP (G10) | parent_session_id + spawn capability + typed handoff + ACP adapter |
| Observability + budgets (G11) | OTel spans + Prometheus + usage events + budget breaker |
| Hooks (G12) | Lifecycle bindings + verdict protocol |
| Hybrid KB (G13) | Indexer + fused retriever + scoped tool |
| Migration (G14) | OpenClaw + Hermes importers with audit log |
| Anthropic API quality without Anthropic lock-in | Two-mode AnthropicProvider (§13.4): `api_key` via raw SDK; `oauth` via claude-agent-sdk-python bridged through MCP tool proxy. Both isolated by §13.5 contract. |
| Subscription-tier economics for dev-loop users (P1) | `oauth` mode (§13.4) → Claude Pro/Max billing instead of per-token API |
| Provider/route configurability without recompile | YAML config (§13.6) + hot reload + JSON Schema for editor support |
| Cross-language type safety (Python daemon ↔ TS UI) | OpenAPI codegen pipeline (§28 below); generated SDKs in `packages/sdk-ts` + `packages/sdk-py` |

### 28.1 The codegen pipeline (Python ↔ TypeScript)

The polyglot stack pays for itself via one mechanical bridge:

```
                            (CI gate)
┌─────────────────┐  export  ┌───────────────────────────────────┐
│ FastAPI app     │ ──────▶  │ packages/schemas/openapi.yaml     │
│ apps/meridiand  │          │  (source of truth, committed)     │
└─────────────────┘          └───────┬───────────────────┬───────┘
                                     │                   │
                          openapi-   │   datamodel-      │
                          typescript │   code-generator  │
                                     ▼                   ▼
                          ┌──────────────────┐  ┌──────────────────┐
                          │ packages/sdk-ts/ │  │ packages/sdk-py/ │
                          │ (generated, RO)  │  │ (generated, RO)  │
                          └────────┬─────────┘  └────────┬─────────┘
                                   │                     │
                                   ▼                     ▼
                          ┌──────────────────┐  ┌──────────────────┐
                          │ apps/meridian-ui │  │ External Python  │
                          │ (TypeScript)     │  │ clients          │
                          └──────────────────┘  └──────────────────┘
```

CI rule (per §27.2 L6):
1. Re-run codegen against the daemon's current schema.
2. Compare against the committed `openapi.yaml` and `sdk-ts/` and
   `sdk-py/` outputs.
3. Fail the PR if any of them is stale.

Local dev rule:
- `make codegen` regenerates everything; commit the result.
- A pre-commit hook reminds you if you forget.

This is the operational cost of polyglot. ~5% engineering tax,
predictable, mostly invisible after the first iteration.

Build these primitives well once; every product capability falls out
of their composition.

---

**End of Architecture v0.4.**

Resolved decisions in [`PRD.md` §9](./PRD.md). Two-track delivery in
[`PRD.md` §8](./PRD.md). Resource map in [`PRD.md` §11](./PRD.md).
Language + YAML decisions in [`PRD.md` v0.4 header note](./PRD.md) and
[`PRD.md` §5.9 F-MD-5..8](./PRD.md).
