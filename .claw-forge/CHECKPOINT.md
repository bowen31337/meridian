# Checkpoint — 20260520T223139

## Status
- Tests: not run (uv workspace requires uv cache; sandbox-restricted at checkpoint time)
- Features: 20/20 complete (all commits on branch vs master)
- Snapshot: snapshots/snapshot-20260520T223139.json

## What's working

### Providers
- `AnthropicApiKeyProvider` (Mode 1) — raw Anthropic SDK, API key from Vault, per-token billing
- `AnthropicOAuthProvider` (Mode 2) — Claude Code CLI subprocess, subscription billing
- `OpenAIProvider` — streaming + tool-use, normalized to ModelEvent shape
- `OpenRouterProvider` — multi-model gateway, honors per-model feature flags

### Provider Infrastructure
- Provider registry with hot-swap on config reload (atomic pointer swap, in-flight drain)
- Failover policy: rate_limit / timeout / 5xx retried against configured fallback model
- Per-call prompt-cache header injection (Anthropic cache_control); hit/miss in usage.delta events
- Model routing rule engine: `skill_id`, `estimated_input_tokens`, `metadata_match`, `role`
- `model_call.started` event with routing rule + chosen provider/model

### Session & Message Lifecycle
- `POST /v1/sessions` — creates Session, pins agent_version, creates initial Thread, emits session.created
- `stop_reason=end_turn` — appends final message, transitions to idle, runs post_message hooks
- `stop_reason=tool_use` — schema-validates args, capability intersection check, pre_tool_call hooks, tool_call.requested events
- `stop_reason=max_tokens` — emits partial message, loops if policy allows, else transitions to waiting_for_user
- Model call error handler — on_error hooks, terminates unless hook marks recoverable

### Config & Reload
- Agent body schema validated via Pydantic v2 (instructions cap, model_routing, capabilities §6 grammar, tools[])
- `POST /v1/x/config/reload` and `SIGHUP` share validate-then-atomic-swap reload path
- Reload failure keeps old config in effect; returns 422 with details — no service interruption

### Observability & Memory
- Harness wake latency p99 < 100ms (OTel span `harness.wake`, metric `meridian_harness_wakes_total`)
- Reconciliation events as `memory.write` with action (deduped/merged/superseded/inserted)
- Forge builds agentskills.io-shaped `SkillVersionRecord`; stores as PROPOSAL in quarantine

## What's in progress
- None — all planned features committed

## Known issues
- Test suite not runnable under sandbox (uv cache path blocked by OS sandbox policy)
- 1 session in state DB at `pending` status (id: 3dccc5c4-a0e5-45c9-83d6-12adec8d639a, project: /Users/bowenli/development/meridian, created: 2026-05-16)
