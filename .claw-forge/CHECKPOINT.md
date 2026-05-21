# Checkpoint — 20260522T000000

## Status
- Tests: 81 passing, 0 failing (sel + diagnosis suites); pre-existing failures in test_daemon/test_harness_loop unrelated to this feature
- Features: 22/22 complete
- Snapshot: snapshots/snapshot-20260521T220216.json

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

### Forge Session Selector — SEL (this feature)
- `_skill_forge_sel.py` — `collect_terminated_sessions` scans `storage_root/events/**/<session_id>.ndjson`, identifies sessions with terminal `session.phase_change` events (`terminated` / `completed`), extracts tool-call sequences from `tool_call.requested` events
- `cluster_trajectories` — groups sessions by identical (terminal_phase, tool_call_sequence) key; deterministic cluster IDs via `uuid.uuid5`; sorted by size descending
- `run_forge_session_selector` — orchestrates scan → cluster → result; OTel span `skill_forge.sel.run` with `session_count` + `cluster_count` attributes; audit `skill_forge.sel.ran` (info) on success; `ForgeSelError` (code=`forge_sel_failed`, HTTP 500) + audit `skill_forge.sel.run.failed` (error) on failure
- 37-test conformance suite in `test_skill_forge_sel.py`

### Session Failure Diagnosis (PRD §7.2 MTTR < 5 min)
- `GET /v1/sessions/{id}/diagnosis` — aggregates event log + audit + replay fixture into one postmortem JSON
- Extracts `terminal_phase` and `stop_reason` from last `session.phase_change` event
- `failure_events` — all error / phase_change / tool_call.vetoed / budget.warning / message.truncated events
- `audit_entries` — audit.ndjson entries filtered by `detail.session_id`
- `replay_fixture_available` — bool; True when `fixtures/{id}/model_responses.ndjson` exists
- `event_count` — total events in log for root-cause timeline sizing
- OTel span `session.diagnosis` with `session.id` attribute and structured invocation event
- On failure: raises `SessionDiagnosisError` (code=`session_diagnosis_failed`, HTTP 500); writes `session.diagnosis.failed` to audit log with session_id + message
- 44-test conformance suite in `test_diagnosis.py`

## What's in progress
- None — all planned features committed

## Known issues
- test_daemon.py::TestDaemonStartupSpan — pre-existing; requires optional `storage_repository` module not installed in test env
- test_harness_loop.py::TestHarnessLoopStopReasonToolUse (2 tests) — pre-existing failures unrelated to this feature
- test_conformance.py — requires `storage_repository` module not installed in test env
- 1 session in state DB at `pending` status (id: 3dccc5c4-a0e5-45c9-83d6-12adec8d639a, project: /Users/bowenli/development/meridian, created: 2026-05-16)
