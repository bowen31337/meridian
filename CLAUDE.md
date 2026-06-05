# CLAUDE.md

## Project Overview
Meridian is a self-hostable, local-first agent runtime — a daemon (`meridiand`)
plus CLI (`meridian`) and web UI that runs LLM agents. It mirrors Anthropic's
managed-agents API shape (Agents, Sessions, Skills, etc.) but stores everything
locally and stays provider-agnostic. Core ideas:

- **The session is the truth.** Each session is an append-only NDJSON event log;
  relational tables and the session *phase* are derived projections.
- **The harness is cattle.** No session state lives in the harness; any worker
  can `wake(session_id)` and resume from the event log.
- **One dispatch surface.** Every executable action (built-in tool, MCP server,
  HTTP tool, container, subprocess) goes through the Sandbox's
  `execute(name, input) -> result`.
- **Capabilities by intersection.** Tools declare required caps, agents declare
  grants, dispatch enforces the intersection. No upward escalation.
- **Provider-polymorphic Model Router.** Anthropic (api_key + oauth modes),
  OpenAI, OpenRouter, Ollama in v1, configured declaratively via YAML.

Authoritative design docs live in `docs/`: `PRD.md` (what/why),
`ARCHITECTURE.md` (how — read this first, section refs like §13.4 are used
throughout the codebase), plus `TOOL_AUTHOR_GUIDE.md` and ACP notes.

Note: the daemon source files use a leading-underscore convention
(`apps/meridiand/src/meridiand/_sessions.py`, `_app.py`, `_sandbox`, etc.).

## Stack
- **Daemon (`apps/meridiand`):** Python 3.11+, FastAPI, Pydantic v2, uvicorn,
  OpenTelemetry + Prometheus. SSE for streaming.
- **CLI (`apps/meridian-cli`):** Python, Click. Entry point: `meridian`.
- **UI (`apps/meridian-ui`):** TypeScript, Vite, Vitest.
- **Storage (v1):** SQLite (WAL) via `sqlean-py`/`aiosqlite`, FTS5 + sqlite-vec
  for KB/memory, NDJSON event-log files, local-FS blobs. Postgres/S3/pgvector
  are later-version tiers behind the Repository pattern.
- **Packages:** polyglot monorepo. The **FastAPI app is the API
  source-of-truth**: `make codegen` exports it to `packages/schemas/openapi.yaml`,
  then generates `packages/sdk-py` (datamodel-code-generator) and
  `packages/sdk-ts` (openapi-typescript). All three are *generated artifacts* —
  don't hand-edit `openapi.yaml`, `sdk-py/src`, or `sdk-ts/src`; change the API
  routes and regenerate.

## Monorepo layout & boundaries
- `apps/` — `meridiand` (daemon), `meridian-cli`, `meridian-ui`.
- `packages/` — foundation SDKs (`sdk-*`, `storage-*`, `core-errors`,
  `system-ulid`), composed packages (`api-capabilities`, `builtin-tools`,
  `plugin-loader`, `storage-reposit`), providers, and generated SDKs.
- **Import boundaries are enforced** by import-linter (`.importlinter`):
  - Layering: apps → composed → foundation (downward only).
  - Foundation and composed packages are independent of their siblings.
  - **Provider isolation (§13.5):** provider adapters (`meridian_sdk_provider`)
    must NOT import the Session Service, Vaults, credential proxy, Sandbox, or
    event log. Custom scripts also guard provider/db/agent-sdk imports.

## Build & Test
Python is managed by **uv** (workspace); the TS/UI side uses **pnpm**
(`pnpm-workspace.yaml`). The Makefile wraps `scripts/make_runner.py`, which is
OTel-instrumented and writes to `.meridian/make-audit.ndjson`:

- `make lint` — ruff format+check, pyright, import-linter, the custom import
  guards (`scripts/check-*-imports.py`), then Biome + tsc.
- `make codegen` — export OpenAPI, then regenerate `sdk-py` and `sdk-ts`.
- `make ci` — codegen + lint + test.
- `make dev` — runs daemon (`python -m meridiand`) + UI (`npm run dev`).
- `make test` — runs root `uv run pytest` + UI vitest.

**Prerequisites / gotchas (verified):**
- **Tests run one pytest process per package.** Each member's `tests/` is its
  own top-level `tests` package (e.g. meridiand tests import
  `tests._otel_shared`), so two members can't be collected in one pytest process
  — their `tests` packages collide. `make test` handles this by invoking pytest
  once per member (with that dir as cwd). Do NOT run a bare `uv run pytest` from
  the repo root — it fails with `ImportPathMismatchError`. For ad-hoc runs:
  - `cd apps/meridiand && uv run pytest` (≈5.6k tests)
  - `cd apps/meridiand && uv run pytest tests/<file>.py -k <name>` for one test
  - `cd packages/<pkg> && uv run pytest`
- **JS/TS steps need deps installed first:** `pnpm install` at the repo root.
  Without it the Biome/tsc/vitest steps in `make lint|test` and the UI in
  `make dev` will fail (no `node_modules`). The Python steps of each target run
  regardless.
- `uv run …` auto-syncs the Python workspace on first use.

## Config & Runtime
- Daemon config: `~/.meridian/config.yml` (or `$MERIDIAN_CONFIG`), a Pydantic
  `MeridianConfig`. Declares providers, routing rules, vaults, storage, daemon
  bind. Secrets use `secret_ref://vault/{id}/{key}` indirection; hot reload via
  `POST /v1/x/config/reload` or SIGHUP (validate-then-atomic-swap).
- API base path `/v1`; Meridian extensions under `/v1/x`. Audit log at
  `~/.meridian/audit.ndjson` (or `$MERIDIAN_AUDIT_LOG`).
- `./run.sh` wraps commands with 1Password (`op`) secret injection — this repo
  is also driven by claw-forge (`claw-forge.yaml`); `app_spec.txt` is the
  feature spec corpus.

## Conventions
- Tool handlers must **never raise** — catch everything and return
  `ToolResult(is_error=True, ...)`; also write failures to the audit log.
- Many features follow a standard shape: emit an OpenTelemetry span, log a
  structured event, and on failure surface an error to the caller + write the
  audit log (see recent git history for the pattern).
- Don't hand-edit generated SDKs (`packages/sdk-py`, `packages/sdk-ts`).
