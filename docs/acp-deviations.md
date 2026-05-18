# ACP Deviations from Hermes Reference Spec

Per PRD Decision D1: CI runs Hermes's reference ACP compliance suite where one exists;
any deliberate deviation from Hermes's ACP is documented here with a rationale.

The deviations below are **additive** — they extend or instrument the protocol without
removing or redefining any Hermes-specified behavior.

---

## DEV-1: Capability gating on outbound calls

**Hermes behavior:** ACP outbound calls are routed to any registered target once a
transport-level connection is established. No capability check is defined in the
Hermes ACP spec.

**Meridian behavior:** An outbound call requires the caller to present `acp.outbound[target]`
(or the unrestricted `acp.outbound`) in its active capability set. Missing or mismatched
capabilities return HTTP 403 with error code `acp_outbound_denied`.

**Rationale:** Meridian's capability-by-intersection enforcement (PRD F-CA-2) applies to
all externally-observable actions, including cross-system ACP calls. Without this gate a
capability-sandboxed session could delegate to arbitrary external agents, violating the
"no upward escalation" invariant (PRD G8).

---

## DEV-2: Session-scoped outbound endpoint

**Hermes behavior:** A single top-level endpoint handles all ACP outbound calls.

**Meridian behavior:** Two endpoints coexist:
- `POST /v1/x/acp/outbound` — top-level, mirrors Hermes's endpoint semantics
- `POST /v1/x/sessions/{session_id}/acp/outbound` — session-scoped; correlates the ACP
  call to a specific session and echoes `session_id` in the response

**Rationale:** The session-scoped endpoint allows ACP calls to be recorded as
`acp.outbound` events in the session event log (PRD F-MA-4), satisfying the observability
requirement (F-OB-1) and enabling deterministic replay (G9). The top-level endpoint
preserves compatibility with callers that have no session context.

---

## DEV-3: `call_id` in every response

**Hermes behavior:** Not specified; the Hermes response does not document a call
correlation identifier.

**Meridian behavior:** Every successful ACP response includes a `call_id` (UUID v4) that
appears in the OTel span attribute `acp.call_id` and in the audit log entry, enabling
end-to-end correlation across observability surfaces.

**Rationale:** PRD F-OB-1 requires a span per ACP exchange; the `call_id` is the join key
that links the HTTP response, the OTel span, and the audit log record.

---

## DEV-4: Structured error envelope

**Hermes behavior:** Error response shape is not formally specified in the Hermes ACP
reference implementation.

**Meridian behavior:** All error responses follow the Meridian error envelope:
```json
{"error": {"code": "<string>", "message": "<string>", "timestamp": "<ISO-8601>"}}
```
Error codes: `acp_outbound_denied` (403) for capability or registry failures;
`acp_outbound_failed` (502) for transport failures.

**Rationale:** Uniform error structure across all Meridian endpoints enables programmatic
error handling, audit correlation, and consistent OTel span status marking.
