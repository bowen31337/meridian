from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from core_errors import (
    AuditLog,
    AuditLogEntry,
    HandlerOptions,
    MeridianError,
    NoopAuditLog,
    StructuredEvent,
    get_tracer,
    install_error_handler,
    record_error,
    record_invocation_event,
)
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from ._acp import AcpInboundHandler, AcpPeerClient, make_acp_router


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class AcpComplianceError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="acp_compliance_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 422


# ---------------------------------------------------------------------------
# In-process compliance peer clients
# ---------------------------------------------------------------------------

_COMPLIANCE_TARGET = "hermes"
_COMPLIANCE_TARGETS: dict[str, str] = {
    _COMPLIANCE_TARGET: "http://hermes.compliance.internal/acp"
}


class _SuccessPeerClient:
    async def call(self, url: str, message: dict[str, Any]) -> dict[str, Any]:
        return {"ack": True, "peer": "hermes"}


class _FailingPeerClient:
    async def call(self, url: str, message: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Connection refused: compliance transport test")


class _FailingInboundHandler:
    async def handle(self, target: str, message: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Inbound processing failed: compliance failure test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    name: str, description: str, passed: bool, reason: str | None = None
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "name": name,
        "description": description,
        "status": "passed" if passed else "failed",
    }
    if not passed and reason:
        r["reason"] = reason
    return r


def _make_acp_test_app(
    peer_client: AcpPeerClient,
    inbound_handler: AcpInboundHandler | None = None,
) -> FastAPI:
    app = FastAPI()
    install_error_handler(app, HandlerOptions(audit_log=NoopAuditLog()))
    app.include_router(
        make_acp_router(
            audit_log=NoopAuditLog(),
            targets=_COMPLIANCE_TARGETS,
            peer_client=peer_client,
            inbound_handler=inbound_handler,
        )
    )
    return app


# ---------------------------------------------------------------------------
# Compliance test suite — 20 Hermes ACP spec assertions
# ---------------------------------------------------------------------------


async def _run_compliance_suite() -> list[dict[str, Any]]:
    """Run all Hermes ACP conformance checks in-process via httpx ASGITransport."""
    import httpx

    results: list[dict[str, Any]] = []
    success_app = _make_acp_test_app(_SuccessPeerClient())
    failure_app = _make_acp_test_app(_FailingPeerClient())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=success_app), base_url="http://test"
    ) as c:
        # --- session-scoped: success path ---

        r = await c.post(
            "/v1/x/sessions/ci-s1/acp/outbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.outbound[hermes]"],
                "message": {"action": "ping"},
            },
        )
        results.append(
            _result(
                "message_delivery",
                "POST /v1/x/sessions/{id}/acp/outbound returns 200 on a valid call",
                r.status_code == 200,
                f"status {r.status_code}",
            )
        )
        body = r.json() if r.status_code == 200 else {}

        results.append(
            _result(
                "response_has_call_id",
                "200 response contains a non-empty call_id string",
                isinstance(body.get("call_id"), str) and len(body.get("call_id", "")) > 0,
                "call_id missing or empty",
            )
        )
        results.append(
            _result(
                "session_id_echoed",
                "200 response session_id matches the session path parameter",
                body.get("session_id") == "ci-s1",
                f"got {body.get('session_id')!r}",
            )
        )
        results.append(
            _result(
                "target_echoed",
                "200 response target matches the request body target",
                body.get("target") == "hermes",
                f"got {body.get('target')!r}",
            )
        )
        results.append(
            _result(
                "status_delivered",
                "200 response status is 'delivered'",
                body.get("status") == "delivered",
                f"got {body.get('status')!r}",
            )
        )
        results.append(
            _result(
                "peer_response_forwarded",
                "200 response includes the peer's response as a dict",
                isinstance(body.get("response"), dict),
                "response field missing or not a dict",
            )
        )

        # call_id uniqueness across two calls
        r2 = await c.post(
            "/v1/x/sessions/ci-s2/acp/outbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.outbound[hermes]"],
                "message": {},
            },
        )
        cid1 = body.get("call_id")
        cid2 = r2.json().get("call_id") if r2.status_code == 200 else None
        results.append(
            _result(
                "call_ids_unique",
                "Repeated calls produce distinct call_ids",
                cid1 is not None and cid2 is not None and cid1 != cid2,
                f"call_ids not unique: {cid1!r} == {cid2!r}",
            )
        )

        # unrestricted capability
        r3 = await c.post(
            "/v1/x/sessions/ci-s3/acp/outbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.outbound"],
                "message": {},
            },
        )
        results.append(
            _result(
                "unrestricted_cap_grants_any_target",
                "acp.outbound (no param) is accepted for any registered target",
                r3.status_code == 200,
                f"status {r3.status_code}",
            )
        )

        # parameterized capability
        r4 = await c.post(
            "/v1/x/sessions/ci-s4/acp/outbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.outbound[hermes]"],
                "message": {},
            },
        )
        results.append(
            _result(
                "parameterized_cap_grants_target",
                "acp.outbound[target] is accepted for the named target",
                r4.status_code == 200,
                f"status {r4.status_code}",
            )
        )

        # capability denial
        r5 = await c.post(
            "/v1/x/sessions/ci-s5/acp/outbound",
            json={
                "target": "hermes",
                "session_capabilities": ["exec.shell"],
                "message": {},
            },
        )
        results.append(
            _result(
                "missing_cap_denied",
                "Missing acp.outbound capability returns 403",
                r5.status_code == 403,
                f"status {r5.status_code}",
            )
        )
        denied_code = (
            r5.json().get("error", {}).get("code") if r5.status_code == 403 else None
        )
        results.append(
            _result(
                "denial_error_code",
                "Capability denial error code is 'acp_outbound_denied'",
                denied_code == "acp_outbound_denied",
                f"got {denied_code!r}",
            )
        )

        r6 = await c.post(
            "/v1/x/sessions/ci-s6/acp/outbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.outbound[other]"],
                "message": {},
            },
        )
        results.append(
            _result(
                "wrong_target_cap_denied",
                "acp.outbound[other] is denied when calling a different target",
                r6.status_code == 403,
                f"status {r6.status_code}",
            )
        )

        r7 = await c.post(
            "/v1/x/sessions/ci-s7/acp/outbound",
            json={
                "target": "hermes",
                "session_capabilities": ["INVALID!!"],
                "message": {},
            },
        )
        results.append(
            _result(
                "invalid_cap_string_denied",
                "Invalid capability string returns 403",
                r7.status_code == 403,
                f"status {r7.status_code}",
            )
        )

        r8 = await c.post(
            "/v1/x/sessions/ci-s8/acp/outbound",
            json={
                "target": "unknown-peer",
                "session_capabilities": ["acp.outbound[unknown-peer]"],
                "message": {},
            },
        )
        results.append(
            _result(
                "unregistered_target_denied",
                "Call to an unregistered target returns 403",
                r8.status_code == 403,
                f"status {r8.status_code}",
            )
        )

        # schema validation
        r9 = await c.post(
            "/v1/x/sessions/ci-s9/acp/outbound",
            json={"session_capabilities": ["acp.outbound[hermes]"], "message": {}},
        )
        results.append(
            _result(
                "missing_required_field_422",
                "Missing required field in request body returns 422",
                r9.status_code == 422,
                f"status {r9.status_code}",
            )
        )

        # --- top-level endpoint ---

        r10 = await c.post(
            "/v1/x/acp/outbound",
            json={
                "target": "hermes",
                "capabilities": ["acp.outbound[hermes]"],
                "message": {},
            },
        )
        results.append(
            _result(
                "toplevel_message_delivery",
                "POST /v1/x/acp/outbound returns 200 on a valid call",
                r10.status_code == 200,
                f"status {r10.status_code}",
            )
        )
        tl_body = r10.json() if r10.status_code == 200 else {}
        results.append(
            _result(
                "toplevel_no_session_id",
                "Top-level endpoint response does not include session_id",
                "session_id" not in tl_body,
                "session_id unexpectedly present",
            )
        )

        r11 = await c.post(
            "/v1/x/acp/outbound",
            json={
                "target": "hermes",
                "capabilities": ["exec.shell"],
                "message": {},
            },
        )
        results.append(
            _result(
                "toplevel_cap_denial",
                "Top-level endpoint returns 403 when capability is missing",
                r11.status_code == 403,
                f"status {r11.status_code}",
            )
        )

    # --- transport failure path (separate app instance) ---

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=failure_app), base_url="http://test"
    ) as c:
        r12 = await c.post(
            "/v1/x/sessions/ci-s12/acp/outbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.outbound[hermes]"],
                "message": {},
            },
        )
        results.append(
            _result(
                "transport_failure_502",
                "Peer transport error returns 502",
                r12.status_code == 502,
                f"status {r12.status_code}",
            )
        )
        tf_code = (
            r12.json().get("error", {}).get("code") if r12.status_code == 502 else None
        )
        results.append(
            _result(
                "transport_failure_error_code",
                "Transport failure error code is 'acp_outbound_failed'",
                tf_code == "acp_outbound_failed",
                f"got {tf_code!r}",
            )
        )

    # --- inbound: success path ---

    inbound_app = _make_acp_test_app(_SuccessPeerClient())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=inbound_app), base_url="http://test"
    ) as c:
        r13 = await c.post(
            "/v1/x/sessions/ci-s13/acp/inbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.inbound[hermes]"],
                "message": {"action": "ping"},
            },
        )
        results.append(
            _result(
                "inbound_session_accepted",
                "POST /v1/x/sessions/{id}/acp/inbound returns 200 on a valid call",
                r13.status_code == 200,
                f"status {r13.status_code}",
            )
        )
        ib_body = r13.json() if r13.status_code == 200 else {}
        results.append(
            _result(
                "inbound_response_has_call_id",
                "Inbound 200 response contains a non-empty call_id string",
                isinstance(ib_body.get("call_id"), str) and len(ib_body.get("call_id", "")) > 0,
                "call_id missing or empty",
            )
        )
        results.append(
            _result(
                "inbound_session_id_echoed",
                "Inbound 200 response session_id matches the session path parameter",
                ib_body.get("session_id") == "ci-s13",
                f"got {ib_body.get('session_id')!r}",
            )
        )
        results.append(
            _result(
                "inbound_target_echoed",
                "Inbound 200 response target matches the request body target",
                ib_body.get("target") == "hermes",
                f"got {ib_body.get('target')!r}",
            )
        )
        results.append(
            _result(
                "inbound_status_accepted",
                "Inbound 200 response status is 'accepted'",
                ib_body.get("status") == "accepted",
                f"got {ib_body.get('status')!r}",
            )
        )

        # unrestricted inbound capability
        r14 = await c.post(
            "/v1/x/sessions/ci-s14/acp/inbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.inbound"],
                "message": {},
            },
        )
        results.append(
            _result(
                "inbound_unrestricted_cap_grants_any_target",
                "acp.inbound (no param) is accepted for any registered target",
                r14.status_code == 200,
                f"status {r14.status_code}",
            )
        )

        # inbound capability denial
        r15 = await c.post(
            "/v1/x/sessions/ci-s15/acp/inbound",
            json={
                "target": "hermes",
                "session_capabilities": ["exec.shell"],
                "message": {},
            },
        )
        results.append(
            _result(
                "inbound_missing_cap_denied",
                "Missing acp.inbound capability returns 403",
                r15.status_code == 403,
                f"status {r15.status_code}",
            )
        )
        ib_denied_code = (
            r15.json().get("error", {}).get("code") if r15.status_code == 403 else None
        )
        results.append(
            _result(
                "inbound_denial_error_code",
                "Inbound capability denial error code is 'acp_inbound_denied'",
                ib_denied_code == "acp_inbound_denied",
                f"got {ib_denied_code!r}",
            )
        )

        r16 = await c.post(
            "/v1/x/sessions/ci-s16/acp/inbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.inbound[other]"],
                "message": {},
            },
        )
        results.append(
            _result(
                "inbound_wrong_target_cap_denied",
                "acp.inbound[other] is denied when receiving from a different target",
                r16.status_code == 403,
                f"status {r16.status_code}",
            )
        )

        r17 = await c.post(
            "/v1/x/sessions/ci-s17/acp/inbound",
            json={
                "target": "unknown-peer",
                "session_capabilities": ["acp.inbound[unknown-peer]"],
                "message": {},
            },
        )
        results.append(
            _result(
                "inbound_unregistered_target_denied",
                "Inbound call from an unregistered target returns 403",
                r17.status_code == 403,
                f"status {r17.status_code}",
            )
        )

        # top-level inbound endpoint
        r18 = await c.post(
            "/v1/x/acp/inbound",
            json={
                "target": "hermes",
                "capabilities": ["acp.inbound[hermes]"],
                "message": {},
            },
        )
        results.append(
            _result(
                "inbound_toplevel_accepted",
                "POST /v1/x/acp/inbound returns 200 on a valid call",
                r18.status_code == 200,
                f"status {r18.status_code}",
            )
        )
        tl_ib_body = r18.json() if r18.status_code == 200 else {}
        results.append(
            _result(
                "inbound_toplevel_no_session_id",
                "Top-level inbound response does not include session_id",
                "session_id" not in tl_ib_body,
                "session_id unexpectedly present",
            )
        )

        r19 = await c.post(
            "/v1/x/acp/inbound",
            json={
                "target": "hermes",
                "capabilities": ["exec.shell"],
                "message": {},
            },
        )
        results.append(
            _result(
                "inbound_toplevel_cap_denial",
                "Top-level inbound endpoint returns 403 when capability is missing",
                r19.status_code == 403,
                f"status {r19.status_code}",
            )
        )

    # --- inbound: processing failure path ---

    inbound_failure_app = _make_acp_test_app(_SuccessPeerClient(), _FailingInboundHandler())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=inbound_failure_app), base_url="http://test"
    ) as c:
        r20 = await c.post(
            "/v1/x/sessions/ci-s20/acp/inbound",
            json={
                "target": "hermes",
                "session_capabilities": ["acp.inbound[hermes]"],
                "message": {},
            },
        )
        results.append(
            _result(
                "inbound_processing_failure_500",
                "Inbound handler error returns 500",
                r20.status_code == 500,
                f"status {r20.status_code}",
            )
        )
        ib_fail_code = (
            r20.json().get("error", {}).get("code") if r20.status_code == 500 else None
        )
        results.append(
            _result(
                "inbound_processing_failure_error_code",
                "Inbound processing failure error code is 'acp_inbound_failed'",
                ib_fail_code == "acp_inbound_failed",
                f"got {ib_fail_code!r}",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def make_acp_compliance_router(
    *,
    audit_log: AuditLog,
    suite_fn: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
) -> APIRouter:
    _suite = suite_fn if suite_fn is not None else _run_compliance_suite
    router = APIRouter()

    @router.post("/v1/x/ci/acp-compliance")
    async def run_acp_compliance() -> JSONResponse:
        now = _now()
        tracer = get_tracer()
        run_id = str(uuid.uuid4())

        with tracer.start_as_current_span(
            "ci.acp.compliance",
            attributes={"compliance.run_id": run_id},
        ) as span:
            record_invocation_event(
                span,
                StructuredEvent(
                    name="ci.acp.compliance.invocation",
                    code="ci_acp_compliance",
                    timestamp=now,
                ),
            )

            tests = await _suite()
            failed = [t for t in tests if t["status"] == "failed"]

            if failed:
                first = failed[0]
                reason = first.get("reason", "")
                msg = f"ACP compliance failed: test '{first['name']}' failed" + (
                    f": {reason}" if reason else ""
                )
                err = AcpComplianceError(message=msg, timestamp=_now())
                record_error(span, err)
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="ci.acp.compliance.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={
                            "run_id": run_id,
                            "failed_test": first["name"],
                            "tests": tests,
                        },
                    )
                )
                raise err

        return JSONResponse(
            content={
                "run_id": run_id,
                "status": "passed",
                "test_count": len(tests),
                "tests": tests,
            }
        )

    return router
