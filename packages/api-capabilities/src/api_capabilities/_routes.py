from __future__ import annotations

import datetime

from fastapi import APIRouter

from core_errors import SchemaInvalidError
from sdk_capabilities import KNOWN_CAPABILITIES, param_expected

from ._audit import AuditLog, NoopAuditLog
from ._registry import CapabilityRegistry, is_valid_identifier
from ._telemetry import get_tracer, record_failure, record_list_event, record_register_event
from ._types import (
    AuditLogEntry,
    CapabilityInfo,
    ListCapabilitiesResponse,
    RegisterNamespaceRequest,
    RegisterNamespaceResponse,
)

_BUILT_IN_NAMESPACES: frozenset[str] = frozenset(ns for (ns, _) in KNOWN_CAPABILITIES)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _build_builtin_capabilities() -> list[CapabilityInfo]:
    result: list[CapabilityInfo] = []
    for ns, name in sorted(KNOWN_CAPABILITIES):
        pe = param_expected(ns, name)
        result.append(
            CapabilityInfo(
                id=f"{ns}.{name}",
                namespace=ns,
                name=name,
                param_expected=bool(pe),
            )
        )
    return result


def make_router(
    registry: CapabilityRegistry | None = None,
    audit_log: AuditLog | None = None,
) -> APIRouter:
    """Return an APIRouter mounting GET and POST /v1/x/capabilities.

    Both routes emit an OpenTelemetry span and log a structured event.  On
    failure the error is surfaced to the caller via the MeridianError envelope
    and written to *audit_log*.
    """
    _registry = registry or CapabilityRegistry()
    _audit = audit_log or NoopAuditLog()
    router = APIRouter()

    @router.get("/v1/x/capabilities", response_model=ListCapabilitiesResponse)
    def list_capabilities() -> ListCapabilitiesResponse:
        tracer = get_tracer()
        with tracer.start_as_current_span("capabilities.list") as span:
            try:
                built_in = _build_builtin_capabilities()
                plugin = _registry.all_capabilities()
                all_caps = built_in + plugin
                record_list_event(span, count=len(all_caps))
                return ListCapabilitiesResponse(capabilities=all_caps)
            except Exception as exc:
                now = _now_iso()
                record_failure(span, exc, operation="list")
                _audit.write(
                    AuditLogEntry(
                        level="error",
                        event="capabilities.list.failed",
                        operation="list",
                        timestamp=now,
                        detail={"error_type": type(exc).__name__, "error": str(exc)},
                    )
                )
                raise

    @router.post(
        "/v1/x/capabilities",
        response_model=RegisterNamespaceResponse,
        status_code=201,
    )
    def register_namespace(body: RegisterNamespaceRequest) -> RegisterNamespaceResponse:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "capabilities.register",
            attributes={"namespace": body.namespace},
        ) as span:
            now = _now_iso()

            validation_error: str | None = None
            if not is_valid_identifier(body.namespace):
                validation_error = (
                    f"Namespace {body.namespace!r} is not a valid identifier "
                    "(must match [a-z][a-z0-9_]*)."
                )
            elif body.namespace in _BUILT_IN_NAMESPACES:
                validation_error = (
                    f"Namespace {body.namespace!r} conflicts with a built-in system namespace."
                )
            elif _registry.is_registered(body.namespace):
                validation_error = f"Namespace {body.namespace!r} is already registered."
            elif not body.capabilities:
                validation_error = "capabilities list must not be empty."
            else:
                for spec in body.capabilities:
                    if not is_valid_identifier(spec.name):
                        validation_error = (
                            f"Capability name {spec.name!r} is not a valid identifier "
                            "(must match [a-z][a-z0-9_]*)."
                        )
                        break

            if validation_error is not None:
                err = SchemaInvalidError(message=validation_error, timestamp=now)
                record_failure(span, err, operation="register")
                _audit.write(
                    AuditLogEntry(
                        level="error",
                        event="capabilities.register.failed",
                        operation="register",
                        timestamp=now,
                        detail={"namespace": body.namespace, "reason": validation_error},
                    )
                )
                raise err

            capabilities = [
                CapabilityInfo(
                    id=f"{body.namespace}.{spec.name}",
                    namespace=body.namespace,
                    name=spec.name,
                    param_expected=spec.param_expected,
                )
                for spec in body.capabilities
            ]
            _registry.register(body.namespace, capabilities)
            record_register_event(
                span,
                namespace=body.namespace,
                capability_count=len(capabilities),
            )
            return RegisterNamespaceResponse(namespace=body.namespace, capabilities=capabilities)

    return router
