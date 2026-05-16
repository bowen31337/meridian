"""
Network policy test suite.

Covers:
  NetworkEnforcer
    - default-deny (egress_allowed=False): only allowlisted hosts pass.
    - default-allow (egress_allowed=True): all hosts pass unless constrained.
    - default-allow with allowlist: only listed hosts pass.
    - blocked_hosts: always denied regardless of other rules.
    - agent allowlist: further constrains (intersection, no escalation).
    - assert_allowed: raises NetworkViolation on denied host.

  OutboundProxyTransport
    - allowed request: audit entry at level="info", span emitted, forwards to inner.
    - denied request: audit entry at level="error", span set to ERROR, raises NetworkViolation.
    - NetworkViolation carries host, agent_id, environment_id, session_id.
    - no inner transport + allowed host: RuntimeError raised (misconfiguration guard).
    - span name is "net.outbound".
    - span attributes include net.host, environment.id, session.id, agent.id.
    - span ended on both success and failure paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from sdk_environment import (
    AgentNetworkPolicy,
    AuditLogEntry,
    NetworkEnforcer,
    NetworkPolicy,
    NetworkViolation,
    OutboundProxyTransport,
)

from .conftest import CapturingAuditLog, MockSpan, MockTracer


# ---------------------------------------------------------------------------
# Minimal httpx request stub — avoids importing httpx in the test suite.
# ---------------------------------------------------------------------------

@dataclass
class _URL:
    host: str


@dataclass
class _Request:
    url: _URL


def _req(host: str) -> _Request:
    return _Request(url=_URL(host=host))


# ---------------------------------------------------------------------------
# Inner transport stub
# ---------------------------------------------------------------------------

class _OkTransport:
    """Records forwarded requests and returns a sentinel."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def handle_request(self, request: Any) -> str:
        self.calls.append(request)
        return "ok"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enforcer(
    *,
    egress_allowed: bool = False,
    allowed_hosts: tuple[str, ...] = (),
    blocked_hosts: tuple[str, ...] = (),
    agent: AgentNetworkPolicy | None = None,
) -> NetworkEnforcer:
    return NetworkEnforcer(
        NetworkPolicy(
            egress_allowed=egress_allowed,
            allowed_hosts=allowed_hosts,
            blocked_hosts=blocked_hosts,
        ),
        agent,
    )


def _proxy(
    enforcer: NetworkEnforcer,
    audit: CapturingAuditLog,
    inner: Any = None,
    *,
    environment_id: str = "env1",
    session_id: str = "sess1",
    agent_id: str = "agent1",
) -> OutboundProxyTransport:
    return OutboundProxyTransport(
        enforcer,
        environment_id=environment_id,
        session_id=session_id,
        agent_id=agent_id,
        audit_log=audit,
        inner=inner,
    )


# ===========================================================================
# NetworkEnforcer — default-deny (egress_allowed=False)
# ===========================================================================

class TestEnforcerDefaultDeny:
    def test_denies_unlisted_host(self) -> None:
        e = _enforcer(egress_allowed=False)
        assert e.is_allowed("evil.com") is False

    def test_allows_host_in_allowlist(self) -> None:
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        assert e.is_allowed("api.example.com") is True

    def test_denies_host_not_in_allowlist(self) -> None:
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        assert e.is_allowed("cdn.example.com") is False

    def test_empty_allowlist_denies_all(self) -> None:
        e = _enforcer(egress_allowed=False, allowed_hosts=())
        assert e.is_allowed("anything.com") is False


# ===========================================================================
# NetworkEnforcer — default-allow (egress_allowed=True)
# ===========================================================================

class TestEnforcerDefaultAllow:
    def test_allows_any_host_when_no_allowlist(self) -> None:
        e = _enforcer(egress_allowed=True)
        assert e.is_allowed("arbitrary.io") is True

    def test_allows_listed_host_with_allowlist(self) -> None:
        e = _enforcer(egress_allowed=True, allowed_hosts=("api.example.com",))
        assert e.is_allowed("api.example.com") is True

    def test_denies_unlisted_host_with_allowlist(self) -> None:
        e = _enforcer(egress_allowed=True, allowed_hosts=("api.example.com",))
        assert e.is_allowed("other.com") is False


# ===========================================================================
# NetworkEnforcer — blocked_hosts always denied
# ===========================================================================

class TestEnforcerBlockedHosts:
    def test_blocked_host_denied_even_if_in_allowlist(self) -> None:
        e = _enforcer(
            egress_allowed=True,
            allowed_hosts=("evil.com",),
            blocked_hosts=("evil.com",),
        )
        assert e.is_allowed("evil.com") is False

    def test_blocked_host_denied_in_default_allow(self) -> None:
        e = _enforcer(egress_allowed=True, blocked_hosts=("evil.com",))
        assert e.is_allowed("evil.com") is False

    def test_non_blocked_host_passes(self) -> None:
        e = _enforcer(egress_allowed=True, blocked_hosts=("evil.com",))
        assert e.is_allowed("good.com") is True


# ===========================================================================
# NetworkEnforcer — agent allowlist (intersection, no escalation)
# ===========================================================================

class TestEnforcerAgentPolicy:
    def test_agent_allowlist_further_constrains(self) -> None:
        agent = AgentNetworkPolicy(agent_id="a1", allowed_hosts=("api.example.com",))
        e = _enforcer(
            egress_allowed=False,
            allowed_hosts=("api.example.com", "cdn.example.com"),
            agent=agent,
        )
        assert e.is_allowed("api.example.com") is True
        assert e.is_allowed("cdn.example.com") is False

    def test_agent_cannot_escalate_beyond_env(self) -> None:
        agent = AgentNetworkPolicy(agent_id="a1", allowed_hosts=("cdn.example.com",))
        e = _enforcer(
            egress_allowed=False,
            allowed_hosts=("api.example.com",),
            agent=agent,
        )
        assert e.is_allowed("cdn.example.com") is False

    def test_empty_agent_allowlist_does_not_further_restrict(self) -> None:
        agent = AgentNetworkPolicy(agent_id="a1", allowed_hosts=())
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",), agent=agent)
        assert e.is_allowed("api.example.com") is True

    def test_no_agent_policy_uses_env_only(self) -> None:
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        assert e.is_allowed("api.example.com") is True
        assert e.is_allowed("other.com") is False


# ===========================================================================
# NetworkEnforcer — assert_allowed
# ===========================================================================

class TestEnforcerAssertAllowed:
    def test_raises_network_violation_when_denied(self) -> None:
        e = _enforcer(egress_allowed=False)
        with pytest.raises(NetworkViolation) as exc_info:
            e.assert_allowed("evil.com", environment_id="env1", session_id="sess1")
        v = exc_info.value
        assert v.host == "evil.com"
        assert v.environment_id == "env1"
        assert v.session_id == "sess1"

    def test_does_not_raise_when_allowed(self) -> None:
        e = _enforcer(egress_allowed=False, allowed_hosts=("ok.com",))
        e.assert_allowed("ok.com")  # no exception

    def test_violation_message_contains_host(self) -> None:
        e = _enforcer(egress_allowed=False)
        with pytest.raises(NetworkViolation) as exc_info:
            e.assert_allowed("blocked.io")
        assert "blocked.io" in str(exc_info.value)

    def test_violation_carries_agent_id(self) -> None:
        agent = AgentNetworkPolicy(agent_id="my-agent", allowed_hosts=())
        e = NetworkEnforcer(
            NetworkPolicy(egress_allowed=True, allowed_hosts=("api.example.com",)),
            agent,
        )
        with pytest.raises(NetworkViolation) as exc_info:
            e.assert_allowed("other.com")
        assert exc_info.value.agent_id == "my-agent"


# ===========================================================================
# OutboundProxyTransport — allowed request
# ===========================================================================

class TestProxyAllowed:
    def test_forwards_to_inner_transport(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        inner = _OkTransport()
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        p = _proxy(e, audit_log, inner)
        result = p.handle_request(_req("api.example.com"))
        assert result == "ok"
        assert len(inner.calls) == 1

    def test_audit_entry_level_info(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        inner = _OkTransport()
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        p = _proxy(e, audit_log, inner)
        p.handle_request(_req("api.example.com"))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "info"

    def test_audit_entry_event_name(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        inner = _OkTransport()
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        p = _proxy(e, audit_log, inner)
        p.handle_request(_req("api.example.com"))
        assert audit_log.entries[0].event == "net.outbound"

    def test_audit_entry_detail_allowed_true(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        inner = _OkTransport()
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        p = _proxy(e, audit_log, inner)
        p.handle_request(_req("api.example.com"))
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.detail is not None
        assert entry.detail["allowed"] is True
        assert entry.detail["host"] == "api.example.com"

    def test_span_name(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        inner = _OkTransport()
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        p = _proxy(e, audit_log, inner)
        p.handle_request(_req("api.example.com"))
        assert mock_proxy_span.name == "net.outbound"

    def test_span_attributes(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        inner = _OkTransport()
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        p = _proxy(e, audit_log, inner, environment_id="env1", session_id="sess1", agent_id="agent1")
        p.handle_request(_req("api.example.com"))
        assert mock_proxy_span.attributes["net.host"] == "api.example.com"
        assert mock_proxy_span.attributes["environment.id"] == "env1"
        assert mock_proxy_span.attributes["session.id"] == "sess1"
        assert mock_proxy_span.attributes["agent.id"] == "agent1"

    def test_span_ended_on_success(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        inner = _OkTransport()
        e = _enforcer(egress_allowed=False, allowed_hosts=("api.example.com",))
        p = _proxy(e, audit_log, inner)
        p.handle_request(_req("api.example.com"))
        assert mock_proxy_span.ended


# ===========================================================================
# OutboundProxyTransport — denied request
# ===========================================================================

class TestProxyDenied:
    def test_raises_network_violation(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        e = _enforcer(egress_allowed=False)
        p = _proxy(e, audit_log)
        with pytest.raises(NetworkViolation) as exc_info:
            p.handle_request(_req("evil.com"))
        assert exc_info.value.host == "evil.com"

    def test_violation_carries_context(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        e = _enforcer(egress_allowed=False)
        p = _proxy(e, audit_log, environment_id="env1", session_id="sess1", agent_id="agent1")
        with pytest.raises(NetworkViolation) as exc_info:
            p.handle_request(_req("evil.com"))
        v = exc_info.value
        assert v.environment_id == "env1"
        assert v.session_id == "sess1"
        assert v.agent_id == "agent1"

    def test_audit_entry_level_error(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        e = _enforcer(egress_allowed=False)
        p = _proxy(e, audit_log)
        with pytest.raises(NetworkViolation):
            p.handle_request(_req("evil.com"))
        assert len(audit_log.entries) == 1
        assert audit_log.entries[0].level == "error"

    def test_audit_entry_detail_allowed_false(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        e = _enforcer(egress_allowed=False)
        p = _proxy(e, audit_log)
        with pytest.raises(NetworkViolation):
            p.handle_request(_req("evil.com"))
        entry: AuditLogEntry = audit_log.entries[0]
        assert entry.detail is not None
        assert entry.detail["allowed"] is False
        assert entry.detail["host"] == "evil.com"

    def test_span_set_to_error(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        e = _enforcer(egress_allowed=False)
        p = _proxy(e, audit_log)
        with pytest.raises(NetworkViolation):
            p.handle_request(_req("evil.com"))
        assert mock_proxy_span.status is not None
        assert mock_proxy_span.status.status_code == StatusCode.ERROR

    def test_span_ended_on_denial(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        e = _enforcer(egress_allowed=False)
        p = _proxy(e, audit_log)
        with pytest.raises(NetworkViolation):
            p.handle_request(_req("evil.com"))
        assert mock_proxy_span.ended

    def test_inner_not_called_on_denial(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        inner = _OkTransport()
        e = _enforcer(egress_allowed=False)
        p = _proxy(e, audit_log, inner)
        with pytest.raises(NetworkViolation):
            p.handle_request(_req("evil.com"))
        assert inner.calls == []


# ===========================================================================
# OutboundProxyTransport — misconfiguration guard
# ===========================================================================

class TestProxyNoInner:
    def test_raises_runtime_error_when_no_inner(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        e = _enforcer(egress_allowed=True)
        p = _proxy(e, audit_log, inner=None)
        with pytest.raises(RuntimeError, match="no inner transport"):
            p.handle_request(_req("ok.com"))


# ===========================================================================
# Integration: default-deny per env, agent allowlist further constrains
# ===========================================================================

class TestIntegrationDefaultDenyWithAgentAllowlist:
    def test_allowed_by_both_env_and_agent(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        agent = AgentNetworkPolicy(agent_id="a1", allowed_hosts=("api.example.com",))
        env_policy = NetworkPolicy(
            egress_allowed=False,
            allowed_hosts=("api.example.com", "cdn.example.com"),
        )
        enforcer = NetworkEnforcer(env_policy, agent)
        inner = _OkTransport()
        p = _proxy(enforcer, audit_log, inner, agent_id="a1")
        result = p.handle_request(_req("api.example.com"))
        assert result == "ok"
        assert audit_log.entries[0].level == "info"

    def test_allowed_by_env_but_not_agent_is_denied(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        agent = AgentNetworkPolicy(agent_id="a1", allowed_hosts=("api.example.com",))
        env_policy = NetworkPolicy(
            egress_allowed=False,
            allowed_hosts=("api.example.com", "cdn.example.com"),
        )
        enforcer = NetworkEnforcer(env_policy, agent)
        p = _proxy(enforcer, audit_log, agent_id="a1")
        with pytest.raises(NetworkViolation) as exc_info:
            p.handle_request(_req("cdn.example.com"))
        assert exc_info.value.host == "cdn.example.com"
        assert audit_log.entries[0].level == "error"

    def test_denied_by_env_even_if_agent_would_allow(
        self, mock_proxy_span: MockSpan, audit_log: CapturingAuditLog
    ) -> None:
        agent = AgentNetworkPolicy(agent_id="a1", allowed_hosts=("other.com",))
        env_policy = NetworkPolicy(egress_allowed=False, allowed_hosts=("api.example.com",))
        enforcer = NetworkEnforcer(env_policy, agent)
        p = _proxy(enforcer, audit_log, agent_id="a1")
        with pytest.raises(NetworkViolation):
            p.handle_request(_req("other.com"))
