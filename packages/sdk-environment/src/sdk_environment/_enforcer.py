from __future__ import annotations

from datetime import UTC, datetime

from ._types import AgentNetworkPolicy, NetworkPolicy, NetworkViolation


def _now() -> str:
    return datetime.now(UTC).isoformat()


class NetworkEnforcer:
    """
    Enforces the environment-level NetworkPolicy, optionally further constrained
    by an agent-level AgentNetworkPolicy.

    Evaluation order (first matching rule wins):
      1. blocked_hosts (environment): always denied, regardless of other rules.
      2. egress_allowed=False: only hosts in environment.allowed_hosts pass;
         all others are denied (default-deny semantics).
      3. egress_allowed=True with non-empty allowed_hosts: only those hosts pass.
      4. egress_allowed=True with empty allowed_hosts: allow all (minus blocked).
      5. Agent allowlist: if present, host must also appear in agent.allowed_hosts
         (agents cannot escalate beyond what the environment grants).
    """

    def __init__(
        self,
        environment_policy: NetworkPolicy,
        agent_policy: AgentNetworkPolicy | None = None,
    ) -> None:
        self._env = environment_policy
        self._agent = agent_policy

    def is_allowed(self, host: str) -> bool:
        """Return True if the host is permitted by all active policies."""
        env = self._env

        if host in env.blocked_hosts:
            return False

        if (not env.egress_allowed or env.allowed_hosts) and host not in env.allowed_hosts:
            return False

        return not (
            self._agent is not None
            and self._agent.allowed_hosts
            and host not in self._agent.allowed_hosts
        )

    def assert_allowed(
        self,
        host: str,
        *,
        environment_id: str = "",
        session_id: str = "",
    ) -> None:
        """Raise NetworkViolation if the host is not permitted."""
        if not self.is_allowed(host):
            raise NetworkViolation(
                host=host,
                agent_id=self._agent.agent_id if self._agent else "",
                environment_id=environment_id,
                session_id=session_id,
                timestamp=_now(),
            )
