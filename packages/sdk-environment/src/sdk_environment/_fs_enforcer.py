from __future__ import annotations

from datetime import UTC, datetime
import os
import re

from ._types import AgentFilesystemPolicy, FilesystemPolicy, FilesystemViolation


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _glob_to_regex(pattern: str) -> str:
    """
    Convert a glob pattern to a regex string.

    - ``**`` matches any sequence of characters including path separators.
    - ``*``  matches any sequence of characters except ``/``.
    - ``?``  matches any single character except ``/``.
    - All other characters are matched literally.
    """
    regex = ""
    i = 0
    while i < len(pattern):
        if pattern[i : i + 2] == "**":
            regex += ".*"
            i += 2
        elif pattern[i] == "*":
            regex += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            regex += "[^/]"
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return regex


def _glob_matches(pattern: str, path: str) -> bool:
    """Return True if *path* matches *pattern* (glob with ** support)."""
    return bool(re.fullmatch(_glob_to_regex(pattern), path))


class FilesystemEnforcer:
    """
    Enforces the environment-level FilesystemPolicy, optionally further
    constrained by an agent-level AgentFilesystemPolicy.

    Evaluation order (all must pass):
      1. Path canonicalisation: symlinks are resolved via os.path.realpath();
         any path that resolves outside $WORKSPACE (symlink escape or ``..``
         traversal) is rejected unconditionally.
      2. Environment policy: the canonical path must match at least one glob
         in the operation's allowlist (read_globs / write_globs / delete_globs).
      3. Agent allowlist: if present and non-empty, the canonical path must
         also match at least one agent glob (agents cannot escalate beyond what
         the environment grants).
    """

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        environment_policy: FilesystemPolicy,
        agent_policy: AgentFilesystemPolicy | None = None,
    ) -> None:
        self._workspace = os.path.realpath(workspace)
        self._env = environment_policy
        self._agent = agent_policy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> str | None:
        """
        Return the canonical absolute path, or None if it falls outside the
        workspace (symlink escape, ``..`` traversal, or absolute path outside).
        """
        real = os.path.realpath(path)
        if real != self._workspace and not real.startswith(self._workspace + os.sep):
            return None
        return real

    def _env_globs(self, operation: str) -> tuple[str, ...]:
        if operation == "read":
            return self._env.read_globs
        if operation == "write":
            return self._env.write_globs
        if operation == "delete":
            return self._env.delete_globs
        return ()

    def _agent_globs(self, operation: str) -> tuple[str, ...] | None:
        if self._agent is None:
            return None
        if operation == "read":
            return self._agent.read_globs
        if operation == "write":
            return self._agent.write_globs
        if operation == "delete":
            return self._agent.delete_globs
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_allowed(self, operation: str, path: str) -> bool:
        """Return True if *operation* on *path* is permitted by all active policies."""
        canonical = self._resolve(path)
        if canonical is None:
            return False

        env_globs = self._env_globs(operation)
        if not any(_glob_matches(g, canonical) for g in env_globs):
            return False

        agent_globs = self._agent_globs(operation)
        return not agent_globs or any(_glob_matches(g, canonical) for g in agent_globs)

    def assert_allowed(
        self,
        operation: str,
        path: str,
        *,
        environment_id: str = "",
        session_id: str = "",
    ) -> None:
        """Raise FilesystemViolation if the operation on path is not permitted."""
        if not self.is_allowed(operation, path):
            raise FilesystemViolation(
                operation=operation,
                path=path,
                agent_id=self._agent.agent_id if self._agent else "",
                environment_id=environment_id,
                session_id=session_id,
                timestamp=_now(),
            )
