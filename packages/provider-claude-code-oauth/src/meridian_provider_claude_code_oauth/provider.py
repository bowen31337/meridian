"""SystemOAuthProvider — Claude Code CLI subprocess-backed model provider.

The provider manages the full lifecycle of a Claude Code CLI subprocess:
spawn, periodic health-check, restart on hang, kill on cancel.  The pinned
CLI version is read from ``meridian.lock`` at the workspace root.

Emits a ``claude_code_oauth.model.call`` OTel span with a
``provider.invocation`` event on every ``call()``.  On failure the span is
marked ERROR, the audit log receives a ``claude_code_oauth.call.failed``
entry, and the exception is surfaced to the caller as a
:class:`~meridian_sdk_provider.errors.ProviderCallError` subclass.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

from meridian_sdk_provider.audit import AuditLog, AuditLogEntry, NoopAuditLog
from meridian_sdk_provider.errors import ProviderCallError
from meridian_sdk_provider.protocol import ModelCapabilities, ModelEntry, ProviderCapabilities
from meridian_sdk_provider.telemetry import (
    get_tracer,
    record_invocation_event,
    record_provider_failure,
)
from meridian_sdk_provider.types import ModelCallOpts, ModelCountReq, ModelEvent, TokenCount

from ._lock import LockFileFormatError, LockFileNotFoundError, read_lock
from ._subprocess import CliSubprocessManager

_LOG = logging.getLogger(__name__)

_DEFAULT_LOCK_NAME = "meridian.lock"
_DEFAULT_CLI_NAME = "claude"

# Known models served via the Claude Code OAuth path.
_KNOWN_MODELS: list[tuple[str, int, bool, bool]] = [
    # (model_id, context_window, thinking, vision)
    ("claude-opus-4-7", 200_000, True, True),
    ("claude-sonnet-4-6", 200_000, True, True),
    ("claude-haiku-4-5-20251001", 200_000, False, True),
    ("claude-3-7-sonnet-20250219", 200_000, True, True),
    ("claude-3-5-sonnet-20241022", 200_000, False, True),
    ("claude-3-5-haiku-20241022", 200_000, False, True),
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SystemOAuthProvider:
    """ModelProvider that delegates calls to a managed Claude Code CLI subprocess.

    The provider owns the subprocess lifecycle: it spawns the ``claude``
    binary in ``--server`` mode on first use (or via ``start()``), runs
    periodic health-check pings in a background task, restarts the process
    when it stops responding, and kills it on call cancellation or ``close()``.

    The pinned CLI version is read from ``meridian.lock``.  If the lock file
    is absent the provider logs a warning and continues with version
    ``"unknown"``.

    Parameters
    ----------
    cli_path:
        Path to the ``claude`` CLI binary.  Defaults to the first ``claude``
        found on ``$PATH``.
    lock_path:
        Path to ``meridian.lock``.  Defaults to ``./meridian.lock`` relative
        to the current working directory.
    name:
        Provider instance identifier surfaced in OTel attributes and audit log
        entries.
    audit_log:
        Audit log sink.  Defaults to :class:`NoopAuditLog`.
    health_interval_s:
        Seconds between background health-check pings (default 30 s).
    health_timeout_s:
        Seconds to wait for a pong before declaring the process hung (default
        5 s).
    call_timeout_s:
        Per-readline timeout for streaming model-call responses (default
        120 s).
    _manager:
        Inject a pre-built :class:`CliSubprocessManager` for testing.
    """

    kind: str = "claude_code_oauth"

    def __init__(
        self,
        *,
        cli_path: str | None = None,
        lock_path: Path | None = None,
        name: str = "claude_code_oauth",
        audit_log: AuditLog | None = None,
        health_interval_s: float = 30.0,
        health_timeout_s: float = 5.0,
        call_timeout_s: float = 120.0,
        _manager: CliSubprocessManager | None = None,
    ) -> None:
        self.name = name
        self.capabilities = ProviderCapabilities(
            streaming=True,
            thinking=True,
            cache_control=False,
            count_tokens=False,
        )
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()

        _cli_path = cli_path or shutil.which(_DEFAULT_CLI_NAME) or _DEFAULT_CLI_NAME

        _lock_path = lock_path or Path(_DEFAULT_LOCK_NAME)
        cli_version = "unknown"
        try:
            entry = read_lock(_lock_path)
            cli_version = entry.cli_version
        except LockFileNotFoundError:
            _LOG.warning(
                "meridian.lock not found at %s; CLI version unpinned — "
                "run 'meridian lock' to pin it",
                _lock_path,
            )
        except LockFileFormatError as exc:
            _LOG.warning("meridian.lock is malformed: %s; CLI version unpinned", exc)

        self._manager = _manager or CliSubprocessManager(
            _cli_path,
            cli_version,
            provider_name=name,
            health_interval_s=health_interval_s,
            health_timeout_s=health_timeout_s,
            call_timeout_s=call_timeout_s,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Spawn the CLI subprocess and start the health-check loop."""
        await self._manager.start()

    async def close(self) -> None:
        """Kill the CLI subprocess and stop the health-check loop."""
        await self._manager.stop()

    # ------------------------------------------------------------------
    # ModelProvider protocol
    # ------------------------------------------------------------------

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        """Stream model events from the Claude Code CLI subprocess.

        Emits an OTel span ``claude_code_oauth.model.call``.  On failure the
        span is marked ERROR, the audit log is written, and the exception is
        re-raised as a :class:`ProviderCallError`.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "claude_code_oauth.model.call",
            attributes={"provider.name": self.name, "model": opts.model},
        ) as span:
            record_invocation_event(
                span,
                provider_name=self.name,
                provider_kind=self.kind,
                model=opts.model,
                session_id=opts.session_id,
                routing_rule=None,
            )
            try:
                async for event in self._manager.call(opts):
                    yield event
            except ProviderCallError as exc:
                record_provider_failure(span, exc, provider_name=self.name, model=opts.model)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="claude_code_oauth.call.failed",
                        provider_name=self.name,
                        provider_kind=self.kind,
                        model=opts.model,
                        session_id=opts.session_id,
                        timestamp=_now(),
                        detail={"error": str(exc), "error_type": type(exc).__name__},
                    )
                )
                raise
            except Exception as exc:
                err = ProviderCallError(str(exc), provider_name=self.name)
                record_provider_failure(span, err, provider_name=self.name, model=opts.model)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="claude_code_oauth.call.failed",
                        provider_name=self.name,
                        provider_kind=self.kind,
                        model=opts.model,
                        session_id=opts.session_id,
                        timestamp=_now(),
                        detail={"error": str(exc), "error_type": type(exc).__name__},
                    )
                )
                raise err from exc

    def list_models(self) -> list[ModelEntry]:
        """Return the set of Claude models served via the OAuth path."""
        return [
            ModelEntry(
                provider=self.name,
                model=model_id,
                context_window=ctx,
                capabilities=ModelCapabilities(
                    streaming=True,
                    thinking=thinking,
                    vision=vision,
                    tools=True,
                    cache=False,
                ),
            )
            for model_id, ctx, thinking, vision in _KNOWN_MODELS
        ]

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        raise NotImplementedError(
            "count_tokens is not supported by SystemOAuthProvider; "
            "set capabilities.count_tokens=False in your routing policy"
        )
