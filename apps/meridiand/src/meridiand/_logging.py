from __future__ import annotations

import contextlib
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from core_errors import AuditLog, AuditLogEntry, MeridianError

_CONTEXT_FIELDS = ("session_id", "agent_id", "tool_name", "provider")

_LEVEL_NAMES: dict[int, str] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class LoggingConfigError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="logging_config_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class JsonFormatter(logging.Formatter):
    """Single-line JSON formatter for application logs written to stderr.

    Emits: ts, level, component, msg, and optional context fields
    (session_id, agent_id, tool_name, provider) when present on the record.
    Distinct from the domain event log (NDJSON in storage/events/).
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": _LEVEL_NAMES.get(record.levelno, record.levelname.lower()),
            "component": record.name,
            "msg": record.getMessage(),
        }
        for field in _CONTEXT_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                entry[field] = val
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, separators=(",", ":"))


def configure_json_logging(level: str, *, audit_log: AuditLog | None = None) -> None:
    """Replace root logger handlers with a single JSON stderr handler.

    Raises LoggingConfigError on failure; writes the failure to *audit_log*
    before raising so the caller can surface it via the standard error path.
    """
    now = _now()
    try:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level.upper())
    except Exception as exc:
        err = LoggingConfigError(
            message=f"Failed to configure JSON logging: {exc}",
            timestamp=now,
            cause=exc,
        )
        if audit_log is not None:
            with contextlib.suppress(Exception):
                audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="logging.configure.failed",
                        code=err.code,
                        timestamp=err.timestamp,
                        detail={"message": str(exc)},
                    )
                )
        raise err


def emit_early_error(component: str, msg: str) -> None:
    """Write a JSON error line to stderr before logging is configured."""
    print(
        json.dumps(
            {"ts": _now(), "level": "error", "component": component, "msg": msg},
            separators=(",", ":"),
        ),
        file=sys.stderr,
    )
