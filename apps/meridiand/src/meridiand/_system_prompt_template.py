"""System prompt template expansion.

Resolves {{ memory.KEY }} references in agent instructions and skill
instructions against memory files stored in ``storage_root/memory/``.
Expansion is performed at run start (wake time) or per turn for short-TTL
memories.  Emits an OpenTelemetry span and a structured event on every
invocation; on failure writes an audit-log entry and re-raises so the
caller can surface the error.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from core_errors import (
    AuditLog,
    AuditLogEntry,
    MeridianError,
    StructuredEvent,
    get_tracer,
    record_error,
    record_invocation_event,
)

_TEMPLATE_REF_RE = re.compile(r"\{\{\s*memory\.([\w.]+)\s*\}\}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sanitize_key(key: str) -> str:
    return key.replace("/", "_").replace("\x00", "_")


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class TemplateExpandError(MeridianError):
    def __init__(
        self,
        *,
        message: str,
        timestamp: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(
            code="template_expand_failed",
            message=message,
            timestamp=timestamp,
            cause=cause,
        )

    def http_status(self) -> int:
        return 500


class TemplateMemoryNotFoundError(MeridianError):
    def __init__(self, *, memory_key: str, timestamp: str) -> None:
        super().__init__(
            code="template_memory_not_found",
            message=f"Memory key '{memory_key}' not found for template expansion",
            timestamp=timestamp,
        )

    def http_status(self) -> int:
        return 404


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expand_system_prompt(
    template: str,
    *,
    storage_root: Path,
    audit_log: AuditLog,
) -> str:
    """Return *template* with every ``{{ memory.KEY }}`` reference replaced.

    Each KEY maps to ``storage_root/memory/{safe_key}.json``; the ``value``
    field of that JSON file is substituted.  Raises
    :class:`TemplateMemoryNotFoundError` when a referenced key has no
    corresponding file, and :class:`TemplateExpandError` for unexpected I/O
    or parse failures.
    """
    now = _now()
    tracer = get_tracer()
    keys = _TEMPLATE_REF_RE.findall(template)

    with tracer.start_as_current_span(
        "system_prompt.template.expand",
        attributes={"template.ref_count": len(keys)},
    ) as span:
        record_invocation_event(
            span,
            StructuredEvent(
                name="system_prompt.template.expand.invocation",
                code="system_prompt_template_expand",
                timestamp=now,
            ),
        )

        try:
            if not keys:
                return template

            resolved: dict[str, str] = {}
            for key in dict.fromkeys(keys):
                safe_key = _sanitize_key(key)
                memory_file = storage_root / "memory" / f"{safe_key}.json"
                if not memory_file.exists():
                    raise TemplateMemoryNotFoundError(memory_key=key, timestamp=_now())
                data = json.loads(memory_file.read_text())
                resolved[key] = str(data.get("value", ""))

            return _TEMPLATE_REF_RE.sub(lambda m: resolved[m.group(1)], template)

        except (TemplateMemoryNotFoundError, TemplateExpandError) as err:
            record_error(span, err)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="system_prompt.template.expand.failed",
                    code=err.code,
                    timestamp=err.timestamp,
                    detail={"message": err.message},
                )
            )
            raise

        except Exception as exc:
            err2 = TemplateExpandError(
                message=f"Failed to expand system prompt template: {exc}",
                timestamp=_now(),
                cause=exc,
            )
            record_error(span, err2)
            audit_log.write(
                AuditLogEntry(
                    level="error",
                    event="system_prompt.template.expand.failed",
                    code=err2.code,
                    timestamp=err2.timestamp,
                    detail={"message": err2.message},
                )
            )
            raise err2
