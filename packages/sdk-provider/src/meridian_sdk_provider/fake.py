"""FakeModelAdapter — replay-based ModelProvider for deterministic testing.

Reads canned ModelEvent sequences from fixtures/{model_slug}.ndjson. Each
non-blank line must be a JSON-serialized ModelEvent (discriminated by "type").
The fixture file is resolved by replacing ":" and "/" in opts.model with "_".

Emits a "fake.model.call" OTel span with a provider.invocation event on every
call(). On fixture load failure the span is marked ERROR, the audit log receives
a "fake_model.fixture.failed" entry, and the exception is re-raised to the
caller as a plain error (never silent).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from .audit import AuditLog, AuditLogEntry, NoopAuditLog
from .protocol import ProviderCapabilities
from .telemetry import get_tracer, record_invocation_event, record_provider_failure
from .types import ModelCallOpts, ModelCountReq, ModelEvent, TokenCount

_MODEL_EVENT_ADAPTER: TypeAdapter[ModelEvent] = TypeAdapter(ModelEvent)


def _model_to_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_model_fixture(path: Path) -> list[ModelEvent]:
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    events: list[ModelEvent] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            events.append(_MODEL_EVENT_ADAPTER.validate_python(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path}:{lineno}: invalid ModelEvent: {exc}") from exc
    return events


class FakeModelAdapter:
    """ModelProvider that replays canned responses from fixtures/*.ndjson.

    Fixture file: ``{fixtures_dir}/{model_slug}.ndjson``
    Each non-blank line must be a JSON-serialized ModelEvent.
    Every call() replays the full event sequence (stateless, deterministic).

    Parameters
    ----------
    fixtures_dir:
        Directory containing fixture NDJSON files. Defaults to ``"fixtures"``
        relative to the process working directory.
    name:
        Provider name surfaced in OTel attributes and audit log entries.
    audit_log:
        Audit log sink. Defaults to ``NoopAuditLog``.
    """

    def __init__(
        self,
        fixtures_dir: Path | str = "fixtures",
        *,
        name: str = "fake",
        audit_log: AuditLog | None = None,
    ) -> None:
        self.name = name
        self.kind = "fake"
        self.capabilities = ProviderCapabilities()
        self._fixtures_dir = Path(fixtures_dir)
        self._audit_log: AuditLog = audit_log if audit_log is not None else NoopAuditLog()

    async def call(self, opts: ModelCallOpts) -> AsyncIterator[ModelEvent]:
        fixture_path = self._fixtures_dir / f"{_model_to_slug(opts.model)}.ndjson"
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "fake.model.call",
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
                events = _load_model_fixture(fixture_path)
            except Exception as exc:
                record_provider_failure(span, exc, provider_name=self.name, model=opts.model)
                self._audit_log.write(
                    AuditLogEntry(
                        level="error",
                        event="fake_model.fixture.failed",
                        provider_name=self.name,
                        provider_kind=self.kind,
                        model=opts.model,
                        session_id=opts.session_id,
                        timestamp=_now(),
                        detail={"error": str(exc), "fixture": str(fixture_path)},
                    )
                )
                raise
            for event in events:
                yield event

    async def count_tokens(self, req: ModelCountReq) -> TokenCount:
        return TokenCount(input_tokens=0)

    async def close(self) -> None:
        pass


def write_model_fixture(path: Path, events: list[dict[str, Any]]) -> None:
    """Write a list of ModelEvent dicts to an NDJSON fixture file.

    Helper for test setup: creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )
