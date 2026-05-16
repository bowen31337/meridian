from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import StatusCode

from core_errors import AuditLog, AuditLogEntry

from ._discovery import DEFAULT_PLUGINS_YML, discover_from_entry_points, discover_from_yml
from ._manifest import PluginLoadError, PluginLoadResult, PluginManifest

_LOG = logging.getLogger("meridian.plugin_loader")
_TRACER_NAME = "meridian.plugin_loader"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PluginLoader:
    """Discovers and loads installed Meridian plugins from two sources:

    1. ``entry_points(group="meridian.plugins")`` — installed Python packages
       that advertise themselves via setuptools entry points.
    2. ``~/.meridian/plugins.yml`` — explicit user declarations (overrides or
       additions that are not installed as packages).

    Each ``load_all()`` call:
    - Emits an OpenTelemetry span named ``plugin_loader.load_all``.
    - Logs a structured ``plugin_loader.invocation`` event at the span level.
    - On per-plugin failure: writes an audit log entry and surfaces the error
      message in the returned ``PluginLoadResult.errors`` list.
    """

    def __init__(
        self,
        audit_log: AuditLog,
        plugins_yml: Path | None = None,
    ) -> None:
        self._audit_log = audit_log
        self._plugins_yml = plugins_yml or DEFAULT_PLUGINS_YML

    def load_all(self) -> PluginLoadResult:
        """Discover plugins from all sources and return parsed manifests.

        Never raises — all per-source and per-plugin failures are captured in
        ``PluginLoadResult.errors`` and recorded to the audit log.
        """
        tracer = trace.get_tracer(_TRACER_NAME)
        with tracer.start_as_current_span("plugin_loader.load_all") as span:
            span.add_event(
                "plugin_loader.invocation",
                {
                    "plugins_yml": str(self._plugins_yml),
                    "entry_point_group": "meridian.plugins",
                },
            )
            _LOG.info(
                "plugin_loader.load_all started",
                extra={"plugin_loader.plugins_yml": str(self._plugins_yml)},
            )

            raw_entries: list[dict[str, Any]] = []
            errors: list[PluginLoadError] = []

            # Source 1: installed entry_points
            try:
                ep_entries = discover_from_entry_points()
                raw_entries.extend(ep_entries)
                span.set_attribute("plugin_loader.entry_points_count", len(ep_entries))
            except Exception as exc:
                msg = f"entry_points discovery failed: {exc}"
                _LOG.error(msg)
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, msg)
                errors.append(
                    PluginLoadError(
                        plugin_name="<entry_points>",
                        message=msg,
                        code="entry_points_discovery_failed",
                    )
                )
                self._write_audit_error("plugin_loader.entry_points_failed", msg)

            # Source 2: ~/.meridian/plugins.yml
            try:
                yml_entries = discover_from_yml(self._plugins_yml)
                raw_entries.extend(yml_entries)
                span.set_attribute("plugin_loader.yml_count", len(yml_entries))
            except Exception as exc:
                msg = f"plugins.yml discovery failed ({self._plugins_yml}): {exc}"
                _LOG.error(msg)
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, msg)
                errors.append(
                    PluginLoadError(
                        plugin_name="<plugins.yml>",
                        message=msg,
                        code="yml_discovery_failed",
                    )
                )
                self._write_audit_error("plugin_loader.yml_failed", msg)

            # Parse and validate manifests
            manifests: list[PluginManifest] = []
            for entry in raw_entries:
                plugin_name = (
                    entry.get("name", "<unknown>") if isinstance(entry, dict) else "<unknown>"
                )
                try:
                    manifest = PluginManifest.model_validate(entry)
                    manifests.append(manifest)
                    span.add_event(
                        "plugin_loader.plugin_loaded",
                        {
                            "plugin.name": manifest.name,
                            "plugin.kind": manifest.kind,
                            "plugin.sandbox_mode": manifest.sandbox_mode,
                        },
                    )
                    _LOG.info(
                        "plugin.loaded",
                        extra={
                            "plugin.name": manifest.name,
                            "plugin.kind": manifest.kind,
                            "plugin.sandbox_mode": manifest.sandbox_mode,
                        },
                    )
                except Exception as exc:
                    msg = f"invalid plugin manifest for {plugin_name!r}: {exc}"
                    _LOG.error(msg)
                    span.record_exception(exc)
                    errors.append(
                        PluginLoadError(
                            plugin_name=plugin_name,
                            message=msg,
                            code="manifest_invalid",
                        )
                    )
                    self._write_audit_error(
                        "plugin_loader.manifest_invalid", msg, plugin_name=plugin_name
                    )

            span.set_attribute("plugin_loader.loaded_count", len(manifests))
            span.set_attribute("plugin_loader.error_count", len(errors))
            _LOG.info(
                "plugin_loader.load_all complete",
                extra={
                    "plugin_loader.loaded_count": len(manifests),
                    "plugin_loader.error_count": len(errors),
                },
            )
            return PluginLoadResult(manifests=manifests, errors=errors)

    def _write_audit_error(self, code: str, message: str, **extra_detail: object) -> None:
        with contextlib.suppress(Exception):
            self._audit_log.write(
                AuditLogEntry(
                    level="error",
                    event=code,
                    code=code,
                    timestamp=_now(),
                    detail={"message": message, **extra_detail},
                )
            )
