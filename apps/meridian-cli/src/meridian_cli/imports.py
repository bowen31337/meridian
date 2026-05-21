"""CLI commands for importing data from OpenClaw and Hermes into Meridian."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from ._audit import write_audit
from ._client import DaemonClient, DaemonError
from ._telemetry import get_tracer, record_failure, record_invocation_event


def _client(ctx: click.Context) -> DaemonClient:
    return ctx.find_root().obj  # type: ignore[return-value]


@click.group()
def imports() -> None:
    """Import data from external systems into Meridian."""


@imports.command("openclaw")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def import_openclaw(ctx: click.Context, file: Path) -> None:
    """Import channels from an OpenClaw export FILE (JSON with a 'records' array)."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "import.openclaw",
        attributes={"import.source": "openclaw", "import.file": str(file)},
    ) as span:
        record_invocation_event(
            span,
            {"event.name": "import.openclaw.invocation", "import.source": "openclaw", "import.file": str(file)},
        )
        write_audit("info", "import.openclaw.invoked", {"file": str(file)})

        try:
            raw = file.read_text()
        except OSError as exc:
            record_failure(span, "import_read_failed", str(exc))
            write_audit("error", "import.openclaw.failed", {"code": "import_read_failed", "message": str(exc)})
            click.echo(f"error: [import_read_failed] {exc}", err=True)
            sys.exit(1)

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            record_failure(span, "import_invalid_json", str(exc))
            write_audit("error", "import.openclaw.failed", {"code": "import_invalid_json", "message": str(exc)})
            click.echo(f"error: [import_invalid_json] {exc}", err=True)
            sys.exit(1)

        try:
            result = _client(ctx).request("POST", "/v1/x/imports/openclaw", json_body=body)
        except DaemonError as exc:
            record_failure(span, exc.code, exc.message)
            write_audit("error", "import.openclaw.failed", {"code": exc.code, "message": exc.message})
            click.echo(f"error: [{exc.code}] {exc.message}", err=True)
            sys.exit(1)

        span.add_event("import.openclaw.completed")
        if result is not None:
            click.echo(json.dumps(result, indent=2))


@imports.command("hermes")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def import_hermes(ctx: click.Context, file: Path) -> None:
    """Import skills from a Hermes export FILE (JSON with a 'records' array)."""
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "import.hermes",
        attributes={"import.source": "hermes", "import.file": str(file)},
    ) as span:
        record_invocation_event(
            span,
            {"event.name": "import.hermes.invocation", "import.source": "hermes", "import.file": str(file)},
        )
        write_audit("info", "import.hermes.invoked", {"file": str(file)})

        try:
            raw = file.read_text()
        except OSError as exc:
            record_failure(span, "import_read_failed", str(exc))
            write_audit("error", "import.hermes.failed", {"code": "import_read_failed", "message": str(exc)})
            click.echo(f"error: [import_read_failed] {exc}", err=True)
            sys.exit(1)

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            record_failure(span, "import_invalid_json", str(exc))
            write_audit("error", "import.hermes.failed", {"code": "import_invalid_json", "message": str(exc)})
            click.echo(f"error: [import_invalid_json] {exc}", err=True)
            sys.exit(1)

        try:
            result = _client(ctx).request("POST", "/v1/x/imports/hermes", json_body=body)
        except DaemonError as exc:
            record_failure(span, exc.code, exc.message)
            write_audit("error", "import.hermes.failed", {"code": exc.code, "message": exc.message})
            click.echo(f"error: [{exc.code}] {exc.message}", err=True)
            sys.exit(1)

        span.add_event("import.hermes.completed")
        if result is not None:
            click.echo(json.dumps(result, indent=2))
