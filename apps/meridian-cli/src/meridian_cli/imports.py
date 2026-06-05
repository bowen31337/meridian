"""CLI commands for importing data from OpenClaw and Hermes into Meridian."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import click

from ._audit import write_audit
from ._client import DaemonClient, DaemonError
from ._telemetry import get_tracer, record_failure, record_invocation_event

# Subsystem files present in an OpenClaw installation directory.
_OPENCLAW_SUBSYSTEM_FILES: dict[str, str] = {
    "channels": "channels.json",
    "sessions": "sessions.json",
    "tools": "tools.json",
}

# Subsystem files present in a Hermes installation directory.
_HERMES_SUBSYSTEM_FILES: dict[str, str] = {
    "skills": "skills.json",
    "environments": "environments.json",
    "providers": "providers.json",
    "sessions": "sessions.json",
    "user_profiles": "user_profiles.json",
    "cron": "cron.json",
    "acp_registry": "acp_registry.json",
}


def _client(ctx: click.Context) -> DaemonClient:
    return ctx.find_root().obj  # type: ignore[return-value]


def _read_json_file(path: Path, span: object, event_prefix: str) -> dict | None:
    """Read and parse a JSON file; return None and exit on error."""
    try:
        raw = path.read_text()
    except OSError as exc:
        record_failure(span, "import_read_failed", str(exc))  # type: ignore[arg-type]
        write_audit(
            "error", f"{event_prefix}.failed", {"code": "import_read_failed", "message": str(exc)}
        )
        click.echo(f"error: [import_read_failed] {exc}", err=True)
        sys.exit(1)
    try:
        return json.loads(raw)  # type: ignore[return-value]
    except json.JSONDecodeError as exc:
        record_failure(span, "import_invalid_json", str(exc))  # type: ignore[arg-type]
        write_audit(
            "error", f"{event_prefix}.failed", {"code": "import_invalid_json", "message": str(exc)}
        )
        click.echo(f"error: [import_invalid_json] {exc}", err=True)
        sys.exit(1)


@click.group()
def imports() -> None:
    """Import data from external systems into Meridian."""


@imports.command("openclaw")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def import_openclaw(ctx: click.Context, path: Path) -> None:
    """Import from OpenClaw.

    PATH may be:
      - A JSON file with a 'records' array — imports channels only (legacy).
      - A directory (OpenClaw installation) — imports all subsystems: channels,
        sessions, memory (MEMORY.md), and tools.
        channels.json, sessions.json, and tools.json are read if present.
        MEMORY.md is read if present and converted to memory store records.
    """
    tracer = get_tracer()
    if path.is_file():
        _import_openclaw_file(ctx, path, tracer)
    elif path.is_dir():
        _import_openclaw_dir(ctx, path, tracer)
    else:
        click.echo(f"error: [import_path_invalid] {path} is not a file or directory", err=True)
        sys.exit(1)


def _import_openclaw_file(ctx: click.Context, file: Path, tracer: object) -> None:
    """Backward-compatible: import channels from an OpenClaw JSON export file."""
    with tracer.start_as_current_span(  # type: ignore[union-attr]
        "import.openclaw",
        attributes={"import.source": "openclaw", "import.file": str(file)},
    ) as span:
        record_invocation_event(
            span,
            {
                "event.name": "import.openclaw.invocation",
                "import.source": "openclaw",
                "import.file": str(file),
            },
        )
        write_audit("info", "import.openclaw.invoked", {"file": str(file)})

        body = _read_json_file(file, span, "import.openclaw")

        try:
            result = _client(ctx).request("POST", "/v1/x/imports/openclaw", json_body=body)
        except DaemonError as exc:
            record_failure(span, exc.code, exc.message)
            write_audit(
                "error", "import.openclaw.failed", {"code": exc.code, "message": exc.message}
            )
            click.echo(f"error: [{exc.code}] {exc.message}", err=True)
            sys.exit(1)

        span.add_event("import.openclaw.completed")
        if result is not None:
            click.echo(json.dumps(result, indent=2))


def _import_openclaw_dir(ctx: click.Context, install_path: Path, tracer: object) -> None:
    """Import a full OpenClaw installation from a directory."""
    with tracer.start_as_current_span(  # type: ignore[union-attr]
        "import.openclaw_install",
        attributes={"import.source": "openclaw_install", "import.path": str(install_path)},
    ) as span:
        record_invocation_event(
            span,
            {
                "event.name": "import.openclaw_install.invocation",
                "import.source": "openclaw_install",
                "import.path": str(install_path),
            },
        )
        write_audit("info", "import.openclaw_install.invoked", {"path": str(install_path)})

        body: dict[str, object] = {}
        found: list[str] = []

        # JSON subsystem files
        for subsystem, filename in _OPENCLAW_SUBSYSTEM_FILES.items():
            fpath = install_path / filename
            if not fpath.exists():
                body[subsystem] = []
                continue
            data = _read_json_file(fpath, span, "import.openclaw_install")
            if not isinstance(data, dict) or "records" not in data:
                click.echo(
                    f"error: [import_invalid_json] {fpath}: expected object with 'records' array",
                    err=True,
                )
                write_audit(
                    "error",
                    "import.openclaw_install.failed",
                    {"code": "import_invalid_json", "message": f"{fpath}: missing 'records' key"},
                )
                sys.exit(1)
            body[subsystem] = data["records"]
            found.append(subsystem)

        # MEMORY.md — convert to memory record list
        memory_path = install_path / "MEMORY.md"
        if memory_path.exists():
            try:
                content = memory_path.read_text()
            except OSError as exc:
                record_failure(span, "import_read_failed", str(exc))  # type: ignore[arg-type]
                write_audit(
                    "error",
                    "import.openclaw_install.failed",
                    {"code": "import_read_failed", "message": str(exc)},
                )
                click.echo(f"error: [import_read_failed] {exc}", err=True)
                sys.exit(1)
            body["memory"] = [{"key": "MEMORY.md", "content": content}]
            found.append("memory")
        else:
            body["memory"] = []

        if not found:
            click.echo(
                f"warning: no OpenClaw subsystem files found in {install_path}; "
                "expected one or more of: channels.json, sessions.json, MEMORY.md, tools.json",
                err=True,
            )

        try:
            result = _client(ctx).request("POST", "/v1/x/imports/openclaw/install", json_body=body)
        except DaemonError as exc:
            record_failure(span, exc.code, exc.message)
            write_audit(
                "error",
                "import.openclaw_install.failed",
                {"code": exc.code, "message": exc.message},
            )
            click.echo(f"error: [{exc.code}] {exc.message}", err=True)
            sys.exit(1)

        span.add_event("import.openclaw_install.completed")
        if result is not None:
            click.echo(json.dumps(result, indent=2))


@imports.command("hermes")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def import_hermes(ctx: click.Context, path: Path) -> None:
    """Import from Hermes.

    PATH may be:
      - A JSON file with a 'records' array — imports skills only (legacy).
      - A directory (Hermes installation) — imports all subsystems: skills,
        environments, providers, sessions, user_profiles, cron, acp_registry.
        Each subsystem is read from a same-named .json file in the directory
        (e.g. skills.json, environments.json).  Missing files are skipped.
    """
    tracer = get_tracer()
    if path.is_file():
        _import_hermes_file(ctx, path, tracer)
    elif path.is_dir():
        _import_hermes_dir(ctx, path, tracer)
    else:
        click.echo(f"error: [import_path_invalid] {path} is not a file or directory", err=True)
        sys.exit(1)


def _import_hermes_file(ctx: click.Context, file: Path, tracer: object) -> None:
    """Backward-compatible: import skills from a Hermes JSON export file."""
    with tracer.start_as_current_span(  # type: ignore[union-attr]
        "import.hermes",
        attributes={"import.source": "hermes", "import.file": str(file)},
    ) as span:
        record_invocation_event(
            span,
            {
                "event.name": "import.hermes.invocation",
                "import.source": "hermes",
                "import.file": str(file),
            },
        )
        write_audit("info", "import.hermes.invoked", {"file": str(file)})

        body = _read_json_file(file, span, "import.hermes")

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


def _import_hermes_dir(ctx: click.Context, install_path: Path, tracer: object) -> None:
    """Import a full Hermes installation from a directory."""
    with tracer.start_as_current_span(  # type: ignore[union-attr]
        "import.hermes_install",
        attributes={"import.source": "hermes_install", "import.path": str(install_path)},
    ) as span:
        record_invocation_event(
            span,
            {
                "event.name": "import.hermes_install.invocation",
                "import.source": "hermes_install",
                "import.path": str(install_path),
            },
        )
        write_audit("info", "import.hermes_install.invoked", {"path": str(install_path)})

        body: dict[str, list] = {}
        found: list[str] = []
        for subsystem, filename in _HERMES_SUBSYSTEM_FILES.items():
            fpath = install_path / filename
            if not fpath.exists():
                body[subsystem] = []
                continue
            data = _read_json_file(fpath, span, "import.hermes_install")
            if not isinstance(data, dict) or "records" not in data:
                click.echo(
                    f"error: [import_invalid_json] {fpath}: expected object with 'records' array",
                    err=True,
                )
                write_audit(
                    "error",
                    "import.hermes_install.failed",
                    {"code": "import_invalid_json", "message": f"{fpath}: missing 'records' key"},
                )
                sys.exit(1)
            body[subsystem] = data["records"]
            found.append(subsystem)

        if not found:
            click.echo(
                f"warning: no Hermes subsystem files found in {install_path}; "
                "expected one or more of: " + ", ".join(_HERMES_SUBSYSTEM_FILES.values()),
                err=True,
            )

        try:
            result = _client(ctx).request("POST", "/v1/x/imports/hermes/install", json_body=body)
        except DaemonError as exc:
            record_failure(span, exc.code, exc.message)
            write_audit(
                "error",
                "import.hermes_install.failed",
                {"code": exc.code, "message": exc.message},
            )
            click.echo(f"error: [{exc.code}] {exc.message}", err=True)
            sys.exit(1)

        span.add_event("import.hermes_install.completed")
        if result is not None:
            click.echo(json.dumps(result, indent=2))
