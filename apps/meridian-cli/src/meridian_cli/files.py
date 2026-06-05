"""files resource — standard CRUD plus a binary upload command."""

from __future__ import annotations

from pathlib import Path
import sys

import click

from ._audit import write_audit
from ._client import DaemonError
from ._resource import _client, make_crud_group
from ._telemetry import get_tracer, record_failure, record_invocation_event

files = make_crud_group("files")

_API_PATH = "/v1/x/files"


@files.command("upload")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--name", default=None, help="Override the stored file name.")
@click.pass_context
def _upload(ctx: click.Context, path: Path, name: str | None) -> None:
    """Upload a local file to the daemon file store."""
    import json

    resource = "files"
    operation = "upload"
    span_name = f"{resource}.{operation}"
    stored_name = name or path.name

    tracer = get_tracer()
    with tracer.start_as_current_span(
        span_name,
        attributes={"resource": resource, "operation": operation},
    ) as span:
        record_invocation_event(
            span,
            {
                "event.name": f"{span_name}.invocation",
                "resource": resource,
                "operation": operation,
                "file.path": str(path),
                "file.name": stored_name,
            },
        )
        write_audit(
            "info",
            f"{span_name}.invoked",
            {
                "resource": resource,
                "operation": operation,
                "file.path": str(path),
                "file.name": stored_name,
            },
        )

        try:
            data = path.read_bytes()
            result = _client(ctx).request(
                "POST",
                _API_PATH,
                json_body={"name": stored_name, "size": len(data)},
            )
            # second call: stream the binary content
            if isinstance(result, dict) and "id" in result:
                _client(ctx).request(
                    "PUT",
                    f"{_API_PATH}/{result['id']}/content",
                    content=data,
                    headers={"Content-Type": "application/octet-stream"},
                )
        except DaemonError as exc:
            record_failure(span, exc.code, exc.message)
            write_audit("error", f"{span_name}.failed", {"code": exc.code, "message": exc.message})
            click.echo(f"error: [{exc.code}] {exc.message}", err=True)
            sys.exit(1)

        span.add_event(f"{span_name}.completed")
        if result is not None:
            click.echo(json.dumps(result, indent=2))
