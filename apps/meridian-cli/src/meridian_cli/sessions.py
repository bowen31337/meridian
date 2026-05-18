import click

from ._resource import _run, make_crud_group

sessions = make_crud_group("sessions")

_API_PREFIX = "/v1/x/sessions"


@sessions.command("archive")
@click.argument("id")
@click.pass_context
def _archive(ctx: click.Context, id: str) -> None:
    """Archive a session's event log to the blob store."""
    _run(
        "sessions",
        "archive",
        "POST",
        f"{_API_PREFIX}/{id}/archive",
        ctx=ctx,
        event_attrs={"id": id},
    )


@sessions.command("restore")
@click.argument("id")
@click.pass_context
def _restore(ctx: click.Context, id: str) -> None:
    """Restore a previously archived session's event log."""
    _run(
        "sessions",
        "restore",
        "POST",
        f"{_API_PREFIX}/{id}/restore",
        ctx=ctx,
        event_attrs={"id": id},
    )
