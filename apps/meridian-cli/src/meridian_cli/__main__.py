from __future__ import annotations

from pathlib import Path
import sys

import click

from ._client import client_from_env
from .agents import agents
from .channels import channels
from .cron import cron
from .environments import environments
from .files import files
from .hooks import hooks
from .imports import imports
from .memory_stores import memory_stores
from .meridianconfig import meridianconfig
from .meridianrun import meridianrun
from .sessions import sessions
from .skills import skills
from .tui import meridiantui
from .user_profiles import user_profiles
from .vaults import vaults
from .webhooks import webhooks
from .workspace import UvWorkspaceInitializer, WorkspaceError


@click.group()
@click.option(
    "--socket",
    envvar="MERIDIAN_SOCKET",
    default=None,
    metavar="PATH",
    help="Unix domain socket path to the daemon (overrides --host/--port).",
)
@click.option(
    "--host",
    envvar="MERIDIAN_HOST",
    default=None,
    metavar="HOST",
    help="Daemon TCP host (default: 127.0.0.1).",
)
@click.option(
    "--port",
    envvar="MERIDIAN_PORT",
    default=None,
    type=int,
    metavar="PORT",
    help="Daemon TCP port (default: 7432).",
)
@click.pass_context
def cli(ctx: click.Context, socket: str | None, host: str | None, port: int | None) -> None:
    """Meridian command-line interface."""
    ctx.ensure_object(dict)
    ctx.obj = client_from_env(socket=socket, host=host, port=port)


# Resource subcommands
cli.add_command(agents)
cli.add_command(sessions)
cli.add_command(skills)
cli.add_command(environments)
cli.add_command(channels)
cli.add_command(vaults)
cli.add_command(memory_stores)
cli.add_command(user_profiles)
cli.add_command(webhooks)
cli.add_command(files)
cli.add_command(hooks)
cli.add_command(imports)
cli.add_command(cron)
cli.add_command(meridianconfig)
cli.add_command(meridianrun)
cli.add_command(meridiantui)


@cli.command()
@click.option(
    "--root", type=click.Path(path_type=Path), default=None, help="Repo root (default: cwd)"
)
def workspace_init(root: Path | None) -> None:
    """Initialize the uv workspace at repo root."""
    try:
        UvWorkspaceInitializer(repo_root=root).init()
    except WorkspaceError:
        sys.exit(1)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
