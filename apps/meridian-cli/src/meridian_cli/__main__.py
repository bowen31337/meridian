from __future__ import annotations

import sys
from pathlib import Path

import click

from .workspace import UvWorkspaceInitializer, WorkspaceError


@click.group()
def cli() -> None:
    """Meridian command-line interface."""


@cli.command()
@click.option("--root", type=click.Path(path_type=Path), default=None, help="Repo root (default: cwd)")
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
