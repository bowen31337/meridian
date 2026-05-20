"""
File-based SQL migration loader for the storage-repository package.

Migration files live in db/migrations/ alongside this package and are named
NNNN_description.sql where NNNN is a 4-digit zero-padded version number.
Each file may contain one or more statements separated by semicolons.

SCHEMA_VERSION is the highest version number bundled with this release.
The migration runner in SqliteRepositoryDriver.migrate() refuses to open a
database whose recorded schema version exceeds SCHEMA_VERSION.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION: int = 17

_MIGRATIONS_DIR = Path(__file__).parent / "db" / "migrations"


def load_migration_files() -> list[tuple[int, str, str]]:
    """Return (version, filename, sql) tuples sorted ascending by version."""
    result: list[tuple[int, str, str]] = []
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        version = int(path.stem.split("_")[0])
        sql = path.read_text(encoding="utf-8")
        result.append((version, path.name, sql))
    return result
