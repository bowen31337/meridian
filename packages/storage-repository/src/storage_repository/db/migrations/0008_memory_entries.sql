CREATE TABLE IF NOT EXISTS memory_entries (
    id         TEXT PRIMARY KEY,
    scope      TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (scope, key)
);

CREATE INDEX IF NOT EXISTS memory_entries_scope ON memory_entries (scope)
