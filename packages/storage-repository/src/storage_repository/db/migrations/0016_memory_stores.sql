CREATE TABLE IF NOT EXISTS memory_stores (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    backend    TEXT NOT NULL,
    scope      TEXT NOT NULL,
    metadata   TEXT,
    created_at TEXT NOT NULL
)
