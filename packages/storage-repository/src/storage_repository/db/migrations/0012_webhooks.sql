CREATE TABLE IF NOT EXISTS webhooks (
    id         TEXT PRIMARY KEY,
    url        TEXT NOT NULL,
    events     TEXT NOT NULL DEFAULT '[]',
    secret_ref TEXT,
    status     TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
