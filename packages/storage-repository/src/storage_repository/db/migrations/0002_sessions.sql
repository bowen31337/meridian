CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    agent_id   TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'active',
    metadata   TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sessions_agent_id ON sessions (agent_id)
