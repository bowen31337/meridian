CREATE TABLE IF NOT EXISTS threads (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS threads_session_id ON threads (session_id)
