CREATE TABLE IF NOT EXISTS tool_calls (
    id         TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    tool_name  TEXT NOT NULL,
    input      TEXT NOT NULL DEFAULT '{}',
    output     TEXT,
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS tool_calls_message_id ON tool_calls (message_id);

CREATE INDEX IF NOT EXISTS tool_calls_session_id ON tool_calls (session_id)
