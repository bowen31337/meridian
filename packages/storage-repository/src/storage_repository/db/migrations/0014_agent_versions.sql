CREATE TABLE IF NOT EXISTS agent_versions (
    id             TEXT PRIMARY KEY,
    agent_id       TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL,
    config         TEXT NOT NULL DEFAULT '{}',
    capabilities   TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS agent_versions_agent_id ON agent_versions (agent_id)
