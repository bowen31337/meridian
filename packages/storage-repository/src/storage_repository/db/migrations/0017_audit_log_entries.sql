CREATE TABLE IF NOT EXISTS audit_log_entries (
    id          TEXT PRIMARY KEY,
    level       TEXT NOT NULL,
    event       TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    operation   TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    detail      TEXT,
    signature   TEXT
);

CREATE INDEX IF NOT EXISTS audit_log_entries_timestamp  ON audit_log_entries (timestamp);
CREATE INDEX IF NOT EXISTS audit_log_entries_event      ON audit_log_entries (event);
CREATE INDEX IF NOT EXISTS audit_log_entries_entity_id  ON audit_log_entries (entity_id)
