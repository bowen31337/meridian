CREATE TABLE IF NOT EXISTS skill_versions (
    id                       TEXT PRIMARY KEY,
    skill_id                 TEXT NOT NULL,
    version_number           INTEGER NOT NULL,
    instructions             TEXT NOT NULL DEFAULT '',
    tools                    TEXT NOT NULL DEFAULT '[]',
    tests                    TEXT NOT NULL DEFAULT '[]',
    source_type              TEXT NOT NULL,
    source_url               TEXT,
    source                   TEXT NOT NULL,
    derived_from_session_ids TEXT,
    created_at               TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS skill_versions_skill_id ON skill_versions (skill_id)
