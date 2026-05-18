CREATE TABLE IF NOT EXISTS user_profiles (
    id           TEXT PRIMARY KEY,
    username     TEXT NOT NULL UNIQUE,
    display_name TEXT,
    email        TEXT,
    metadata     TEXT,
    is_primary   INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
)
