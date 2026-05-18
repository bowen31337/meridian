CREATE VIRTUAL TABLE IF NOT EXISTS memory_entries_vec
    USING vec0(embedding FLOAT[128] distance_metric=cosine)
