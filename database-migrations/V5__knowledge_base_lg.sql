CREATE TABLE knowledge_base_lg (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT,
    embedding VECTOR(3072),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
