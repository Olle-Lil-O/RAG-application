
CREATE TABLE knowledge_base (
    id SERIAL PRIMARY KEY,
    file TEXT NOT NULL,
    type TEXT NOT NULL, -- 'function', 'class', 'document', etc.
    name TEXT,
    docstring TEXT,
    start_line INTEGER,
    content TEXT,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
