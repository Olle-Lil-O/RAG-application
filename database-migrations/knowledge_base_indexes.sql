-- Reusable index definitions for knowledge_base tables (non-versioned include file)
-- Choose ONE strategy for production: HNSW (recommended) or IVFFLAT.

-- ==========================================================
-- OPTION A: HNSW (recommended for RAG recall/latency)
-- ==========================================================
CREATE INDEX IF NOT EXISTS idx_kb_sm_embedding_hnsw
ON knowledge_base_sm
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_kb_md_embedding_hnsw
ON knowledge_base_md
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_kb_lg_embedding_hnsw
ON knowledge_base_lg
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ==========================================================
-- OPTION B: IVFFLAT (faster build, requires probe tuning)
-- Uncomment if you prefer IVFFLAT and remove/avoid HNSW above.
-- ==========================================================
-- CREATE INDEX IF NOT EXISTS idx_kb_sm_embedding_ivfflat
-- ON knowledge_base_sm
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- CREATE INDEX IF NOT EXISTS idx_kb_md_embedding_ivfflat
-- ON knowledge_base_md
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- CREATE INDEX IF NOT EXISTS idx_kb_lg_embedding_ivfflat
-- ON knowledge_base_lg
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- Refresh planner statistics after bulk loads and index creation.
-- i.e., if you use python scripts to load data, run these ANALYZE commands afterward

-- ANALYZE knowledge_base_sm;
-- ANALYZE knowledge_base_md;
-- ANALYZE knowledge_base_lg;
