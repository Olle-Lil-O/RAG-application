---
marp: true
theme: default
paginate: true
class: lead
---

# Goal for the App/Project

- Offer an LLM chat interface where you can upload books/manuals and trust the answers are anchored in those documents
- Surface relevant snippets, similarity confidence, and reranking so the assistant stays grounded instead of hallucinating
- Keep the workflow accessible via Gradio loader + CLI while centralizing embedding/data handling

---

# RAG-application demo

- Interactive loader lets you upload books/docs and choose the target profile (mini/sm/md)
- Loaded content is chunked, embedded, and stored so it can back future conversations
- Query app allows querying LLM, and the model grounds it's answers on relevant context.

---


## Demo goal

- Collect long-form documents (books, manuals) via the Gradio loader (or pipeline CLI)
- Chunk, embed, and index those docs into pgvector/pg_search backed tables
- Query UI uses the stored docs as the source material for an LLM-powered answer

---

## Tech stack overview

- **Docker Compose**: boots `pgvector_database` (PostgreSQL 18 + pgvector + pg_search) and Flyway migrations
- **Embeddings**: prefer Hugging Face local models for mini/sm; fallback to OpenAI/Azure OpenAI for larger profiles
- **LangChain + Gradio**: The gradio chat app supports retriever tools, reranking, and a streaming the final answer
- **Utils**: `scripts/preprocess.py`, `utils/loader_utils.py`, and `chat_manager.py` share env-driven steps for consistency

---

## Architecture in brief

- **Loader apps** (`app_loader.py`, `pipeline.py`) run `scripts/preprocess.py` with shared `utils/loader_utils.py`
- **Preprocess pipeline** chunking via spaCy/semantic chunker + local/Azure embeddings, then writes to `knowledge_base_mini/sm/md`
- **Database** stores vectors (pgvector) and lexical indexes (pg_search); migrations live under `database-migrations/`

---

## Chat + retrieval

1. `app_query.py` + `ChatManagerWithTools` select a `PROFILE` (`mini`/`sm`/`md`)
2. Embeddings (Hugging Face, Azure, OpenAI) feed pgvector/pg_search retrievers + hybrid rerankers
3. Reranked context is passed to `gpt-4o-mini` and returned to the user with citations/tools

---

## Technical notes

- **Docker Compose** boots the `pgvector_database` (PostgreSQL 18 + pgvector) + Flyway migrations (`database-migrations` service)
- **Env hierarchy**: `project.env` → `.env` → `--env-file` overrides; controls PG, embeddings, Azure keys, chunking behavior
- **Key libs**: `pdfplumber`, spaCy, `sentence-transformers`, `psycopg2`, `pgvector`, `langchain`, `gradio`, `dotenv`
- **Run commands**: `uv sync`, `docker compose up -d database`, `uv run pipeline.py` / `uv run app_loader.py`, `uv run app_query.py`

