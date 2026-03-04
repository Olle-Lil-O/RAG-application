from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Mapping, Sequence

import psycopg2
from psycopg2 import Error as PsycopgError

Profile = Literal["mini", "sm", "md"]


@dataclass(frozen=True)
class RetrievalRow:
    id: int
    source: str
    content: str
    vector_similarity: float | None = None
    text_score: float | None = None
    rrf_score: float | None = None


_PROFILE_TABLE_DIM: Mapping[Profile, tuple[str, int]] = {
    "mini": ("knowledge_base_mini", 384),
    "sm": ("knowledge_base_sm", 1024),
    "md": ("knowledge_base_md", 1536),
}


def _profile_meta(profile: Profile) -> tuple[str, int]:
    return _PROFILE_TABLE_DIM[profile]


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _as_float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _rows_to_objects(rows: Sequence[tuple[Any, ...]]) -> list[RetrievalRow]:
    return [
        RetrievalRow(
            id=row[0],
            source=row[1],
            content=row[2],
            vector_similarity=row[3] if len(row) > 3 else None,
            text_score=row[4] if len(row) > 4 else None,
            rrf_score=row[5] if len(row) > 5 else None,
        )
        for row in rows
    ]


def retrieval_rows_to_dicts(rows: Sequence[RetrievalRow]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def semantic_retrieve(
    conn: psycopg2.extensions.connection,
    *,
    profile: Profile,
    query_embedding: Sequence[float],
    top_k: int = 8,
) -> list[RetrievalRow]:
    table_name, vector_dim = _profile_meta(profile)
    vector_list = _as_float_list(query_embedding)

    sql = f"""
        SELECT
            id,
            source,
            content,
            1.0 - (embedding <=> %s::vector({vector_dim})) AS vector_similarity,
            NULL::double precision AS text_score,
            NULL::double precision AS rrf_score
        FROM {table_name}
        ORDER BY embedding <=> %s::vector({vector_dim}), id
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute("SAVEPOINT semantic_vector_bind")
        try:
            cur.execute(sql, (vector_list, vector_list, top_k))
            cur.execute("RELEASE SAVEPOINT semantic_vector_bind")
        except PsycopgError:
            vector_literal = _vector_literal(query_embedding)
            cur.execute("ROLLBACK TO SAVEPOINT semantic_vector_bind")
            cur.execute(sql, (vector_literal, vector_literal, top_k))
            cur.execute("RELEASE SAVEPOINT semantic_vector_bind")
        rows = cur.fetchall()
    return _rows_to_objects(rows)


def lexical_retrieve(
    conn: psycopg2.extensions.connection,
    *,
    profile: Profile,
    query_text: str,
    top_k: int = 8,
    ts_config: str = "english",
) -> list[RetrievalRow]:
    table_name, _ = _profile_meta(profile)

    sql = f"""
        SELECT
            id,
            source,
            content,
            NULL::double precision AS vector_similarity,
            ts_rank_cd(content_tsv, websearch_to_tsquery(%s, %s)) AS text_score,
            NULL::double precision AS rrf_score
        FROM {table_name}
        WHERE content_tsv @@ websearch_to_tsquery(%s, %s)
        ORDER BY text_score DESC, id
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, (ts_config, query_text, ts_config, query_text, top_k))
        rows = cur.fetchall()
    return _rows_to_objects(rows)


def hybrid_retrieve(
    conn: psycopg2.extensions.connection,
    *,
    profile: Profile,
    query_text: str,
    query_embedding: Sequence[float],
    top_k: int = 8,
    candidate_k: int = 100,
    rrf_k: int = 60,
) -> list[RetrievalRow]:
    _, vector_dim = _profile_meta(profile)
    vector_list = _as_float_list(query_embedding)

    function_name = f"search_knowledge_base_{profile}_hybrid"
    sql = f"""
        SELECT id, source, content, vector_similarity, text_score, rrf_score
        FROM {function_name}(%s, %s::vector({vector_dim}), %s, %s, %s)
    """

    with conn.cursor() as cur:
        cur.execute("SAVEPOINT hybrid_vector_bind")
        try:
            cur.execute(sql, (query_text, vector_list, top_k, candidate_k, rrf_k))
            cur.execute("RELEASE SAVEPOINT hybrid_vector_bind")
        except PsycopgError:
            vector_literal = _vector_literal(query_embedding)
            cur.execute("ROLLBACK TO SAVEPOINT hybrid_vector_bind")
            cur.execute(sql, (query_text, vector_literal, top_k, candidate_k, rrf_k))
            cur.execute("RELEASE SAVEPOINT hybrid_vector_bind")
        rows = cur.fetchall()
    return _rows_to_objects(rows)


def open_connection(dsn: str) -> psycopg2.extensions.connection:
    return psycopg2.connect(dsn)
