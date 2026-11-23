"""
Lightweight additive schema reconciliation.

SQLAlchemy's create_all() only creates MISSING tables — it never ALTERs an
existing table, so a new column on an already-created table would silently be
absent. Until a full Alembic setup lands, this applies idempotent, additive-only
DDL (ADD COLUMN IF NOT EXISTS) so new nullable columns appear on existing tables.

Additive and reversible by design: it never drops or retypes anything.
"""
from sqlalchemy import text

# (table, column, type) — all nullable/defaulted so adding them is always safe.
ADDITIVE_COLUMNS = [
    ("batches", "assignment_id", "UUID REFERENCES assignments(id)"),
    ("batches", "dedupe_key", "VARCHAR"),          # idempotency: signature of an identical submission
    ("batches", "submitter", "VARCHAR"),           # external student ref (e.g. Moodle userid) for attribution
    ("documents", "ocr_used", "BOOLEAN DEFAULT FALSE"),
    ("documents", "ocr_confidence", "DOUBLE PRECISION"),
    # LLM-gateway transcript filled by /analyze (see services/transcription_tasks.py)
    ("documents", "transcript", "TEXT"),
    # Web-source check result (services/web_search.py), filled by the batch worker
    ("documents", "web_search_results", "JSON"),

]

# Indexes to create if missing. HNSW on the embedding turns cross-batch candidate
# search from a sequential scan into an approximate-NN lookup — the key scale fix.
INDEXES = [
    ("ix_batches_dedupe_key", "CREATE INDEX IF NOT EXISTS ix_batches_dedupe_key ON batches (dedupe_key)"),
    ("ix_documents_embedding_hnsw",
     "CREATE INDEX IF NOT EXISTS ix_documents_embedding_hnsw ON documents "
     "USING hnsw (embedding vector_cosine_ops)"),
]


async def ensure_schema(engine):
    async with engine.begin() as conn:
        for table, column, coltype in ADDITIVE_COLUMNS:
            await conn.execute(text(
                f'ALTER TABLE IF EXISTS {table} ADD COLUMN IF NOT EXISTS {column} {coltype}'
            ))
        for name, ddl in INDEXES:
            try:
                await conn.execute(text(ddl))
            except Exception as e:
                # Index build is best-effort (e.g. pgvector without HNSW support);
                # never block startup on it.
                print(f"schema_guard: index {name} skipped: {e}")
