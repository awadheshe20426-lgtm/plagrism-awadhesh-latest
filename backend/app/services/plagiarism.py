from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import Document, Batch
from app.services.embedding import EmbeddingService
from app.services.text_normalize import normalize_for_compare


@dataclass
class CompareConfig:
    """Per-assignment detection tuning. Defaults preserve prior global behaviour."""
    chunk_threshold: float = 0.75
    doc_threshold: float = 0.10
    strip_quotes: bool = False
    strip_references: bool = False
    boilerplate_text: Optional[str] = None
    assignment_id: Optional[str] = None  # scope cross-batch to this assignment when set
    corpus_limit: int = 20

    def normalize(self, text: str) -> str:
        return normalize_for_compare(
            text,
            strip_quotes=self.strip_quotes,
            strip_references=self.strip_references,
            boilerplate_text=self.boilerplate_text,
        )


class PlagiarismService:
    def __init__(self, db_session: AsyncSession = None):
        self.db_session = db_session
        self.embedding_service = EmbeddingService()

    def calculate_similarity(self, embedding_a, embedding_b) -> float:
        """Calculate cosine similarity between two embeddings"""
        if embedding_a is None or embedding_b is None:
            return 0.0
        vec_a = np.array(embedding_a)
        vec_b = np.array(embedding_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    async def compare_documents(self, doc_a_text: str, doc_b_text: str,
                                precomputed_a=None, config: CompareConfig = None) -> Dict[str, Any]:
        """
        Compare two documents using chunk-based analysis.
        Returns overall similarity and specific matching passages.
        precomputed_a: optional (chunks_a, embeddings_a) tuple to avoid re-encoding
        the source document when comparing it against many candidates.
        """
        config = config or CompareConfig()
        if precomputed_a is not None:
            chunks_a, embeddings_a = precomputed_a
        else:
            chunks_a, embeddings_a = self.embedding_service.encode_chunks(config.normalize(doc_a_text))
        chunks_b, embeddings_b = self.embedding_service.encode_chunks(config.normalize(doc_b_text))

        if not embeddings_a or not embeddings_b:
            return {"score": 0.0, "matches": []}

        matches = []
        total_similarity = 0.0

        # Compare every chunk in A against every chunk in B. O(N*M); candidates are
        # already bounded to top-K by the caller, so this stays cheap at scale.
        for i, emb_a in enumerate(embeddings_a):
            best_match_score = 0.0
            best_match_idx = -1
            for j, emb_b in enumerate(embeddings_b):
                score = self.calculate_similarity(emb_a, emb_b)
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = j

            if best_match_score > config.chunk_threshold:
                matches.append({
                    "source_chunk": chunks_a[i],
                    "target_chunk": chunks_b[best_match_idx],
                    "score": round(best_match_score, 4),
                    "source_index": i,
                    "target_index": best_match_idx
                })
                total_similarity += best_match_score

        # "How much of A is found in B" = matched chunk score sum / total chunks in A.
        overall_score = total_similarity / len(chunks_a) if chunks_a else 0.0

        return {
            "score": round(overall_score, 4),
            "matches": matches,
            "details": {"chunks_a": len(chunks_a), "chunks_b": len(chunks_b)},
        }

    async def find_similar_in_batch(self, document: Document, batch_id: str,
                                    config: CompareConfig = None) -> List[Dict[str, Any]]:
        """
        Find documents similar to `document`, Turnitin-style:
        1. Exact digital copies via content hash (whole corpus, instant, no model).
        2. All other documents in the same batch (full pairwise comparison).
        3. Cross-batch semantic candidates via pgvector top-K, then detailed
           chunk-level comparison — scoped to the same ASSIGNMENT when the batch
           belongs to one, else global. Bounded cost regardless of corpus size.
        """
        if not self.db_session:
            raise ValueError("Database session required for batch search")
        config = config or CompareConfig()

        # Encode the (normalized) source document once; reuse across all comparisons.
        precomputed_a = self.embedding_service.encode_chunks(config.normalize(document.text_content))

        results = []
        seen_ids = {str(document.id)}

        # --- 1. Exact digital copies across the whole corpus (hash match) ---
        if document.content_hash:
            exact_dupes = (await self.db_session.execute(
                select(Document).where(
                    Document.content_hash == document.content_hash,
                    Document.id != document.id,
                )
            )).scalars().all()
            for dupe in exact_dupes:
                seen_ids.add(str(dupe.id))
                results.append({
                    "document_id": str(dupe.id),
                    "filename": dupe.filename,
                    "similarity": 1.0,
                    "matches": [{
                        "type": "exact_duplicate",
                        "source_chunk": "(entire document)",
                        "target_chunk": "(entire document)",
                        "score": 1.0,
                        "note": "Files are byte-identical (same content hash)"
                    }]
                })

        # --- 2. Same batch: full pairwise comparison (batch sizes are small) ---
        other_docs = (await self.db_session.execute(
            select(Document).where(
                Document.batch_id == batch_id,
                Document.id != document.id,
            )
        )).scalars().all()

        # --- 3. Cross-batch semantic candidates (scoped to assignment when set) ---
        corpus_docs = []
        doc_embedding = document.embedding
        if doc_embedding is not None and len(doc_embedding) > 0:
            corpus_query = (
                select(Document)
                .where(
                    Document.batch_id != batch_id,
                    Document.id != document.id,
                    Document.embedding.isnot(None),
                    Document.text_content.isnot(None),
                )
            )
            if config.assignment_id:
                # Only compare against submissions of the SAME assignment.
                corpus_query = corpus_query.join(Batch, Document.batch_id == Batch.id).where(
                    Batch.assignment_id == config.assignment_id
                )
            corpus_query = corpus_query.order_by(
                Document.embedding.cosine_distance(doc_embedding)
            ).limit(config.corpus_limit)
            corpus_docs = (await self.db_session.execute(corpus_query)).scalars().all()

        for other_doc in list(other_docs) + list(corpus_docs):
            if str(other_doc.id) in seen_ids:
                continue
            seen_ids.add(str(other_doc.id))
            comparison = await self.compare_documents(
                document.text_content, other_doc.text_content,
                precomputed_a=precomputed_a, config=config,
            )
            if comparison["score"] > config.doc_threshold:
                results.append({
                    "document_id": str(other_doc.id),
                    "filename": other_doc.filename,
                    "similarity": comparison["score"],
                    "matches": comparison["matches"]
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results
