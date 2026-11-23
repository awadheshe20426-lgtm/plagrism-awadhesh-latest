from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from app.core.config import settings
from app.core.celery import app as celery  # shared Celery app — one broker, one task registry
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from app.models.batch import Batch
from app.models.document import Document
from app.models.comparison import Comparison
from app.services.embedding import EmbeddingService
from app.services.ai_detection import AIDetectionService
# from app.services.comparison import ComparisonService # Deleted
import asyncio

embedding_service = EmbeddingService()
ai_service = AIDetectionService()

@celery.task(bind=True, max_retries=2, default_retry_delay=30)
def process_batch(self, batch_id: str, provider: str = "local", ai_threshold: float = 0.5,
                  check_web: bool = True):
    """
    Process a batch of documents for plagiarism, web-source and/or AI detection.
    Retries on infrastructure errors; already-completed documents are skipped,
    so a retry never reprocesses or double-stores results.
    """
    try:
        asyncio.run(_process_batch_async(batch_id, provider, ai_threshold, check_web))
    except Exception as exc:
        raise self.retry(exc=exc)

async def _process_batch_async(batch_id: str, provider: str, ai_threshold: float, check_web: bool = True):
    # Engine must be created inside the task's event loop: asyncio.run() starts a
    # fresh loop per task, and pooled asyncpg connections cannot cross loops.
    # NullPool ensures no connection outlives this run.
    engine = create_async_engine(settings.DATABASE_URL, echo=False, poolclass=NullPool)
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        await _run_batch(SessionLocal, batch_id, provider, ai_threshold, check_web)
    finally:
        await engine.dispose()

async def _run_batch(SessionLocal, batch_id: str, provider: str, ai_threshold: float, check_web: bool = True):
    async with SessionLocal() as session:
        # Get batch and documents
        batch = await session.get(Batch, batch_id)
        if not batch:
            print(f"Batch {batch_id} not found")
            return
        
        batch.status = "processing"
        await session.commit()
        
        # Get all documents in this batch
        from sqlalchemy import select
        result = await session.execute(
            select(Document).where(Document.batch_id == batch_id)
        )
        documents = result.scalars().all()
        
        analysis_type = batch.analysis_type or "plagiarism"  # default to plagiarism

        # Instantiate PlagiarismService, with per-assignment tuning if this batch
        # belongs to an assignment (thresholds, boilerplate/quote handling, scope).
        from app.services.plagiarism import PlagiarismService, CompareConfig
        plagiarism_service = PlagiarismService(session)

        config = CompareConfig()
        if batch.assignment_id:
            from app.models.assignment import Assignment
            assignment = await session.get(Assignment, batch.assignment_id)
            if assignment:
                config = CompareConfig(
                    chunk_threshold=assignment.chunk_threshold if assignment.chunk_threshold is not None else 0.75,
                    doc_threshold=assignment.doc_threshold if assignment.doc_threshold is not None else 0.10,
                    strip_quotes=bool(assignment.strip_quotes),
                    strip_references=bool(assignment.strip_references),
                    boilerplate_text=assignment.boilerplate_text,
                    assignment_id=str(batch.assignment_id),
                )

        # Process each document
        for doc in documents:
            if doc.status == "completed":
                continue  # idempotent on retry: never reprocess or double-store
            try:
                doc.status = "processing"
                await session.commit()
                
                # AI Detection
                if analysis_type in ["ai", "both", "mixed"]:
                    if doc.text_content:
                        ai_result = ai_service.detect(doc.text_content, provider=provider, threshold=ai_threshold)
                        doc.ai_score = ai_result.get("score", 0.0)
                        doc.is_ai_generated = ai_result.get("is_ai", False)
                        doc.ai_confidence = ai_result.get("confidence", 0.0)
                        doc.ai_provider = ai_result.get("provider", "unknown")
                        
                        # Store detailed AI detection result in AIDetection table
                        from app.models.ai_detection import AIDetection
                        ai_detection_record = AIDetection(
                            document_id=doc.id,
                            model_version=ai_result.get("details", {}).get("model", "unknown"),
                            probability=ai_result.get("score", 0.0),
                            meta_data={
                                "provider": ai_result.get("provider", "unknown"),
                                "confidence": ai_result.get("confidence", 0.0),
                                "label": ai_result.get("label", "unknown"),
                                "details": ai_result.get("details", {})
                            }
                        )
                        session.add(ai_detection_record)
                
                # Plagiarism Detection (semantic similarity)
                if analysis_type in ["plagiarism", "both", "mixed"]:
                    if doc.text_content and embedding_service.model:
                        # Generate embedding (average) for legacy compatibility/search
                        embedding = embedding_service.generate_text_embedding(doc.text_content)
                        doc.embedding = embedding
                        
                        # Find similar documents (in-batch + assignment-scoped cross-batch)
                        similar_results = await plagiarism_service.find_similar_in_batch(doc, batch_id, config)
                        
                        # Store comparisons
                        for res in similar_results:
                            comparison = Comparison(
                                doc_a=doc.id,
                                doc_b=res["document_id"],
                                similarity=res["similarity"],
                                matches=res.get("matches", [])  # Store detailed matches in JSONB field
                            )
                            session.add(comparison)

                # Web-source check: which web pages was this text copied from?
                # Uses the transcript when present (it includes text inside images).
                # web_search.analyze never raises, so a web/search problem can't fail the document.
                if check_web and analysis_type in ["plagiarism", "both", "mixed"]:
                    web_text = doc.transcript or doc.text_content
                    if web_text:
                        from app.services import web_search
                        doc.web_search_results = await asyncio.to_thread(web_search.analyze, web_text)

                doc.status = "completed"
                await session.commit()
            except Exception as e:
                print(f"Error processing document {doc.id}: {e}")
                doc.status = "failed"
                await session.commit()
        
        # Update batch status
        batch.status = "completed"
        batch.processed_docs = len([d for d in documents if d.status == "completed"])
        await session.commit()
