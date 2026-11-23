import uuid
from sqlalchemy import Column, String, Text, DateTime, func, UUID, Float, Boolean, ForeignKey, JSON
from pgvector.sqlalchemy import Vector
from .base import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(UUID(as_uuid=True), ForeignKey("batches.id"))
    filename = Column(String, nullable=False)
    content_hash = Column(String)
    mime_type = Column(String)
    text_content = Column(Text)
    embedding = Column(Vector(384))  # Assuming sentence-transformers/all-MiniLM-L6-v2 embedding dim
    storage_path = Column(String)
    uploaded_by = Column(UUID(as_uuid=True))
    status = Column(String, default="queued")  # queued, processing, completed, failed
    ai_score = Column(Float, default=0.0)  # AI detection confidence score
    is_ai_generated = Column(Boolean, default=False)  # Is the text AI-generated?
    ai_confidence = Column(Float, default=0.0)  # AI detection confidence level
    ai_provider = Column(String)  # AI detection provider used
    # OCR provenance: when text came from a scan/image, surface how reliable it is so a
    # garbage extraction (low confidence) is visible instead of silently scoring ~0% similar.
    ocr_used = Column(Boolean, default=False)
    ocr_confidence = Column(Float)  # 0-100, None when no OCR was needed
    transcript = Column(Text)  # LLM-gateway transcript, set by /analyze when OCR was used
    web_search_results = Column(JSON)  # web-source check (services/web_search.py), set by the batch worker
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
