import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, func, UUID, ForeignKey
from .base import Base

class Batch(Base):
    __tablename__ = "batches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    # Nullable for backward compatibility: batches without an assignment fall back
    # to global cross-batch comparison (previous behaviour).
    assignment_id = Column(UUID(as_uuid=True), ForeignKey("assignments.id"), nullable=True, index=True)
    name = Column(String)
    total_docs = Column(Integer)
    processed_docs = Column(Integer, default=0)
    status = Column(String)
    analysis_type = Column(String, default="plagiarism")  # plagiarism, ai, or both
    ai_provider = Column(String, default="local")  # AI detection provider
    ai_threshold = Column(Float, default=0.5)  # AI detection threshold
    dedupe_key = Column(String, index=True)  # idempotency: signature of an identical submission
    submitter = Column(String)  # external student ref (e.g. Moodle userid) for attribution
    created_at = Column(DateTime(timezone=True), server_default=func.now())
