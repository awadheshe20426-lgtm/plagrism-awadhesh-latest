import uuid
from sqlalchemy import Column, String, Float, Boolean, Text, DateTime, func, UUID, ForeignKey
from .base import Base


class Assignment(Base):
    """
    An assignment/activity groups the batches submitted for the same task
    (e.g. "What is thermodynamics"). Cross-batch plagiarism is scoped to the
    assignment: every submission is compared against all other submissions of
    the SAME assignment, across batches and users. Also carries the per-assignment
    detection tuning so thresholds and boilerplate handling are not global magic numbers.
    """
    __tablename__ = "assignments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    # Optional external key so a Moodle activity (cmid) can map 1:1 to an assignment.
    external_ref = Column(String, index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))

    # Per-assignment detection tuning (replaces hardcoded 0.75 / 0.10)
    chunk_threshold = Column(Float, default=0.75)   # chunk cosine sim to count as a match
    doc_threshold = Column(Float, default=0.10)     # document score to report at all

    # Legitimate-overlap mitigation (see PlagiarismService.normalize_for_compare)
    strip_quotes = Column(Boolean, default=True)        # ignore quoted passages
    strip_references = Column(Boolean, default=True)    # ignore reference/bibliography lines
    boilerplate_text = Column(Text)                     # e.g. the assignment prompt, masked out

    created_at = Column(DateTime(timezone=True), server_default=func.now())
