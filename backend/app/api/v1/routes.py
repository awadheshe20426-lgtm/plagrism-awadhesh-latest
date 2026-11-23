from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import hashlib
import loguru
import os
import re
import uuid
import json

from app.core.db import get_db
from app.models.user import User
from app.api.auth import fastapi_users
from app.services.ai_detection import AIDetectionService
from app.services.plagiarism import PlagiarismService
from app.services import web_search
from app.core.provider_router import ProviderRouter, ProviderType
from app.core.config import settings

router = APIRouter()
ai_service = AIDetectionService()
# PlagiarismService needs a session, so we instantiate it per request

# Upload constraints — reject early, before any expensive work
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_FILES_PER_BATCH = settings.MAX_FILES_PER_BATCH


def sanitize_filename(name: Optional[str]) -> str:
    """Strip directory components and unsafe characters from a client-supplied filename."""
    base = os.path.basename(name or "upload").replace("\x00", "")
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip().lstrip(".")
    return base[:255] or "upload"

def _llm_transcript(content: bytes, safe_name: str, ocr_used: bool, extracted_text: str) -> Optional[str]:
    """
    LLM-gateway transcript for files that are, or contain, images (photos, handwriting):
      - image file                              -> transcribe_image
      - PDF that is scanned or has images       -> transcribe_pdf (model reads each page)
      - DOCX with embedded images               -> typed text + transcript of each image, in order
    Returns None for files without images or if the gateway fails. Never raises.
    """
    from app.api.v1.upload import llm_gateway_transcript, ALLOWED_TYPES
    from app.services.parsing import docx_images, pdf_has_images

    kind, mime = ALLOWED_TYPES.get(os.path.splitext(safe_name)[1].lower(), (None, None))
    try:
        if kind == "image":
            return llm_gateway_transcript.transcribe_image(content, mime, model="gemma4-26b")["transcript"] or None
        if kind == "pdf" and (ocr_used or pdf_has_images(content)):
            return llm_gateway_transcript.transcribe_pdf(content, model="gemma4-26b")["transcript"] or None
        if kind == "document":
            if not docx_images(content):
                return None
            # Typed text kept verbatim; each image's transcript placed where the image sits.
            r = llm_gateway_transcript.transcribe_docx(content, model="gemma4-26b", clean_text=False)
            return r["transcript"] or None
    except Exception as e:  # gateway down / timeout: never break /analyze
        loguru.logger.warning(f"LLM transcript failed for {safe_name}: {e}")
    return None


class AnalysisOptions(BaseModel):
    provider: str = Field(default=ProviderType.LOCAL, description="AI detection provider (local, openai, together)")
    ai_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    check_plagiarism: bool = True
    check_ai: bool = True
    # Web-source check (find the website the text was copied from). On by default because
    # the Moodle plugin does not send it; runs with the plagiarism check.
    check_web: bool = True
    assignment_id: Optional[str] = Field(default=None, description="Group this batch under an assignment for scoped cross-batch comparison")

class AnalysisResponse(BaseModel):
    batch_id: str
    status: str
    message: str

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(
    files: List[UploadFile] = File(default=[]),
    text: Optional[str] = Form(default=None),
    options: str = Form(default='{"provider": "local", "check_plagiarism": true, "check_ai": true}'),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """
    Unified endpoint for analyzing content (files or text).
    Supports Plagiarism and AI Detection with configurable providers.
    """
    try:
        parsed_options = json.loads(options)
        opts = AnalysisOptions(**parsed_options)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {e}")

    if not files and not text:
        raise HTTPException(status_code=400, detail="Must provide either files or text")

    if len(files) > MAX_FILES_PER_BATCH:
        raise HTTPException(status_code=400, detail=f"Too many files: max {MAX_FILES_PER_BATCH} per batch")

    # Validate every file before persisting anything, so a bad file rejects the whole batch cheaply
    for file in files:
        safe_name = sanitize_filename(file.filename)
        ext = os.path.splitext(safe_name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}' ({safe_name}). Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

    # Determine analysis type from the user's selection (was previously hardcoded)
    if opts.check_plagiarism and opts.check_ai:
        analysis_type = "both"
    elif opts.check_ai:
        analysis_type = "ai"
    else:
        analysis_type = "plagiarism"

    # Validate assignment (if given): must exist and be owned by the user (admins may use any).
    from app.models import Assignment
    assignment_uuid = None
    if opts.assignment_id:
        try:
            assignment_uuid = uuid.UUID(opts.assignment_id)
        except (ValueError, AttributeError):
            raise HTTPException(status_code=400, detail="Invalid assignment_id")
        # Any authenticated user may submit to an existing assignment (faculty create
        # the activity; students submit to it). Ownership only gates management/listing.
        assignment = (await db.execute(
            select(Assignment).where(Assignment.id == assignment_uuid)
        )).scalar_one_or_none()
        if not assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")

    # Read + hash all uploads up front so we can (a) build an idempotency signature and
    # (b) reject oversized files before persisting anything.
    file_payloads = []  # (safe_name, content, sha256)
    for file in files:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File '{sanitize_filename(file.filename)}' exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit"
            )
        file_payloads.append((sanitize_filename(file.filename), content, hashlib.sha256(content).hexdigest()))

    # Idempotency: a signature over (user, assignment, options, sorted content hashes).
    # A duplicate submit (double-click / plugin retry) within the recent window returns
    # the SAME batch instead of creating a redundant one.
    from app.models import Batch, Document
    sig_parts = [str(user.id), str(assignment_uuid), analysis_type, opts.provider, f"web={opts.check_web}",
                 hashlib.sha256((text or "").encode()).hexdigest()]
    sig_parts += sorted(h for _, _, h in file_payloads)
    dedupe_key = hashlib.sha256("|".join(sig_parts).encode()).hexdigest()

    existing = (await db.execute(
        select(Batch).where(
            Batch.dedupe_key == dedupe_key,
            Batch.user_id == user.id,
            Batch.status != "failed",
        ).order_by(Batch.created_at.desc()).limit(1)
    )).scalar_one_or_none()
    if existing:
        return AnalysisResponse(
            batch_id=str(existing.id),
            status=existing.status,
            message="Duplicate submission — returning the existing batch."
        )

    # Create Batch
    batch_id = uuid.uuid4()
    batch = Batch(
        id=batch_id,
        user_id=user.id,
        assignment_id=assignment_uuid,
        total_docs=0,
        status="queued",
        analysis_type=analysis_type,
        ai_provider=opts.provider,
        ai_threshold=opts.ai_threshold,
        dedupe_key=dedupe_key,
    )
    db.add(batch)

    # Process Text Input
    docs_to_process = []
    if text:
        doc_id = uuid.uuid4()
        doc = Document(
            id=doc_id,
            batch_id=batch_id,
            filename="input_text.txt",
            storage_path=f"{batch_id}/input_text.txt",
            text_content=text,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            uploaded_by=user.id,
            status="queued"
        )
        db.add(doc)
        docs_to_process.append(doc)

    # Process Uploaded Files (already read + hashed above)
    from app.services.parsing import extract_document
    from app.services.storage import StorageService
    storage_service = StorageService()

    for safe_name, content, content_hash in file_payloads:
        storage_path = f"{batch_id}/{safe_name}"
        storage_service.save(storage_path, content)

        # Extract text off the event loop (PDF/OCR parsing is CPU-bound)
        extracted = await run_in_threadpool(extract_document, content, safe_name)

        # File is (or contains) an image / handwriting -> also get the LLM gateway transcript
        transcript = await run_in_threadpool(
            _llm_transcript, content, safe_name, extracted["ocr_used"], extracted["text"]
        )

        doc = Document(
            batch_id=batch_id,
            filename=safe_name,
            storage_path=storage_path,
            text_content=extracted["text"],
            content_hash=content_hash,
            uploaded_by=user.id,
            ocr_used=extracted["ocr_used"],
            ocr_confidence=extracted["ocr_confidence"],
            transcript=transcript,
            status="queued"
        )
        db.add(doc)
        docs_to_process.append(doc)

    batch.total_docs = len(docs_to_process)
    await db.commit()

    # Trigger Processing (Async) - Options stored in batch
    from app.services.batch_processing import process_batch

    process_batch.delay(str(batch_id), provider=opts.provider, ai_threshold=opts.ai_threshold,
                        check_web=opts.check_web)

    return AnalysisResponse(
        batch_id=str(batch_id),
        status="queued",
        message="Analysis started successfully"
    )

# ---------------------------------------------------------------------------
# Assignments (activities) — group batches for scoped cross-batch comparison
# ---------------------------------------------------------------------------

class AssignmentCreate(BaseModel):
    name: str
    external_ref: Optional[str] = None          # e.g. Moodle cmid
    chunk_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    doc_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    strip_quotes: bool = True
    strip_references: bool = True
    boilerplate_text: Optional[str] = None


@router.post("/assignments")
async def create_assignment(
    payload: AssignmentCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """Create an assignment/activity. Batches uploaded under it are compared only against each other."""
    from app.models import Assignment
    assignment = Assignment(
        name=payload.name.strip() or "Untitled assignment",
        external_ref=payload.external_ref,
        created_by=user.id,
        chunk_threshold=payload.chunk_threshold,
        doc_threshold=payload.doc_threshold,
        strip_quotes=payload.strip_quotes,
        strip_references=payload.strip_references,
        boilerplate_text=payload.boilerplate_text,
    )
    db.add(assignment)
    await db.commit()
    return {"assignment_id": str(assignment.id), "name": assignment.name}


@router.get("/assignments")
async def list_assignments(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """List assignments (own for regular users, all for admins), with batch/submission counts."""
    from app.models import Assignment, Batch, Document
    from sqlalchemy import func

    conditions = [] if _is_admin(user) else [Assignment.created_by == user.id]
    batch_counts = (
        select(Batch.assignment_id, func.count(Batch.id).label("batch_count"))
        .group_by(Batch.assignment_id).subquery()
    )
    doc_counts = (
        select(Batch.assignment_id, func.count(Document.id).label("doc_count"))
        .join(Document, Document.batch_id == Batch.id)
        .group_by(Batch.assignment_id).subquery()
    )
    rows = (await db.execute(
        select(
            Assignment,
            func.coalesce(batch_counts.c.batch_count, 0),
            func.coalesce(doc_counts.c.doc_count, 0),
        )
        .outerjoin(batch_counts, batch_counts.c.assignment_id == Assignment.id)
        .outerjoin(doc_counts, doc_counts.c.assignment_id == Assignment.id)
        .where(*conditions)
        .order_by(Assignment.created_at.desc())
    )).all()

    return {"data": [{
        "assignment_id": str(a.id),
        "name": a.name,
        "external_ref": a.external_ref,
        "batch_count": bc,
        "submission_count": dc,
        "chunk_threshold": a.chunk_threshold,
        "doc_threshold": a.doc_threshold,
        "strip_quotes": a.strip_quotes,
        "strip_references": a.strip_references,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    } for a, bc, dc in rows]}


class AssignmentUpdate(BaseModel):
    name: Optional[str] = None
    chunk_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    doc_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    strip_quotes: Optional[bool] = None
    strip_references: Optional[bool] = None
    boilerplate_text: Optional[str] = None


async def _get_assignment_authorized(assignment_id: uuid.UUID, db: AsyncSession, user: User):
    from app.models import Assignment
    a = (await db.execute(select(Assignment).where(Assignment.id == assignment_id))).scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Assignment not found")
    if a.created_by != user.id and not _is_admin(user):
        raise HTTPException(status_code=403, detail="Not authorized for this assignment")
    return a


@router.put("/assignments/{assignment_id}")
async def update_assignment(
    assignment_id: uuid.UUID,
    payload: AssignmentUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """Edit an assignment's name/thresholds/boilerplate (owner or admin)."""
    a = await _get_assignment_authorized(assignment_id, db, user)
    for field, value in payload.dict(exclude_unset=True).items():
        setattr(a, field, value)
    await db.commit()
    return {"assignment_id": str(a.id), "name": a.name, "updated": True}


@router.delete("/assignments/{assignment_id}")
async def delete_assignment(
    assignment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """Delete an assignment (owner or admin). Its batches are detached, not deleted."""
    from app.models import Assignment, Batch
    from sqlalchemy import update as sa_update
    a = await _get_assignment_authorized(assignment_id, db, user)
    await db.execute(sa_update(Batch).where(Batch.assignment_id == assignment_id).values(assignment_id=None))
    await db.delete(a)
    await db.commit()
    return {"deleted": True}


async def _delete_batch_rows(batch_id: uuid.UUID, db: AsyncSession) -> List[str]:
    """Delete a batch and every row that references it, in FK-safe order.

    Children (embeddings, results, comparisons, ai_detection) are removed BEFORE
    the documents they reference, then the documents, then the batch itself.
    Does NOT commit — the caller commits. Returns the storage paths of the deleted
    documents so their files can be purged after the commit succeeds.
    """
    from app.models import Document, Comparison, AIDetection, Batch
    from sqlalchemy import delete as sa_delete

    rows = (await db.execute(
        select(Document.id, Document.storage_path).where(Document.batch_id == batch_id)
    )).all()
    doc_ids = [r[0] for r in rows]

    if doc_ids:
        # Rows that FK-reference documents must be deleted first, or the DELETE errors.
        # (The document's embedding is a pgvector column ON the document row, so it is
        # removed with the document — there is no separate embeddings table here.)
        await db.execute(sa_delete(Comparison).where(
            (Comparison.doc_a.in_(doc_ids)) | (Comparison.doc_b.in_(doc_ids))
        ))
        await db.execute(sa_delete(AIDetection).where(AIDetection.document_id.in_(doc_ids)))
        await db.execute(sa_delete(Document).where(Document.batch_id == batch_id))

    await db.execute(sa_delete(Batch).where(Batch.id == batch_id))
    return [path for (_id, path) in rows if path]


def _purge_files(paths: List[str]) -> None:
    """Best-effort removal of stored files (runs off the event loop)."""
    from app.services.storage import StorageService
    storage = StorageService()  # same backend the upload path used
    for path in paths:
        storage.delete(path)


@router.delete("/batches/{batch_id}")
async def delete_batch(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """Delete a single batch and ALL of its data — documents, comparisons,
    embeddings, results, AI records and the stored files (owner or admin)."""
    await _get_batch_authorized(batch_id, db, user)
    paths = await _delete_batch_rows(batch_id, db)
    await db.commit()
    await run_in_threadpool(_purge_files, paths)   # only after the DB delete committed
    return {"deleted": True, "batch_id": str(batch_id)}


class BulkDeleteRequest(BaseModel):
    batch_ids: List[uuid.UUID] = Field(..., description="Batch ids to delete")


@router.post("/batches/bulk-delete")
async def bulk_delete_batches(
    payload: BulkDeleteRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """Delete several batches in one call (owner or admin per batch).

    Partial success: each id is committed independently, so a not-found or
    unauthorized id never rolls back the others. The response reports exactly what
    was deleted and why anything was skipped.
    """
    # De-duplicate while preserving order; cap the batch size.
    seen, ids = set(), []
    for bid in payload.batch_ids:
        if bid not in seen:
            seen.add(bid)
            ids.append(bid)
    if not ids:
        raise HTTPException(status_code=400, detail="batch_ids must not be empty")
    if len(ids) > 200:
        raise HTTPException(status_code=400, detail="Too many batch_ids (max 200 per request)")

    deleted: List[str] = []
    errors: Dict[str, str] = {}
    all_paths: List[str] = []

    for bid in ids:
        try:
            await _get_batch_authorized(bid, db, user)
            all_paths.extend(await _delete_batch_rows(bid, db))
            await db.commit()               # isolate each deletion
            deleted.append(str(bid))
        except HTTPException as e:
            await db.rollback()
            errors[str(bid)] = str(e.detail)
        except Exception as e:              # unexpected — don't sink the whole request
            await db.rollback()
            errors[str(bid)] = f"delete_failed: {type(e).__name__}"

    await run_in_threadpool(_purge_files, all_paths)
    return {
        "requested": len(ids),
        "deleted_count": len(deleted),
        "deleted": deleted,
        "errors": errors,
    }


@router.get("/assignments/{assignment_id}/cohort")
async def assignment_cohort_report(
    assignment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """
    Cohort view for an assignment: every similar pair of submissions across all its
    batches, ranked by similarity — the "who copied whom" table a teacher wants at
    grading time. Deduplicated so A↔B appears once, with both owners named.
    """
    from app.models import Assignment, Batch, Document, Comparison
    from sqlalchemy.orm import aliased

    await _get_assignment_authorized(assignment_id, db, user)

    from sqlalchemy import func
    DocA, DocB = aliased(Document), aliased(Document)
    BatchA, BatchB = aliased(Batch), aliased(Batch)
    UserA, UserB = aliased(User), aliased(User)

    rows = (await db.execute(
        select(
            DocA.filename, func.coalesce(BatchA.submitter, UserA.email), BatchA.id,
            DocB.filename, func.coalesce(BatchB.submitter, UserB.email), BatchB.id,
            Comparison.similarity,
        )
        .join(DocA, Comparison.doc_a == DocA.id)
        .join(DocB, Comparison.doc_b == DocB.id)
        .join(BatchA, DocA.batch_id == BatchA.id)
        .join(BatchB, DocB.batch_id == BatchB.id)
        .outerjoin(UserA, BatchA.user_id == UserA.id)
        .outerjoin(UserB, BatchB.user_id == UserB.id)
        .where(BatchA.assignment_id == assignment_id, BatchB.assignment_id == assignment_id)
        .order_by(Comparison.similarity.desc())
    )).all()

    seen, pairs = set(), []
    for fa, ea, ba, fb, eb, bb, sim in rows:
        key = tuple(sorted([f"{ba}:{fa}", f"{bb}:{fb}"]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append({
            "submission_a": fa, "owner_a": ea,
            "submission_b": fb, "owner_b": eb,
            "similarity": round(sim, 4),
            "cross_batch": ba != bb,
        })

    return {"assignment_id": str(assignment_id), "pair_count": len(pairs), "pairs": pairs}


@router.get("/ai-detection/health")
async def ai_health_check():
    """Health check for AI detection service."""
    health_status = ai_service.health_check()
    return {"service": "ai_detection", "health": health_status}

@router.post("/ai-detection")
async def detect_ai_only(
    text: str = Body(..., embed=True),
    provider: str = Body("local", embed=True),
    threshold: float = Body(0.5, embed=True),
    user: User = Depends(fastapi_users.current_user())
):
    """
    Direct AI detection endpoint for text.
    """
    try:
        ai_result = ai_service.detect(text, provider=provider, threshold=threshold)
        
        # Create temporary document to store result
        from app.models.document import Document
        from app.models.ai_detection import AIDetection
        from sqlalchemy.orm import Session
        from app.core.db import get_db
        
        # Note: For this endpoint, we're returning the result directly
        # In a full implementation, we might want to store this in the DB
        return {
            "is_ai": ai_result["is_ai"],
            "score": ai_result["score"],
            "confidence": ai_result["confidence"],
            "label": ai_result["label"],
            "provider": ai_result["provider"],
            "details": ai_result["details"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI detection failed: {str(e)}")

def _is_admin(user: User) -> bool:
    return getattr(user, "role", None) == "admin"


async def _get_batch_authorized(batch_id: uuid.UUID, db: AsyncSession, user: User):
    """Fetch a batch the user may access: owners see their own, admins see any."""
    from app.models import Batch
    from sqlalchemy import select

    conditions = [Batch.id == batch_id]
    if not _is_admin(user):
        conditions.append(Batch.user_id == user.id)
    batch = (await db.execute(select(Batch).where(*conditions))).scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch


async def _build_batch_results(batch_id: uuid.UUID, db: AsyncSession, user: User) -> List[Dict[str, Any]]:
    """
    Shared result builder: per-document AI scores plus plagiarism matches,
    each match annotated with the owner (user/batch) of the matched source document.
    """
    from app.models import Batch, Document, Comparison
    from sqlalchemy import select
    from sqlalchemy.orm import aliased

    batch = await _get_batch_authorized(batch_id, db, user)

    documents = (await db.execute(
        select(Document).where(Document.batch_id == batch_id)
    )).scalars().all()

    results = []
    for doc in documents:
        # Plagiarism comparisons, joined to the matched document, its batch and its owner
        DocB = aliased(Document)
        comparisons = (await db.execute(
            select(
                Comparison,
                DocB.filename.label("match_filename"),
                DocB.batch_id.label("match_batch_id"),
                DocB.created_at.label("match_uploaded_at"),
                User.email.label("match_owner_email"),
            )
            .join(DocB, Comparison.doc_b == DocB.id)
            .outerjoin(Batch, DocB.batch_id == Batch.id)
            .outerjoin(User, Batch.user_id == User.id)
            .where(Comparison.doc_a == doc.id)
            .order_by(Comparison.similarity.desc())
        )).all()

        plagiarism_details = []
        for comp, match_filename, match_batch_id, match_uploaded_at, match_owner_email in comparisons:
            plagiarism_details.append({
                "similar_document": match_filename,
                "similarity": comp.similarity,
                "matches": comp.matches or [],  # Detailed chunks
                "source_batch_id": str(match_batch_id) if match_batch_id else None,
                "source_owner": match_owner_email,
                "source_uploaded_at": match_uploaded_at.isoformat() if match_uploaded_at else None,
                "cross_batch": match_batch_id != doc.batch_id,
            })

        # OCR provenance: warn when text came from a low-confidence scan.
        ocr_info = None
        if getattr(doc, "ocr_used", False):
            conf = doc.ocr_confidence
            ocr_info = {
                "used": True,
                "confidence": conf,
                "low_confidence": conf is not None and conf < 60,
            }

        results.append({
            "document_id": str(doc.id),
            "filename": doc.filename,
            "status": doc.status,
            "ocr": ocr_info,
            # LLM transcript (gemma4-26b) for images, scanned/image PDFs and DOCX with
            # images; null for plain-text files.
            "transcript": doc.transcript or None,
            "transcript_length": len(doc.transcript) if doc.transcript else 0,
            "ai_analysis": {
                "score": doc.ai_score,
                "is_ai": doc.is_ai_generated,
                "confidence": doc.ai_confidence,
                "provider": doc.ai_provider,
                # AI detection is probabilistic and imperfect — always advisory, never proof.
                "advisory": True,
                "note": "AI-detection scores are indicative only and should be reviewed by a human, not used as sole proof.",
            },
            "plagiarism_analysis": plagiarism_details,
            # Web pages this text was found on (services/web_search.py). Key/field names
            # match what the Moodle plugin's WEBSOURCE tab reads.
            # Same keys whatever the status, at every level (see web_search.normalize_result);
            # rows stored by older code get missing keys filled with defaults.
            "web_search_analysis": web_search.normalize_result(doc.web_search_results),
        })

    return results


@router.get("/batches/{batch_id}/status")
async def get_batch_status(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """
    Lightweight progress endpoint for live polling: batch status plus per-document
    status counts and a percentage. Cheap enough to poll every couple of seconds.
    """
    from app.models import Batch, Document
    from sqlalchemy import func

    batch = await _get_batch_authorized(batch_id, db, user)
    rows = (await db.execute(
        select(Document.status, func.count(Document.id))
        .where(Document.batch_id == batch_id)
        .group_by(Document.status)
    )).all()
    counts = {status: n for status, n in rows}
    total = sum(counts.values())
    done = counts.get("completed", 0) + counts.get("failed", 0)
    return {
        "batch_id": str(batch_id),
        "status": batch.status,
        "total": total,
        "completed": counts.get("completed", 0),
        "failed": counts.get("failed", 0),
        "processing": counts.get("processing", 0),
        "queued": counts.get("queued", 0),
        "percent": round(100 * done / total) if total else 0,
        "done": batch.status in ("completed", "failed") or (total > 0 and done == total),
    }


@router.get("/batches")
async def list_batches(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """
    List analysis batches, newest first.
    Regular users see their own batches; admins see all batches with owner info.
    Paginated (limit/offset) so it stays cheap at large scale.
    """
    from app.models import Batch, Document
    from sqlalchemy import select, func

    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    doc_counts = (
        select(
            Document.batch_id,
            func.count(Document.id).label("doc_count"),
            func.count(func.nullif(Document.transcript, "")).label("transcript_count"),
        )
        .group_by(Document.batch_id)
        .subquery()
    )

    conditions = [] if _is_admin(user) else [Batch.user_id == user.id]

    total = (await db.execute(
        select(func.count(Batch.id)).where(*conditions)
    )).scalar() or 0

    rows = (await db.execute(
        select(
            Batch,
            User.email.label("owner_email"),
            func.coalesce(doc_counts.c.doc_count, 0).label("doc_count"),
            func.coalesce(doc_counts.c.transcript_count, 0).label("transcript_count"),
        )
        .outerjoin(User, Batch.user_id == User.id)
        .outerjoin(doc_counts, doc_counts.c.batch_id == Batch.id)
        .where(*conditions)
        .order_by(Batch.created_at.desc())
        .limit(limit)
        .offset(offset)
    )).all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": [{
            "batch_id": str(batch.id),
            "owner": owner_email,
            "status": batch.status,
            "analysis_type": batch.analysis_type,
            "total_docs": doc_count,
            "transcript_count": transcript_count,
            "created_at": batch.created_at.isoformat() if batch.created_at else None,
        } for batch, owner_email, doc_count, transcript_count in rows]
    }


@router.get("/batches/{batch_id}/transcripts")
async def get_batch_transcripts(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """LLM-gateway transcripts of every document in the batch (null where none was made)."""
    from app.models import Document
    from sqlalchemy import select

    await _get_batch_authorized(batch_id, db, user)
    docs = (await db.execute(
        select(Document.id, Document.filename, Document.ocr_used, Document.transcript)
        .where(Document.batch_id == batch_id)
        .order_by(Document.filename)
    )).all()
    return {
        "batch_id": str(batch_id),
        "documents": [{
            "document_id": str(doc_id),
            "filename": filename,
            "ocr_used": bool(ocr_used),
            "transcript": transcript or None,
        } for doc_id, filename, ocr_used, transcript in docs],
    }


async def _top_matches_for_batch(batch_id: uuid.UUID, db: AsyncSession) -> Dict[str, Dict[str, Any]]:
    """Best plagiarism match per document in the batch, from stored comparisons."""
    from app.models import Batch, Document, Comparison
    from sqlalchemy import select
    from sqlalchemy.orm import aliased

    DocB = aliased(Document)
    rows = (await db.execute(
        select(
            Comparison.doc_a,
            Comparison.similarity,
            DocB.filename,
            User.email.label("owner_email"),
        )
        .join(DocB, Comparison.doc_b == DocB.id)
        .outerjoin(Batch, DocB.batch_id == Batch.id)
        .outerjoin(User, Batch.user_id == User.id)
        .where(Comparison.doc_a.in_(select(Document.id).where(Document.batch_id == batch_id)))
        .order_by(Comparison.doc_a, Comparison.similarity.desc())
    )).all()

    top: Dict[str, Dict[str, Any]] = {}
    for doc_a, similarity, filename, owner_email in rows:
        key = str(doc_a)
        if key not in top:  # rows are sorted best-first per document
            top[key] = {
                "similarity": similarity,
                "matched_filename": filename,
                "matched_owner": owner_email,
            }
    return top


@router.get("/batches/{batch_id}/report.csv")
async def export_batch_csv(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """Download the batch report as CSV (owner or admin)."""
    from fastapi.responses import Response
    from app.models import Document
    from sqlalchemy import select
    from app.services.report import ReportService

    await _get_batch_authorized(batch_id, db, user)
    documents = (await db.execute(
        select(Document).where(Document.batch_id == batch_id)
    )).scalars().all()
    top_matches = await _top_matches_for_batch(batch_id, db)

    csv_content = ReportService.generate_csv_report(documents, top_matches)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="batch_{batch_id}_report.csv"'},
    )


@router.get("/batches/{batch_id}/report.pdf")
async def export_batch_pdf(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """Download the batch report as PDF (owner or admin)."""
    from fastapi.responses import Response
    from app.models import Document
    from sqlalchemy import select
    from app.services.report import ReportService

    batch = await _get_batch_authorized(batch_id, db, user)
    documents = (await db.execute(
        select(Document).where(Document.batch_id == batch_id)
    )).scalars().all()
    top_matches = await _top_matches_for_batch(batch_id, db)

    pdf_content = await run_in_threadpool(
        ReportService.generate_pdf_report, batch, documents, top_matches
    )
    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="batch_{batch_id}_report.pdf"'},
    )


@router.get("/batches/{batch_id}/results")
async def get_batch_results(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """
    Get detailed results for a batch, including AI scores and plagiarism matches.
    """
    results = await _build_batch_results(batch_id, db, user)
    return {"status": "ok", "data": results}


@router.get("/batch/{batch_id}/results")
async def get_batch_results_flat(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(fastapi_users.current_user())
):
    """
    Flat view of batch results (shape expected by the BatchResultsPage frontend route).
    One row per (document, match) pair.
    """
    results = await _build_batch_results(batch_id, db, user)
    flat = []
    for doc in results:
        if doc["plagiarism_analysis"]:
            for match in doc["plagiarism_analysis"]:
                flat.append({
                    "document_name": doc["filename"],
                    "similarity": match["similarity"],
                    "similar_document_name": (
                        f'{match["similar_document"]}'
                        + (f' (by {match["source_owner"]})' if match.get("source_owner") else '')
                        + (' [other batch]' if match.get("cross_batch") else '')
                    ),
                    "transcript": doc["transcript"],
                    "transcript_length": doc["transcript_length"],
                })
        else:
            flat.append({
                "document_name": doc["filename"],
                "similarity": 0.0,
                "similar_document_name": "no matches found",
                "transcript": doc["transcript"],
                "transcript_length": doc["transcript_length"],
            })
    return {"data": flat}
