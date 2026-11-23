"""
Moodle server-to-server integration.

A Moodle `plagiarism_*` plugin authenticates with a static X-API-Key (not a
per-user JWT) and posts each Assignment submission here. We map the Moodle
activity (cmid) to an Assignment so all submissions for the same activity are
compared against each other (in-batch + cross-batch), reusing the exact same
pipeline as the web UI. Results (and an embeddable HTML report) come back keyed
by batch id, which the plugin stores against the submission.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Header
from fastapi.responses import HTMLResponse
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import hashlib
import html
import secrets
import uuid

from app.core.db import get_db
from app.core.config import settings
from app.models import Assignment, Batch, Document, User

router = APIRouter(prefix="/moodle", tags=["moodle"])


async def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Gate every Moodle endpoint behind the shared secret (constant-time compare)."""
    if not settings.MOODLE_API_KEY:
        raise HTTPException(status_code=503, detail="Moodle integration is not configured (MOODLE_API_KEY unset)")
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.MOODLE_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
    return True


async def _service_user(db: AsyncSession) -> User:
    """
    The batches created by Moodle are owned by a dedicated service account,
    auto-provisioned via the ORM. We bypass the UserCreate/EmailStr path because
    this is an internal, non-deliverable account (the address may use a reserved
    TLD like .local) and it never logs in interactively.
    """
    from app.api.auth import pwd_context
    user = (await db.execute(select(User).where(User.email == settings.MOODLE_SERVICE_EMAIL))).scalar_one_or_none()
    if user:
        return user
    user = User(
        email=settings.MOODLE_SERVICE_EMAIL,
        hashed_password=pwd_context.hash(secrets.token_urlsafe(32)),
        role="admin",
        is_active=True,
    )
    db.add(user)
    await db.flush()
    return user


async def _assignment_for_cmid(db: AsyncSession, cmid: str, name: Optional[str], owner_id) -> Assignment:
    """Map a Moodle course-module id (cmid) 1:1 to an Assignment via external_ref."""
    a = (await db.execute(select(Assignment).where(Assignment.external_ref == str(cmid)))).scalar_one_or_none()
    if a:
        return a
    a = Assignment(name=(name or f"Moodle activity {cmid}"), external_ref=str(cmid), created_by=owner_id)
    db.add(a)
    await db.flush()
    return a


@router.post("/submit")
async def moodle_submit(
    file: UploadFile = File(...),
    cmid: str = Form(...),
    userid: str = Form(...),
    activity_name: Optional[str] = Form(None),
    check_ai: bool = Form(True),
    _: bool = Depends(require_api_key),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept one student submission for a Moodle activity. Creates/reuses the
    assignment for `cmid`, creates a batch for this submission, and queues analysis.
    Returns {batch_id, assignment_id} — poll /moodle/status/{batch_id}.
    """
    from app.api.v1.routes import sanitize_filename, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
    from app.services.parsing import extract_document
    from app.services.storage import StorageService
    from app.services.batch_processing import process_batch
    import os

    safe_name = sanitize_filename(file.filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")

    service = await _service_user(db)
    assignment = await _assignment_for_cmid(db, cmid, activity_name, service.id)

    content_hash = hashlib.sha256(content).hexdigest()
    dedupe_key = hashlib.sha256(f"moodle|{cmid}|{userid}|{content_hash}|{check_ai}".encode()).hexdigest()
    existing = (await db.execute(
        select(Batch).where(Batch.dedupe_key == dedupe_key, Batch.status != "failed")
        .order_by(Batch.created_at.desc()).limit(1)
    )).scalar_one_or_none()
    if existing:
        return {"batch_id": str(existing.id), "assignment_id": str(assignment.id), "status": existing.status, "duplicate": True}

    batch_id = uuid.uuid4()
    batch = Batch(
        id=batch_id, user_id=service.id, assignment_id=assignment.id, total_docs=1,
        status="queued", analysis_type=("both" if check_ai else "plagiarism"),
        ai_provider="local", ai_threshold=0.5, dedupe_key=dedupe_key,
        name=f"moodle:cmid={cmid}:user={userid}", submitter=f"student:{userid}",
    )
    db.add(batch)

    storage_path = f"{batch_id}/{safe_name}"
    StorageService().save(storage_path, content)
    extracted = await run_in_threadpool(extract_document, content, safe_name)
    db.add(Document(
        batch_id=batch_id, filename=safe_name, storage_path=storage_path,
        text_content=extracted["text"], content_hash=content_hash, uploaded_by=service.id,
        ocr_used=extracted["ocr_used"], ocr_confidence=extracted["ocr_confidence"], status="queued",
    ))
    await db.commit()

    process_batch.delay(str(batch_id), provider="local", ai_threshold=0.5)
    return {"batch_id": str(batch_id), "assignment_id": str(assignment.id), "status": "queued"}


@router.get("/status/{batch_id}")
async def moodle_status(batch_id: uuid.UUID, _: bool = Depends(require_api_key), db: AsyncSession = Depends(get_db)):
    from sqlalchemy import func
    batch = (await db.execute(select(Batch).where(Batch.id == batch_id))).scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    rows = (await db.execute(
        select(Document.status, func.count(Document.id)).where(Document.batch_id == batch_id).group_by(Document.status)
    )).all()
    counts = {s: n for s, n in rows}
    total = sum(counts.values())
    done = counts.get("completed", 0) + counts.get("failed", 0)
    return {"batch_id": str(batch_id), "status": batch.status, "percent": round(100 * done / total) if total else 0,
            "done": batch.status in ("completed", "failed") or (total and done == total)}


async def _results(db: AsyncSession, batch_id: uuid.UUID):
    from sqlalchemy.orm import aliased
    from sqlalchemy import func
    from app.models import Comparison
    docs = (await db.execute(select(Document).where(Document.batch_id == batch_id))).scalars().all()
    out = []
    for doc in docs:
        DocB = aliased(Document)
        comps = (await db.execute(
            # Prefer the external submitter (Moodle student id) for attribution; fall
            # back to the account email for web-UI submissions.
            select(Comparison.similarity, DocB.filename,
                   func.coalesce(Batch.submitter, User.email))
            .join(DocB, Comparison.doc_b == DocB.id)
            .outerjoin(Batch, DocB.batch_id == Batch.id)
            .outerjoin(User, Batch.user_id == User.id)
            .where(Comparison.doc_a == doc.id).order_by(Comparison.similarity.desc())
        )).all()
        out.append({
            "filename": doc.filename, "status": doc.status,
            "ai_score": doc.ai_score, "is_ai": doc.is_ai_generated,
            "top_similarity": comps[0][0] if comps else 0.0,
            "matches": [{"similar_document": f, "owner": e, "similarity": round(s, 4)} for s, f, e in comps],
        })
    return out


@router.get("/results/{batch_id}")
async def moodle_results(batch_id: uuid.UUID, _: bool = Depends(require_api_key), db: AsyncSession = Depends(get_db)):
    if not (await db.execute(select(Batch.id).where(Batch.id == batch_id))).scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Batch not found")
    return {"batch_id": str(batch_id), "results": await _results(db, batch_id)}


@router.get("/report/{batch_id}", response_class=HTMLResponse)
async def moodle_report(batch_id: uuid.UUID, _: bool = Depends(require_api_key), db: AsyncSession = Depends(get_db)):
    """Self-contained HTML report suitable for embedding in the Moodle grader view (iframe)."""
    if not (await db.execute(select(Batch.id).where(Batch.id == batch_id))).scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Batch not found")
    results = await _results(db, batch_id)

    def color(v):
        return "#c0392b" if v >= 0.7 else "#e67e22" if v >= 0.3 else "#27ae60"

    rows = ""
    for r in results:
        sim = r["top_similarity"] or 0.0
        ai = r["ai_score"] or 0.0
        matches = "; ".join(
            f'{html.escape(m["similar_document"])} ({m["similarity"]*100:.0f}%'
            + (f', {html.escape(m["owner"])}' if m["owner"] else '') + ')'
            for m in r["matches"][:5]
        ) or "—"
        rows += (
            f'<tr><td>{html.escape(r["filename"])}</td>'
            f'<td style="color:{color(sim)};font-weight:700">{sim*100:.1f}%</td>'
            f'<td style="color:{color(ai)}">{ai*100:.1f}%</td>'
            f'<td>{matches}</td></tr>'
        )

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Similarity report</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:16px;color:#222}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}}
th{{background:#f4f4f4}} caption{{text-align:left;font-weight:700;margin-bottom:8px}}
.note{{color:#888;font-size:12px;margin-top:10px}}</style></head><body>
<table><caption>Plagiarism &amp; AI similarity report</caption>
<tr><th>Submission</th><th>Plagiarism</th><th>AI (advisory)</th><th>Top matches</th></tr>
{rows}</table>
<p class="note">AI scores are advisory and should be reviewed by a human, not used as sole proof.</p>
</body></html>"""
    return HTMLResponse(content=page)
