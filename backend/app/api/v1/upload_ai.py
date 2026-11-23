"""
Upload + AI detection route.

POST /api/v1/upload-ai  — accepts one file (multipart/form-data, field "file"),
validates and stores it, extracts its text, then runs AI-generated-text detection
on that text and returns the verdict together with the upload metadata.
Synchronous (no batch / Celery): the caller gets the result in the same response.
"""
import hashlib
import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.api.auth import fastapi_users
from app.api.v1.upload import MAX_FILE_SIZE_BYTES, PREVIEW_CHARS, sanitize_filename
from app.core.config import settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
from app.core.provider_router import ProviderType
from app.models.user import User
from app.services.ai_detection import AIDetectionService
from app.services.parsing import extract_document
from app.services.storage import StorageService

router = APIRouter(prefix="/upload-ai", tags=["upload-ai"])
ai_service = AIDetectionService()


class AIVerdict(BaseModel):
    is_ai: bool
    score: float
    confidence: float
    label: str
    provider: str
    details: dict
    advisory: bool = True
    note: str = "AI-detection scores are indicative only and should be reviewed by a human, not used as sole proof."


class UploadAIResponse(BaseModel):
    upload_id: str
    filename: str
    size_bytes: int
    sha256: str
    storage_path: str
    text_length: int
    text_preview: str
    ocr_used: bool
    ocr_confidence: Optional[float]
    ai_analysis: AIVerdict


@router.post("", response_model=UploadAIResponse)
async def upload_and_detect_ai(
    file: UploadFile = File(..., description="The file to upload and check"),
    provider: str = Form(default=ProviderType.LOCAL, description="local, openai or together"),
    threshold: float = Form(default=0.5, ge=0.0, le=1.0, description="Score above which text is labelled AI"),
    user: User = Depends(fastapi_users.current_user()),
):
    """
    Upload a single file and run AI-generated-text detection on its contents.
    Validates type and size, stores the file, extracts text (OCR for images /
    scanned PDFs), then classifies the text with the chosen provider.
    """
    safe_name = sanitize_filename(file.filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")

    upload_id = uuid.uuid4()
    storage_path = f"uploads-ai/{user.id}/{upload_id}/{safe_name}"
    StorageService().save(storage_path, content)

    extracted = await run_in_threadpool(extract_document, content, safe_name)
    text = (extracted["text"] or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="No text could be extracted from the file")

    # Model inference is CPU-bound and may call an external API: keep it off the event loop.
    result = await run_in_threadpool(ai_service.detect, text, provider, threshold)
    if result.get("label") == "Error":
        raise HTTPException(status_code=502, detail=result.get("details", {}).get("error", "AI detection failed"))

    return UploadAIResponse(
        upload_id=str(upload_id),
        filename=safe_name,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        storage_path=storage_path,
        text_length=len(text),
        text_preview=text[:PREVIEW_CHARS],
        ocr_used=bool(extracted["ocr_used"]),
        ocr_confidence=extracted["ocr_confidence"],
        ai_analysis=AIVerdict(
            is_ai=result["is_ai"],
            score=result["score"],
            confidence=result["confidence"],
            label=result["label"],
            provider=getattr(result["provider"], "value", str(result["provider"])),
            details=result.get("details", {}),
        ),
    )
