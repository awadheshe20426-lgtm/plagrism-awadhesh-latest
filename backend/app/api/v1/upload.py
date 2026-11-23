"""
Transcription route.

POST /api/v1/upload  — accepts one file (multipart/form-data, field "file") that is
an image (PNG, JPG/JPEG), a plain-text file (TXT) or a document (DOCX, PDF). The file is validated,
sent to the Gemma model on the CU AI gateway with instructions to extract the text,
and the transcript is returned as JSON. Nothing is stored and no batch is created.
"""
import hashlib
import os
import re
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.api.auth import fastapi_users
from app.core.config import settings
from app.models.user import User
from app.services.llm_gateway_transcript import LLMGatewayError, LLMGatewayService

router = APIRouter(prefix="/upload", tags=["upload"])
llm_gateway_transcript = LLMGatewayService()

# Extension -> (kind, mime type). Only these are accepted.
#   image    -> file sent to the model as base64; the model reads it and returns the text
#   pdf      -> each page rendered to an image and sent to the model the same way
#   text     -> file content sent to the model as-is
#   document -> DOCX: text extracted locally and sent; each embedded image is sent to the
#               vision model and its transcript placed where the image sits ("[Image N]")
ALLOWED_TYPES = {
    ".png": ("image", "image/png"),
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".pdf": ("pdf", "application/pdf"),
    ".txt": ("text", "text/plain"),
    ".docx": ("document", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
}
MAX_FILE_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
PREVIEW_CHARS = 500  # kept for other modules that import it

# Magic-number sniffing so a renamed binary cannot pass as .png/.jpg
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"  # .docx is a zip container


def sanitize_filename(name: Optional[str]) -> str:
    """Strip directory components and unsafe characters from a client-supplied filename."""
    base = os.path.basename(name or "upload").replace("\x00", "")
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip().lstrip(".")
    return base[:255] or "upload"


def _validate(safe_name: str, content: bytes):
    """Return (kind, mime) or raise 400/413."""
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit")

    kind, mime = ALLOWED_TYPES[ext]
    if ext == ".png" and not content.startswith(_PNG_MAGIC):
        raise HTTPException(status_code=400, detail="File is not a valid PNG image")
    if ext in (".jpg", ".jpeg") and not content.startswith(_JPEG_MAGIC):
        raise HTTPException(status_code=400, detail="File is not a valid JPEG image")
    if ext == ".pdf" and not content.startswith(_PDF_MAGIC):
        raise HTTPException(status_code=400, detail="File is not a valid PDF")
    if ext == ".docx" and not content.startswith(_ZIP_MAGIC):
        raise HTTPException(status_code=400, detail="File is not a valid DOCX")
    return kind, mime


class TranscriptResponse(BaseModel):
    filename: str
    file_type: str            # "image", "pdf", "text" or "document"
    pages: Optional[int] = None   # number of pages sent to the model (PDF only)
    chunks: Optional[int] = None  # number of text chunks sent to the model (TXT/DOCX only)
    images: Optional[int] = None  # number of embedded images transcribed (DOCX only)
    mime_type: str
    size_bytes: int
    sha256: str
    model: str                              # model that produced the transcript (always the one requested)
    transcript: str
    transcript_length: int
    truncated: bool = False   # true if the model hit its output limit before finishing
    usage: Optional[dict] = None


@router.post("", response_model=TranscriptResponse)
async def transcribe_upload(
    file: UploadFile = File(..., description="PNG, JPG/JPEG, TXT, DOCX or PDF file to transcribe"),
    model_query: Optional[str] = Query(
        default=None,
        alias="model",
        description="Optional. Gateway model: qwen2.5-14b, gemma4-26b, nemotron-vl or qwen3-30b-a3b. Defaults to LLM_GATEWAY_MODEL.",
    ),
    model_form: Optional[str] = Form(
        default=None,
        alias="model",
        description="Same as the `model` query parameter, but sent as a form-data field. The query parameter wins if both are given.",
    ),
    user: User = Depends(fastapi_users.current_user()),
):
    model = model_query or model_form
    """
    Upload a file and get its text back, transcribed by the selected gateway model.
    Images and PDF pages are posted to the model directly as images and the model
    returns what it reads; text files are posted as their content. DOCX is the one
    type the model cannot view, so its text is extracted locally and then posted.
    """
    safe_name = sanitize_filename(file.filename)
    content = await file.read()
    kind, mime = _validate(safe_name, content)

    try:
        chosen_model = llm_gateway_transcript.resolve_model(model)
        if kind in ("image", "pdf") and chosen_model not in llm_gateway_transcript.vision_models:
            raise ValueError(
                f"Model '{chosen_model}' is text-only and cannot read {'images' if kind == 'image' else 'PDF pages'}. "
                f"Use one of: {', '.join(sorted(llm_gateway_transcript.vision_models))}"
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        if kind == "image":
            result = await run_in_threadpool(llm_gateway_transcript.transcribe_image, content, mime, chosen_model)
        elif kind == "pdf":
            result = await run_in_threadpool(llm_gateway_transcript.transcribe_pdf, content, chosen_model)
        elif kind == "document":
            # Text + embedded images, each image transcribed in place by the vision model.
            result = await run_in_threadpool(llm_gateway_transcript.transcribe_docx, content, chosen_model)
            if not result["transcript"] and not result["truncated"]:
                raise HTTPException(status_code=422, detail="No readable text or images found in the file")
        else:
            text = content.decode("utf-8", errors="replace")
            if not text.strip():
                raise HTTPException(status_code=422, detail="No readable text found in the file")
            result = await run_in_threadpool(llm_gateway_transcript.transcribe_text, text, chosen_model)
    except ValueError as e:  # e.g. text-only model chosen for a DOCX that contains images
        raise HTTPException(status_code=400, detail=str(e))
    except LLMGatewayError as e:
        raise HTTPException(status_code=502, detail=str(e))

    transcript = result["transcript"]
    if not transcript and result.get("truncated"):
        # The model used its whole output budget without producing text (reasoning overrun).
        raise HTTPException(
            status_code=502,
            detail=f"Model '{result['model']}' returned no transcript before hitting its output limit. "
                   f"Try a smaller file, raise LLM_GATEWAY_TRANSCRIBE_MAX_TOKENS, or choose a different model.",
        )
    return TranscriptResponse(
        filename=safe_name,
        file_type=kind,
        pages=result.get("pages"),
        chunks=result.get("chunks"),
        images=result.get("images"),
        mime_type=mime,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        model=result["model"],
        transcript=transcript,
        transcript_length=len(transcript),
        truncated=result.get("truncated", False),
        usage=result["usage"],
    )
