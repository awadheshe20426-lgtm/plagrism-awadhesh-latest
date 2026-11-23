from fastapi import UploadFile
from pdfminer.high_level import extract_text
import docx
import inspect
import io
import os
import re
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import tempfile
from typing import Optional


async def extract_text_from_file(file: UploadFile) -> str:
    """
    Extracts text from a file, supporting .txt, .docx, .pdf, and image formats (.png, .jpg, .jpeg).
    Accepts a FastAPI UploadFile (async read) or any file-like object (sync read).
    """
    content = file.read()
    if inspect.isawaitable(content):
        content = await content
    filename = getattr(file, "filename", None) or getattr(file, "name", "") or ""
    return extract_text_from_bytes(content, filename)


def extract_text_from_bytes(content: bytes, filename: str) -> str:
    """Backward-compatible: returns only the extracted text."""
    return extract_document(content, filename)["text"]


def _ocr_image(image) -> tuple:
    """OCR one image, returning (text, mean_confidence 0-100)."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        if word and word.strip():
            words.append(word)
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1
            if c >= 0:
                confs.append(c)
    text = " ".join(words)
    confidence = round(sum(confs) / len(confs), 1) if confs else 0.0
    return text, confidence


# --- Embedded-image detection (images / handwriting inside documents) ---------------
_PDF_IMAGE_RE = re.compile(rb"/Subtype\s*/Image")
MIN_IMAGE_SIDE = 150   # px: smaller embedded images are icons/logos, not content
MAX_DOC_IMAGES = 20    # cap images sent to the model per document


def pdf_has_images(content: bytes) -> bool:
    """True if the PDF contains any image (photo, scan, handwriting...), including
    inline images inside compressed content streams that the raw-bytes regex can't see."""
    if _PDF_IMAGE_RE.search(content):
        return True
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTImage, LTContainer

        def has_image(obj) -> bool:
            if isinstance(obj, LTImage):
                return True
            return isinstance(obj, LTContainer) and any(has_image(c) for c in obj)

        return any(has_image(page) for page in extract_pages(io.BytesIO(content)))
    except Exception:
        return False


def _docx_image(part) -> Optional[tuple]:
    """(bytes, mime) for an image part, or None for icons/logos and unreadable formats."""
    data, mime = part.blob, part.content_type
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None  # unreadable / vector format (EMF/WMF)
    if max(img.size) < MIN_IMAGE_SIDE:
        return None
    if mime not in ("image/png", "image/jpeg"):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        data, mime = buf.getvalue(), "image/png"
    return data, mime


def docx_segments(content: bytes) -> list:
    """
    Walk a DOCX body in reading order (paragraphs, tables, text boxes) and return
    [("text", str) | ("image", bytes, mime)], so image transcripts can be placed
    where the image actually sits in the document. Handles DrawingML (<a:blip>) and
    legacy VML (<v:imagedata>) images; skips tiny icons/logos.
    """
    from docx.oxml.ns import qn
    try:
        doc = docx.Document(io.BytesIO(content))
    except Exception:
        return []

    W_P, W_T, W_TAB, W_BR, W_CR = qn("w:p"), qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:cr")
    A_BLIP, R_EMBED = qn("a:blip"), qn("r:embed")
    V_IMAGEDATA = "{urn:schemas-microsoft-com:vml}imagedata"
    R_ID = qn("r:id")
    # Text boxes / shapes are stored twice (mc:Choice + mc:Fallback); read only the Choice.
    MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"

    segments, buf, seen = [], [], set()
    n_images = 0

    def flush():
        text = re.sub(r"\n{3,}", "\n\n", "".join(buf)).strip()
        if text:
            segments.append(("text", text))
        buf.clear()

    def image(rid):
        nonlocal n_images
        if not rid or rid in seen or n_images >= MAX_DOC_IMAGES:
            return
        seen.add(rid)
        part = doc.part.related_parts.get(rid)
        img = _docx_image(part) if part is not None else None
        if img:
            flush()
            segments.append(("image", *img))
            n_images += 1

    def walk(el):
        tag = el.tag
        if tag == MC_FALLBACK:
            return
        if tag == W_T:
            buf.append(el.text or "")
        elif tag == W_TAB:
            buf.append("\t")
        elif tag in (W_BR, W_CR):
            buf.append("\n")
        elif tag == A_BLIP:
            image(el.get(R_EMBED))
        elif tag == V_IMAGEDATA:
            image(el.get(R_ID))
        for child in el:
            walk(child)
        if tag == W_P:
            buf.append("\n")

    walk(doc.element.body)
    flush()
    return segments


def docx_images(content: bytes) -> list:
    """Images embedded in a DOCX body, in document order, as [(bytes, mime)]."""
    return [(s[1], s[2]) for s in docx_segments(content) if s[0] == "image"]


def extract_document(content: bytes, filename: str) -> dict:
    """
    Extract text from raw file bytes. Sync — safe to run in a worker thread.
    Returns {"text", "ocr_used", "ocr_confidence"} so callers can surface a
    low-confidence scan instead of silently treating garbage as a clean document.
    """
    filename = (filename or "").lower()

    if filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        return {"text": " ".join(p.text for p in doc.paragraphs), "ocr_used": False, "ocr_confidence": None}

    elif filename.endswith(".pdf"):
        text = extract_text(io.BytesIO(content))
        if len(text.strip()) < 10:  # Likely a scanned PDF -> OCR
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                images = convert_from_path(tmp_path)
                parts, confs = [], []
                for img in images:
                    t, c = _ocr_image(img)
                    parts.append(t)
                    confs.append(c)
                mean_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
                return {"text": "\n".join(parts), "ocr_used": True, "ocr_confidence": mean_conf}
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return {"text": text, "ocr_used": False, "ocr_confidence": None}

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(content))
        text, confidence = _ocr_image(image)
        return {"text": text, "ocr_used": True, "ocr_confidence": confidence}

    elif filename.endswith(".txt"):
        return {"text": content.decode("utf-8", errors="replace"), "ocr_used": False, "ocr_confidence": None}

    else:
        try:
            return {"text": content.decode("utf-8"), "ocr_used": False, "ocr_confidence": None}
        except UnicodeDecodeError:
            return {"text": "", "ocr_used": False, "ocr_confidence": None}
