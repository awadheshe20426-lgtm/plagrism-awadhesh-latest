"""
Client for the CU self-hosted AI gateway (OpenAI-compatible, Gemma 4 / Nemotron VL).

Configured via LLM_GATEWAY_* settings. Gemma 4 is a reasoning model: it "thinks"
before answering, so max_tokens must stay >= 512 or the content comes back empty.
"""
import base64
import json
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from app.core.config import settings

DEFAULT_PROMPT = """You are an academic-integrity assistant. Analyse the document below and respond ONLY with a JSON object of this shape:
{
  "summary": "<2-3 sentence summary of the document>",
  "ai_likelihood": <float 0.0 (clearly human-written) to 1.0 (clearly AI-generated)>,
  "ai_reasoning": "<brief justification for the ai_likelihood>",
  "key_topics": ["<topic>", "..."],
  "quality_notes": "<brief remarks on writing quality, structure and originality>"
}

Document:
"""

MAX_DOC_CHARS = 12000  # analyze_text: keep the prompt well inside the model's context window

# Transcription: text is sent in chunks of this many characters (~600-700 tokens).
# Small chunks keep Gemma's reasoning short so it actually emits the transcript.
TRANSCRIBE_CHUNK_CHARS = 2500
MAX_TOTAL_CHARS = 200_000  # hard cap per request (~80 chunks)


# Photos from phones are often 3000-4000 px and several MB. Vision models read text
# just as well at this size, and a smaller payload keeps Gemma's reasoning short.
IMAGE_MAX_SIDE = 1600
IMAGE_JPEG_QUALITY = 85


def _shrink_image(image_bytes: bytes, mime_type: str):
    """Downscale to IMAGE_MAX_SIDE on the longest edge and re-encode as JPEG if large.
    Returns (bytes, mime). Falls back to the original on any decoding problem."""
    try:
        import io
        from PIL import Image, ImageOps
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)  # honour phone orientation tags
        w, h = img.size
        if max(w, h) <= IMAGE_MAX_SIDE and len(image_bytes) <= 1_000_000:
            return image_bytes, mime_type
        scale = min(1.0, IMAGE_MAX_SIDE / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, mime_type


def _split_chunks(text: str, size: int) -> list:
    """Split on paragraph / line / sentence boundaries, never mid-word where avoidable."""
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks, buf = [], ""
    for para in re.split(r"(\n\s*\n)", text):  # keep the separators
        if len(buf) + len(para) <= size:
            buf += para
            continue
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""
        while len(para) > size:  # a single huge paragraph: cut at last sentence/line/space
            cut = max(para.rfind(". ", 0, size), para.rfind("\n", 0, size), para.rfind(" ", 0, size))
            cut = cut + 1 if cut > size // 2 else size
            chunks.append(para[:cut].strip())
            para = para[cut:]
        buf = para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

# Gemma 4 is a reasoning model. On handwritten pages it can loop forever "re-reading"
# lines inside its hidden reasoning and never emit the transcript (finish_reason=length
# with empty content, even at 16k tokens). The gateway ignores enable_thinking /
# reasoning_effort, but this system message reliably stops the deliberation.
TRANSCRIBE_SYSTEM_PROMPT = "Answer directly without deliberation. Keep any thinking under 3 sentences."

TRANSCRIBE_IMAGE_PROMPT = (
    "Transcribe ALL text visible in this image exactly as written, preserving line breaks, "
    "paragraphs, headings and lists. Do not summarise, translate, correct or add anything. "
    "If there is no readable text, reply with exactly: [NO TEXT FOUND]. Output only the transcript."
)

TRANSCRIBE_TEXT_PROMPT = (
    "Below is the raw content of a text file. Return a clean transcript of it: keep every sentence "
    "and the original wording and order, fix only broken line-wrapping, stray control characters "
    "and encoding artefacts. Do not summarise, translate or add anything. Output only the transcript.\n\n"
    "File content:\n"
)


class LLMGatewayError(Exception):
    pass


class LLMGatewayService:
    def __init__(self):
        self.base_url = settings.LLM_GATEWAY_BASE_URL
        self.api_key = settings.LLM_GATEWAY_API_KEY
        self.model = settings.LLM_GATEWAY_MODEL
        self.allowed_models = {m.strip() for m in settings.LLM_GATEWAY_ALLOWED_MODELS.split(",") if m.strip()}
        self.allowed_models.add(self.model)
        self.vision_models = {m.strip() for m in settings.LLM_GATEWAY_VISION_MODELS.split(",") if m.strip()}
        self.max_tokens = max(512, settings.LLM_GATEWAY_MAX_TOKENS)
        self.timeout = settings.LLM_GATEWAY_TIMEOUT_SECONDS

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def resolve_model(self, model: Optional[str]) -> str:
        """Return the model to use: the requested one if allowed, else the configured default."""
        if not model or not model.strip():
            return self.model
        model = model.strip()
        if model not in self.allowed_models:
            raise ValueError(f"Unknown model '{model}'. Allowed: {', '.join(sorted(self.allowed_models))}")
        return model

    def analyze_text(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Send `text` to the gateway model and return
        {"model", "raw", "parsed" (dict or None), "usage"}.
        Raises LLMGatewayError when the gateway is unconfigured or the call fails.
        """
        if not self.configured:
            raise LLMGatewayError("LLM gateway not configured: set LLM_GATEWAY_API_KEY")

        instruction = (prompt or DEFAULT_PROMPT).rstrip() + "\n"
        content = instruction + text[:MAX_DOC_CHARS]

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
        except Exception as e:  # auth, network, model-not-allowed, etc.
            raise LLMGatewayError(f"Gateway call failed: {e}") from e

        raw, usage, _ = _unpack(resp)
        return {"model": self.model, "raw": raw, "parsed": _extract_json(raw), "usage": usage}

    def transcribe_image(self, image_bytes: bytes, mime_type: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Send an image (PNG/JPEG) to the gateway model as a base64 data URI and ask it to
        transcribe every visible word. Returns {"model", "transcript", "usage", "truncated"}.
        `model` overrides the configured default (must be in LLM_GATEWAY_ALLOWED_MODELS)
        and is the only model called: there is no automatic fallback.
        """
        if not self.configured:
            raise LLMGatewayError("LLM gateway not configured: set LLM_GATEWAY_API_KEY")
        chosen = self.resolve_model(model)
        if chosen not in self.vision_models:
            raise ValueError(
                f"Model '{chosen}' is text-only and cannot read images. "
                f"Use one of: {', '.join(sorted(self.vision_models))}"
            )
        image_bytes, mime_type = _shrink_image(image_bytes, mime_type)
        data_uri = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
        messages = [{"role": "user", "content": [
            {"type": "text", "text": TRANSCRIBE_IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]}]
        # No fallback: the caller's chosen model is the only one called. If it runs out
        # of output budget without producing text, the route reports that as a 502.
        return self._transcribe(messages, chosen)

    def transcribe_pdf(self, pdf_bytes: bytes, model: Optional[str] = None,
                       max_pages: int = 20, dpi: int = 150) -> Dict[str, Any]:
        """
        Send a PDF to the model *as images*: every page is rendered with poppler and
        transcribed by the vision model, then the page transcripts are joined in order.
        No local text extraction / OCR is involved; the model reads the page itself.
        """
        import io
        from pdf2image import convert_from_bytes

        chosen = self.resolve_model(model)
        if chosen not in self.vision_models:
            raise ValueError(
                f"Model '{chosen}' is text-only and cannot read PDF pages as images. "
                f"Use one of: {', '.join(sorted(self.vision_models))}"
            )
        try:
            pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=1, last_page=max_pages)
        except Exception as e:
            raise LLMGatewayError(f"Could not render PDF pages: {e}") from e
        if not pages:
            raise LLMGatewayError("PDF has no pages")

        transcripts, usage_total, truncated = [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, False
        for idx, page in enumerate(pages, start=1):
            buf = io.BytesIO()
            page.convert("RGB").save(buf, format="PNG")
            r = self.transcribe_image(buf.getvalue(), "image/png", chosen)
            transcripts.append(r["transcript"])
            truncated = truncated or r["truncated"]
            if r["usage"]:
                for k in usage_total:
                    usage_total[k] += r["usage"].get(k, 0)

        return {
            "model": chosen,
            "transcript": "\n\n".join(t for t in transcripts if t).strip(),
            "usage": usage_total,
            "truncated": truncated,
            "pages": len(pages),
        }

    def transcribe_text(self, text: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Send text-file content to the gateway model and get a clean transcript back.

        Long inputs are split into paragraph-aligned chunks and transcribed separately,
        then joined in order. This matters for Gemma: it is a reasoning model, and on a
        large input it can burn the whole output budget "thinking" and return nothing.
        """
        if not self.configured:
            raise LLMGatewayError("LLM gateway not configured: set LLM_GATEWAY_API_KEY")
        chosen = self.resolve_model(model)
        text = text[:MAX_TOTAL_CHARS]
        chunks = _split_chunks(text, TRANSCRIBE_CHUNK_CHARS)

        def one(chunk: str) -> Dict[str, Any]:
            return self._transcribe([{"role": "user", "content": TRANSCRIBE_TEXT_PROMPT + chunk}], chosen)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
            results = list(pool.map(one, chunks))

        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for r in results:
            if r["usage"]:
                for k in usage_total:
                    usage_total[k] += r["usage"].get(k, 0)
        return {
            "model": chosen,
            "transcript": "\n\n".join(r["transcript"] for r in results if r["transcript"]).strip(),
            "usage": usage_total,
            "truncated": any(r["truncated"] for r in results),
            "chunks": len(chunks),
        }

    def transcribe_docx(self, docx_bytes: bytes, model: Optional[str] = None,
                        clean_text: bool = True) -> Dict[str, Any]:
        """
        Transcribe a DOCX including its embedded images. The body is walked in reading
        order; each image is sent to the vision model and its transcript is placed where
        the image sits, marked "[Image N]". Typed text is cleaned by the model
        (clean_text=True) or kept verbatim. A failing image is reported inline instead of
        losing the rest of the document.
        """
        from app.services.parsing import docx_segments

        if not self.configured:
            raise LLMGatewayError("LLM gateway not configured: set LLM_GATEWAY_API_KEY")
        chosen = self.resolve_model(model)
        segments = docx_segments(docx_bytes)
        n_images = sum(1 for s in segments if s[0] == "image")
        if n_images and chosen not in self.vision_models:
            raise ValueError(
                f"Model '{chosen}' is text-only and cannot read the {n_images} image(s) in this document. "
                f"Use one of: {', '.join(sorted(self.vision_models))}"
            )

        parts, truncated, img_no = [], False, 0
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        chunks = 0
        for seg in segments:
            if seg[0] == "text":
                if not clean_text:
                    parts.append(seg[1])
                    continue
                r = self.transcribe_text(seg[1], chosen)
                chunks += r.get("chunks", 0)
                text = r["transcript"]
            else:
                img_no += 1
                try:
                    r = self.transcribe_image(seg[1], seg[2], chosen)
                    body = r["transcript"] or "[no readable text in image]"
                except LLMGatewayError as e:
                    r, body = None, f"[image could not be transcribed: {e}]"
                text = f"[Image {img_no}]\n{body}"
            if r:
                truncated = truncated or r["truncated"]
                for k in usage_total:
                    usage_total[k] += (r["usage"] or {}).get(k, 0)
            if text:
                parts.append(text)

        return {
            "model": chosen,
            "transcript": "\n\n".join(parts).strip(),
            "usage": usage_total,
            "truncated": truncated,
            "chunks": chunks or None,
            "images": n_images,
        }

    def _transcribe(self, messages, model: str) -> Dict[str, Any]:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": TRANSCRIBE_SYSTEM_PROMPT}, *messages],
                # Transcripts must reproduce the whole file, and Gemma reasons before
                # answering, so give it a much larger budget than a normal chat reply.
                max_tokens=max(self.max_tokens, settings.LLM_GATEWAY_TRANSCRIBE_MAX_TOKENS),
                temperature=0.0,
            )
        except Exception as e:
            raise LLMGatewayError(f"Gateway call failed: {e}") from e
        transcript, usage, finish_reason = _unpack(resp)
        if transcript.strip() == "[NO TEXT FOUND]":
            transcript = ""
        return {
            "model": model,
            "transcript": transcript,
            "usage": usage,
            "truncated": finish_reason == "length",
        }


def _unpack(resp):
    raw = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    finish_reason = resp.choices[0].finish_reason if resp.choices else None
    usage = None
    if getattr(resp, "usage", None):
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    return raw, usage, finish_reason


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort: the model may wrap JSON in ```json fences or add prose around it."""
    if not raw:
        return None
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
