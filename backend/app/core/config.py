from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@db:5432/plagiarism_db")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Upload limits (DoS protection)
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    MAX_FILES_PER_BATCH: int = int(os.getenv("MAX_FILES_PER_BATCH", "50"))

    # Moodle server-to-server integration (X-API-Key). Empty = integration disabled.
    MOODLE_API_KEY: Optional[str] = os.getenv("MOODLE_API_KEY")
    MOODLE_SERVICE_EMAIL: str = os.getenv("MOODLE_SERVICE_EMAIL", "moodle-service@moodle.local")

    # Storage settings
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # s3 or local
    S3_ENDPOINT_URL: Optional[str] = os.getenv("S3_ENDPOINT_URL")
    S3_ACCESS_KEY: Optional[str] = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY: Optional[str] = os.getenv("S3_SECRET_KEY")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "plagiarism-uploads")
    
    # AI Detection settings
    USE_EXTERNAL_AI_DETECTION: bool = os.getenv("USE_EXTERNAL_AI_DETECTION", "false").lower() == "true"
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    TOGETHER_API_KEY: Optional[str] = os.getenv("TOGETHER_API_KEY")
    
    # CU AI gateway (OpenAI-compatible). Used by /api/v1/upload to send uploaded text to Gemma.
    LLM_GATEWAY_BASE_URL: str = os.getenv("LLM_GATEWAY_BASE_URL", "https://llm.unicornfortunes.com/v1")
    LLM_GATEWAY_API_KEY: Optional[str] = os.getenv("LLM_GATEWAY_API_KEY")
    LLM_GATEWAY_MODEL: str = os.getenv("LLM_GATEWAY_MODEL", "gemma4-26b")  # default when the request sends no `model`
    # Models a request may select via the `model` form field (comma-separated). Must be in the key's scope.
    LLM_GATEWAY_ALLOWED_MODELS: str = os.getenv(
        "LLM_GATEWAY_ALLOWED_MODELS", "qwen2.5-14b,gemma4-26b,nemotron-vl,qwen3-30b-a3b"
    )
    # Subset of the above that accepts images. Text-only models (the Qwen ones) reject image input.
    LLM_GATEWAY_VISION_MODELS: str = os.getenv("LLM_GATEWAY_VISION_MODELS", "gemma4-26b,nemotron-vl")
    LLM_GATEWAY_MAX_TOKENS: int = int(os.getenv("LLM_GATEWAY_MAX_TOKENS", "1024"))  # Gemma reasons first: keep >= 512
    LLM_GATEWAY_TIMEOUT_SECONDS: int = int(os.getenv("LLM_GATEWAY_TIMEOUT_SECONDS", "300"))
    # Transcription needs the full file echoed back plus Gemma's reasoning tokens.
    LLM_GATEWAY_TRANSCRIBE_MAX_TOKENS: int = int(os.getenv("LLM_GATEWAY_TRANSCRIBE_MAX_TOKENS", "8192"))

    # Web-source check (services/web_search.py): finds the web pages submitted text was copied
    # from. Provider: searxng (free, self-hosted, no key) | google_cse | brave | serpapi.
    # Unset or missing key/URL = web check disabled.
    # Comma-separated; every listed provider is queried and results are merged, e.g.
    # "searxng,wikipedia,openalex" (all free, no key). wikipedia = MediaWiki search API,
    # openalex = open index of research papers.
    WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "")
    SEARXNG_URL: str = os.getenv("SEARXNG_URL", "http://searxng:8080")
    WEB_WIKIPEDIA_LANG: str = os.getenv("WEB_WIKIPEDIA_LANG", "en")
    GOOGLE_CSE_KEY: Optional[str] = os.getenv("GOOGLE_CSE_KEY")
    GOOGLE_CSE_CX: Optional[str] = os.getenv("GOOGLE_CSE_CX")
    BRAVE_API_KEY: Optional[str] = os.getenv("BRAVE_API_KEY")
    SERPAPI_KEY: Optional[str] = os.getenv("SERPAPI_KEY")
    # Searches per document scale with length: 1 per WEB_SEARCH_WORDS_PER_QUERY words (min 4),
    # capped at WEB_SEARCH_MAX_QUERIES. E.g. 300 words -> 8 searches, 1000+ words -> 25.
    # More searches = more copied sentences found, but more search-API cost.
    WEB_SEARCH_MAX_QUERIES: int = int(os.getenv("WEB_SEARCH_MAX_QUERIES", "25"))
    WEB_SEARCH_WORDS_PER_QUERY: int = int(os.getenv("WEB_SEARCH_WORDS_PER_QUERY", "40"))
    WEB_SEARCH_MAX_PAGES: int = int(os.getenv("WEB_SEARCH_MAX_PAGES", "25"))      # pages fetched per document (all rounds)
    WEB_MIN_MATCH_WORDS: int = int(os.getenv("WEB_MIN_MATCH_WORDS", "12"))        # shortest exact passage that counts
    # Reworded copies: a sentence counts when meaning >= MIN_SIMILARITY AND word overlap >= MIN_OVERLAP.
    # Both are required so sentences that only share a topic never match.
    WEB_PARAPHRASE_ENABLED: bool = os.getenv("WEB_PARAPHRASE_ENABLED", "true").lower() == "true"
    WEB_PARAPHRASE_MIN_SIMILARITY: float = float(os.getenv("WEB_PARAPHRASE_MIN_SIMILARITY", "0.80"))
    WEB_PARAPHRASE_MIN_OVERLAP: float = float(os.getenv("WEB_PARAPHRASE_MIN_OVERLAP", "0.45"))
    WEB_FETCH_TIMEOUT: int = int(os.getenv("WEB_FETCH_TIMEOUT", "10"))            # seconds per search/page request
    WEB_CACHE_TTL_DAYS: int = int(os.getenv("WEB_CACHE_TTL_DAYS", "7"))
    # Document-sharing mirrors: still reported, but the original site wins attribution.
    WEB_AGGREGATOR_DOMAINS: str = os.getenv(
        "WEB_AGGREGATOR_DOMAINS",
        "scribd.com,coursehero.com,studocu.com,chegg.com,slideshare.net,brainly.com,brainly.in,"
        "quizlet.com,academia.edu,pdfcoffee.com,dokumen.pub,documents.pub,vdocuments.net,doku.pub",
    )
    WEB_EXCLUDE_DOMAINS: str = os.getenv("WEB_EXCLUDE_DOMAINS", "")  # never reported (e.g. your own LMS)

    # Celery settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")


settings = Settings()