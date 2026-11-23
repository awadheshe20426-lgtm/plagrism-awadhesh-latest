"""
Web-source detection: find the web pages a submitted text was copied from.

Pipeline (per document), run in up to MAX_ROUNDS rounds:
  1. queries     distinctive sentences turned into 12-word exact-phrase searches. The number
                 of searches scales with length (1 per WEB_SEARCH_WORDS_PER_QUERY words, capped
                 at WEB_SEARCH_MAX_QUERIES). Round 1 spreads them over the whole document; later
                 rounds search only sentences that no source explains yet, so a paragraph copied
                 from a second/third website is not missed.
  2. search      web search API (Google CSE / Brave / SerpAPI)
  3. fetch       download candidate pages, extract text (HTML / PDF); search snippet as fallback
  4. match       a) exact: 8-word shingles -> word-aligned passages (small edits bridged)
                 b) reworded: sentence pairs with meaning >= WEB_PARAPHRASE_MIN_SIMILARITY AND
                    word overlap >= WEB_PARAPHRASE_MIN_OVERLAP. Both must hold, so sentences that
                    merely share a topic never match (measured: topic-only pairs score <= 0.71 /
                    0.31, reworded copies >= 0.96 / 0.57).
  5. attribute   every matched word is credited to ONE source (most coverage wins; original
                 sites beat mirrors such as scribd); other URLs go to `also_found_at`
  6. score       per source: credited words / document words (split exact vs reworded)
                 overall:    all matched words / document words

The result is stored on Document.web_search_results and returned by the results API as
`web_search_analysis`, in the shape the Moodle plugin (local_plagiarism_ai) renders:
  sources[]: {source_title, source_url, similarity, matches[{source_chunk, target_chunk}]}
  where target_chunk is the exact student text (used for highlighting) and source_chunk
  the matching text on the web page (shown under "View passages").

Nothing here raises to the caller: analyze() always returns a dict with a `status`.
"""
import bisect
import difflib
import hashlib
import io
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from app.core.config import settings

SHINGLE = 8                 # words per shingle for exact matching
MAX_GAP_WORDS = 5           # merge matched runs separated by at most this many words (light rewording)
MAX_INPUT_CHARS = 100_000   # cap on submission text that is analysed
MAX_PAGE_BYTES = 5 * 1024 * 1024
MAX_SOURCE_CHUNK_CHARS = 3000
QUERY_WORDS = 12            # words per exact-phrase query
MAX_ROUNDS = 3              # search rounds (round 2+ targets text not yet explained)
PAGES_PER_ROUND = 10        # new pages fetched per round
PARA_MIN_WORDS = 8          # sentences shorter than this are not checked for rewording
PARA_MIN_SHARED_CONTENT = 6 # reworded pair must share this many meaningful (non stop-) words
MAX_PAGE_SENTENCES = 2000   # cap per page for the rewording check
USER_AGENT = "Mozilla/5.0 (compatible; PlagiarismChecker/1.0; +web-source-check)"

_WORD_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*")
_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works cited|sources|citations)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TRACKING_PARAMS = {"gclid", "fbclid", "ref", "ref_src", "mc_cid", "mc_eid", "_hsenc", "_hsmi"}
_STOPWORDS = set("""
a an the and or but if then else of to in on at by for with from as is are was were be been being
this that these those it its it's into than so such not no can could should would will may might must
do does did done have has had having i you he she we they them their our your his her my me us
which who whom whose what when where why how all any each other some more most very also just only
there here about over under between through during before after above below up down out off again
""".split())


# --------------------------------------------------------------------------- tokens
def _tokens(text: str) -> List[Tuple[str, int, int]]:
    """Words of `text` as (normalized_word, start_offset, end_offset)."""
    return [(m.group(0).lower().replace("’", "'"), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


_TRAILING_PUNCT = ".,;:!?)]}\"'”’"


def _extend_punct(text: str, end: int) -> int:
    """Move `end` past punctuation that directly follows the last word (e.g. the full stop),
    so a passage reads '...period of time.' rather than '...period of time'."""
    while end < len(text) and text[end] in _TRAILING_PUNCT:
        end += 1
    return end


def _strip_reference_section(text: str) -> str:
    """Drop a trailing references/bibliography section so cited sources don't self-match.
    Only a heading line that is exactly 'References' (etc.) in the second half of the
    text counts, so ordinary prose and numbered lists are never removed."""
    last = None
    for m in _REFERENCES_HEADING_RE.finditer(text):
        last = m
    if last and last.start() > len(text) * 0.5:
        return text[:last.start()]
    return text


# --------------------------------------------------------------------------- queries
def _sentence_spans(text: str, min_words: int) -> List[Tuple[int, int]]:
    """(start, end) character offsets of sentences/lines with at least `min_words` words."""
    out = []
    for m in _SENTENCE_RE.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if len(_WORD_RE.findall(text[s:e])) >= min_words:
            out.append((s, e))
    return out


def _tok_range(student: List[Tuple[str, int, int]], start: int, end: int) -> Tuple[int, int]:
    """Token index range [a, b) of the student tokens lying inside characters [start, end)."""
    starts = [t[1] for t in student]
    return bisect.bisect_left(starts, start), bisect.bisect_left(starts, end)


def _query_candidates(text: str, student: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
    """Every sentence usable as a search query, with a distinctiveness score and the
    QUERY_WORDS-word exact phrase taken from its middle."""
    cands = []
    for s, e in _sentence_spans(text, 10):
        words = _WORD_RE.findall(text[s:e])
        lw = [w.lower() for w in words]
        content = [w for w in lw if w not in _STOPWORDS]
        score = len(set(content)) / len(lw)                      # few stop-words, varied words
        score += 0.15 * sum(1 for w in words if any(c.isdigit() for c in w))
        score += 0.05 * sum(1 for w in content if len(w) >= 8)  # rarer, longer words
        n = min(QUERY_WORDS, len(words))
        a = (len(words) - n) // 2
        ts, te = _tok_range(student, s, e)
        cands.append({"pos": s, "score": score, "phrase": " ".join(words[a:a + n]),
                      "tok_s": ts, "tok_e": te})
    return cands


def _spread(cands: List[Dict[str, Any]], k: int, text_len: int) -> List[Dict[str, Any]]:
    """Pick up to k candidates spread evenly by position (best per region), topped up by score."""
    k = min(k, len(cands))
    if k <= 0:
        return []
    regions: List[List[dict]] = [[] for _ in range(k)]
    for c in cands:
        regions[min(k - 1, int(c["pos"] / max(1, text_len) * k))].append(c)
    picked, seen = [], set()
    for region in regions:
        for c in sorted(region, key=lambda c: -c["score"]):
            if c["phrase"].lower() not in seen:
                picked.append(c)
                seen.add(c["phrase"].lower())
                break
    for c in sorted(cands, key=lambda c: -c["score"]):  # empty regions: top up
        if len(picked) >= k:
            break
        if c["phrase"].lower() not in seen:
            picked.append(c)
            seen.add(c["phrase"].lower())
    picked.sort(key=lambda c: c["pos"])
    return picked


def query_budget(total_words: int) -> int:
    """Searches for a document: one per WEB_SEARCH_WORDS_PER_QUERY words, min 4, capped."""
    per = max(10, settings.WEB_SEARCH_WORDS_PER_QUERY)
    return max(1, min(settings.WEB_SEARCH_MAX_QUERIES, max(4, math.ceil(total_words / per))))


def select_queries(text: str, max_queries: int) -> List[str]:
    """Distinctive sentences spread evenly over the document, as exact phrases (round-1 picks)."""
    student = _tokens(text)
    return [c["phrase"] for c in _spread(_query_candidates(text, student), max_queries, len(text))]


# --------------------------------------------------------------------------- cache
_redis = None


def _cache():
    global _redis
    if _redis is None:
        try:
            import redis
            _redis = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=2, socket_connect_timeout=2)
            _redis.ping()
        except Exception:
            _redis = False  # cache unavailable: run uncached
    return _redis or None


def _cache_get(kind: str, key: str):
    r = _cache()
    if not r:
        return None
    try:
        raw = r.get(f"websearch:{kind}:{hashlib.sha256(key.encode()).hexdigest()}")
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _cache_set(kind: str, key: str, value) -> None:
    r = _cache()
    if not r:
        return
    try:
        r.setex(f"websearch:{kind}:{hashlib.sha256(key.encode()).hexdigest()}",
                settings.WEB_CACHE_TTL_DAYS * 86400, json.dumps(value))
    except Exception:
        pass


# --------------------------------------------------------------------------- search providers
class SearchError(Exception):
    pass


def providers() -> List[str]:
    """Usable providers from WEB_SEARCH_PROVIDER (comma-separated; all are queried and their
    results merged). Free, no key: searxng, wikipedia, openalex. With key: google_cse, brave, serpapi.
    A provider whose key/URL is missing is skipped."""
    ready = {
        "searxng": bool(settings.SEARXNG_URL),
        "wikipedia": True,
        "openalex": True,
        "google_cse": bool(settings.GOOGLE_CSE_KEY and settings.GOOGLE_CSE_CX),
        "brave": bool(settings.BRAVE_API_KEY),
        "serpapi": bool(settings.SERPAPI_KEY),
    }
    out = []
    for p in (settings.WEB_SEARCH_PROVIDER or "").lower().split(","):
        p = p.strip()
        if ready.get(p) and p not in out:
            out.append(p)
    return out


def provider_name() -> Optional[str]:
    """Configured provider(s) as 'a,b', or None when none is usable (web check disabled)."""
    return ",".join(providers()) or None


_API_UA = {"User-Agent": "PlagiarismDetection/1.0 (self-hosted academic integrity checker)"}


def _openalex_abstract(inv: Optional[dict]) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the plain text."""
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _search_raw(provider: str, query: str) -> List[Dict[str, str]]:
    t = settings.WEB_FETCH_TIMEOUT
    if provider == "searxng":
        # Self-hosted SearXNG (free, open source): it queries Google/Bing/DuckDuckGo/... itself
        # and merges the results. Upstream engines can be slow, so allow a bit more time.
        r = requests.get(f"{settings.SEARXNG_URL.rstrip('/')}/search", timeout=max(t, 20), params={
            "q": query, "format": "json", "safesearch": 0, "language": "all"})
        if r.status_code != 200:
            raise SearchError(f"searxng HTTP {r.status_code}: {r.text[:200]}")
        items = r.json().get("results", [])[:10]
        return [{"url": i.get("url", ""), "title": i.get("title", ""), "snippet": i.get("content", "")} for i in items]
    if provider == "wikipedia":
        # Official MediaWiki search API (free, no key). Quoted query = exact phrase in article text.
        lang = (settings.WEB_WIKIPEDIA_LANG or "en").strip()
        r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", timeout=t, headers=_API_UA, params={
            "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 5})
        if r.status_code != 200:
            raise SearchError(f"wikipedia HTTP {r.status_code}: {r.text[:200]}")
        items = (r.json().get("query") or {}).get("search", [])
        return [{"url": f"https://{lang}.wikipedia.org/wiki/{i['title'].replace(' ', '_')}",
                 "title": f"{i['title']} - Wikipedia",
                 "snippet": re.sub(r"<[^>]+>", "", i.get("snippet", ""))} for i in items if i.get("title")]
    if provider == "openalex":
        # OpenAlex (free, open scholarly index of 250M+ papers, no key). Searches title,
        # abstract and full text; links to the open-access PDF when there is one. The
        # abstract is returned as the snippet, so a copied abstract matches even when the
        # publisher page itself cannot be downloaded.
        r = requests.get("https://api.openalex.org/works", timeout=max(t, 20), headers=_API_UA, params={
            "search": query, "per-page": 5,
            "select": "id,title,doi,primary_location,best_oa_location,abstract_inverted_index"})
        if r.status_code != 200:
            raise SearchError(f"openalex HTTP {r.status_code}: {r.text[:200]}")
        out = []
        for w in r.json().get("results", []):
            oa, pl = w.get("best_oa_location") or {}, w.get("primary_location") or {}
            url = oa.get("pdf_url") or oa.get("landing_page_url") or pl.get("landing_page_url") or w.get("doi") or w.get("id")
            if url:
                out.append({"url": url, "title": w.get("title") or "OpenAlex work",
                            "snippet": _openalex_abstract(w.get("abstract_inverted_index"))})
        return out
    if provider == "google_cse":
        r = requests.get("https://www.googleapis.com/customsearch/v1", timeout=t, params={
            "key": settings.GOOGLE_CSE_KEY, "cx": settings.GOOGLE_CSE_CX, "q": query, "num": 10})
        if r.status_code != 200:
            raise SearchError(f"google_cse HTTP {r.status_code}: {r.text[:200]}")
        items = r.json().get("items", [])
        return [{"url": i.get("link", ""), "title": i.get("title", ""), "snippet": i.get("snippet", "")} for i in items]
    if provider == "brave":
        r = requests.get("https://api.search.brave.com/res/v1/web/search", timeout=t,
                         params={"q": query, "count": 10},
                         headers={"Accept": "application/json", "X-Subscription-Token": settings.BRAVE_API_KEY})
        if r.status_code != 200:
            raise SearchError(f"brave HTTP {r.status_code}: {r.text[:200]}")
        items = (r.json().get("web") or {}).get("results", [])
        return [{"url": i.get("url", ""), "title": i.get("title", ""),
                 "snippet": re.sub(r"<[^>]+>", "", i.get("description", ""))} for i in items]
    if provider == "serpapi":
        r = requests.get("https://serpapi.com/search.json", timeout=t, params={
            "engine": "google", "q": query, "num": 10, "api_key": settings.SERPAPI_KEY})
        if r.status_code != 200:
            raise SearchError(f"serpapi HTTP {r.status_code}: {r.text[:200]}")
        items = r.json().get("organic_results", [])
        return [{"url": i.get("link", ""), "title": i.get("title", ""), "snippet": i.get("snippet", "")} for i in items]
    raise SearchError(f"unknown provider {provider}")


_api_calls = Counter()  # per-thread-safe enough for reporting: incremented under the GIL


def search(provider: str, phrase: str) -> List[Dict[str, str]]:
    """Exact-phrase search; one unquoted (keyword) retry if the quoted query finds nothing,
    which is what lets reworded copies be discovered. Cached."""
    cached = _cache_get("q", f"{provider}|{phrase}")
    if cached is not None:
        return cached
    _api_calls["n"] += 1
    results = _search_raw(provider, f'"{phrase}"')
    if not results:
        _api_calls["n"] += 1
        results = _search_raw(provider, phrase)
    results = [r for r in results if r.get("url", "").startswith(("http://", "https://"))]
    _cache_set("q", f"{provider}|{phrase}", results)
    return results


# --------------------------------------------------------------------------- urls / domains
def normalize_url(url: str) -> str:
    """Canonical form for deduplication: lower host without www., no fragment,
    no tracking params, no trailing slash."""
    try:
        p = urlsplit(url.strip())
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        query = urlencode([(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                           if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS])
        path = p.path.rstrip("/") or "/"
        return urlunsplit((p.scheme.lower(), host, path, query, ""))
    except Exception:
        return url


def _domain(url: str) -> str:
    host = urlsplit(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _domain_in(url: str, domains: List[str]) -> bool:
    d = _domain(url)
    return any(d == x or d.endswith("." + x) for x in domains)


def _csv(value: str) -> List[str]:
    return [x.strip().lower() for x in (value or "").split(",") if x.strip()]


# --------------------------------------------------------------------------- page fetch
_DROP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe", "template")


def _html_to_text(raw: bytes) -> Tuple[str, str]:
    import lxml.html
    doc = lxml.html.fromstring(raw)
    title = (doc.findtext(".//title") or "").strip()
    for bad in doc.xpath("//" + " | //".join(_DROP_TAGS)):
        bad.drop_tree()
    body = doc.find("body")
    text = " ".join((body if body is not None else doc).itertext())
    return re.sub(r"\s+", " ", text).strip(), re.sub(r"\s+", " ", title)


def _pdf_to_text(raw: bytes) -> str:
    from pdfminer.high_level import extract_text
    return re.sub(r"\s+", " ", extract_text(io.BytesIO(raw))).strip()


def fetch_page(url: str) -> Dict[str, Any]:
    """{'ok': bool, 'text': str, 'title': str, 'error': str}. Cached (successes only)."""
    cached = _cache_get("page", url)
    if cached is not None:
        return cached
    try:
        r = requests.get(url, timeout=settings.WEB_FETCH_TIMEOUT, stream=True, allow_redirects=True,
                         headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.5"})
        if r.status_code != 200:
            return {"ok": False, "text": "", "title": "", "error": f"HTTP {r.status_code}"}
        raw = b""
        for part in r.iter_content(65536):
            raw += part
            if len(raw) > MAX_PAGE_BYTES:
                break
        ctype = r.headers.get("Content-Type", "").lower()
        if "pdf" in ctype or raw[:5] == b"%PDF-":
            text, title = _pdf_to_text(raw), ""
        elif "html" in ctype or "xml" in ctype or not ctype:
            text, title = _html_to_text(raw)
        elif ctype.startswith("text/"):
            text, title = re.sub(r"\s+", " ", raw.decode(r.encoding or "utf-8", "replace")), ""
        else:
            return {"ok": False, "text": "", "title": "", "error": f"unsupported content-type {ctype}"}
        page = {"ok": bool(text), "text": text, "title": title, "error": "" if text else "no text"}
        if page["ok"]:
            _cache_set("page", url, page)
        return page
    except Exception as e:
        return {"ok": False, "text": "", "title": "", "error": f"{type(e).__name__}: {e}"[:200]}


# --------------------------------------------------------------------------- matching
def match_spans(student: List[Tuple[str, int, int]], page_text: str,
                min_words: int) -> List[Dict[str, Any]]:
    """
    Exact word-level matches between the submission tokens and a page.
    Returns spans [{'start_tok', 'end_tok' (exclusive), 'page_start', 'page_end' (char offsets)}],
    each at least `min_words` long, word-aligned on both sides.
    """
    page = _tokens(page_text)
    if len(page) < SHINGLE or len(student) < SHINGLE:
        return []
    index: Dict[tuple, int] = {}
    for j in range(len(page) - SHINGLE + 1):
        index.setdefault(tuple(t[0] for t in page[j:j + SHINGLE]), j)

    hit_at: Dict[int, int] = {}  # student shingle start -> page token index
    covered = [False] * len(student)
    for i in range(len(student) - SHINGLE + 1):
        j = index.get(tuple(t[0] for t in student[i:i + SHINGLE]))
        if j is not None:
            hit_at[i] = j
            for x in range(i, i + SHINGLE):
                covered[x] = True

    # Runs of covered tokens, merging small gaps (a changed word or two).
    spans, i, n = [], 0, len(student)
    while i < n:
        if not covered[i]:
            i += 1
            continue
        s = e = i
        while e < n:
            if covered[e]:
                e += 1
                continue
            gap_end = e
            while gap_end < n and not covered[gap_end] and gap_end - e < MAX_GAP_WORDS + 1:
                gap_end += 1
            if gap_end < n and covered[gap_end] and gap_end - e <= MAX_GAP_WORDS:
                e = gap_end
                continue
            break
        i = e
        if e - s < min_words:
            continue
        starts = [k for k in hit_at if s <= k < e]
        p_first = hit_at[min(starts)]
        p_last = hit_at[max(starts)] + SHINGLE - 1
        if p_last < p_first:  # passage reordered on the page: show the first matching stretch
            p_last = min(len(page) - 1, p_first + (e - s) - 1)
        spans.append({"start_tok": s, "end_tok": e,
                      "page_start": page[p_first][1], "page_end": _extend_punct(page_text, page[p_last][2])})
    return spans


# --------------------------------------------------------------------------- reworded copies
_embed_model = None


def _embedder():
    """Sentence-embedding model (same all-MiniLM-L6-v2 the plagiarism check uses), loaded once.
    Returns None if unavailable; the reworded check is then skipped, exact matching still runs."""
    global _embed_model
    if _embed_model is None:
        try:
            from app.services.embedding import EmbeddingService
            _embed_model = EmbeddingService().model or False
        except Exception:
            _embed_model = False
    return _embed_model or None


def _overlap(a: List[str], b: List[str]) -> float:
    """Word overlap of two sentences: best of in-order similarity and shared-word ratio
    (the second keeps reordered clauses matching)."""
    seq = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    sa, sb = set(a), set(b)
    jac = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
    return max(seq, jac)


class StudentSentences:
    """Sentences of the submission, embedded lazily once and reused for every page."""

    def __init__(self, text: str, student: List[Tuple[str, int, int]]):
        self.items = []
        for s, e in _sentence_spans(text, PARA_MIN_WORDS):
            a, b = _tok_range(student, s, e)
            if b > a:
                self.items.append({"start": s, "end": e, "tok_s": a, "tok_e": b,
                                   "words": [t[0] for t in student[a:b]]})
        self._emb = None

    def embeddings(self, model):
        if self._emb is None:
            self._emb = model.encode([" ".join(i["words"]) for i in self.items],
                                     batch_size=64, normalize_embeddings=True)
        return self._emb


def reworded_matches(sents: StudentSentences, exact_tokens: set, page_text: str) -> List[Dict[str, Any]]:
    """
    Student sentences that are reworded copies of a sentence on the page.
    Only sentences not already (mostly) matched exactly are checked. A pair counts when
    meaning similarity >= WEB_PARAPHRASE_MIN_SIMILARITY AND word overlap >= WEB_PARAPHRASE_MIN_OVERLAP.
    """
    if not settings.WEB_PARAPHRASE_ENABLED or not sents.items:
        return []
    model = _embedder()
    if model is None:
        return []
    todo = [i for i, it in enumerate(sents.items)
            if sum(1 for t in range(it["tok_s"], it["tok_e"]) if t in exact_tokens) < 0.5 * (it["tok_e"] - it["tok_s"])]
    if not todo:
        return []
    page_spans = _sentence_spans(page_text, 6)[:MAX_PAGE_SENTENCES]
    if not page_spans:
        return []
    page_words = [[t[0] for t in _tokens(page_text[s:e])] for s, e in page_spans]
    p_emb = model.encode([" ".join(w) for w in page_words], batch_size=64, normalize_embeddings=True)
    s_emb = sents.embeddings(model)
    sims = s_emb[todo] @ p_emb.T
    out = []
    for row, i in enumerate(todo):
        it = sents.items[i]
        # Best few page sentences by meaning, then require real word overlap.
        for j in sims[row].argsort()[::-1][:3]:
            sim = float(sims[row][j])
            if sim < settings.WEB_PARAPHRASE_MIN_SIMILARITY:
                break
            ov = _overlap(it["words"], page_words[j])
            shared = (set(it["words"]) & set(page_words[j])) - _STOPWORDS
            # Meaning + word overlap + enough shared meaningful words: an ordinary short
            # technical phrase ("return air ... below the seat") is not enough on its own.
            if ov >= settings.WEB_PARAPHRASE_MIN_OVERLAP and len(shared) >= PARA_MIN_SHARED_CONTENT:
                out.append({"tok_s": it["tok_s"], "tok_e": it["tok_e"],
                            "start": it["start"], "end": it["end"],
                            "page_start": page_spans[j][0], "page_end": page_spans[j][1],
                            "similarity": round(sim, 4), "overlap": round(ov, 4)})
                break
    return out


# --------------------------------------------------------------------------- main
def empty_result(status: str, **kw) -> Dict[str, Any]:
    """
    The web_search_analysis response. EVERY status (completed / disabled / skipped /
    error / not_run) returns exactly these keys, so clients can parse it one way:
      status          completed | disabled | skipped | error | not_run
      provider        search source(s) used, e.g. "searxng,wikipedia,openalex" (null if none)
      score           share of the document found on the web, 0-1
      exact_score     share copied word-for-word, 0-1
      reworded_score  share copied with rewording, 0-1
      words / matched_words / exact_words / reworded_words   word counts
      queries / query_budget / api_calls / rounds / pages_checked / search_errors
      reason          why the check did not complete (disabled / skipped), else null
      error           error message when status = error, else null
      sources         list of sources (fixed keys, see _source); empty when none
    Keys passed in **kw must already exist in the template.
    """
    base = {
        "status": status, "provider": provider_name(),
        "score": 0.0, "exact_score": 0.0, "reworded_score": 0.0,
        "words": 0, "matched_words": 0, "exact_words": 0, "reworded_words": 0,
        "queries": 0, "query_budget": 0, "api_calls": 0, "rounds": 0,
        "pages_checked": 0, "search_errors": 0,
        "reason": None, "error": None,
        "sources": [],
    }
    unknown = set(kw) - set(base)
    if unknown:  # keep the schema fixed: a new field must be added to the template first
        raise KeyError(f"unknown web_search_analysis field(s): {sorted(unknown)}")
    base.update(kw)
    return base


_result = empty_result

_SOURCE_DEFAULTS = {
    "source_title": "", "source_url": "", "similarity": 0.0,
    "matched_words": 0, "exact_words": 0, "reworded_words": 0,
    "evidence": "page", "document_sharing_site": False, "also_found_at": [], "matches": [],
}
_MATCH_DEFAULTS = {
    "type": "web_exact", "source_chunk": "", "target_chunk": "",
    "score": 1.0, "word_overlap": 1.0, "words": 0,
}


def normalize_result(stored: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Stored result (or None = check never ran) -> the fixed response shape, with every
    key present at every level. Results saved by older code get the missing keys filled."""
    if not stored:
        return empty_result("not_run", provider=None)
    out = empty_result(stored.get("status") or "completed", provider=stored.get("provider"))
    out.update({k: v for k, v in stored.items() if k in out})
    sources = []
    for s in stored.get("sources") or []:
        src = {**_SOURCE_DEFAULTS, **{k: v for k, v in s.items() if k in _SOURCE_DEFAULTS}}
        if "exact_words" not in s:  # older rows: only exact matching existed
            src["exact_words"] = src["matched_words"]
        src["matches"] = [{**_MATCH_DEFAULTS, **{k: v for k, v in m.items() if k in _MATCH_DEFAULTS}}
                          for m in s.get("matches") or []]
        sources.append(src)
    out["sources"] = sources
    return out


def analyze(text: str) -> Dict[str, Any]:
    """Find web sources for `text`. Always returns a dict; never raises."""
    try:
        return _analyze(text)
    except Exception as e:  # last-resort guard: web problems must never fail a document
        return _result("error", error=f"{type(e).__name__}: {e}"[:300])


def _analyze(text: str) -> Dict[str, Any]:
    provs = providers()
    if not provs:
        return _result("disabled", reason="Set WEB_SEARCH_PROVIDER (e.g. searxng,wikipedia,openalex) to enable web-source checking")

    text = _strip_reference_section((text or "")[:MAX_INPUT_CHARS])
    student = _tokens(text)
    total_words = len(student)
    min_words = max(SHINGLE, settings.WEB_MIN_MATCH_WORDS)
    cands = _query_candidates(text, student)
    if not cands or total_words < min_words:
        return _result("skipped", reason="Text too short for web-source checking", words=total_words)

    budget = query_budget(total_words)
    max_pages = max(1, settings.WEB_SEARCH_MAX_PAGES)
    calls_before = _api_calls["n"]
    sents = StudentSentences(text, student)
    exclude = _csv(settings.WEB_EXCLUDE_DOMAINS)
    aggregators = _csv(settings.WEB_AGGREGATOR_DOMAINS)

    errors: List[str] = []
    attempts = [0]  # provider calls attempted (for the "every search failed" check)
    hits: Counter = Counter()
    rank_score: Counter = Counter()  # sum of 1/(1+position) over all searches: engine's own ranking
    info: Dict[str, Dict[str, Any]] = {}
    pages: Dict[str, Dict[str, Any]] = {}
    found: Dict[str, Dict[str, Any]] = {}
    queried: set = set()
    searches = rounds = 0

    def run(q):
        """One phrase, sent to every configured provider. Returns one result list per
        provider; a failing provider is recorded and skipped, the others still count."""
        lists = []
        for p in provs:
            attempts[0] += 1
            try:
                lists.append(search(p, q))
            except Exception as e:
                errors.append(f"{p}: {str(e)[:180]}")
        return lists

    def covered_tokens() -> set:
        out = set()
        for f in found.values():
            out |= f["tokens"]
        return out

    # Rounds: round 1 spreads half the budget over the whole document; later rounds spend
    # the rest on sentences that no page explains yet.
    while searches < budget and rounds < MAX_ROUNDS and len(pages) < max_pages:
        covered = covered_tokens()
        pool = [c for c in cands if c["phrase"].lower() not in queried
                and sum(1 for t in range(c["tok_s"], c["tok_e"]) if t in covered) < 0.5 * max(1, c["tok_e"] - c["tok_s"])]
        if not pool:
            break
        remaining = budget - searches
        k = remaining if rounds else min(remaining, max(4, math.ceil(budget / 2)))
        batch = _spread(pool, k, len(text))
        queried.update(c["phrase"].lower() for c in batch)
        with ThreadPoolExecutor(max_workers=4) as ex:
            batch_results = list(ex.map(run, [c["phrase"] for c in batch]))
        searches += len(batch)
        rounds += 1

        for results in (lst for per_phrase in batch_results for lst in per_phrase):
            for pos, r in enumerate(results):
                if _domain_in(r["url"], exclude):
                    continue
                key = normalize_url(r["url"])
                hits[key] += 1
                rank_score[key] += 1.0 / (1 + pos)
                entry = info.setdefault(key, {"url": r["url"], "title": r.get("title", ""), "snippets": []})
                if r.get("snippet"):
                    entry["snippets"].append(r["snippet"])

        new_keys = [k for k, _ in rank_score.most_common() if k not in pages][:min(PAGES_PER_ROUND, max_pages - len(pages))]
        with ThreadPoolExecutor(max_workers=6) as ex:
            pages.update(zip(new_keys, ex.map(lambda k: fetch_page(info[k]["url"]), new_keys)))

        for key in new_keys:
            page, meta = pages[key], info[key]
            if page["ok"]:
                body, evidence = page["text"], "page"
            else:
                body, evidence = " ... ".join(meta["snippets"]), "snippet"
            if not body:
                continue
            units = []  # matched pieces: exact spans and reworded sentences
            for sp in match_spans(student, body, min_words):
                units.append({"kind": "exact", "tok_s": sp["start_tok"], "tok_e": sp["end_tok"],
                              "page_start": sp["page_start"], "page_end": sp["page_end"], "score": 1.0})
            exact_toks = set()
            for u in units:
                exact_toks.update(range(u["tok_s"], u["tok_e"]))
            for rw in reworded_matches(sents, exact_toks, body):
                units.append({"kind": "reworded", "tok_s": rw["tok_s"], "tok_e": rw["tok_e"],
                              "page_start": rw["page_start"], "page_end": rw["page_end"],
                              "score": rw["similarity"], "overlap": rw["overlap"]})
            if not units:
                continue
            toks = set()
            for u in units:
                toks.update(range(u["tok_s"], u["tok_e"]))
            found[key] = {
                "key": key, "url": meta["url"], "title": page.get("title") or meta["title"] or _domain(meta["url"]),
                "body": body, "units": units, "tokens": toks, "exact_tokens": exact_toks,
                "evidence": evidence, "aggregator": _domain_in(meta["url"], aggregators),
            }

    if errors and len(errors) == attempts[0] and not found:
        return _result("error", queries=searches, error=errors[0], search_errors=len(errors))

    # Attribution: every matched word is credited to ONE source. Most coverage first;
    # mirrors/aggregators are penalised so the original site wins unless the mirror
    # covers clearly more of the text. Coverage differing by less than min_words counts
    # as a tie (popular text is reposted word-for-word on many sites); ties go to the page
    # with fetched text, then to the page the search engines rank highest across all
    # searches, which is usually the original/most authoritative one.
    def rank(f):
        cov = len(f["tokens"]) * (0.85 if f["aggregator"] else 1.0)
        return (-int(cov // min_words), f["evidence"] != "page", -rank_score[f["key"]], -cov)

    def unit_toks(u):
        return set(range(u["tok_s"], u["tok_e"]))

    mirror_min = min(min_words, PARA_MIN_WORDS)
    assigned: Dict[int, int] = {}  # student token -> index in `primaries`
    primaries: List[Dict[str, Any]] = []
    for f in sorted(found.values(), key=rank):
        # Text this page shares with sources already credited -> list it there as a mirror.
        shared = Counter(assigned[t] for t in f["tokens"] if t in assigned)
        for idx, n in shared.items():
            if n >= mirror_min and f["url"] not in primaries[idx]["also_found_at"]:
                primaries[idx]["also_found_at"].append(f["url"])
        # Pieces mostly not credited yet -> this page becomes their source.
        new_units = [u for u in f["units"]
                     if len(unit_toks(u) - assigned.keys()) >= 0.5 * len(unit_toks(u))]
        if not new_units:
            continue
        own = set()
        for u in new_units:
            own |= unit_toks(u) - assigned.keys()
        idx = len(primaries)
        for t in own:
            assigned[t] = idx
        f.update(also_found_at=[], own=own, own_units=new_units)
        primaries.append(f)

    # Output in the plugin's shape.
    sources, exact_total = [], 0
    for f in primaries:
        matches = []
        for u in sorted(f["own_units"], key=lambda u: u["tok_s"]):
            end_char = _extend_punct(text, student[u["tok_e"] - 1][2])
            # Every match has the same keys (exact matches have word_overlap 1.0).
            matches.append({
                "type": "web_exact" if u["kind"] == "exact" else "web_reworded",
                "source_chunk": f["body"][u["page_start"]:u["page_end"]][:MAX_SOURCE_CHUNK_CHARS],
                "target_chunk": text[student[u["tok_s"]][1]:end_char],
                "score": u["score"],
                "word_overlap": u.get("overlap", 1.0),
                "words": u["tok_e"] - u["tok_s"],
            })
        exact_words = len(f["own"] & f["exact_tokens"])
        exact_total += exact_words
        sources.append({
            "source_title": f["title"],
            "source_url": f["url"],
            "similarity": round(len(f["own"]) / total_words, 4),
            "matched_words": len(f["own"]),
            "exact_words": exact_words,
            "reworded_words": len(f["own"]) - exact_words,
            "evidence": f["evidence"],
            "document_sharing_site": f["aggregator"],
            "also_found_at": f["also_found_at"],
            "matches": matches,
        })
    sources.sort(key=lambda s: -s["similarity"])

    return _result(
        "completed",
        score=round(len(assigned) / total_words, 4),
        exact_score=round(exact_total / total_words, 4),
        reworded_score=round((len(assigned) - exact_total) / total_words, 4),
        queries=searches,
        query_budget=budget,
        api_calls=_api_calls["n"] - calls_before,  # billable search calls (cache hits excluded)
        rounds=rounds,
        pages_checked=len(pages),
        words=total_words,
        matched_words=len(assigned),
        exact_words=exact_total,
        reworded_words=len(assigned) - exact_total,
        search_errors=len(errors),
        sources=sources,
    )
