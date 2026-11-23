"""
Text normalization to reduce false-positive similarity from legitimately shared
content — quotations, reference/bibliography lines, and assignment boilerplate
(e.g. the prompt every student pastes). Applied before chunking/embedding so the
similarity score reflects the student's OWN writing, Turnitin-style.

All steps are opt-in per assignment and conservative: they remove clearly-shared
spans, never the student's original prose.
"""
import re
from difflib import SequenceMatcher

# Quotation spans: "..." “...” '...' (3+ chars inside, to avoid nuking apostrophes)
_QUOTE_RE = re.compile(r'[""\'"]{1}.{3,}?[""\'"]{1}', re.DOTALL)

# Reference/citation lines: bibliography entries and inline citations.
_REFERENCE_LINE_RE = re.compile(
    r'^\s*(\[\d+\]|\d+\.\s|references?|bibliography|works cited)\b.*$',
    re.IGNORECASE | re.MULTILINE,
)
_INLINE_CITATION_RE = re.compile(r'\((?:[A-Z][A-Za-z]+(?:\s(?:et al\.?|and|&)\s?[A-Z][A-Za-z]+)*,?\s*\d{4})\)')


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.?!])\s+|\n+')


def _mask_boilerplate(text: str, boilerplate: str) -> str:
    """
    Remove sentences from `text` that closely match a sentence of `boilerplate`
    (the assignment prompt / shared template), whether the student pasted it on
    its own line or inline with their answer. Fuzzy so minor edits don't defeat it.
    """
    if not boilerplate:
        return text
    bp_sents = [s.strip().lower() for s in _SENTENCE_SPLIT_RE.split(boilerplate) if len(s.strip()) >= 12]
    if not bp_sents:
        return text
    kept = []
    for sent in _SENTENCE_SPLIT_RE.split(text):
        probe = sent.strip().lower()
        if len(probe) >= 12 and any(
            SequenceMatcher(None, probe, bp).ratio() >= 0.82 for bp in bp_sents
        ):
            continue  # drop boilerplate sentence
        kept.append(sent)
    return " ".join(kept)


def normalize_for_compare(text: str, strip_quotes: bool = True,
                          strip_references: bool = True, boilerplate_text: str = None) -> str:
    """Return `text` with shared/expected content removed, per the given options."""
    if not text:
        return text
    if boilerplate_text:
        text = _mask_boilerplate(text, boilerplate_text)
    if strip_references:
        text = _REFERENCE_LINE_RE.sub(" ", text)
        text = _INLINE_CITATION_RE.sub(" ", text)
    if strip_quotes:
        text = _QUOTE_RE.sub(" ", text)
    # Collapse the whitespace left behind so chunk boundaries stay stable.
    return re.sub(r'\s+', ' ', text).strip()
