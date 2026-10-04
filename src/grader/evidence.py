"""Verification that evidence quoted by the judge really exists in the conversation."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\[\]]")


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def verify_evidence(evidence: list[str], source: str, *, min_length: int = 3) -> bool:
    """Return True only if every evidence excerpt is a (normalized) substring of `source`.

    An empty evidence list is considered unverified: the judge must cite something.
    """
    if not evidence:
        return False
    normalized_source = normalize(source)
    for excerpt in evidence:
        normalized = normalize(excerpt)
        if len(normalized) < min_length or normalized not in normalized_source:
            return False
    return True
