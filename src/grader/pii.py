"""PII redaction applied before any text reaches the LLM.

Patterns target Brazilian formats (CPF, phones) plus e-mails and payment cards.
Order matters: cards are matched before CPF/phones so their digit groups are not
partially consumed by the shorter patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)),
    ("CARD", re.compile(r"\b(?:\d{4}[ .-]?){3}\d{4}\b")),
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    (
        "PHONE",
        re.compile(r"(?<!\d)(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}[ -]?\d{4}(?!\d)"),
    ),
]


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact_pii(text: str) -> RedactionResult:
    """Replace PII occurrences with `[TOKEN]` placeholders and count them by type."""
    counts: dict[str, int] = {}
    for token, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{token}]", text)
        if n:
            counts[token] = n
    return RedactionResult(text=text, counts=counts)
