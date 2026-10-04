"""Prompt construction for the judge.

The user prompt uses explicit XML-style markers so that both the real model and the
`FakeLLMClient` can locate the criterion and the transcript deterministically.
"""

from __future__ import annotations

from grader.rubric import Criterion, Rubric

SYSTEM_PROMPT = """You are a rigorous quality auditor for sales and customer-service \
conversations written in Brazilian Portuguese.

You grade ONE criterion at a time using the rubric provided. Rules:
- Use only what is written in the transcript. Never assume facts that are not there.
- The score must follow the 0-5 anchors literally.
- `evidence` must contain short excerpts copied VERBATIM from the transcript (same words, \
same accents). Do not paraphrase. Prefer 1-3 excerpts.
- `rationale` is 1-3 sentences, in Portuguese, explaining how the evidence maps to the anchor.
- Personal data in the transcript was replaced by tokens such as [CPF] or [PHONE]; treat them \
as redacted values, not as content to grade.
"""


def build_user_prompt(rubric: Rubric, criterion: Criterion, transcript: str) -> str:
    """Build the user turn for grading `criterion` over `transcript`."""
    anchors = "\n".join(f"  {level}: {text}" for level, text in criterion.anchors.items())
    return (
        f'<rubric id="{rubric.id}" version="{rubric.version}">{rubric.name}</rubric>\n'
        f'<criterion id="{criterion.id}" weight="{criterion.weight}">\n'
        f"name: {criterion.name}\n"
        f"description: {criterion.description.strip()}\n"
        f"anchors:\n{anchors}\n"
        f"</criterion>\n"
        f"<transcript>\n{transcript}\n</transcript>\n"
        "Grade the criterion above and return the structured result."
    )
