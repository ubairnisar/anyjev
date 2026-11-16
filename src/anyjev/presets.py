"""Ready-made tasks. Each function returns a fresh Task you can tweak."""

from __future__ import annotations

from typing import Callable

from .task import Option, Task

_STRUCTURED_DEFINITIONS = """\
DEFINITIONS

STRUCTURED content is present when the crop contains one or more of:
  * Key-value pairs - a label paired with a field/value (e.g. "Name: John Doe",
    a label next to a box/line/dropdown/chip).
  * Checkboxes or radio buttons before or after a phrase - any selectable
    indicator, such as a box or circle that may be checked, filled, marked,
    selected, or not. IMPORTANT NOTE: Due to cropping, these indicators may
    appear incomplete or partially visible, but if any part of a
    checkbox/radio button is present, it counts as structured content.
  * Signature fields - a signature line or box, signed or blank.
  * Ordered lists - items preceded by incrementing numeric/alphanumeric
    markers (1., 2., i., ii., A., B., ...).
  * Unordered lists - items preceded by bullet markers (-, *, bullet dots, ...).
  * Image, Symbol, Icon - if the crop contains any non-text visual element
    (image fragment, icon, decorative graphic, rectangle or marker), it counts
    as structured content. Ignore the content of the image; just note its
    presence.

PURE PROSE is when the crop has:
  * Only paragraphs, sentences, phrases, or standalone headings/subheadings.
  * No labeled fields, no boxes/lines to fill in, no checkboxes/radios, no
    signature areas, no bulleted/numbered list markers.

When in doubt, choose STRUCTURED (safer to run full extraction than to lose
structured data)."""


def structured_vs_prose(threshold: float = 0.30) -> Task:
    """Document crop: structured content or pure prose? (biased to structured)

    Structured = key-value fields, checkboxes/radios, signatures, lists, icons."""
    return Task(
        name="structured-vs-prose",
        question=(
            "Analyse the text-block image and decide whether it contains any "
            "STRUCTURED content, or is PURE PROSE."
        ),
        context=_STRUCTURED_DEFINITIONS,
        options=[
            Option("structured", "STRUCTURED - the crop contains any structured content as defined above"),
            Option("prose", "PURE PROSE - only paragraphs, sentences, phrases or standalone headings"),
        ],
        positive="structured",
        threshold=threshold,
    )


def sentiment() -> Task:
    """Text sentiment: positive / neutral / negative."""
    return Task(
        name="sentiment",
        question="What is the overall sentiment of the input text?",
        options={
            "positive": "Positive",
            "neutral": "Neutral or mixed",
            "negative": "Negative",
        },
    )


def answer_correct(threshold: float | None = None) -> Task:
    """LLM-as-judge: is the answer in the input correct for its question?

    Pass text formatted as "Question: ...\\nReference: ...\\nAnswer: ..."."""
    return Task(
        name="answer-correct",
        question=(
            "The input contains a question, a reference answer and a candidate answer. "
            "Is the candidate answer correct and consistent with the reference?"
        ),
        options={
            "correct": "Correct - it matches the reference in substance",
            "incorrect": "Incorrect - it contradicts or misses the reference",
        },
        positive="correct" if threshold is not None else None,
        threshold=threshold,
    )


PRESETS: dict[str, Callable[[], Task]] = {
    "structured-vs-prose": structured_vs_prose,
    "sentiment": sentiment,
    "answer-correct": answer_correct,
}


def get(name: str) -> Task:
    try:
        return PRESETS[name]()
    except KeyError:
        raise KeyError(f"Unknown preset {name!r}. Available: {sorted(PRESETS)}") from None
