"""FUNSD semantic tags.

The published LayoutLMv3 FUNSD setup scores header, question, and answer.
Words labeled "other" in the dataset are the outside class, so they are not
a separate entity type.
"""

from __future__ import annotations

from deeptrace.types import Word

LABELS: list[str] = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]
LABEL2ID: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {index: label for label, index in LABEL2ID.items()}

_ENTITY_LABELS = {"header", "question", "answer"}


def bio_tags(words: list[Word]) -> list[str]:
    """IOB2 tags for words already sorted in the order the model will see."""

    tags: list[str] = []
    previous_id: int | None = None
    previous_name: str | None = None
    for word in words:
        raw = (word.entity_label or "").strip().lower()
        if raw not in _ENTITY_LABELS:
            if raw not in {"", "other", "o"}:
                raise ValueError(f"Unexpected FUNSD label {word.entity_label!r}")
            tags.append("O")
            previous_id = None
            previous_name = None
            continue
        name = raw.upper()
        if word.entity_id != previous_id or name != previous_name:
            tags.append(f"B-{name}")
        else:
            tags.append(f"I-{name}")
        previous_id = word.entity_id
        previous_name = name
    return tags


def bio_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """Return (start, end, label) spans. `end` is exclusive."""

    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    label: str | None = None
    for index, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None and label is not None:
                spans.append((start, index, label))
            start = index
            label = tag[2:]
            continue
        if tag.startswith("I-") and label is not None and tag[2:] == label and start is not None:
            continue
        if start is not None and label is not None:
            spans.append((start, index, label))
        start = None
        label = None
    if start is not None and label is not None:
        spans.append((start, len(tags), label))
    return spans
