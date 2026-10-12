"""Shared records for one document page."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    """One OCR token and the pixel box that contains it. Box is (x0, y0, x1, y1)."""

    text: str
    box: tuple[int, int, int, int]
    confidence: float = 1.0
    entity_id: int | None = None
    entity_label: str | None = None
    pred_tag: str | None = None


@dataclass
class Region:
    """A layout block built by grouping OCR boxes."""

    kind: str
    words: list[Word]
    box: tuple[int, int, int, int]


@dataclass
class Entity:
    """A contiguous header, question, or answer span in reading order."""

    label: str
    start: int
    end: int
    box: tuple[int, int, int, int]

    def text(self, words: list[Word]) -> str:
        return " ".join(word.text for word in words[self.start : self.end]).strip()


@dataclass
class FieldLink:
    """A question (the field name) paired with the answer span that fills it."""

    answer: Entity
    question: Entity | None = None


@dataclass
class Page:
    image_width: int
    image_height: int
    words: list[Word]
    page_id: str = ""
    links: list[tuple[int, int]] = field(default_factory=list)
    source: str = ""


@dataclass
class Explanation:
    """Signed Integrated Gradients score for every word on the page."""

    scores: list[float]
    convergence_delta: float
    target_word_indexes: list[int]
