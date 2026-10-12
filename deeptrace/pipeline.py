"""Run OCR or gold words, extract fields, and optionally explain one answer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

from deeptrace.encode import encode_page
from deeptrace.explain import explain_span
from deeptrace.fields import entities_from_tags, pair_fields
from deeptrace.geometry import reading_order
from deeptrace.layout import group_regions
from deeptrace.model import predict_word_tags
from deeptrace.preprocess import prepare_page
from deeptrace.types import Entity, Explanation, FieldLink, Region, Word
from deeptrace.visualize import render_page


@dataclass
class PagePrediction:
    image: Image.Image
    words: list[Word]
    tags: list[str]
    regions: list[Region]
    fields: list[FieldLink]
    deskew_angle: float = 0.0
    explanations: dict[int, Explanation] = field(default_factory=dict)

    def field_rows(self) -> list[dict[str, str]]:
        rows = []
        for index, link in enumerate(self.fields):
            question = link.question.text(self.words) if link.question is not None else ""
            rows.append(
                {
                    "index": str(index),
                    "field": question,
                    "value": link.answer.text(self.words),
                }
            )
        return rows


def evidence_records(prediction: PagePrediction, field_index: int) -> list[dict[str, str]]:
    """Words that make one answer inspectable.

    The field name and the answer are always listed. Other words appear only
    when their Integrated Gradients score is positive, because that score is
    what pushed the model toward the predicted answer label.
    """

    link = prediction.fields[field_index]
    explanation = prediction.explanations.get(field_index)
    records: list[dict[str, str]] = []
    for index, word in enumerate(prediction.words):
        if link.question is not None and link.question.start <= index < link.question.end:
            role = "field name"
        elif link.answer.start <= index < link.answer.end:
            role = "answer"
        else:
            role = "context"
        score = None if explanation is None else explanation.scores[index]
        if role == "context" and (score is None or score <= 0):
            continue
        records.append(
            {
                "word": word.text,
                "tag": word.pred_tag or "",
                "role": role,
                "evidence": "" if score is None else f"{score:.4f}",
                "box": f"{word.box[0]}, {word.box[1]}, {word.box[2]}, {word.box[3]}",
            }
        )
    if explanation is not None:
        records.sort(key=lambda row: float(row["evidence"] or "-1e9"), reverse=True)
    return records


class DeepTrace:
    def __init__(self, model, processor, device: torch.device, ocr=None):
        self.model = model
        self.processor = processor
        self.device = device
        self.ocr = ocr

    def extract(
        self,
        image: Image.Image,
        words: list[Word] | None = None,
        prepare: bool | None = None,
    ) -> PagePrediction:
        """Extract fields. Pass `words` to skip OCR, which is how FUNSD is scored.

        `prepare` deskews the page. Leave it off when `words` already belong to
        the image, and on when Baidu has not read the page yet.
        """

        if prepare is None:
            prepare = words is None
        if prepare and words is not None:
            raise ValueError("Deskew would move the page but the supplied boxes would not move with it")
        angle = 0.0
        page_image = image.convert("RGB")
        if prepare:
            page_image, angle = prepare_page(page_image)
        if words is None:
            if self.ocr is None:
                raise RuntimeError("No OCR backend was provided")
            words = self.ocr.read(page_image)
        ordered = reading_order(words)
        for word in ordered:
            word.pred_tag = None
        if not ordered:
            return PagePrediction(
                image=page_image,
                words=[],
                tags=[],
                regions=[],
                fields=[],
                deskew_angle=angle,
            )
        _same_order, windows = encode_page(self.processor, page_image, ordered, sort=False)
        tags = predict_word_tags(self.model, windows, len(ordered), self.device)
        for word, tag in zip(ordered, tags):
            word.pred_tag = tag
        regions = group_regions(ordered, page_image.size)
        entities = entities_from_tags(tags, ordered)
        return PagePrediction(
            image=page_image,
            words=ordered,
            tags=tags,
            regions=regions,
            fields=pair_fields(entities, ordered),
            deskew_angle=angle,
        )

    def explain(self, prediction: PagePrediction, field_index: int, n_steps: int = 50) -> Explanation:
        link = prediction.fields[field_index]
        explanation = explain_span(
            self.model,
            self.processor,
            prediction.image,
            prediction.words,
            link.answer.start,
            link.answer.end,
            self.device,
            n_steps=n_steps,
        )
        prediction.explanations[field_index] = explanation
        return explanation

    def draw(self, prediction: PagePrediction, field_index: int | None = None, show_tags: bool = False) -> Image.Image:
        scores = None
        answer: Entity | None = None
        if field_index is not None and field_index in prediction.explanations:
            scores = prediction.explanations[field_index].scores
            answer = prediction.fields[field_index].answer
        elif field_index is not None:
            answer = prediction.fields[field_index].answer
        return render_page(
            prediction.image,
            prediction.words,
            scores=scores,
            regions=prediction.regions,
            answer_box=None if answer is None else answer.box,
            show_tags=show_tags,
        )
