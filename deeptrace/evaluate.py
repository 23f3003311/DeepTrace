"""Score a trained checkpoint on the FUNSD test split.

Field quality is entity-level precision, recall, and F1. With
--evidence-pages, Integrated Gradients is run on that many test pages and a
hit is counted when the highest-scoring word of an answer sits inside the
gold answer box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from deeptrace.fields import entities_from_tags
from deeptrace.funsd import iter_pages
from deeptrace.geometry import reading_order
from deeptrace.labels import bio_tags
from deeptrace.metrics import entity_scores
from deeptrace.model import load_model_and_processor, pick_device
from deeptrace.pipeline import DeepTrace


def _center_inside(box: tuple[int, int, int, int], region: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    rx0, ry0, rx1, ry1 = region
    return rx0 <= center_x <= rx1 and ry0 <= center_y <= ry1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepTrace on FUNSD")
    parser.add_argument("--checkpoint", default="outputs/layoutlmv3-funsd")
    parser.add_argument("--funsd-dir", default="data/funsd")
    parser.add_argument("--device", default=None)
    parser.add_argument("--evidence-pages", type=int, default=0)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, processor = load_model_and_processor(args.checkpoint, device=device)
    pipeline = DeepTrace(model, processor, device)
    truths: list[list[str]] = []
    predictions: list[list[str]] = []
    evidence_hits = 0
    evidence_total = 0
    for page_index, page in enumerate(iter_pages(args.funsd_dir, "test")):
        with Image.open(page.source) as image:
            prediction = pipeline.extract(image.convert("RGB"), words=page.words, prepare=False)
        ordered_gold = reading_order(page.words)
        if [word.box for word in ordered_gold] != [word.box for word in prediction.words]:
            raise RuntimeError(f"{page.page_id} reading order changed between gold and prediction")
        truths.append(bio_tags(ordered_gold))
        predictions.append(prediction.tags)
        if page_index >= args.evidence_pages:
            continue
        gold_answers = [entity for entity in entities_from_tags(bio_tags(ordered_gold), ordered_gold) if entity.label == "ANSWER"]
        for gold in gold_answers:
            evidence_total += 1
            overlapping = [
                link
                for link in prediction.fields
                if not (link.answer.end <= gold.start or link.answer.start >= gold.end)
            ]
            if not overlapping:
                continue
            chosen = max(overlapping, key=lambda link: min(link.answer.end, gold.end) - max(link.answer.start, gold.start))
            field_index = prediction.fields.index(chosen)
            explanation = pipeline.explain(prediction, field_index, n_steps=args.ig_steps)
            span_scores = explanation.scores[chosen.answer.start : chosen.answer.end]
            if not span_scores:
                continue
            best_offset = max(range(len(span_scores)), key=lambda offset: span_scores[offset])
            word = prediction.words[chosen.answer.start + best_offset]
            if _center_inside(word.box, gold.box):
                evidence_hits += 1
    scores = entity_scores(truths, predictions)
    if evidence_total:
        scores["evidence_hit_rate"] = evidence_hits / evidence_total
        scores["evidence_pages"] = args.evidence_pages
        scores["evidence_answers"] = evidence_total
    print(json.dumps(scores, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
