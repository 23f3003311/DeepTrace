"""Integrated Gradients from the predicted answer back onto OCR words.

The interpolated input is the word embedding. Boxes and the page image stay
fixed, because the claim is about which words drove the answer. The baseline
is the pad-token embedding: a blank token, not a zero vector the model never
saw in pretraining.

Scores are the sum of token attributions across the embedding axis. That sum
is the completeness term from Sundararajan, Taly, and Yan (ICML 2017). Subword
pieces of one OCR word are added together. Positive scores supported the
predicted label.
"""

from __future__ import annotations

import torch
from captum.attr import IntegratedGradients
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from deeptrace.encode import _token_length, encode_page
from deeptrace.labels import LABEL2ID
from deeptrace.types import Explanation, Word


def aggregate_token_scores(
    token_scores: list[float],
    word_ids: list[int | None],
    n_words: int,
) -> list[float]:
    scores = [0.0] * n_words
    if len(token_scores) != len(word_ids):
        raise ValueError("token scores and word ids have different lengths")
    for score, word_id in zip(token_scores, word_ids):
        if word_id is None:
            continue
        if word_id < 0 or word_id >= n_words:
            raise IndexError(f"word id {word_id} is outside 0..{n_words - 1}")
        scores[word_id] += float(score)
    return scores


def first_subword_positions(word_ids: list[int | None], local_word_indexes: list[int]) -> list[int]:
    wanted = set(local_word_indexes)
    found: dict[int, int] = {}
    for token_index, word_id in enumerate(word_ids):
        if word_id in wanted and word_id not in found:
            found[word_id] = token_index
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"Answer words missing from the explanation window: {sorted(missing)}")
    return [found[index] for index in local_word_indexes]


def _context_bounds(
    processor: LayoutLMv3Processor,
    words: list[Word],
    start: int,
    end: int,
    width: int,
    height: int,
    max_length: int,
) -> tuple[int, int]:
    if end <= start:
        raise ValueError("Answer span is empty")
    while end - start > 1 and _token_length(processor, words[start:end], width, height) > max_length:
        end -= 1
    if _token_length(processor, words[start:end], width, height) > max_length:
        raise ValueError("A single answer word does not fit in the explanation window")

    left = start
    right = end
    while left > 0 and _token_length(processor, words[left - 1 : right], width, height) <= max_length:
        left -= 1
    while right < len(words) and _token_length(processor, words[left : right + 1], width, height) <= max_length:
        right += 1
    return left, right


def explain_span(
    model: LayoutLMv3ForTokenClassification,
    processor: LayoutLMv3Processor,
    image: Image.Image,
    words: list[Word],
    start: int,
    end: int,
    device: torch.device,
    n_steps: int = 50,
) -> Explanation:
    """Attribute the predicted tags on words[start:end] to every word in context."""

    if n_steps < 2:
        raise ValueError("Integrated Gradients needs at least 2 steps")
    if not 0 <= start < end <= len(words):
        raise IndexError("Answer span is outside the page")
    image = image.convert("RGB")
    width, height = image.size
    left, right = _context_bounds(processor, words, start, end, width, height, max_length=512)
    _ordered, windows = encode_page(
        processor,
        image,
        words[left:right],
        max_length=512,
        stride_words=0,
        sort=False,
    )
    if len(windows) != 1:
        raise RuntimeError("Explanation context did not fit in one window")
    window = windows[0]
    local_indexes = list(range(start - left, end - left))
    positions = first_subword_positions(window.word_ids, local_indexes)
    class_ids = []
    for offset, _position in enumerate(positions):
        tag = words[start + offset].pred_tag
        if tag not in LABEL2ID or tag == "O":
            raise ValueError(f"Cannot explain a word tagged {tag!r}")
        class_ids.append(LABEL2ID[tag])

    class_index = torch.tensor(class_ids, dtype=torch.long, device=device)
    position_index = torch.tensor(positions, dtype=torch.long, device=device)
    input_ids = window.tensors["input_ids"].unsqueeze(0).to(device)
    bbox = window.tensors["bbox"].unsqueeze(0).to(device)
    attention_mask = window.tensors["attention_mask"].unsqueeze(0).to(device)
    pixel_values = window.tensors["pixel_values"].unsqueeze(0).to(device)

    was_training = model.training
    grad_flags = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        embedding = model.get_input_embeddings()
        inputs_embeds = embedding(input_ids).detach()
        pad_id = processor.tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_id)
        baseline = embedding(baseline_ids).detach()

        def forward_from_embeds(embeds: torch.Tensor) -> torch.Tensor:
            batch = embeds.shape[0]
            logits = model(
                inputs_embeds=embeds,
                bbox=bbox.expand(batch, -1, -1),
                attention_mask=attention_mask.expand(batch, -1),
                pixel_values=pixel_values.expand(batch, -1, -1, -1),
            ).logits
            chosen = logits[:, position_index, class_index]
            return chosen.sum(dim=-1)

        integrated = IntegratedGradients(forward_from_embeds)
        attributions, delta = integrated.attribute(
            inputs_embeds,
            baselines=baseline,
            n_steps=n_steps,
            internal_batch_size=2,
            return_convergence_delta=True,
        )
        token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()
        delta_value = float(delta.detach().flatten()[0].cpu())
    finally:
        for parameter, flag in zip(model.parameters(), grad_flags):
            parameter.requires_grad_(flag)
        if was_training:
            model.train()
    local_scores = aggregate_token_scores(token_scores, window.word_ids, window.word_end - window.word_start)
    page_scores = [0.0] * len(words)
    for offset, score in enumerate(local_scores):
        page_scores[left + offset] = score
    return Explanation(
        scores=page_scores,
        convergence_delta=delta_value,
        target_word_indexes=list(range(start, end)),
    )
