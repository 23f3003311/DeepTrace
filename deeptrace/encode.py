"""Encode a page into the tensors LayoutLMv3ForTokenClassification consumes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import LayoutLMv3Processor

from deeptrace.geometry import chunk_spans, normalize_box, reading_order
from deeptrace.labels import LABEL2ID, bio_tags
from deeptrace.types import Word

_MODEL_KEYS = ("input_ids", "bbox", "attention_mask", "pixel_values", "labels")


@dataclass
class EncodedWindow:
    tensors: dict[str, torch.Tensor]
    word_start: int
    word_end: int
    word_ids: list[int | None]
    words: list[Word]


def load_processor(name_or_path: str = "microsoft/layoutlmv3-base") -> LayoutLMv3Processor:
    """OCR is supplied by us. The processor must not run its own Tesseract pass."""

    processor = LayoutLMv3Processor.from_pretrained(name_or_path, apply_ocr=False, use_fast=True)
    if processor.image_processor.apply_ocr:
        raise RuntimeError("LayoutLMv3 processor was loaded with apply_ocr=True")
    return processor


def _token_length(processor: LayoutLMv3Processor, words: list[Word], width: int, height: int) -> int:
    if not words:
        return 0
    boxes = [normalize_box(word.box, width, height) for word in words]
    labels = [LABEL2ID[tag] for tag in bio_tags(words)]
    encoded = processor.tokenizer(
        [word.text for word in words],
        boxes=boxes,
        word_labels=labels,
        truncation=False,
        padding=False,
        verbose=False,
    )
    return len(encoded["input_ids"])


def encode_page(
    processor: LayoutLMv3Processor,
    image: Image.Image,
    words: list[Word],
    max_length: int = 512,
    stride_words: int = 32,
    sort: bool = True,
) -> tuple[list[Word], list[EncodedWindow]]:
    """Sort words into reading order and split them into model windows.

    Every word appears in at least one window. Each window is checked so the
    tokenizer did not drop a word to stay under 512 tokens.
    """

    if max_length < 8:
        raise ValueError("max_length is too small for LayoutLMv3")
    image = image.convert("RGB")
    width, height = image.size
    ordered = reading_order(words) if sort else list(words)
    global_tags = bio_tags(ordered)

    def fits(start: int, end: int) -> bool:
        return _token_length(processor, ordered[start:end], width, height) <= max_length

    spans = chunk_spans(len(ordered), fits, stride_words)
    windows: list[EncodedWindow] = []
    for start, end in spans:
        slice_words = ordered[start:end]
        boxes = [normalize_box(word.box, width, height) for word in slice_words]
        label_ids = [LABEL2ID[tag] for tag in global_tags[start:end]]
        encoded = processor(
            image,
            [word.text for word in slice_words],
            boxes=boxes,
            word_labels=label_ids,
            truncation=False,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[0] != 1:
            raise RuntimeError("Expected one window per processor call")
        word_ids = list(encoded.word_ids(0))
        present = {word_id for word_id in word_ids if word_id is not None}
        if present != set(range(len(slice_words))):
            missing = sorted(set(range(len(slice_words))) - present)
            raise RuntimeError(f"Tokenizer dropped words {missing} in window [{start}, {end})")
        if int(encoded["attention_mask"].sum()) > max_length:
            raise RuntimeError("Window exceeds the LayoutLMv3 length of 512")
        tensors = {key: encoded[key].squeeze(0) for key in _MODEL_KEYS}
        tensors["input_ids"] = tensors["input_ids"].long()
        tensors["bbox"] = tensors["bbox"].long()
        tensors["attention_mask"] = tensors["attention_mask"].long()
        tensors["labels"] = tensors["labels"].long()
        tensors["pixel_values"] = tensors["pixel_values"].float()
        windows.append(
            EncodedWindow(
                tensors=tensors,
                word_start=start,
                word_end=end,
                word_ids=word_ids,
                words=slice_words,
            )
        )
    return ordered, windows
