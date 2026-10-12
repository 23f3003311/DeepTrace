"""Load the original FUNSD split. Boxes stay in pixels until encoding."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from deeptrace.types import Page, Word

_SPLIT_DIRS = {"train": "training_data", "test": "testing_data"}


def resolve_funsd_root(path: str | Path) -> Path:
    raw = Path(path)
    candidates = [raw, raw / "dataset", raw / "FUNSD" / "dataset"]
    for candidate in candidates:
        annotations = candidate / "training_data" / "annotations"
        if annotations.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No FUNSD training_data under {raw}. Download it with: python -m deeptrace.download"
    )


def _image_for(annotation_path: Path, image_dir: Path) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = image_dir / f"{annotation_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image for annotation {annotation_path.name}")


def _word_from_annotation(raw: dict, entity_id: int, entity_label: str) -> Word | None:
    text = str(raw.get("text", "")).strip()
    if not text or "box" not in raw:
        return None
    x0, y0, x1, y1 = (int(round(float(value))) for value in raw["box"])
    return Word(
        text=text,
        box=(x0, y0, x1, y1),
        confidence=1.0,
        entity_id=entity_id,
        entity_label=str(entity_label).strip().lower(),
    )


def load_page(annotation_path: Path, image_dir: Path) -> Page:
    payload = json.loads(annotation_path.read_text())
    words: list[Word] = []
    links: list[tuple[int, int]] = []
    for entity in payload.get("form", []):
        entity_id = int(entity["id"])
        entity_label = str(entity.get("label", "other"))
        entity_words = entity.get("words") or []
        if not entity_words and entity.get("text") and entity.get("box"):
            entity_words = [{"text": entity["text"], "box": entity["box"]}]
        for raw_word in entity_words:
            word = _word_from_annotation(raw_word, entity_id, entity_label)
            if word is not None:
                words.append(word)
        for link in entity.get("linking") or []:
            if len(link) == 2:
                links.append((int(link[0]), int(link[1])))
    image_path = _image_for(annotation_path, image_dir)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
    if not words:
        raise ValueError(f"{annotation_path.name} has no OCR words")
    return Page(
        image_width=width,
        image_height=height,
        words=words,
        page_id=annotation_path.stem,
        links=links,
        source=str(image_path),
    )


def iter_pages(root: str | Path, split: str) -> Iterator[Page]:
    if split not in _SPLIT_DIRS:
        raise ValueError("split must be 'train' or 'test'")
    dataset_root = resolve_funsd_root(root)
    folder = dataset_root / _SPLIT_DIRS[split]
    annotation_dir = folder / "annotations"
    image_dir = folder / "images"
    paths = sorted(annotation_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No annotations in {annotation_dir}")
    for path in paths:
        yield load_page(path, image_dir)


def count_pages(root: str | Path, split: str) -> int:
    return sum(1 for _ in iter_pages(root, split))
