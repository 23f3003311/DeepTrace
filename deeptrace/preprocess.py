"""Turn an uploaded file into page images.

Deskew runs before OCR on a new upload. It must not run on FUNSD pages,
because those word boxes were drawn on the original scan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def clean_page(image: Image.Image) -> Image.Image:
    """Return an RGB page. Stretch contrast only when the scan is nearly flat."""

    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    if float(gray.std()) >= 25:
        return rgb
    low, high = np.percentile(gray, [1, 99])
    if high <= low + 1:
        return rgb
    stretched = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return Image.fromarray(stretched, mode="L").convert("RGB")


def _projection_score(gray: np.ndarray, angle: float) -> float:
    rotated = Image.fromarray(gray).rotate(angle, expand=False, fillcolor=255, resample=Image.Resampling.BILINEAR)
    ink = np.asarray(rotated) < 180
    projection = ink.sum(axis=1).astype(np.float32)
    return float(projection.var())


def deskew(image: Image.Image, search_degrees: float = 8.0) -> tuple[Image.Image, float]:
    """Rotate a page so text lines are horizontal.

    The search maximises the variance of the horizontal ink projection, which
    is highest when lines of text sit on their own rows. The returned angle is
    the correction applied, in degrees counter-clockwise.
    """

    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    height, width = gray.shape
    scale = 800 / max(width, 1)
    if scale < 1:
        small = np.asarray(rgb.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BILINEAR).convert("L"))
    else:
        small = gray

    def best_in(angles: np.ndarray, current_angle: float, current_score: float) -> tuple[float, float]:
        for angle in angles:
            score = _projection_score(small, float(angle))
            if score > current_score:
                current_angle = float(angle)
                current_score = score
        return current_angle, current_score

    coarse = np.linspace(-search_degrees, search_degrees, 33)
    angle, score = best_in(coarse, 0.0, _projection_score(small, 0.0))
    fine = np.linspace(angle - 0.8, angle + 0.8, 17)
    angle, _score = best_in(fine, angle, score)
    if abs(angle) < 0.3:
        return rgb, 0.0
    corrected = rgb.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.Resampling.BICUBIC)
    return corrected, angle


def prepare_page(image: Image.Image) -> tuple[Image.Image, float]:
    """Clean, then deskew. Use this only when OCR has not run yet."""

    cleaned = clean_page(image)
    return deskew(cleaned)


def load_upload(path: str | Path, dpi: int = 200) -> list[Image.Image]:
    """Load an image, or render every page of a PDF."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        import fitz

        document = fitz.open(file_path)
        pages: list[Image.Image] = []
        try:
            for page in document:
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
                pages.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
        finally:
            document.close()
        if not pages:
            raise ValueError(f"PDF has no pages: {file_path}")
        return pages
    if suffix not in _IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported file type {suffix!r}. Use a PDF or a page image.")
    with Image.open(file_path) as image:
        return [image.convert("RGB")]
