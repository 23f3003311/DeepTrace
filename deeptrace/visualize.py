"""Draw layout outlines and Integrated Gradients scores on the page."""

from __future__ import annotations

from PIL import Image, ImageDraw

from deeptrace.types import Region, Word

_REGION_COLOR = {
    "header": (166, 124, 61, 255),
    "section": (11, 79, 99, 255),
    "table": (15, 110, 134, 255),
    "paragraph": (170, 170, 170, 220),
}
_TAG_OUTLINE = {
    "HEADER": (166, 124, 61, 230),
    "QUESTION": (11, 31, 58, 230),
    "ANSWER": (15, 110, 134, 230),
}


def _tag_outline(tag: str | None) -> tuple[int, int, int, int]:
    if not tag or tag == "O":
        return (150, 150, 150, 160)
    return _TAG_OUTLINE.get(tag.split("-", 1)[-1], (150, 150, 150, 160))


def render_page(
    image: Image.Image,
    words: list[Word],
    scores: list[float] | None = None,
    regions: list[Region] | None = None,
    answer_box: tuple[int, int, int, int] | None = None,
    show_tags: bool = False,
) -> Image.Image:
    if scores is not None and len(scores) != len(words):
        raise ValueError("One score is required for each word")
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if regions:
        for region in regions:
            color = _REGION_COLOR.get(region.kind, (170, 170, 170, 220))
            draw.rectangle(region.box, outline=color, width=2)
    if show_tags:
        for word in words:
            draw.rectangle(word.box, outline=_tag_outline(word.pred_tag), width=2)
    if scores:
        positive = [score for score in scores if score > 0]
        peak = max(positive) if positive else 1.0
        for word, score in zip(words, scores):
            if score <= 0:
                continue
            alpha = int(50 + 180 * (score / peak))
            draw.rectangle(word.box, fill=(15, 110, 134, alpha))
    if answer_box is not None:
        draw.rectangle(answer_box, outline=(166, 124, 61, 255), width=3)
    return Image.alpha_composite(base, overlay).convert("RGB")
