"""Box math, reading order, and the sliding windows fed to LayoutLMv3."""

from __future__ import annotations

from collections.abc import Callable

from deeptrace.types import Word


def normalize_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> list[int]:
    """Map a pixel box to the integer 0–1000 range LayoutLMv3 requires.

    A zero-area box is expanded by one unit. Coordinates are clamped so the
    position embedding lookup cannot go out of range.
    """

    x0, y0, x1, y1 = (int(value) for value in box)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    width = max(int(width), 1)
    height = max(int(height), 1)

    def scale(value: int, size: int) -> int:
        return int(max(0, min(1000, round(1000 * value / size))))

    nx0, ny0, nx1, ny1 = scale(x0, width), scale(y0, height), scale(x1, width), scale(y1, height)
    if nx1 <= nx0:
        if nx0 >= 1000:
            nx0, nx1 = 999, 1000
        else:
            nx1 = nx0 + 1
    if ny1 <= ny0:
        if ny0 >= 1000:
            ny0, ny1 = 999, 1000
        else:
            ny1 = ny0 + 1
    return [nx0, ny0, nx1, ny1]


def union_box(words: list[Word]) -> tuple[int, int, int, int]:
    if not words:
        raise ValueError("Cannot take a union of an empty word list")
    return (
        min(word.box[0] for word in words),
        min(word.box[1] for word in words),
        max(word.box[2] for word in words),
        max(word.box[3] for word in words),
    )


def group_lines(words: list[Word], center_ratio: float = 0.5) -> list[list[Word]]:
    """Cluster boxes into lines, then sort each line left to right.

    A word joins the current line when its vertical center is within half the
    line height of that line's center. Words are considered in top-to-bottom
    order, so the comparison only needs the latest line.
    """

    ordered = sorted(words, key=lambda word: ((word.box[1] + word.box[3]) / 2, word.box[0]))
    lines: list[list[Word]] = []
    for word in ordered:
        center = (word.box[1] + word.box[3]) / 2
        height = max(1, word.box[3] - word.box[1])
        if lines:
            line = lines[-1]
            line_center = sum((item.box[1] + item.box[3]) / 2 for item in line) / len(line)
            line_height = sum(max(1, item.box[3] - item.box[1]) for item in line) / len(line)
            limit = center_ratio * max(height, line_height)
            if abs(center - line_center) <= limit:
                line.append(word)
                continue
        lines.append([word])
    for line in lines:
        line.sort(key=lambda item: item.box[0])
    return lines


def reading_order(words: list[Word]) -> list[Word]:
    return [word for line in group_lines(words) for word in line]


def chunk_spans(
    n_words: int,
    fits: Callable[[int, int], bool],
    stride: int,
) -> list[tuple[int, int]]:
    """Cover `range(n_words)` with [start, end) windows that each satisfy `fits`.

    Later windows step back by `stride` words so a phrase on a boundary is not
    cut out of every window. A word that cannot fit even alone raises.
    """

    if n_words < 0:
        raise ValueError("n_words must be non-negative")
    if stride < 0:
        raise ValueError("stride must be non-negative")
    if n_words == 0:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n_words:
        if not fits(start, start + 1):
            raise ValueError(f"Word {start} does not fit in one model window")
        low = start + 1
        high = n_words
        best = start + 1
        while low <= high:
            mid = (low + high) // 2
            if fits(start, mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        spans.append((start, best))
        if best == n_words:
            break
        next_start = best - stride
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return spans
