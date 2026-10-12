"""Group OCR lines into header, paragraph, table, and section blocks."""

from __future__ import annotations

from statistics import median

from deeptrace.geometry import group_lines, union_box
from deeptrace.types import Region, Word


def _line_gap(previous: list[Word], current: list[Word]) -> float:
    previous_bottom = max(word.box[3] for word in previous)
    current_top = min(word.box[1] for word in current)
    return current_top - previous_bottom


def _blocks(lines: list[list[Word]]) -> list[list[list[Word]]]:
    if not lines:
        return []
    heights = [max(word.box[3] for word in line) - min(word.box[1] for word in line) for line in lines]
    typical = max(1.0, float(median(heights)))
    blocks: list[list[list[Word]]] = [[lines[0]]]
    for line in lines[1:]:
        if _line_gap(blocks[-1][-1], line) > 0.8 * typical:
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return blocks


def _is_table(lines: list[list[Word]]) -> bool:
    if len(lines) < 3:
        return False
    heights = [max(1, word.box[3] - word.box[1]) for line in lines for word in line]
    tolerance = max(8.0, 0.6 * float(median(heights)))
    anchors: list[float] = []
    for line in lines:
        for word in line:
            edge = float(word.box[0])
            for index, anchor in enumerate(anchors):
                if abs(edge - anchor) <= tolerance:
                    anchors[index] = (anchor + edge) / 2
                    break
            else:
                anchors.append(edge)
    hits = []
    for anchor in anchors:
        count = 0
        for line in lines:
            if any(abs(word.box[0] - anchor) <= tolerance for word in line):
                count += 1
        hits.append(count)
    columns = [count for count in hits if count >= max(3, int(0.6 * len(lines)))]
    if len(columns) < 2:
        return False
    cell_sizes = [len(line) for line in lines]
    return float(median(cell_sizes)) <= 6


def group_regions(words: list[Word], page_size: tuple[int, int]) -> list[Region]:
    """Return layout blocks in reading order. `page_size` is (width, height)."""

    _width, height = page_size
    lines = group_lines(words)
    regions: list[Region] = []
    blocks = _blocks(lines)
    for index, block in enumerate(blocks):
        block_words = [word for line in block for word in line]
        box = union_box(block_words)
        if _is_table(block):
            kind = "table"
        elif index == 0 and box[1] <= 0.15 * height and len(block_words) <= 10:
            kind = "header"
        elif len(block) == 1 and len(block_words) <= 6:
            kind = "section"
        else:
            kind = "paragraph"
        regions.append(Region(kind=kind, words=block_words, box=box))
    return regions
