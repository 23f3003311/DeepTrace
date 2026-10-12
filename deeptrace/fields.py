"""Pair a question span with the answer that sits beside it on the form."""

from __future__ import annotations

from deeptrace.geometry import union_box
from deeptrace.labels import bio_to_spans
from deeptrace.types import Entity, FieldLink, Word


def entities_from_tags(tags: list[str], words: list[Word]) -> list[Entity]:
    if len(tags) != len(words):
        raise ValueError("Tags and words must be the same length")
    entities: list[Entity] = []
    for start, end, label in bio_to_spans(tags):
        entities.append(Entity(label=label, start=start, end=end, box=union_box(words[start:end])))
    return entities


def _same_row(question_box: tuple[int, int, int, int], answer_box: tuple[int, int, int, int]) -> bool:
    _, qy0, _, qy1 = question_box
    _, ay0, _, ay1 = answer_box
    question_height = max(1, qy1 - qy0)
    answer_height = max(1, ay1 - ay0)
    question_center = (qy0 + qy1) / 2
    answer_center = (ay0 + ay1) / 2
    return abs(question_center - answer_center) <= 0.6 * max(question_height, answer_height)


def link_score(question_box: tuple[int, int, int, int], answer_box: tuple[int, int, int, int]) -> float:
    """Prefer an answer on the same row to the right, otherwise the next answer below."""

    qx0, qy0, qx1, qy1 = question_box
    ax0, ay0, _, _ = answer_box
    if _same_row(question_box, answer_box) and ax0 >= qx0 - 5:
        return 1000 - (ax0 - qx1)
    if ay0 >= qy0 - 5:
        return 100 - (ay0 - qy1) - 0.01 * abs(ax0 - qx0)
    return -1.0


def _row_value(question: Entity, words: list[Word], blocked: set[int]) -> Entity | None:
    """Words on the question's row that the tagger left unmarked."""

    if question.end - question.start > 6:
        return None
    chosen: list[int] = []
    for index in range(question.end, len(words)):
        if index in blocked:
            break
        word = words[index]
        if not _same_row(question.box, word.box) or word.box[0] < question.box[0] - 5:
            break
        chosen.append(index)
    if not chosen:
        return None
    start, end = chosen[0], chosen[-1] + 1
    return Entity(label="ANSWER", start=start, end=end, box=union_box(words[start:end]))


def pair_fields(entities: list[Entity], words: list[Word] | None = None) -> list[FieldLink]:
    """Attach each question to the value on its own row before looking downward.

    A question processed first must not take the answer sitting beside a lower
    question. Unmarked words directly to the right of a question still count
    as its value.
    """

    questions = [entity for entity in entities if entity.label == "QUESTION"]
    answers = [entity for entity in entities if entity.label == "ANSWER"]
    used: set[int] = set()
    assigned: dict[int, Entity] = {}

    same_row_pairs: list[tuple[float, int, int]] = []
    for question_index, question in enumerate(questions):
        for answer_index, answer in enumerate(answers):
            answer_x = answer.box[0]
            if _same_row(question.box, answer.box) and answer_x >= question.box[0] - 5:
                same_row_pairs.append((link_score(question.box, answer.box), question_index, answer_index))
    for _, question_index, answer_index in sorted(same_row_pairs, reverse=True):
        if question_index in assigned or answer_index in used:
            continue
        assigned[question_index] = answers[answer_index]
        used.add(answer_index)

    if words is not None:
        blocked = {index for entity in entities for index in range(entity.start, entity.end)}
        for question_index, question in enumerate(questions):
            if question_index in assigned:
                continue
            value = _row_value(question, words, blocked)
            if value is not None:
                assigned[question_index] = value
                blocked.update(range(value.start, value.end))

    for question_index, question in enumerate(questions):
        if question_index in assigned:
            continue
        best_index: int | None = None
        best_score = 0.0
        for answer_index, answer in enumerate(answers):
            if answer_index in used:
                continue
            if _same_row(question.box, answer.box) and answer.box[0] >= question.box[0] - 5:
                continue
            score = link_score(question.box, answer.box)
            if score <= 0:
                continue
            if score > best_score:
                best_score = score
                best_index = answer_index
        if best_index is not None:
            assigned[question_index] = answers[best_index]
            used.add(best_index)

    links = [
        FieldLink(question=question, answer=assigned[question_index])
        for question_index, question in enumerate(questions)
        if question_index in assigned
    ]
    for answer_index, answer in enumerate(answers):
        if answer_index not in used:
            links.append(FieldLink(question=None, answer=answer))
    return links
