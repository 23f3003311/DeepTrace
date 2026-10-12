import torch
from captum.attr import IntegratedGradients
from PIL import Image

from deeptrace.explain import aggregate_token_scores
from deeptrace.fields import entities_from_tags, link_score, pair_fields
from deeptrace.geometry import chunk_spans, group_lines, normalize_box, reading_order
from deeptrace.labels import bio_tags, bio_to_spans
from deeptrace.layout import group_regions
from deeptrace.metrics import entity_scores, tags_from_label_ids
from deeptrace.ocr import image_for_baidu, parse_baidu_response, parse_unlimited_ocr
from deeptrace.preprocess import deskew, prepare_page
from deeptrace.pipeline import PagePrediction, evidence_records
from deeptrace.types import Entity, Explanation, FieldLink, Word
from deeptrace.model import rewrite_legacy_layernorm
from deeptrace.app import build_app


def _word(text, box, entity_id=None, entity_label=None):
    return Word(text, box, entity_id=entity_id, entity_label=entity_label)


def test_normalize_box_clamps_and_keeps_area():
    assert normalize_box((0, 0, 50, 20), 100, 40) == [0, 0, 500, 500]
    assert normalize_box((1000, 1000, 1000, 1000), 100, 100) == [999, 999, 1000, 1000]
    swapped = normalize_box((80, 30, 10, 5), 100, 100)
    assert swapped[0] < swapped[2] and swapped[1] < swapped[3]
    assert all(0 <= value <= 1000 for value in swapped)


def test_reading_order_is_line_then_left_and_is_stable():
    words = [
        _word("b", (40, 30, 55, 42)),
        _word("a", (10, 28, 25, 40)),
        _word("c", (12, 70, 30, 82)),
    ]
    ordered = reading_order(words)
    assert [word.text for word in ordered] == ["a", "b", "c"]
    assert [word.text for word in reading_order(ordered)] == ["a", "b", "c"]
    assert len(group_lines(words)) == 2


def test_chunk_spans_cover_every_word_with_overlap():
    spans = chunk_spans(10, lambda start, end: end - start <= 4, stride=2)
    covered = set()
    for start, end in spans:
        assert end - start <= 4
        covered.update(range(start, end))
    assert covered == set(range(10))
    assert spans[0] == (0, 4)
    assert spans[1][0] == 2


def test_chunk_spans_reject_a_word_that_cannot_fit():
    try:
        chunk_spans(3, lambda start, end: False, stride=1)
    except ValueError as error:
        assert "Word 0" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_bio_follows_entity_ids_and_maps_other_to_outside():
    words = [
        _word("name", (0, 0, 1, 1), 1, "question"),
        _word(":", (1, 0, 2, 1), 1, "question"),
        _word("ada", (3, 0, 4, 1), 2, "answer"),
        _word("logo", (0, 2, 1, 3), 3, "other"),
    ]
    assert bio_tags(words) == ["B-QUESTION", "I-QUESTION", "B-ANSWER", "O"]
    assert bio_to_spans(bio_tags(words)) == [(0, 2, "QUESTION"), (2, 3, "ANSWER")]


def test_pair_fields_prefers_the_answer_on_the_same_row():
    question = Entity("QUESTION", 0, 1, (0, 0, 20, 10))
    right = Entity("ANSWER", 1, 2, (30, 0, 50, 10))
    below = Entity("ANSWER", 2, 3, (0, 40, 20, 50))
    links = pair_fields([question, below, right])
    assert links[0].answer is right
    assert link_score(question.box, right.box) > link_score(question.box, below.box)
    assert any(link.question is None for link in links)


def test_pair_fields_keeps_each_value_on_its_own_row():
    upper = Entity("QUESTION", 0, 1, (0, 0, 40, 10))
    lower = Entity("QUESTION", 1, 2, (0, 40, 40, 50))
    lower_answer = Entity("ANSWER", 2, 3, (50, 40, 90, 50))
    links = pair_fields([upper, lower, lower_answer])
    assert links[0].question is lower
    assert links[0].answer is lower_answer
    assert all(link.question is not upper for link in links)


def test_pair_fields_reads_an_untagged_value_beside_the_label():
    words = [
        _word("FAX", (0, 0, 20, 10)),
        _word("NO.", (22, 0, 40, 10)),
        _word("(614)", (50, 0, 80, 10)),
        _word("466-5087", (82, 0, 130, 10)),
        _word("TO:", (0, 30, 20, 40)),
        _word("Ada", (30, 30, 60, 40)),
    ]
    tags = ["B-QUESTION", "I-QUESTION", "O", "O", "B-QUESTION", "B-ANSWER"]
    for word, tag in zip(words, tags):
        word.pred_tag = tag
    links = pair_fields(entities_from_tags(tags, words), words)
    paired = {link.question.text(words): link.answer.text(words) for link in links if link.question}
    assert paired["FAX NO."] == "(614) 466-5087"
    assert paired["TO:"] == "Ada"


def test_pair_fields_leaves_a_long_sentence_unsplit():
    words = [_word(f"w{index}", (index * 12, 0, index * 12 + 10, 10)) for index in range(9)]
    tags = ["B-QUESTION", *["I-QUESTION"] * 6, "O", "O"]
    for word, tag in zip(words, tags):
        word.pred_tag = tag
    links = pair_fields(entities_from_tags(tags, words), words)
    assert links == []


def test_layout_marks_a_grid_as_a_table_and_prose_as_a_paragraph():
    grid = []
    for row in range(3):
        for column in range(2):
            x = 10 + column * 80
            y = 10 + row * 24
            grid.append(_word(f"{row}{column}", (x, y, x + 30, y + 14)))
    kinds = [region.kind for region in group_regions(grid, (300, 200))]
    assert "table" in kinds

    prose = [_word("sentence", (10, 40 + row * 18, 220, 52 + row * 18)) for row in range(4)]
    assert [region.kind for region in group_regions(prose, (300, 220))] == ["paragraph"]


def test_baidu_parser_maps_boxes_back_to_the_original_page():
    payload = {
        "words_result": [
            {
                "words": " Hemoglobin ",
                "location": {"left": 10, "top": 20, "width": 30, "height": 8},
                "probability": {"average": 0.91},
            }
        ]
    }
    words = parse_baidu_response(payload, scale=0.5)
    assert words[0].text == "Hemoglobin"
    assert words[0].box == (20, 40, 80, 56)
    assert abs(words[0].confidence - 0.91) < 1e-6
    line = parse_baidu_response(
        {
            "words_result": [
                {
                    "words": "TO: Ada",
                    "location": {"left": 0, "top": 0, "width": 60, "height": 10},
                }
            ]
        }
    )
    assert [word.text for word in line] == ["TO:", "Ada"]
    assert line[0].box[2] <= line[1].box[0]
    characters = parse_baidu_response(
        {
            "words_result": [
                {
                    "words": "TO Ada",
                    "location": {"left": 0, "top": 0, "width": 50, "height": 10},
                    "chars": [
                        {"char": "T", "location": {"left": 0, "top": 0, "width": 8, "height": 10}},
                        {"char": "O", "location": {"left": 8, "top": 0, "width": 8, "height": 10}},
                        {"char": " ", "location": {"left": 16, "top": 0, "width": 4, "height": 10}},
                        {"char": "A", "location": {"left": 20, "top": 1, "width": 8, "height": 9}},
                        {"char": "d", "location": {"left": 28, "top": 1, "width": 8, "height": 9}},
                        {"char": "a", "location": {"left": 36, "top": 1, "width": 8, "height": 9}},
                    ],
                }
            ]
        }
    )
    assert [word.text for word in characters] == ["TO", "Ada"]
    assert characters[0].box == (0, 0, 16, 10)
    assert characters[1].box == (20, 1, 44, 10)
    try:
        parse_baidu_response({"error_code": 17, "error_msg": "quota"})
    except Exception as error:
        assert "17" in str(error)
    else:
        raise AssertionError("expected OCR error")


def test_unlimited_ocr_blocks_become_word_boxes():
    raw = "\n".join(
        [
            "<|det|>title [0, 0, 999, 99]<|/det|>TO: Ada",
            "<|det|>text [0, 100, 999, 299]<|/det|>DATE:",
            "12/10/98",
            "<|det|>image [0, 0, 100, 100]<|/det|>",
        ]
    )
    words = parse_unlimited_ocr(raw, width=999, height=999)
    assert [word.text for word in words] == ["TO:", "Ada", "DATE:", "12/10/98"]
    assert words[0].box[2] <= words[1].box[0]
    assert words[0].box[3] <= words[2].box[1]
    assert words[2].box[1] < words[3].box[1]
    assert all(0 <= value <= 999 for word in words for value in word.box)


def test_baidu_downscale_reports_the_scale():
    image = Image.new("RGB", (200, 80), "white")
    _payload, scale = image_for_baidu(image, max_side=100, max_bytes=10_000_000)
    assert abs(scale - 0.5) < 1e-6


def test_deskew_undoes_a_known_tilt_and_leaves_a_straight_page():
    canvas = Image.new("L", (360, 240), 255)
    pixels = canvas.load()
    for row in range(8):
        y = 30 + row * 24
        for x in range(40, 300):
            for thickness in range(4):
                pixels[x, y + thickness] = 0
    straight, angle = deskew(canvas.convert("RGB"))
    assert straight.mode == "RGB"
    assert abs(angle) < 0.5

    tilted = canvas.rotate(-6, expand=True, fillcolor=255).convert("RGB")
    fixed, correction = deskew(tilted)
    assert abs(correction - 6) < 1.0
    assert fixed.size[0] > 10


def test_prepare_page_returns_rgb():
    image, angle = prepare_page(Image.new("L", (80, 60), 255))
    assert image.mode == "RGB"
    assert isinstance(angle, float)


def test_entity_f1_is_exact_for_a_perfect_span():
    scores = entity_scores(
        [["B-ANSWER", "I-ANSWER", "O"]],
        [["B-ANSWER", "I-ANSWER", "O"]],
    )
    assert scores["f1"] == 1.0
    missed = entity_scores([["B-ANSWER", "O"]], [["O", "O"]])
    assert missed["f1"] == 0.0
    truth, pred = tags_from_label_ids([1, -100, 0], [1, 4, 0], {0: "O", 1: "B-HEADER", 4: "B-ANSWER"})
    assert truth == ["B-HEADER", "O"]
    assert pred == ["B-HEADER", "O"]


def test_token_scores_sum_into_their_word():
    scores = aggregate_token_scores([0.2, 0.3, 5.0, -1.0], [0, 0, None, 1], n_words=2)
    assert scores == [0.5, -1.0]


def test_integrated_gradients_completeness_on_a_linear_map():
    weight = torch.tensor([[1.5, -2.0, 0.5, 0.25]])

    def forward(inputs):
        return (inputs * weight).sum(dim=-1)

    inputs = torch.tensor([[0.4, -0.2, 0.7, 0.1]], requires_grad=True)
    baseline = torch.zeros_like(inputs)
    attributions, delta = IntegratedGradients(forward).attribute(
        inputs, baselines=baseline, n_steps=64, return_convergence_delta=True
    )
    gap = (forward(inputs) - forward(baseline)).detach()
    assert torch.allclose(attributions.sum(dim=-1), gap, atol=1e-4)
    assert float(delta.abs()) < 1e-3


def test_entities_use_the_union_of_word_boxes():
    words = [_word("a", (0, 0, 4, 2)), _word("b", (6, 1, 9, 4))]
    entities = entities_from_tags(["B-ANSWER", "I-ANSWER"], words)
    assert entities[0].box == (0, 0, 9, 4)


def test_rewrite_layernorm_names(tmp_path):
    from safetensors.torch import load_file, save_file

    save_file(
        {
            "layoutlmv3.embeddings.LayerNorm.gamma": torch.ones(4),
            "layoutlmv3.embeddings.LayerNorm.beta": torch.zeros(4),
            "classifier.weight": torch.ones(2, 2),
        },
        tmp_path / "model.safetensors",
    )
    assert rewrite_legacy_layernorm(tmp_path) is True
    keys = set(load_file(tmp_path / "model.safetensors"))
    assert "layoutlmv3.embeddings.LayerNorm.weight" in keys
    assert "layoutlmv3.embeddings.LayerNorm.bias" in keys
    assert "layoutlmv3.embeddings.LayerNorm.gamma" not in keys
    assert rewrite_legacy_layernorm(tmp_path) is False


def test_evidence_records_keep_the_answer_and_hide_negative_context():
    words = [_word("TO:", (0, 0, 10, 10)), _word("Ada", (12, 0, 30, 10)), _word("note", (0, 20, 20, 30))]
    tags = ["B-QUESTION", "B-ANSWER", "O"]
    for word, tag in zip(words, tags):
        word.pred_tag = tag
    page = PagePrediction(
        image=Image.new("RGB", (40, 40), "white"),
        words=words,
        tags=tags,
        regions=[],
        fields=[FieldLink(answer=Entity("answer", 1, 2, words[1].box), question=Entity("question", 0, 1, words[0].box))],
    )
    page.explanations[0] = Explanation(scores=[0.2, 0.5, -0.1], convergence_delta=0.01, target_word_indexes=[1])
    rows = evidence_records(page, 0)
    assert [row["word"] for row in rows] == ["Ada", "TO:"]
    assert rows[0]["role"] == "answer"
    assert rows[1]["role"] == "field name"


def test_demo_blocks_build_without_a_checkpoint():
    demo = build_app(pipeline=None, ig_steps=2)
    assert demo is not None
    demo.close()
