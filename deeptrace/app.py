"""Local screen: drop a page or a PDF and see the evidence for each field.

  python -m deeptrace.app --checkpoint outputs/layoutlmv3-funsd
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
from PIL import Image

from deeptrace.funsd import iter_pages
from deeptrace.model import load_model_and_processor, pick_device
from deeptrace.ocr import BaiduOcr, OcrError, UnlimitedOcr
from deeptrace.pipeline import DeepTrace, PagePrediction, evidence_records
from deeptrace.preprocess import load_upload

_CSS = """
body, .gradio-container {
  background: #f4f1eb !important;
  color: #0b1f3a !important;
}
"""
_HEAD = """
<script>
if (!window.location.search.includes("__theme=light")) {
  const url = new URL(window.location.href);
  url.searchParams.set("__theme", "light");
  window.location.replace(url.href);
}
document.documentElement.style.colorScheme = "light";
</script>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepTrace demo")
    parser.add_argument("--checkpoint", default="outputs/layoutlmv3-funsd")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--ocr", choices=("baidu", "unlimited"), default="baidu")
    parser.add_argument("--ocr-text", default="", help="Raw Unlimited-OCR transcript used for every upload.")
    parser.add_argument("--funsd-dir", default="data/funsd")
    parser.add_argument("--sample-doc", default="82092117")
    return parser.parse_args()


def _file_paths(files) -> list[str]:
    if not files:
        return []
    items = files if isinstance(files, (list, tuple)) else [files]
    paths: list[str] = []
    for item in items:
        if isinstance(item, str):
            path = item
        elif isinstance(item, dict):
            path = item.get("path") or item.get("name") or ""
        else:
            path = getattr(item, "path", None) or getattr(item, "name", "") or ""
        if path:
            paths.append(str(path))
    return paths


def _explainable(page: PagePrediction, index: int) -> bool:
    link = page.fields[index]
    return all(word.pred_tag not in (None, "O") for word in page.words[link.answer.start : link.answer.end])


def _field_choice(page_number: int, row: dict[str, str]) -> tuple[str, str]:
    name = row["field"] or "(value only)"
    label = f"Page {page_number} · {name} = {row['value']}"
    return label, f"p{page_number}|{row['index']}"


def _parse_choice(choice: str) -> tuple[int, int]:
    page_token, field_token, *_rest = choice.split("|")
    return int(page_token[1:]) - 1, int(field_token)


def _field_table(pages: list[PagePrediction]) -> list[list[str]]:
    rows: list[list[str]] = []
    for page_number, page in enumerate(pages, start=1):
        for row in page.field_rows():
            rows.append([str(page_number), row["field"], row["value"]])
    return rows


def _evidence_table(page: PagePrediction, field_index: int) -> list[list[str]]:
    return [
        [row["word"], row["tag"], row["role"], row["evidence"], row["box"]]
        for row in evidence_records(page, field_index)
    ]


def _trace(pages: list[PagePrediction], source: str) -> str:
    lines = [
        f"Word boxes come from {source}.",
        "Every word is tagged header, question, answer, or other. Navy outline is a field name, teal is an answer, gold is a header.",
        "A field is that question paired with the value on the same row, or on the row below.",
        "Evidence is Integrated Gradients on the word embeddings. The page image and the boxes stay fixed. A positive score supported the predicted answer, and that word is filled in teal.",
        "",
    ]
    for number, page in enumerate(pages, start=1):
        counts: dict[str, int] = {}
        for region in page.regions:
            counts[region.kind] = counts.get(region.kind, 0) + 1
        layout = ", ".join(f"{count} {kind}" for kind, count in counts.items()) or "no layout blocks"
        lines.append(
            f"Page {number}: {len(page.words)} words, {layout}, {len(page.fields)} fields, deskew {page.deskew_angle:.2f} degrees."
        )
    return "\n".join(lines)


def _blank(message: str):
    return (
        None,
        [],
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        [],
        "",
        message,
        [],
    )


def build_app(
    pipeline: DeepTrace | None,
    ig_steps: int,
    funsd_dir: str = "data/funsd",
    sample_doc: str = "82092117",
    ocr_label: str = "Baidu OCR",
) -> gr.Blocks:
    def _selected(pages: list[PagePrediction]) -> tuple[list[tuple[str, str]], str | None]:
        choices: list[tuple[str, str]] = []
        selected: str | None = None
        for page_number, page in enumerate(pages, start=1):
            for row in page.field_rows():
                label, value = _field_choice(page_number, row)
                choices.append((label, value))
                if selected is None and _explainable(page, int(row["index"])):
                    selected = value
        if selected is None and choices:
            selected = choices[0][1]
        return choices, selected

    def _view(pages: list[PagePrediction], choice: str | None) -> tuple[Image.Image | None, list[list[str]], str]:
        if not pages or not choice:
            image = pipeline.draw(pages[0], show_tags=True) if pages else None
            return image, [], "No field was found. The outlines are still the model's tags."
        page_index, field_index = _parse_choice(choice)
        page = pages[page_index]
        question = page.fields[field_index].question
        name = question.text(page.words) if question is not None else "(value only)"
        value = page.fields[field_index].answer.text(page.words)
        if not _explainable(page, field_index):
            image = pipeline.draw(page, field_index=field_index, show_tags=True)
            note = (
                f"Page {page_index + 1}. {name} = {value}. "
                "The answer words are tagged other, so Integrated Gradients has no answer label to explain. "
                "The gold rectangle is still where that value sits."
            )
            return image, _evidence_table(page, field_index), note
        explanation = page.explanations.get(field_index)
        image = pipeline.draw(page, field_index=field_index, show_tags=True)
        if explanation is None:
            note = f"Page {page_index + 1}. {name} = {value}. Scoring which words supported this answer."
        else:
            note = (
                f"Page {page_index + 1}. {name} = {value}. "
                f"Convergence delta {explanation.convergence_delta:.4f}. "
                "Teal fill is positive evidence. The table lists the field name, the answer, and every other word that supported it."
            )
        return image, _evidence_table(page, field_index), note

    def _publish(pages: list[PagePrediction], source: str, progress=gr.Progress()) -> Iterator[tuple]:
        choices, selected = _selected(pages)
        page_choices = [f"Page {number}" for number in range(1, len(pages) + 1)]
        page_value = "Page 1"
        if selected:
            page_value = f"Page {_parse_choice(selected)[0] + 1}"
        image, evidence, status = _view(pages, selected)
        yield (
            image,
            _field_table(pages),
            gr.update(choices=choices, value=selected),
            gr.update(choices=page_choices, value=page_value),
            evidence,
            _trace(pages, source),
            status,
            pages,
        )
        if not selected:
            return
        page_index, field_index = _parse_choice(selected)
        page = pages[page_index]
        if not _explainable(page, field_index) or field_index in page.explanations:
            return
        progress(0.85, desc="Scoring evidence")
        try:
            pipeline.explain(page, field_index, n_steps=ig_steps)
        except (ValueError, RuntimeError, IndexError) as error:
            yield (
                pipeline.draw(page, field_index=field_index, show_tags=True),
                _field_table(pages),
                gr.update(choices=choices, value=selected),
                gr.update(choices=page_choices, value=page_value),
                _evidence_table(page, field_index),
                _trace(pages, source),
                str(error),
                pages,
            )
            return
        image, evidence, status = _view(pages, selected)
        yield (
            image,
            _field_table(pages),
            gr.update(choices=choices, value=selected),
            gr.update(choices=page_choices, value=page_value),
            evidence,
            _trace(pages, source),
            status,
            pages,
        )

    def read_files(files, progress=gr.Progress()):
        if pipeline is None:
            yield _blank("The screen was built without a model.")
            return
        paths = _file_paths(files)
        if not paths:
            yield _blank("Drop a PDF or one or more page images.")
            return
        pages: list[PagePrediction] = []
        try:
            images: list[Image.Image] = []
            for path in paths:
                images.extend(load_upload(path))
            for index, image in enumerate(images):
                progress(index / max(len(images), 1), desc=f"Reading page {index + 1} of {len(images)}")
                pages.append(pipeline.extract(image, words=None, prepare=True))
        except (OcrError, ValueError, RuntimeError, OSError) as error:
            yield _blank(str(error))
            return
        yield from _publish(pages, ocr_label, progress)

    def load_sample(progress=gr.Progress()):
        if pipeline is None:
            yield _blank("The screen was built without a model.")
            return
        progress(0.2, desc="Opening the sample form")
        try:
            match = next((page for page in iter_pages(funsd_dir, "test") if page.page_id == sample_doc), None)
            if match is None:
                match = next((page for page in iter_pages(funsd_dir, "train") if page.page_id == sample_doc), None)
            if match is None:
                yield _blank(f"No sample form named {sample_doc}.")
                return
            with Image.open(match.source) as image:
                prediction = pipeline.extract(image.convert("RGB"), words=list(match.words), prepare=False)
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
            yield _blank(str(error))
            return
        yield from _publish([prediction], "the words printed on the sample form", progress)

    def show_field(choice: str | None, pages: list[PagePrediction] | None, progress=gr.Progress()):
        if pipeline is None or not pages or not choice:
            return None, gr.update(), [], "Read a document first."
        page_index, field_index = _parse_choice(choice)
        page = pages[page_index]
        if _explainable(page, field_index) and field_index not in page.explanations:
            progress(0.5, desc="Scoring evidence")
            try:
                pipeline.explain(page, field_index, n_steps=ig_steps)
            except (ValueError, RuntimeError, IndexError) as error:
                return pipeline.draw(page, field_index=field_index, show_tags=True), gr.update(), [], str(error)
        image, evidence, status = _view(pages, choice)
        return image, gr.update(value=f"Page {page_index + 1}"), evidence, status

    def show_page(page_label: str | None, choice: str | None, pages: list[PagePrediction] | None):
        if not pages or not page_label:
            return None, "Read a document first."
        page_index = int(str(page_label).replace("Page ", "")) - 1
        page = pages[page_index]
        field_index = None
        if choice:
            chosen_page, chosen_field = _parse_choice(choice)
            if chosen_page == page_index:
                field_index = chosen_field
        image = pipeline.draw(page, field_index=field_index, show_tags=True)
        return image, f"Page {page_index + 1} of {len(pages)}. Outlines are the model's tags for every word."

    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="stone")
    with gr.Blocks(title="DeepTrace") as demo:
        gr.Markdown(
            "Drop a PDF or page images. DeepTrace reads every page, returns each field, "
            "and shows the words that supported that answer."
        )
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(
                    label="PDF or page images",
                    file_count="multiple",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
                )
                read_button = gr.Button("Read pages", variant="primary")
                sample_button = gr.Button("Load sample form")
                page_choice = gr.Dropdown(label="Page", choices=[])
                field_choice = gr.Dropdown(label="Field", choices=[])
                evidence_button = gr.Button("Show evidence for this field")
                status = gr.Textbox(label="What you are looking at", lines=4)
            preview = gr.Image(label="Page, with tags and evidence", type="pil", scale=2)
        fields = gr.Dataframe(headers=["page", "field", "value"], label="Fields on every page", interactive=False)
        evidence = gr.Dataframe(
            headers=["word", "tag", "role", "evidence", "box"],
            label="Evidence for the selected field",
            interactive=False,
        )
        trace = gr.Textbox(label="How the answer was produced", lines=8)
        state = gr.State([])
        outputs = [preview, fields, field_choice, page_choice, evidence, trace, status, state]
        read_button.click(read_files, inputs=upload, outputs=outputs)
        sample_button.click(load_sample, inputs=None, outputs=outputs)
        evidence_button.click(show_field, inputs=[field_choice, state], outputs=[preview, page_choice, evidence, status])
        page_choice.input(show_page, inputs=[page_choice, field_choice, state], outputs=[preview, status])
    demo.theme = theme
    return demo


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, processor = load_model_and_processor(args.checkpoint, device=device)
    if args.ocr_text:
        ocr = UnlimitedOcr(transcript=Path(args.ocr_text).read_text(encoding="utf-8"))
        ocr_label = "Unlimited-OCR text, split into one box per word"
    elif args.ocr == "unlimited":
        ocr = UnlimitedOcr()
        ocr_label = "Unlimited-OCR, split into one box per word"
    else:
        ocr = BaiduOcr()
        ocr_label = "Baidu OCR, split into one box per word"
    pipeline = DeepTrace(model, processor, device, ocr=ocr)
    app = build_app(pipeline, args.ig_steps, args.funsd_dir, args.sample_doc, ocr_label)
    app.launch(
        server_port=args.server_port,
        theme=getattr(app, "theme", None),
        css=_CSS,
        head=_HEAD,
    )


if __name__ == "__main__":
    main()
