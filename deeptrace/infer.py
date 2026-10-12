"""Read one page with a trained checkpoint.

FUNSD forms use the gold OCR words. Any other file uses Baidu OCR, or an
Unlimited-OCR transcript converted into the same word boxes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from deeptrace.funsd import iter_pages
from deeptrace.model import load_model_and_processor, pick_device
from deeptrace.ocr import BaiduOcr, UnlimitedOcr
from deeptrace.pipeline import DeepTrace
from deeptrace.preprocess import load_upload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepTrace on one page")
    parser.add_argument("--checkpoint", default="outputs/layoutlmv3-funsd")
    parser.add_argument("--image", default="", help="PDF or page image.")
    parser.add_argument("--ocr", choices=("baidu", "unlimited"), default="baidu")
    parser.add_argument(
        "--ocr-text",
        default="",
        help="Raw Unlimited-OCR transcript. Converted into word boxes without loading the 3B model.",
    )
    parser.add_argument("--funsd-dir", default="")
    parser.add_argument("--doc", default="", help="FUNSD file stem, for a gold-OCR preview.")
    parser.add_argument("--field", type=int, default=0)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--skip-explain", action="store_true")
    parser.add_argument("--out", default="outputs/preview.png")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.image) == bool(args.doc):
        raise SystemExit("Pass either --image or both --funsd-dir and --doc")
    device = pick_device(args.device)
    model, processor = load_model_and_processor(args.checkpoint, device=device)
    if args.doc:
        ocr = None
    elif args.ocr_text:
        ocr = UnlimitedOcr(transcript=Path(args.ocr_text).read_text(encoding="utf-8"))
    elif args.ocr == "unlimited":
        ocr = UnlimitedOcr()
    else:
        ocr = BaiduOcr()
    pipeline = DeepTrace(model, processor, device, ocr=ocr)
    if args.doc:
        if not args.funsd_dir:
            raise SystemExit("--doc requires --funsd-dir")
        match = next((page for page in iter_pages(args.funsd_dir, "test") if page.page_id == args.doc), None)
        if match is None:
            match = next((page for page in iter_pages(args.funsd_dir, "train") if page.page_id == args.doc), None)
        if match is None:
            raise SystemExit(f"No FUNSD document named {args.doc}")
        with Image.open(match.source) as image:
            prediction = pipeline.extract(image.convert("RGB"), words=match.words, prepare=False)
    else:
        pages = load_upload(args.image)
        prediction = pipeline.extract(pages[0], words=None, prepare=True)
    rows = prediction.field_rows()
    print(json.dumps(rows, indent=2))

    def explainable(index: int) -> bool:
        link = prediction.fields[index]
        return all(word.pred_tag not in (None, "O") for word in prediction.words[link.answer.start : link.answer.end])

    if not rows:
        drawn = pipeline.draw(prediction)
    elif args.skip_explain:
        drawn = pipeline.draw(prediction, field_index=min(args.field, len(rows) - 1))
    else:
        index = min(args.field, len(rows) - 1)
        if not explainable(index):
            index = next((item for item in range(len(rows)) if explainable(item)), None)
        if index is None:
            drawn = pipeline.draw(prediction)
        else:
            explanation = pipeline.explain(prediction, index, n_steps=args.ig_steps)
            print(f"highlighted field: {index}")
            print(f"convergence_delta: {explanation.convergence_delta:.6f}")
            drawn = pipeline.draw(prediction, field_index=index)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    drawn.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
