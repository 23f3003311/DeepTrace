"""Fine-tune LayoutLMv3-base on FUNSD.

The recipe matches the base-model FUNSD run: 1e-5, 1000 steps, effective
batch size 16, linear schedule. Boxes are the word boxes, not the entity
box, so the same encoder can read Baidu OCR output after training.

  python -m deeptrace.download
  python -m deeptrace.train --funsd-dir data/funsd
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Trainer, TrainerCallback, TrainingArguments

from deeptrace.encode import encode_page, load_processor
from deeptrace.funsd import iter_pages
from deeptrace.labels import ID2LABEL, LABELS
from deeptrace.metrics import entity_scores, tags_from_label_ids
from deeptrace.model import load_model, pick_device, rewrite_legacy_layernorm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class _EncodedPages(torch.utils.data.Dataset):
    def __init__(self, items: list[dict[str, torch.Tensor]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def encode_split(processor, root: str, split: str) -> _EncodedPages:
    items: list[dict[str, torch.Tensor]] = []
    pages = 0
    for page in iter_pages(root, split):
        pages += 1
        with Image.open(page.source) as image:
            _ordered, windows = encode_page(processor, image.convert("RGB"), page.words)
        items.extend(window.tensors for window in windows)
    if pages == 0 or not items:
        raise RuntimeError(f"No {split} windows were encoded")
    print(f"{split}: {pages} pages, {len(items)} windows")
    return _EncodedPages(items)


class _KeepLayerNorm(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        rewrite_legacy_layernorm(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        return control


class _Trainer(Trainer):
    def _load_best_model(self) -> None:
        if self.state.best_model_checkpoint:
            rewrite_legacy_layernorm(self.state.best_model_checkpoint)
        super()._load_best_model()


def make_compute_metrics():
    def compute_metrics(eval_prediction) -> dict[str, float]:
        logits, labels = eval_prediction
        predicted = np.argmax(logits, axis=-1)
        truths: list[list[str]] = []
        preds: list[list[str]] = []
        for pred_row, label_row in zip(predicted, labels):
            truth, pred = tags_from_label_ids(label_row.tolist(), pred_row.tolist(), ID2LABEL)
            truths.append(truth)
            preds.append(pred)
        return entity_scores(truths, preds)

    return compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 on FUNSD for DeepTrace")
    parser.add_argument("--funsd-dir", default="data/funsd")
    parser.add_argument("--output-dir", default="outputs/layoutlmv3-funsd")
    parser.add_argument("--model-name", default="microsoft/layoutlmv3-base")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu. Default: pick automatically.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")
    processor = load_processor(args.model_name)
    processor.save_pretrained(output_dir)
    model = load_model(args.model_name, device=None)
    train_data = encode_split(processor, args.funsd_dir, "train")
    eval_data = encode_split(processor, args.funsd_dir, "test")
    use_fp16 = device.type == "cuda"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=20,
        seed=args.seed,
        fp16=use_fp16,
        dataloader_pin_memory=device.type == "cuda",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = _Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=make_compute_metrics(),
        callbacks=[_KeepLayerNorm()],
    )
    (output_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "model": args.model_name,
                "dataset": "FUNSD",
                "labels": LABELS,
                "boxes": "word-level, normalized 0-1000",
                "order": "reading order",
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "effective_batch_size": args.batch_size * args.grad_accum,
                "seed": args.seed,
                "ocr_during_training": "FUNSD gold words, not Baidu",
            },
            indent=2,
        )
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    rewrite_legacy_layernorm(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({key: value for key, value in metrics.items() if not key.startswith("eval_runtime")}, indent=2))


if __name__ == "__main__":
    main()
