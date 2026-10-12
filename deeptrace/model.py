"""Token-classification model used to extract header, question, and answer spans."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from deeptrace.encode import EncodedWindow, load_processor
from deeptrace.labels import ID2LABEL, LABEL2ID


def rewrite_legacy_layernorm(checkpoint_dir: str | Path) -> bool:
    """Rename LayerNorm.gamma/beta to weight/bias inside a saved checkpoint.

    Transformers 5 writes the old names when it saves LayoutLMv3, then the
    Trainer reloads that file with a plain state dict and skips every LayerNorm
    weight. Rewriting the file makes the reload exact.
    """

    path = Path(checkpoint_dir) / "model.safetensors"
    if not path.is_file():
        return False
    state = load_file(path)
    renamed: dict[str, torch.Tensor] = {}
    changed = False
    for key, value in state.items():
        if key.endswith("LayerNorm.gamma"):
            key = key[: -len("gamma")] + "weight"
            changed = True
        elif key.endswith("LayerNorm.beta"):
            key = key[: -len("beta")] + "bias"
            changed = True
        renamed[key] = value
    if changed:
        save_file(renamed, path)
    return changed


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name_or_path: str, device: torch.device | None = None) -> LayoutLMv3ForTokenClassification:
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        name_or_path,
        num_labels=len(LABEL2ID),
        id2label={str(index): label for index, label in ID2LABEL.items()},
        label2id=LABEL2ID,
    )
    if device is not None:
        model.to(device)
    model.eval()
    return model


def load_model_and_processor(
    checkpoint: str,
    device: torch.device | None = None,
) -> tuple[LayoutLMv3ForTokenClassification, LayoutLMv3Processor]:
    path = Path(checkpoint)
    processor_source = str(path if (path / "tokenizer_config.json").exists() else checkpoint)
    processor = load_processor(processor_source)
    model = load_model(checkpoint, device=device)
    return model, processor


def predict_word_tags(
    model: LayoutLMv3ForTokenClassification,
    windows: list[EncodedWindow],
    n_words: int,
    device: torch.device,
) -> list[str]:
    """Average first-subword logits across overlapping windows, then take argmax.

    Only the first subword of each word was given a label in training, so only
    that position votes.
    """

    if n_words == 0:
        return []
    sums: list[torch.Tensor | None] = [None] * n_words
    counts = [0] * n_words
    model.eval()
    with torch.no_grad():
        for window in windows:
            batch = {key: value.unsqueeze(0).to(device) for key, value in window.tensors.items() if key != "labels"}
            logits = model(**batch).logits[0].detach().cpu()
            seen: set[int] = set()
            for token_index, word_id in enumerate(window.word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                global_index = window.word_start + word_id
                piece = logits[token_index]
                sums[global_index] = piece if sums[global_index] is None else sums[global_index] + piece
                counts[global_index] += 1
    missing = [index for index, count in enumerate(counts) if count == 0]
    if missing:
        raise RuntimeError(f"No prediction for word indexes {missing[:8]}")
    tags: list[str] = []
    for index in range(n_words):
        averaged = sums[index] / counts[index]
        tags.append(ID2LABEL[int(averaged.argmax())])
    return tags
