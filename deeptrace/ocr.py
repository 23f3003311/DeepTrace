"""OCR backends.

Training reads the gold FUNSD boxes. A new page can be read by Baidu's cloud
OCR or by baidu/Unlimited-OCR. Both are turned into one pixel box per word,
which is the shape the checkpoint was trained on.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image

from deeptrace.types import Word

_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
_DEFAULT_OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"


class OcrError(RuntimeError):
    pass


def _as_box(item: dict) -> tuple[int, int, int, int] | None:
    if "location" in item:
        location = item["location"]
        left = int(location["left"])
        top = int(location["top"])
        return (left, top, left + int(location["width"]), top + int(location["height"]))
    vertexes = item.get("vertexes_location")
    if vertexes:
        xs = [int(point["x"]) for point in vertexes]
        ys = [int(point["y"]) for point in vertexes]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def _confidence(item: dict) -> float:
    probability = item.get("probability", 1.0)
    if isinstance(probability, dict):
        return float(probability.get("average", 1.0))
    if isinstance(probability, (int, float)):
        return float(probability)
    return 1.0


def _scale_box(box: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (
        int(round(x0 / scale)),
        int(round(y0 / scale)),
        int(round(x1 / scale)),
        int(round(y1 / scale)),
    )


def _tokens(text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        start = index
        while index < len(text) and not text[index].isspace():
            index += 1
        if start < index:
            tokens.append((text[start:index], start, index))
    return tokens


def _split_line(text: str, box: tuple[int, int, int, int], confidence: float) -> list[Word]:
    """Give each whitespace-separated token its own slice of the line box.

    The checkpoint was trained on one box per word. Baidu's line box is divided
    in proportion to where each token sits in the line.
    """

    tokens = _tokens(text)
    if not tokens:
        return []
    if len(tokens) == 1:
        return [Word(text=tokens[0][0], box=box, confidence=confidence)]
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    total = max(len(text), 1)
    words: list[Word] = []
    for token, start, end in tokens:
        left = x0 + int(round(width * start / total))
        right = x0 + int(round(width * end / total))
        if right <= left:
            right = min(x1, left + 1)
        words.append(Word(text=token, box=(left, y0, right, y1), confidence=confidence))
    return words


def _split_block(text: str, box: tuple[int, int, int, int], confidence: float = 1.0) -> list[Word]:
    """Split a block into lines, then into the word boxes the checkpoint expects."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    if len(lines) == 1:
        return _split_line(lines[0], box, confidence)
    x0, y0, x1, y1 = box
    height = max(1, y1 - y0)
    words: list[Word] = []
    for index, line in enumerate(lines):
        top = y0 + int(round(height * index / len(lines)))
        bottom = y0 + int(round(height * (index + 1) / len(lines)))
        if bottom <= top:
            bottom = min(y1, top + 1)
        words.extend(_split_line(line, (x0, top, x1, bottom), confidence))
    return words


_DET_LINE = re.compile(
    r"<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]*\])\s*<\|/det\|>(.*)$"
)
_END_OF_SENTENCE = "<\uFF5Cend\u2581of\u2581sentence\uFF5C>"


def _norm_box(raw: str) -> tuple[int, int, int, int] | None:
    numbers = [int(float(value)) for value in re.findall(r"-?\d+(?:\.\d+)?", raw)]
    if len(numbers) < 4:
        return None
    xs0, ys0, xs1, ys1 = numbers[0::4], numbers[1::4], numbers[2::4], numbers[3::4]
    count = min(len(xs0), len(ys0), len(xs1), len(ys1))
    if count == 0:
        return None
    return (min(xs0[:count]), min(ys0[:count]), max(xs1[:count]), max(ys1[:count]))


def _norm_to_pixel(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Unlimited-OCR draws boxes with coordinate / 999 * page size."""

    def axis(value: int, size: int) -> int:
        return int(max(0, min(size, round(value / 999 * size))))

    x0, y0, x1, y1 = box
    return (axis(x0, width), axis(y0, height), axis(x1, width), axis(y1, height))


def parse_unlimited_ocr(raw: str, width: int, height: int) -> list[Word]:
    """Turn Unlimited-OCR `<|det|>type [box]<|/det|>text` into one word per box.

    The model box is a block, normalized on a 0–999 grid. It is mapped onto the
    page, then cut into a pixel box for each word. Image blocks are dropped.
    """

    if width <= 0 or height <= 0:
        raise ValueError("page size must be positive")
    blocks: list[list] = []
    current: list | None = None
    for line in raw.replace(_END_OF_SENTENCE, "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _DET_LINE.match(stripped)
        if match:
            if current is not None:
                blocks.append(current)
            content = match.group(3).strip()
            current = [match.group(1), _norm_box(match.group(2)), [content] if content else []]
            continue
        if current is not None:
            current[2].append(stripped)
    if current is not None:
        blocks.append(current)

    words: list[Word] = []
    for label, box, lines in blocks:
        if label == "image" or box is None:
            continue
        body = "\n".join(line for line in lines if line)
        if not body:
            continue
        words.extend(_split_block(body, _norm_to_pixel(box, width, height)))
    return words


def _words_from_chars(chars: list[dict], confidence: float) -> list[Word] | None:
    groups: list[list[tuple[str, tuple[int, int, int, int]]]] = []
    current: list[tuple[str, tuple[int, int, int, int]]] = []
    for item in chars:
        text = str(item.get("char", ""))
        if text == "" or text.isspace():
            if current:
                groups.append(current)
                current = []
            continue
        box = _as_box(item)
        if box is None:
            return None
        current.append((text, box))
    if current:
        groups.append(current)
    words: list[Word] = []
    for group in groups:
        piece = "".join(text for text, _box in group)
        boxes = [box for _text, box in group]
        union = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        words.append(Word(text=piece, box=union, confidence=confidence))
    return words


def parse_baidu_response(payload: dict, scale: float = 1.0) -> list[Word]:
    """Convert a Baidu OCR JSON body into one word per box on the original page.

    A line is split into words. Character boxes are used when Baidu sent them.
    `scale` is the factor applied before the request (new size / original size).
    Boxes in the response are divided by it so they land on the page we draw.
    """

    if "error_code" in payload:
        code = payload["error_code"]
        message = payload.get("error_msg", "Baidu OCR request failed")
        raise OcrError(f"Baidu OCR error {code}: {message}")
    if scale <= 0:
        raise ValueError("scale must be positive")
    words: list[Word] = []
    for item in payload.get("words_result", []):
        text = str(item.get("words", ""))
        box = _as_box(item)
        if box is None or not text.strip():
            continue
        confidence = _confidence(item)
        from_chars = _words_from_chars(item.get("chars") or [], confidence)
        line_words = from_chars if from_chars else _split_line(text, box, confidence)
        for word in line_words:
            word.box = _scale_box(word.box, scale)
            words.append(word)
    return words


def _jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def image_for_baidu(image: Image.Image, max_side: int = 4096, max_bytes: int = 3_500_000) -> tuple[bytes, float]:
    """Return JPEG bytes and the scale relative to `image`.

    Baidu's accurate OCR rejects very large images. The scale maps response
    boxes back onto `image`.
    """

    page = image.convert("RGB")
    width, height = page.size
    side = max(width, height, 1)
    scale = 1.0 if side <= max_side else max_side / side
    current = page
    if scale < 1:
        current = page.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)

    for _attempt in range(6):
        for quality in (90, 75, 60):
            payload = _jpeg_bytes(current, quality)
            if len(payload) <= max_bytes:
                return payload, scale
        scale *= 0.8
        current = page.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return _jpeg_bytes(current, 50), scale


class BaiduOcr:
    """Baidu accurate OCR. Credentials come from the environment, not from code."""

    def __init__(self, cache_dir: str | Path | None = "outputs/ocr_cache", session: requests.Session | None = None):
        self.api_key = os.environ.get("BAIDU_API_KEY", "").strip()
        self.secret_key = os.environ.get("BAIDU_SECRET_KEY", "").strip()
        self.ocr_url = os.environ.get("BAIDU_OCR_URL", _DEFAULT_OCR_URL).strip()
        self.language = os.environ.get("BAIDU_OCR_LANG", "ENG").strip() or "ENG"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.session = session or requests.Session()
        self._token = ""
        self._token_deadline = 0.0

    def _require_keys(self) -> None:
        if not self.api_key or not self.secret_key:
            raise OcrError(
                "Baidu OCR needs BAIDU_API_KEY and BAIDU_SECRET_KEY in the environment. "
                "Training on FUNSD does not need them."
            )

    def _access_token(self) -> str:
        self._require_keys()
        now = time.time()
        if self._token and now < self._token_deadline:
            return self._token
        response = self.session.post(
            _TOKEN_URL,
            params={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            detail = payload.get("error_description") or payload.get("error") or payload
            raise OcrError(f"Baidu token request failed: {detail}")
        self._token = str(token)
        self._token_deadline = now + max(0, int(payload.get("expires_in", 0)) - 60)
        return self._token

    def _cache_path(self, digest: str) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{digest}.json"

    def read(self, image: Image.Image) -> list[Word]:
        encoded = io.BytesIO()
        image.convert("RGB").save(encoded, format="PNG")
        digest = hashlib.sha1(encoded.getvalue()).hexdigest()
        cache_path = self._cache_path(digest)
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return [
                Word(text=item["text"], box=tuple(item["box"]), confidence=float(item["confidence"]))
                for item in cached["words"]
            ]

        jpeg, scale = image_for_baidu(image)
        body = {
            "image": base64.b64encode(jpeg).decode("ascii"),
            "language_type": self.language,
            "detect_direction": "false",
            "probability": "true",
            "recognize_granularity": "small",
        }
        payload = self._post_ocr(body)
        if payload.get("error_code") in {110, 111}:
            self._token = ""
            payload = self._post_ocr(body)
        words = parse_baidu_response(payload, scale=scale)
        if cache_path is not None:
            cache_path.write_text(
                json.dumps(
                    {
                        "words": [
                            {"text": word.text, "box": list(word.box), "confidence": word.confidence}
                            for word in words
                        ]
                    }
                )
            )
        return words

    def _post_ocr(self, body: dict) -> dict:
        response = self.session.post(
            self.ocr_url,
            params={"access_token": self._access_token()},
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()


class UnlimitedOcr:
    """baidu/Unlimited-OCR, adapted to the checkpoint's word boxes.

    A saved transcript is parsed locally. Live inference follows the model card
    and runs only when CUDA is available, because that code calls `model.cuda()`.
    """

    def __init__(self, transcript: str | None = None, model_name: str = "baidu/Unlimited-OCR"):
        self.transcript = transcript
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def read(self, image: Image.Image) -> list[Word]:
        raw = self.transcript if self.transcript is not None else self._generate(image)
        return parse_unlimited_ocr(raw, image.width, image.height)

    def _generate(self, image: Image.Image) -> str:
        import torch

        if not torch.cuda.is_available():
            raise OcrError(
                "Unlimited-OCR inference from the model card runs with model.cuda() on an NVIDIA GPU. "
                "This computer has no CUDA device, so the 3B weights are not downloaded. "
                "Save the model's raw text and pass it with --ocr-text. "
                "That text is converted into the word boxes the checkpoint expects."
            )
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
            ).eval().cuda()
        with tempfile.TemporaryDirectory() as directory:
            image_path = str(Path(directory) / "page.jpg")
            image.convert("RGB").save(image_path, format="JPEG")
            return self._model.infer(
                self._tokenizer,
                prompt="<image>document parsing.",
                image_file=image_path,
                output_path=directory,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=False,
                eval_mode=True,
            )
