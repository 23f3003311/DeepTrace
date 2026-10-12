# DeepTrace

Read a form, return the field, and mark the words that produced it.

Baidu OCR reads a new page once. Layout grouping puts those boxes into headers, paragraphs, tables, and sections. LayoutLMv3 reads the words and the boxes and labels each word as header, question, or answer. Integrated Gradients then scores which words drove the chosen answer, and those scores are drawn on the page.

Training does not call Baidu. It uses the gold word boxes in FUNSD (149 train forms, 50 test forms). The boxes are word boxes in reading order, which is the same shape Baidu returns. The published 90 F1 figure used one box for a whole entity; this training run does not, because that box is not available from OCR at inference.

## Train

```bash
pip install -r requirements.txt
python -m deeptrace.download
python -m deeptrace.train --funsd-dir data/funsd
```

The run is 1000 steps, learning rate 1e-5, and an effective batch of 16 (batch 2, gradient accumulation 8). It downloads `microsoft/layoutlmv3-base` on the first launch. A CUDA GPU is the practical place to train. On a laptop, use `--batch-size 1 --grad-accum 16` if memory runs out. The checkpoint is written to `outputs/layoutlmv3-funsd`, and `metrics.json` there is the test-set entity F1.

```bash
python -m deeptrace.evaluate --checkpoint outputs/layoutlmv3-funsd
```

## Explain a FUNSD page before you have a Baidu key

```bash
python -m deeptrace.infer \
  --checkpoint outputs/layoutlmv3-funsd \
  --funsd-dir data/funsd \
  --doc 82092117 \
  --out outputs/preview.png
```

Use a real file stem from `data/funsd`. `--doc` reads the gold OCR words, so the API is not called.

## A new page, with Baidu

```bash
export BAIDU_API_KEY=...
export BAIDU_SECRET_KEY=...
python -m deeptrace.infer --checkpoint outputs/layoutlmv3-funsd --image scan.pdf --out outputs/preview.png
python -m deeptrace.app --checkpoint outputs/layoutlmv3-funsd
```

Each page image is cached under `outputs/ocr_cache`, so running it again does not send it back to Baidu. The page is deskewed before OCR. FUNSD pages are not deskewed, because their boxes were drawn on the original scan.
