# DeepTrace pipeline

Read a form, return the field, and mark the words that supported that answer.

This note is the path from a page to that highlight. Training and inference do not start from the same place. Training uses the words already printed in FUNSD. A new page on a GPU is read by Unlimited-OCR, then reshaped into the same word boxes the checkpoint was trained on.

## What a word is

Every later step consumes the same record:

- `text`, one word, such as `TO:`
- `box`, pixels on the page, `(x0, y0, x1, y1)`
- `confidence`

The field model never sees a whole line as one word. If OCR returns a line, it is split before the model runs.

## Training, already done

FUNSD has 149 train forms and 50 test forms. Each word already has a pixel box. Those pages are not deskewed, and Baidu and Unlimited-OCR are not called.

The run was 1000 steps, learning rate `1e-5`, effective batch 16, seed 42. The checkpoint is `outputs/layoutlmv3-funsd`. Best test entity F1 is **0.802** at step 800. The published LayoutLMv3 figure is higher because that paper used one box for a whole entity. This run uses one box per word, which is what OCR can actually give at inference.

`outputs/` is gitignored, so a fresh clone does not contain the checkpoint. Train it, or copy that folder across.

## Inference on a GPU

Unlimited-OCR is the reader for a new page. It needs an NVIDIA GPU. The model card runs it with `model.cuda()`.

### 1. Split the file

Input: a PDF or an image.

Output: one RGB image per page. A PDF is rendered at 200 dpi.

### 2. Prepare the page

Input: that RGB image.

Output: the same page, contrast-stretched only when the scan is nearly flat, then deskewed. A page tilted by about 6 degrees is rotated back. The angle is kept. This step does not run on FUNSD.

### 3. Unlimited-OCR

Input: the prepared page, on CUDA, with the prompt `<image>document parsing.`

Output: text, not word boxes. One block looks like this:

```text
<|det|>text [100, 200, 700, 260]<|/det|>TO: Ada
```

`text` is the block type. `[100, 200, 700, 260]` is one box for the whole block, on a 0–999 grid, not in pixels. The model's own drawing code maps a coordinate with `value / 999 * page size`.

Call inference with `eval_mode=True`. The ordinary save path strips the `<|det|>` markers, and the next step needs them.

### 4. Turn the block into words

Input: that text, plus the page width and height.

Output: one record per word. Image blocks are dropped. On a 999×999 page the block above becomes:

| word | pixel box |
| --- | --- |
| `TO:` | `(100, 200, 357, 260)` |
| `Ada` | `(443, 200, 700, 260)` |

The block width is split by where each word sits in the line. The space between the words is left out. If Unlimited-OCR also returned a box per character, those character boxes are joined into the word instead of sliced. A sliced box is a share of the block, because the model did not measure each word.

Confidence is 1. Unlimited-OCR does not return a per-word score.

### 5. Reading order

Input: those words.

Output: the same words, lines from top to bottom, and each line from left to right. The tags depend on this order.

### 6. Layout

Input: the ordered words and the page size.

Output: blocks labeled header, section, table, or paragraph. A header is a short block in the top 15% of the page. A table needs several aligned columns. The word boxes do not change.

### 7. Encode for LayoutLMv3

Input: the page image and the ordered words.

Each pixel box is scaled to 0–1000 using the real page width and height. The page is cut into windows of at most 512 tokens, overlapping by 32 words. A word may become more than one token. Only the first piece is labeled. The processor's own OCR stays off.

### 8. Tag each word

Input: those windows and `outputs/layoutlmv3-funsd`.

Output: one tag per word.

- `O` for other
- `B-HEADER` / `I-HEADER`
- `B-QUESTION` / `I-QUESTION`
- `B-ANSWER` / `I-ANSWER`

For the line above, the tags the field model is meant to produce are `TO:` as `B-QUESTION` and `Ada` as `B-ANSWER`.

### 9. Pair a field with its value

Input: the tags and the words.

Output: links from a question to an answer.

1. Same row first, so `TO:` takes `Ada` beside it and does not steal the next line.
2. A question of at most six words can claim unmarked words in the gap on its row.
3. What remains is paired with an answer below.

The link for this line is `TO: → Ada`.

### 10. Score the evidence

Input: one answer span.

Integrated Gradients changes only the word embeddings. The boxes and the page image stay fixed. The baseline is the pad-token embedding. Subword pieces of one word are added together. A positive score supported the predicted answer label.

A word tagged `O` is skipped. There is no answer label to explain.

### 11. Draw it

Input: the page, the tags, the chosen field, and the scores.

Output: the page plus two tables.

- Navy outline: field name
- Teal outline: answer
- Gold outline: header
- Gold rectangle: the chosen value
- Teal fill: a word with a positive evidence score

## A measured page

The fax sample `82092117` used the words printed on the FUNSD form, not a live Unlimited-OCR run. The first field that could be explained was:

- field: `TO:`
- value: `George Baroody`
- `George` evidence score: `0.5294`

`FAX NO.` was filled from words the model tagged `O`, so that value was listed and not explained.

## Commands

From the project folder. Use `python3`.

Train, if you do not already have the checkpoint:

```bash
pip install -r requirements.txt
python3 -m deeptrace.download
python3 -m deeptrace.train --funsd-dir data/funsd
```

Preview the fax form. This does not call OCR:

```bash
python3 -m deeptrace.infer \
  --checkpoint outputs/layoutlmv3-funsd \
  --funsd-dir data/funsd \
  --doc 82092117 \
  --out outputs/preview.png
```

Screen on this laptop. **Load sample form** uses the printed words. A file you drop uses Baidu OCR, which needs `BAIDU_API_KEY` and `BAIDU_SECRET_KEY`:

```bash
python3 -m deeptrace.app --checkpoint outputs/layoutlmv3-funsd
```

A saved Unlimited-OCR transcript, already containing the `<|det|>` lines, can be adapted without loading the 3B model:

```bash
python3 -m deeptrace.infer \
  --checkpoint outputs/layoutlmv3-funsd \
  --image scan.png \
  --ocr-text unlimited.txt \
  --out outputs/preview.png
```

On a CUDA machine, `--ocr unlimited` loads `baidu/Unlimited-OCR` and applies the same word split.
