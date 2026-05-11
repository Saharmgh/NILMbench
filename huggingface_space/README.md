---
title: NILMbench
emoji: ⚡
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: High-frequency NILM disaggregation demo (UK-DALE 16 kHz V/I).
---

# NILMbench demo

This Space runs the FaustineCNN baseline trained on UK-DALE House 1 against a
single 6-second 16 kHz voltage/current frame from House 2.

* Upload a ``(2, 96000)`` float32 NumPy file, or pick one of the built-in
  example frames.
* The model returns a per-category predicted power vector, post-processed with
  the recall-constrained validation cutoffs from the paper.

The demo intentionally exposes a single frame at a time so the result fits in
one screen. For full benchmark scoring use the ``nilmbench`` CLI on the
companion GitHub repo.

## Files

| File              | Purpose                                                  |
| ----------------- | -------------------------------------------------------- |
| `app.py`          | Gradio entry point                                       |
| `requirements.txt`| Pinned runtime dependencies                              |
| `examples/`       | Built-in V/I frames and their ground-truth labels        |
| `model/`          | FaustineCNN checkpoint + class names + cutoffs           |

## Local development

```bash
pip install -r requirements.txt
python app.py
```
