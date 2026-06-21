# Text Summarizer

![CI](https://github.com/delcenjo/text-summarizer/actions/workflows/ci.yml/badge.svg)
[![Live demo](https://img.shields.io/badge/Live_demo-Spaces-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/spaces/delcenjo/text-summarizer)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/delcenjo/text-summarizer/blob/main/notebooks/quickstart.ipynb)

Abstractive text summariser built on a distilled BART model, with **chunking so
it handles long documents**, not just short snippets. Available as a Python
package, a CLI, and a live web demo.

> **Try it live:** paste any long text at the
> [interactive demo on Hugging Face Spaces](https://huggingface.co/spaces/delcenjo/text-summarizer).

## How it works

Transformer summarisers have a limited input window (~1k tokens), so long inputs
get truncated and lose information. This project avoids that:

1. **Split into sentences** and group them into chunks that fit the model's window.
2. **Summarise each chunk** with `distilbart-cnn` (abstractive, not just extractive).
3. **Second pass** - if several chunks were needed, the partial summaries are
   summarised once more so the output reads as one coherent summary.

The text-processing logic (sentence splitting, chunking, key points) is pure
Python and unit-tested; the model is loaded lazily, so the test suite runs
without downloading any weights.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"      # core + torch + transformers
```

## Usage

As a CLI:

```bash
text-summarize article.txt                 # summary paragraph
text-summarize article.txt --points        # key points as bullets
cat notes.txt | text-summarize --max-length 80
```

As a library:

```python
from textsummarizer import summarize, key_points

summary = summarize(long_text, max_length=120)
points = key_points(summary)
```

## Project structure

```
src/textsummarizer/
  summarizer.py   sentence splitting, chunking, lazy model, summarise
  cli.py          command-line interface
tests/            unit tests for the pure text-processing helpers
app.py            Gradio demo (also the Hugging Face Space entry point)
```

## Development

```bash
pip install -e ".[dev]"     # just pytest, no heavy ML deps
pytest
```

## Possible improvements

- Swap in a larger model (e.g. `bart-large-cnn`) when GPU is available.
- Add extractive pre-filtering for very long documents.
- Stream the summary token by token in the demo.
