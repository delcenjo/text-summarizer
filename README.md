# Text Summarizer

![CI](https://github.com/delcenjo/text-summarizer/actions/workflows/ci.yml/badge.svg)
[![Live demo](https://img.shields.io/badge/Live_demo-Spaces-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/spaces/delcenjo/text-summarizer)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/delcenjo/text-summarizer/blob/main/notebooks/quickstart.ipynb)

Abstractive summarisation that doesn't fall apart on long text. Paste a whole
article or report and get back a short, readable summary instead of a truncated
one. It runs as a small Python package, a CLI, and a hosted web demo.

The quickest way to see it is the
[demo on Hugging Face Spaces](https://huggingface.co/spaces/delcenjo/text-summarizer):
drop in some text, tweak the length sliders, read the result. No setup needed.

## Using it from Python

```python
from textsummarizer import summarize, key_points

summary = summarize(long_text, max_length=120)
points = key_points(summary)   # the summary split into one bullet per sentence
```

`summarize` also takes `min_length` and a `model_name` if you want to point it at
a different Hugging Face summarisation model. The default is
`sshleifer/distilbart-cnn-12-6`.

## The long-document problem

Transformer summarisers only look at a limited input window (on the order of a
thousand tokens). Feed them a long article and everything past the window is
quietly dropped, so the summary describes the first few paragraphs and ignores
the rest.

The workaround here is straightforward:

- Break the text into sentences, then group those sentences into chunks that
  each stay under a word budget (`chunk_text`, 700 words by default). Splitting
  on sentence boundaries keeps every chunk readable on its own and avoids
  cutting a thought in half.
- Summarise each chunk with distilBART.
- If the input needed more than one chunk and the joined-up result is still
  long, run that combined text through the model one more time so the final
  output reads as a single summary rather than a list of partial ones.

All of the text wrangling (sentence splitting, chunking, turning a summary into
key points) is plain Python with no model dependency, and it has unit tests. The
model itself is loaded lazily and cached, so importing the package, or running
the tests, never downloads any weights.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml]"      # the package plus torch and transformers
```

The first call to `summarize` will pull down the model weights; after that they
are cached locally.

## Command line

Installing the package adds a `text-summarize` command. It reads a file or, if
you don't pass one, standard input.

```bash
text-summarize article.txt                 # print the summary paragraph
text-summarize article.txt --points        # print it as bullet points instead
cat notes.txt | text-summarize --max-length 80
```

`--min-length` is available too, mirroring the library arguments.

## Layout

```
src/textsummarizer/
  summarizer.py   sentence splitting, chunking, lazy model loading, summarise
  cli.py          the text-summarize command
tests/            tests for the pure text helpers (no ML deps needed)
app.py            Gradio app, also the entry point for the Hugging Face Space
notebooks/        a quickstart notebook (Colab-ready)
```

## Working on it

The dev extra skips the heavy ML packages, so you can run the tests fast:

```bash
pip install -e ".[dev]"
pytest
```

If you have a GPU to spare, swapping the default model for a larger one such as
`bart-large-cnn` (via `model_name`) is the obvious next step for better output.
