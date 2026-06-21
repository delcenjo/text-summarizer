"""Abstractive text summarisation with chunking for long inputs.

The pure helpers (sentence splitting, chunking, key points) have no heavy
dependencies and are unit-tested. The model is loaded lazily, so importing this
module — and running the tests — does not require torch/transformers.
"""

from __future__ import annotations

import re
from functools import lru_cache

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ``.!?`` boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_words: int = 700) -> list[str]:
    """Split text into chunks of at most ``max_words`` words.

    Chunks break on sentence boundaries so each one stays coherent, which keeps
    every chunk within the model's input limit when summarising long documents.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if current and count + words > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def key_points(summary: str) -> list[str]:
    """Turn a summary into a short list of bullet points (one per sentence)."""
    return split_sentences(summary)


@lru_cache(maxsize=2)
def _get_pipeline(model_name: str):
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(
            "Summarisation needs the 'ml' extra: pip install 'text-summarizer[ml]'"
        ) from exc
    return pipeline("summarization", model=model_name)


def summarize(
    text: str,
    model_name: str = DEFAULT_MODEL,
    max_length: int = 130,
    min_length: int = 30,
) -> str:
    """Summarise ``text``.

    Long inputs are split into chunks, each chunk is summarised, and if several
    chunks were needed the partial summaries are summarised once more so the
    final result reads as a single coherent summary.
    """
    text = text.strip()
    if not text:
        return ""
    summarizer = _get_pipeline(model_name)

    def run(piece: str) -> str:
        result = summarizer(piece, max_length=max_length, min_length=min_length, truncation=True)
        return result[0]["summary_text"].strip()

    summaries = [run(chunk) for chunk in chunk_text(text)]
    combined = " ".join(summaries)
    if len(summaries) > 1 and len(combined.split()) > max_length:
        combined = run(combined)
    return combined
