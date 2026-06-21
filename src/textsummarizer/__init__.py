"""text-summarizer: abstractive summarisation with chunking for long inputs."""

from .summarizer import chunk_text, key_points, split_sentences, summarize

__all__ = ["summarize", "chunk_text", "split_sentences", "key_points"]
__version__ = "0.1.0"
