import pytest

from textsummarizer.summarizer import chunk_text, key_points, split_sentences


def test_split_sentences():
    assert split_sentences("Hello world. How are you? Fine!") == [
        "Hello world.",
        "How are you?",
        "Fine!",
    ]


def test_split_sentences_empty():
    assert split_sentences("   ") == []


def test_chunk_text_single_chunk():
    assert chunk_text("Short text here.", max_words=100) == ["Short text here."]


def test_chunk_text_respects_word_limit():
    text = " ".join(f"Sentence number {i}." for i in range(60))  # 3 words each
    chunks = chunk_text(text, max_words=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 20 for c in chunks)
    # No sentence is lost when chunking.
    assert sum(len(c.split()) for c in chunks) == len(text.split())


def test_chunk_text_invalid_max_words():
    with pytest.raises(ValueError):
        chunk_text("anything", max_words=0)


def test_key_points():
    assert key_points("First point. Second point.") == ["First point.", "Second point."]
