"""Gradio demo for the text summariser (also runs as a Hugging Face Space)."""

import gradio as gr

from textsummarizer.summarizer import summarize

EXAMPLE = (
    "Hugging Face Spaces let you host machine learning demos directly from a "
    "repository. A Space is a git repo with an app file and a requirements file; "
    "when you push, the platform builds an environment and runs the app, giving "
    "you a public URL. Spaces support Gradio and Streamlit out of the box, and "
    "the free tier runs on CPU, which is enough for small models. This makes it "
    "easy to share a working demo of a model with anyone, without managing "
    "servers yourself."
)


def run(text, max_length, min_length):
    text = (text or "").strip()
    if len(text.split()) < 30:
        return "Please paste a longer text (at least ~30 words)."
    return summarize(text, max_length=int(max_length), min_length=int(min_length))


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(lines=12, label="Paste a long text (article, report, notes)"),
        gr.Slider(60, 250, value=130, step=10, label="Max summary length"),
        gr.Slider(10, 80, value=30, step=5, label="Min summary length"),
    ],
    outputs=gr.Textbox(lines=6, label="Summary"),
    title="📝 Text Summarizer",
    description=(
        "Abstractive summarisation with distilBART. Long inputs are split into "
        "chunks and summarised, so it handles full articles, not just snippets."
    ),
    article="Code: https://github.com/delcenjo/text-summarizer",
    examples=[[EXAMPLE, 130, 30]],
)

if __name__ == "__main__":
    demo.launch()
