"""Snake++ Gradio demo — owner: Roberto.

Ask a Python question, optionally say which library versions you run ("pandas==1.5.3"), and get code grounded
in retrieved docs / Stack Overflow chunks with numbered citations.
Run: python demo/app.py   (needs data/chunks, VLLM_BASE_URL in .env, pip install -e ".[demo,retrieval]")
"""

from __future__ import annotations

from functools import cache

from snakepp.config import Config, load_config
from snakepp.generation import llm
from snakepp.generation.citations import extract_citations, extract_code
from snakepp.generation.prompts import SYSTEM, build_prompt
from snakepp.retrieval.version_filter import detect_versions, versions_in_text
from snakepp.schemas import Chunk, Task

CONFIGS = {"Full system + version filter": "configs/c5_full_vf.yaml", "Full system": "configs/c5_full.yaml",
           "No retrieval": "configs/c0_none.yaml"}


@cache
def retriever_for(config_path: str):
    cfg = load_config(config_path)
    if not cfg.retrieval.enabled:
        return None
    from snakepp.retrieval import Retriever

    return Retriever.from_disk(cfg)


def parse_versions(text: str) -> dict[str, str]:
    """'pandas==1.5.3, numpy 1.26' -> {'pandas': '1.5.3', 'numpy': '1.26'}."""
    return versions_in_text(text.replace(",", " "))


def source_list(chunks: list[Chunk], cited: list[int]) -> str:
    lines = []
    for n, c in enumerate(chunks, start=1):
        mark = "**cited**" if n in cited else ""
        who = f", by {c.author}" if c.author else ""
        lines.append(f"[{n}] [{c.title}]({c.url}) — {c.library} {c.version or '?'}, {c.source}{who}, "
                     f"{c.license} {mark}")
    return "\n".join(lines) or "_no retrieval_"


def answer(question: str, versions_text: str, config_name: str, model: str | None = None) -> tuple[str, str, str]:
    """(code, sources markdown, raw model output) for one question."""
    cfg: Config = load_config(CONFIGS[config_name])
    if model:
        cfg = cfg.model_copy(update={"generation": cfg.generation.model_copy(update={"model": model})})
    targets = detect_versions(question, parse_versions(versions_text))
    task = Task(task_id="demo", benchmark="version_set", prompt=question, target_versions=targets)
    retriever = retriever_for(CONFIGS[config_name])
    hits = retriever.retrieve(question, targets) if retriever else []
    chunks = [retriever.chunks[h.chunk_id] for h in hits] if retriever else []
    out = llm.generate(build_prompt(task, chunks, cfg.generation), cfg.generation, SYSTEM)
    return extract_code(out), source_list(chunks, extract_citations(out)), out


def build_ui(model: str | None = None):
    import gradio as gr

    with gr.Blocks(title="Snake++") as ui:
        gr.Markdown("# Snake++\nVersion-aware coding assistant for pandas, NumPy, Matplotlib, scikit-learn, SciPy.")
        with gr.Row():
            question = gr.Textbox(label="Question", lines=4, placeholder="How do I add a row to a DataFrame?")
            with gr.Column():
                versions = gr.Textbox(label="Your versions", placeholder="pandas==2.2.2, numpy==1.26.4")
                config = gr.Dropdown(list(CONFIGS), value=next(iter(CONFIGS)), label="System")
                ask = gr.Button("Ask", variant="primary")
        code = gr.Code(label="Answer", language="python")
        sources = gr.Markdown(label="Sources")
        with gr.Accordion("Raw model output", open=False):
            raw = gr.Textbox(lines=8, show_label=False)
        ask.click(lambda q, v, c: answer(q, v, c, model), [question, versions, config], [code, sources, raw])
    return ui


if __name__ == "__main__":
    import typer

    def main(model: str | None = None, port: int = 7860) -> None:
        build_ui(model).launch(server_port=port)

    typer.run(main)
