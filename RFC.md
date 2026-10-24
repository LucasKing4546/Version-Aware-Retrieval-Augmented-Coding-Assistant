# RFC: Version-aware retrieval-augmented coding assistant

**Course:** LLMs · **Milestone 1:** RFC · **Team:** Snake++ (Secara Lucas, Kevin Petras, Enea Sebastian, Dolha Roberto) · **Date:** 18 October 2026

## 1. Problem

Developers now ask LLMs for code all the time, and two problems keep coming up. First, the model's knowledge stops at training time while libraries keep moving. pandas, NumPy and scikit-learn deprecate functions and rename arguments between releases, so the model suggests code that breaks on the version the user has installed, or calls a function that never existed. Second, the answer comes with no source, so the user has no quick way to check it.

We want to build a RAG pipeline that answers Python programming questions and writes code grounded in retrieved documentation and Stack Overflow answers, and then measure with unit tests whether retrieval actually makes the code more correct.

## 2. State of the art

| Approach | Pros | Cons |
|---|---|---|
| Plain LLM, no retrieval | Simple and fast; good on common problems | Stale knowledge, invented APIs, no sources, weak on less popular libraries |
| Keyword search (docs search, Stack Overflow, BM25) | Matches exact function names and error messages; cheap | Misses paraphrased questions; returns links, so the user still has to read and adapt |
| Dense retrieval + LLM (basic RAG) | Handles paraphrases, cites sources, updated by re-indexing | Embeddings can miss exact identifiers; code and docs are often chunked naively; library versions are ignored |
| CodeRAG-Bench (Wang et al., 2024) | Pairs retrieval corpora (docs, tutorials, Stack Overflow, GitHub) with code generation tasks and shows retrieval can help | Also shows retrievers often miss the useful context and generators struggle with long or noisy context; a benchmark, not a usable system |
| Fine-tuning on code and docs | Knowledge lives inside the model | Expensive, goes stale at the next release, no citations |

For evaluation, HumanEval and MBPP test function synthesis with unit tests. DS-1000 has data-science problems across seven Python libraries, also checked by tests. All of these are public, so their solutions may already sit in a scraped corpus, and we have to check for that contamination.

## 3. Proposal

A RAG coding assistant over a Python corpus, evaluated with unit tests. The corpus is picked to match the evaluation sets:

| Tier | Content |
|---|---|
| Core | Docs, tutorials/gallery examples and Stack Overflow Q&A for pandas, NumPy, Matplotlib and scikit-learn |
| Extension | Python standard library docs (needed for HumanEval/MBPP) and SciPy, which overlaps with NumPy and pandas on purpose so retrieval has realistic near-misses to get wrong |
| Extension | Hugging Face `transformers` and `datasets`: docs, tutorials and Stack Overflow/forum Q&A. The API changes fast, which makes it a good test for version-aware retrieval. DS-1000 does not cover it, so we write a small set of questions with executable checks |

Docs come from each library's repository or official archive at a pinned version, plus one older pandas release so we can test version handling. Stack Overflow posts come from the official CC BY-SA data dump, filtered by tag, accepted answer and minimum score. If time allows we reuse the corpora released with CodeRAG-Bench.

**Pipeline.** Docs are split by heading with code blocks kept whole, and each Stack Overflow question and answer becomes one chunk. Every chunk carries metadata (source, library, version, URL, score). Retrieval combines BM25 with dense embeddings, a cross-encoder reranks the results, and the LLM answers from the retrieved context and cites the chunks it used. Implementation: Python with sentence-transformers, FAISS or Chroma, rank_bm25 and Hugging Face models.

**What we add over basic RAG.** The main addition is version awareness: chunks are filtered by library version so the answer fits what the user runs. Hybrid retrieval plus reranking targets the weakness CodeRAG-Bench found, since BM25 catches exact identifiers that embeddings miss. Every answer cites its sources.

**Evaluation.** pass@1 on DS-1000, restricted to the five libraries we index (NumPy, pandas, Matplotlib, scikit-learn, SciPy; the TensorFlow and PyTorch problems are left out), and on HumanEval/MBPP. We compare six configurations: no retrieval, docs only, Stack Overflow only, dense only, hybrid, and hybrid with reranking. Where gold documents exist we also report recall@k and MRR, and we go through failure cases and check for contamination.

**Expected outcome.** A reproducible pipeline and an ablation study showing which sources and which retrieval components actually change pass@1.

**Risks.** Hugging Face has no ready-made benchmark, so we write our own (30 to 50 questions). Compute is limited; we use a 50k to 200k post subset and an API or a small open model. Contamination may inflate scores. Stack Overflow content requires CC BY-SA attribution, and each documentation set has its own license.
