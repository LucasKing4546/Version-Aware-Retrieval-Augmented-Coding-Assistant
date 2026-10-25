# Related work — outline

Owner: Roberto · Target length in the report: ~1–1.5 pages · Status: outline (M0)

Every subsection ends with one sentence on **how Snake++ differs**. Papers marked ✔ have their arXiv id checked (2026-10-25).
Check the others before adding them to the bibliography.

## 1. Retrieval-augmented generation
- Lewis et al., 2020. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS. The original RAG formulation.
- Karpukhin et al., 2020. *Dense Passage Retrieval for Open-Domain QA*. EMNLP. Dense bi-encoder retrieval.
- Liu et al., 2024. *Lost in the Middle: How Language Models Use Long Contexts*. TACL. Argues for a small top-k and reranking.
- Gao et al., 2023. *Enabling Large Language Models to Generate Text with Citations* (ALCE). EMNLP. Citation quality as a metric.
- **Snake++:** RAG for code, where correctness is checked by unit tests instead of by text overlap.

## 2. Retrieval for code generation
- Zhou et al., 2023. *DocPrompting: Generating Code by Retrieving the Docs*. ICLR. The closest prior work: it retrieves docs for code generation.
- Parvez et al., 2021. *Retrieval Augmented Code Generation and Summarization* (REDCODER). Findings of EMNLP.
- Zhang et al., 2023. *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation*. EMNLP.
- Wang et al., 2024. *CodeRAG-Bench: Can Retrieval Augment Code Generation?* arXiv 2406.14497. Shows that retrievers miss useful context and that noisy context hurts generation. This motivates our hybrid retrieval with reranking.
- **Snake++:** we combine docs with Stack Overflow, tag each chunk with its version, and ablate each source and each retrieval stage separately.

## 3. Hybrid retrieval, fusion and reranking
- Robertson & Zaragoza, 2009. *The Probabilistic Relevance Framework: BM25 and Beyond*. FnTIR.
- Cormack, Clarke & Büttcher, 2009. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR.
- Nogueira & Cho, 2019. *Passage Re-ranking with BERT*. arXiv 1901.04085. Cross-encoder reranking.
- Xiao et al., 2023. *C-Pack: Packaged Resources to Advance General Chinese Embedding*. arXiv 2309.07597. The BGE embedding and reranker models we use.
- **Snake++:** an identifier-aware BM25 tokenizer, because exact API names (`DataFrame.append`) matter in code questions.

## 4. Library evolution and version-aware code generation (core section)
- ✔ Wang et al., 2025. *LLMs Meet Library Evolution: Evaluating Deprecated API Usage in LLM-based Code Completion*. ICSE. arXiv 2406.09834.
  Models often prefer deprecated APIs; the paper proposes the ReplaceAPI and InsertPrompt fixes.
- ✔ Wu et al., 2024. *VersiCode: Towards Version-controllable Code Generation*. arXiv 2406.07411.
  Defines version-specific code completion and version-aware code migration, plus the CDC@1 metric.
- ✔ Islah et al., 2024. *GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models*. arXiv 2411.05830.
  116 version-conditioned problems with unit tests. GitChameleon 2.0 (arXiv 2507.12367) expands this to 328 problems and also evaluates RAG systems.
- ✔ Kuhar et al., 2025. *LibEvolutionEval: A Benchmark and Study for Version-Specific Code Generation*. NAACL. arXiv 2412.04478.
  Covers 8 libraries including pandas, matplotlib and scipy. Finds that retrieving version-specific docs helps.
- Liu et al., 2024. *CodeUpdateArena: Benchmarking Knowledge Editing on API Updates*. arXiv 2407.06249.
  The alternative approach: edit the model's knowledge instead of retrieving.
- **Snake++:** a version filter applied at retrieval time, with a soft and a hard mode, inside a working assistant.
  It is evaluated on DS-1000 at its pinned versions and on our own version-sensitive question set.
  Most prior work in this section is benchmarks only.

## 5. API hallucination
- Liu et al., 2024. *Exploring and Evaluating Hallucinations in LLM-Powered Code Generation*. arXiv 2404.00971.
- Spracklen et al., 2025. *We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs*. USENIX Security. arXiv 2406.10279.
- **Snake++:** grounds answers in retrieved chunks and cites them; the error-type analysis counts AttributeError and ImportError cases.

## 6. Benchmarks and contamination
- Chen et al., 2021. *Evaluating Large Language Models Trained on Code* (HumanEval). arXiv 2107.03374.
- Austin et al., 2021. *Program Synthesis with Large Language Models* (MBPP). arXiv 2108.07732.
- Lai et al., 2023. *DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation*. ICML. arXiv 2211.11501.
- Riddell et al., 2024. *Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models*. ACL.
- Jain et al., 2024. *LiveCodeBench: Holistic and Contamination Free Evaluation of LLMs for Code*. arXiv 2403.07974.
- Broder, 1997. *On the Resemblance and Containment of Documents* (MinHash).
  Lee et al., 2022. *Deduplicating Training Data Makes Language Models Better*. ACL.
- **Snake++:** decontaminates the Stack Overflow index against the benchmark prompts and reports a deliberately contaminated config (c5_leaky).
  This measures how much leakage would inflate pass@1.

## 7. Models and serving (one paragraph, or move it to the setup section)
- Hui et al., 2024. *Qwen2.5-Coder Technical Report*. arXiv 2409.12186. Our generator.
- Kwon et al., 2023. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP.

## Writing plan
1. Write §4 first; it is where our contribution sits.
2. Next, §2 and §3 (Kevin can review §3).
3. Then §6 (Enea reviews it, since evaluation is his area).
4. Compress §1, §5 and §7 to two or three sentences each.
