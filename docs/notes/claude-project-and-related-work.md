# Task: Claude Project with RFC, plan, papers; related-work outline (#6)

Owner: Roberto (outline drafted by Kevin with Claude Code) · Milestone: M0 · Due: 2026-10-26 · Branch: `feat/report-related-work`

## What was done
- **`report/related_work.md`:** a related-work outline in 7 sections with about 30 references. It covers:
  - RAG, retrieval for code, and hybrid retrieval with reranking;
  - **library evolution and version-aware generation**, the core section;
  - API hallucination, benchmarks and contamination, and the models we use.

  Every section ends with how Snake++ differs. The arXiv ids of the four core version-aware papers were checked online on 2026-10-25: VersiCode, GitChameleon, LibEvolutionEval and LLMs Meet Library Evolution.
  The other references still need checking before they go into the bibliography.

## Not done: the Claude Project (Roberto)
A Claude Project can only be made in the claude.ai web app; Claude Code can't create one. Steps:
1. **Create it.** On claude.ai go to **Projects → Create project**, name it "Snake++", and share it with the team. Sharing needs a Team plan; otherwise each person makes their own copy.
2. **Upload the project files:** `RFC.md`, `docs/PLAN.md`, `docs/Snakepp_Work_Plan.pdf`, `docs/tasks.yaml`, `CLAUDE.md` and `report/related_work.md`.
3. **Upload the papers** as PDFs from arXiv. Start with the four core ones: 2406.09834, 2406.07411, 2411.05830, 2412.04478.
   Then add CodeRAG-Bench 2406.14497, DocPrompting and DS-1000 2211.11501.
4. **Set the project instructions**, for example: "We are writing a course report on Snake++, a version-aware RAG coding assistant.
   Use the uploaded RFC and plan as ground truth. Cite only papers in the project files. Never invent results."
5. **Keep it current.** Re-upload `PLAN.md` and `related_work.md` whenever they change.
