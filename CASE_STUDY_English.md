# Case study — Building a RAG and QLoRA pipeline from the ground up: mini-RAG NDT

## Context

As an electrical engineering researcher specialized in Non-Destructive Testing (NDT), I wanted to build a practical, verifiable skill set in LLM engineering — not just knowing how to wire up LangChain, but understanding what actually happens at every stage of a RAG pipeline and a QLoRA fine-tuning run, to the point of being able to debug and justify every design decision.

I used my own field of expertise as the testing ground: five scientific papers on railway non-destructive testing (eddy current, GMR, ultrasonic testing, EMAT, MFL, ACFM...). This choice wasn't incidental — knowing the subject matter deeply let me judge the real quality of generated answers, not just their surface plausibility.

## Approach

Built **without a RAG framework** (no LangChain, no LlamaIndex): chunking, embeddings, retrieval, reranking, and prompt engineering all implemented directly. The goal wasn't delivery speed but understanding — a deliberate choice for a learning project, which also had a useful side effect: when something broke, I knew exactly where to look, with no abstraction layer to dig through first.

Working method: iterative, with **objective validation at every step** rather than eyeballing a handful of questions. This principle shaped the entire project, from chunking strategy to fine-tuning evaluation.

## Technical challenges and how they were resolved

### 1. Naive chunking breaks meaning

Fixed-character chunking cut formulas and ideas apart arbitrarily. Solution: two-level semantic chunking — section detection via numbered headings, then sentence-boundary splitting within each section, with overlap at the sentence level rather than the character level.

**Edge case caught and fixed**: a single sentence occasionally exceeded the target chunk size on its own (common with formulas poorly segmented by PDF extraction), which caused that oversized chunk to be fully duplicated into the next one via the overlap mechanism — caught by testing on a real excerpt before running the full ingestion pipeline.

### 2. Metadata over-tags survey articles

First instinct: detect the NDT method (eddy current, ultrasonic testing...) once per whole document. Result: a thesis chapter surveying seven methods picked up all seven tags, even for chunks mentioning only one — making metadata filtering nearly useless on survey-style documents.

**Fix**: two-level detection — the "is this an NDT article" context check still runs on the full document (a global property), but each specific method is now detected **per chunk** (a local property). A chunk is only tagged for what it actually mentions.

### 3. A revealing false positive: physics isn't NDT

A paper on railway induction heating ended up tagged "eddy current" — because induction heating physically *uses* that phenomenon, without being a non-destructive testing paper at all. A single keyword wasn't enough to distinguish the physical principle from its NDT application.

**Fix**: a co-occurrence guard — a method tag is only validated if the document also mentions a generic NDT marker ("non-destructive testing," "NDT," "non-destructive inspection"...). Once fixed, this also revealed the inverse limitation (possible under-detection if a legitimate article never uses these explicit markers) — documented rather than hidden.

### 4. Diagnosing a retrieval failure at the right layer

A question about a technique combining two methods (eddy current + thermography) failed to surface the right article, despite retrieval appearing to work in general. Diagnosis: the bi-encoder captures topical proximity, not the notion of *combination* — it surfaced articles mentioning both methods separately rather than the one article actually combining them.

**Two-stage resolution**, deliberately kept separate to isolate the cause:
- An **AND** metadata filter (instead of the usual OR) fixed retrieval — proof the problem was the bi-encoder's logic, not the underlying data.
- Once retrieval was fixed, generation *still* failed to answer correctly (the local LLM wasn't bridging "courants de Foucault" in French with "eddy current" in the English source text). A second, distinct problem at a different pipeline layer, fixed separately with a prompt enriched with a terminology glossary.

Keeping these as two separate diagnoses (retrieval vs. generation) rather than one blanket fix reflects a debugging discipline that matters in production: don't patch blindly — pinpoint exactly which layer is responsible.

### 5. Closing the loop between RAG and QLoRA

Rather than an artificial fine-tuning dataset, question/answer pairs were generated automatically **directly from the chunks already indexed and tagged** by the RAG pipeline, using the same local LLM already in place. An initial imbalance was caught (some NDT methods over-represented due to tag overlap on enumeration-style chunks) and fixed by over-fetching and randomly shuffling candidates.

Result verified objectively: on questions absent from the training set, the fine-tuned model consistently produces the learned JSON format, while the base model answers in free-form prose — evidence of generalizing a structural pattern, not memorization.

### 6. Dependency management as a skill in its own right

A non-trivial part of the work involved diagnosing version incompatibilities between `bitsandbytes`, `transformers`, and `trl` on Windows — including a known bug (an `AttributeError` on a `frozenset`) documented on GitHub. Rather than working around it by trial and error, I traced the exact root cause (a `trl` v0.9 → v1.x API change) and migrated the code to the current API instead of pinning outdated, mutually incompatible versions together.

## Results

| Metric | Value |
|---|---|
| Recall@5 (retrieval, reranking on) | 100% |
| Precision@5 | 80% (vs. 77.5% with naive chunking, measured objectively) |
| MRR | 0.917 |
| Trainable parameters (QLoRA) | 0.17% of the model (13.6M / 8.04 billion) |
| Fine-tuned format generalization | Verified on unseen questions |

## What this project demonstrates

- Understanding of RAG pipeline internals (not just framework usage)
- Diagnostic discipline: isolating the exact root cause before fixing, distinguishing pipeline layers (retrieval vs. generation, data vs. code)
- Methodological honesty: measuring rather than eyeballing, documenting limitations rather than hiding them
- Autonomy on cross-cutting engineering problems (environment management, version compatibility, Windows-specific quirks)
- Ability to chain multiple AI building blocks (RAG → synthetic data generation → fine-tuning) into a coherent pipeline

## Next steps

Deploying the pipeline as an API (FastAPI + Docker), integrating a cloud LLM alongside the local model, and expanding the evaluation set.

---

*Full source code: see the [technical README](./README_English.md)*
