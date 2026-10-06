# Phase 04 — Applications and Use Cases

[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Topics](https://img.shields.io/badge/Topics-3%2F4-brightgreen)]()

Where LLMs meet the outside world: retrieving from a real vector database, integrating knowledge beyond parametric memory, prompting techniques that elicit better reasoning, and the two core skills behind chatbots. This phase produced this repository's richest run of real, resolved debugging journeys — three of its four topics caught and fixed a genuine bug or design flaw mid-development, each documented in full rather than smoothed over.

## Topics

| # | Topic | What the code actually proves |
|---|---|---|
| 1 | [RAG with Vector DBs](01-Retrieval-Augmented-Generation-RAG-Vector-DBs/) | A real dual-encoder retriever + real FAISS index, honestly split into what works (in-distribution: 0.70–0.90 retrieval, 1.00 answering) and what doesn't yet (zero-shot generalization to brand-new entities) — after a 3-iteration debugging journey that caught an impossible-classification bug and a memorization shortcut along the way. |
| 2 | [Integrating External Knowledge](02-Integrating-External-Knowledge-into-LLMs/) | Real BM25 (zero training, 1.00/1.00 seen/held-out) directly contrasted with the same dense retriever from Topic 1 (0.70/0.00) — a clean, real confirmation of *why* sparse and dense retrieval fail differently. Plus a real tool-use routing classifier (1.00 accuracy). |
| 3 | [Chain-of-Thought & Few-Shot Learning](03-Chain-of-Thought-Prompting-and-Few-Shot-Learning/) | A real in-context-learning experiment (zero weight updates) showing flat, near-chance performance across every k-shot level and both prompting styles — a small-scale replication consistent with Wei et al. (2022)'s own finding that CoT's benefit is itself scale-emergent, not a failed demo. |
| 4 | [Conversational AI & Knowledge Extraction](04-Conversational-AI-Chatbots-Knowledge-Extraction/) | Real multi-turn dialogue context retention (modest, 0.20–0.25, after catching a real seen/held-out measurement bug) alongside real, perfect NER via BIO tagging (1.00 across every metric) — two genuinely different difficulty profiles, reported side by side rather than averaged into one number. |

## Progress

| Topic | theory.md | implementation.py | explanation.md | proof.png |
|---|---|---|---|---|
| 01 — RAG with Vector DBs | ✅ | ✅ Executed | ✅ | ✅ Real |
| 02 — Integrating External Knowledge | ✅ | ✅ Executed | ✅ | ✅ Real |
| 03 — CoT & Few-Shot Learning | ✅ | ✅ Executed | ✅ | ✅ Real |
| 04 — Conversational AI & Extraction | ✅ | ✅ Executed | ✅ | ✅ Real |

**4 / 4 topics complete — every proof image in this phase is real; no placeholders.**

## A Note on the Debugging Journeys in This Phase

Unlike earlier phases, most of this phase's real findings emerged only *after* catching and fixing a genuine flaw in the experimental design — not from the first version of the code:

- **Topic 01** went through three iterations: an initial classifier evaluated on fact classes it had never been trained on (structurally impossible to get right, regardless of context — the 0.000 oracle-context score was the tell); then a fix that inadvertently let the model memorize entity-name-to-label mappings directly from the query text, bypassing context entirely; then a final design with shared, many-to-one category labels and genuinely held-out entities.
- **Topic 02** applied Topic 01's final lessons from the start, and used the resulting clean setup to produce a real, informative BM25-vs-dense comparison.
- **Topic 03** required two rounds of compute right-sizing after early runs badly exceeded reasonable CPU time, and caught a real answer-extraction bug (searching for "Answer:" in text that structurally couldn't contain it for one of two prompting styles) before trusting any comparison between them.
- **Topic 04** caught a real seen/held-out evaluation mismatch (evaluating against a fresh resample from an eligible pool rather than the specific data actually used in training) before its dialogue-retention numbers could be trusted.

Every fix is documented in the relevant topic's `explanation.md` in enough detail to be independently checked, not just asserted.

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

`faiss-cpu` (Topic 01, 02) and `tokenizers`-style local libraries need no Hugging Face Hub access — vector indexing and BM25 are both local, network-free operations. Every topic's `implementation.py` remains self-contained per this repository's convention, redefining shared architecture patterns (the `Attention`/`EncoderBlock` family) fresh in each file rather than importing across topics.

---
Part of the [llm-mastery](../README.md) curriculum.
