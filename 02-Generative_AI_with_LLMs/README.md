# Phase 02 — Generative AI with LLMs

[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Topics](https://img.shields.io/badge/Topics-0%2F4-brightgreen)]()

Fine-tuning, end to end: the techniques, the transfer-learning theory behind them, how they differ across GPT/BERT/T5 architecture families, and three real applied case studies. Every topic in this phase touches fine-tuning pretrained models — exactly the territory where this repository's development sandbox (no Hugging Face Hub access) bites hardest, so each topic's honesty posture is documented explicitly rather than assumed uniform.

## Topics

| # | Topic | What the code actually proves |
|---|---|---|
| 1 | [Fine-Tuning Principles and Techniques](01-Fine-Tuning-Principles-and-Techniques/) | One MLM-pretrained encoder, four fine-tuning techniques (naive/discriminative-LR/warmup-cosine/gradual-unfreeze) — real convergence curves, including an honest surprise where the "protective" techniques converge slower on an easy task within a fixed budget. |
| 2 | [Transfer Learning for Domain-Specific Tasks](02-Transfer-Learning-for-Domain-Specific-Tasks/) | Domain-adaptive pretraining (DAPT) vs. no-adaptation vs. no-pretraining, real 3-arm comparison — includes a real bug caught and fixed mid-development (CLS token never pretrained), documented in full rather than silently corrected. |
| 3 | [Implementing Fine-Tuning (GPT, BERT, T5)](03-Implementing-Fine-Tuning-GPT-BERT-T5/) | Three architecture families (MiniGPT, MiniBERT, MiniT5) built from scratch, each fine-tuned via its own architecture-appropriate method (last-token / [CLS] / text-to-text generation) — real cross-attention, real checkpoint-reuse technique (Rothe et al., 2020), real comparable accuracy across all three. |
| 4 | [Case Studies: Summarization, Translation, Sentiment](04-Case-Studies-Summarization-Translation-Sentiment/) | Real sentiment case study with negation patterns and a full confusion matrix; real toy translation demonstrating genuine word-reordering via cross-attention; summarization shipped as correct Colab-ready code but honestly not executed, since a toy model cannot produce coherent summaries. |

## Progress

| Topic | theory.md | implementation.py | explanation.md | proof.png |
|---|---|---|---|---|
| 01 — Fine-Tuning Principles & Techniques | ✅ | ✅ Executed | ✅ | ✅ Real |
| 02 — Transfer Learning for Domains | ✅ | ✅ Executed | ✅ | ✅ Real |
| 03 — Implementing GPT/BERT/T5 | ✅ | ✅ Executed | ✅ | ✅ Real |
| 04 — Case Studies | ✅ | ✅ Executed (2/3 case studies) | ✅ | ⚠️ Composite (2 real panels + 1 labeled placeholder) |

**4 / 4 topics complete.**

## A Note on Honesty Across This Phase

Every topic in Phase 02 is, in some form, about fine-tuning a pretrained model — the exact territory this sandbox's lack of Hugging Face Hub access affects most. Rather than uniformly degrade every topic to network-free proxies, or uniformly ship placeholders, each topic was handled on its actual merits:

- **Topics 01, 02, 03** don't need a *specific* real-world pretrained checkpoint's knowledge — they need *a* genuinely pretrained checkpoint to fine-tune, which a from-scratch MLM/CLM-pretrained model provides just as validly for studying technique mechanics. All three were fully executed, including one real bug caught and fixed in Topic 02 (documented in that topic's `explanation.md` and `theory.md` in detail, not glossed over) and one deliberately-reported counterintuitive result in Topic 01 (the "safer" fine-tuning techniques converging slower within a fixed budget — a real, explained trade-off, not a tuning failure).
- **Topic 04** splits three ways: sentiment analysis and a toy translation mechanism demo are both genuinely reproducible network-free and were executed for real; summarization inherently requires real pretrained sequence-generation knowledge that no toy model trained in seconds can honestly provide, so that one case study ships as correct, Colab-ready code with a clearly watermarked placeholder rather than fabricated ROUGE scores.

Every topic's own `STATUS.md` documents the specifics.

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Each topic's `implementation.py` is self-contained — no cross-topic imports, even where architecture code (the shared `Attention`/`EncoderBlock`/`DecoderBlock` pattern used throughout this phase) is conceptually identical across topics. Colab-only sections (real HF checkpoints in Topics 03 and 04) are gated behind explicit `RUN_*` flags near the bottom of each script.

---
Part of the [llm-mastery](../README.md) curriculum.
