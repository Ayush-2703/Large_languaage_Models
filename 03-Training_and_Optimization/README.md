# Phase 03 — Training and Optimization of LLMs

[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Topics](https://img.shields.io/badge/Topics-0%2F4-brightgreen)]()

How LLMs actually get trained and made affordable: cleaning and tokenizing raw text, shrinking what needs to be trained and stored (LoRA, quantization, pruning), teaching a model to follow human preference (RLHF's reward model, and DPO), and the two biggest memory/compute bottlenecks (gradient checkpointing, Flash Attention). Unlike Phase 02, most of this phase's techniques are algorithmic — they don't need a *specific* real pretrained checkpoint's knowledge — so most of it runs fully real and network-free.

## Topics

| # | Topic | What the code actually proves |
|---|---|---|
| 1 | [Data Collection and Preprocessing](01-Data-Collection-and-Preprocessing/) | A real 5-stage cleaning pipeline (dedup, near-dedup, quality filters, PII scrubbing) plus a real BPE tokenizer trained from scratch — 902→750 real documents, 2.96x real compression. Caught and fixed a real regex bug that silently missed one of two synthetic PII formats. |
| 2 | [Efficient Training: LoRA, QLoRA, Quantization, Pruning](02-Efficient-Training-LoRA-QLoRA-Quantization-Pruning/) | All four techniques implemented from their actual math and applied to one real fine-tuned model: LoRA trains 2% of params for comparable accuracy; INT8/NF4 quantization give 4x/8x size reduction at zero measured accuracy loss; pruning traces the real accuracy-cliff curve. |
| 3 | [Alignment: RLHF & DPO](03-Alignment-RLHF-and-DPO/) | A real Bradley-Terry reward model (0.587→1.000 pairwise accuracy) and real DPO training (0.275→1.000 preference accuracy, smoothly growing implicit reward margin) — including an honestly-reported length-bias artifact in the preference metric. RLHF's PPO stage is deliberately conceptual only, exactly as scoped before this repository's build began. |
| 4 | [Memory and Computational Challenges](04-Memory-and-Computational-Challenges/) | Real peak-memory measurement (isolated subprocesses) showing gradient checkpointing's real 35% memory reduction at an 18.5% time cost, plus real attention memory/time scaling showing why avoiding the materialized attention matrix matters more as sequences grow — the general principle behind Flash Attention, honestly distinguished from its GPU-specific mechanism, which this CPU sandbox cannot run under any configuration. |

## Progress

| Topic | theory.md | implementation.py | explanation.md | proof.png |
|---|---|---|---|---|
| 01 — Data Collection & Preprocessing | ✅ | ✅ Executed | ✅ | ✅ Real |
| 02 — Efficient Training | ✅ | ✅ Executed | ✅ | ✅ Real |
| 03 — Alignment (RLHF & DPO) | ✅ | ✅ Executed (RM + DPO); PPO conceptual | ✅ | ✅ Real |
| 04 — Memory & Computational Challenges | ✅ | ✅ Executed | ✅ | ✅ Real |

**4 / 4 topics complete — the first fully-real phase in this repository, with zero placeholder proof images.**

## A Note on Honesty Across This Phase

Every technique in this phase turned out to be a genuine algorithm applicable to any pretrained-then-fine-tuned model, not a property of one *specific* real-world checkpoint — which meant this sandbox's lack of Hugging Face Hub access, the dominant constraint in Phase 02, barely mattered here. The one deliberate scope limit is RLHF's PPO stage (Topic 03): implementing rollout sampling, advantage estimation, and clipped policy updates correctly carries real risk of subtle bugs that would produce code that runs without actually demonstrating correct reinforcement learning — a worse outcome than the rigorous conceptual treatment `theory.md` gives it instead. This was decided and communicated before this repository's very first file was written, not a fallback reached for after running into difficulty.

Two real bugs were also caught and fixed during this phase's development, both documented in full in their respective topics rather than silently corrected: a PII-detection regex that missed one of two synthetic phone formats (Topic 01), and an ambiguous console-output label that made a real memory *reduction* read like an increase (Topic 04). A length-bias artifact in DPO's preference metric (Topic 03) was reduced but not fully eliminated, and is reported as the real, partially-unresolved finding it is.

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Topic 01 additionally installs `tokenizers` (already listed in the root `requirements.txt`) for real local BPE training — no Hugging Face Hub access required, since tokenizer *training* is a local operation. Topic 04's memory measurements launch isolated subprocesses internally; no extra setup is needed to reproduce them.

---
Part of the [llm-mastery](../README.md) curriculum.
