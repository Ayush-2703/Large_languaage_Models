# Phase 05 — Frontiers and Future of LLMs

Status: **COMPLETE — 5/5 topics, real executed code behind every proof image.**

This phase closes the curriculum by looking past "how do current LLMs
work" (Phases 01-04) toward the open research questions and emerging
model families that define where the field is headed. Every topic
below ships the standard four deliverables (`theory.md`,
`implementation.py`, `explanation.md`, `proof.png`) plus a `STATUS.md`
documenting exactly what was executed and a raw results file
(`history.json` / `results.json`) backing the chart.

## Sandbox network constraint (same as Phase 01)

This build environment cannot reach Hugging Face Hub
(`huggingface.co` -> `403 host_not_allowed`; PyPI is open). No topic in
this phase needed a pretrained HF checkpoint to produce a real result,
so **all five topics ship fully real, executed proof images — zero
placeholders in this phase.** Where a topic (5.2, 5.5) would ideally
use a real multilingual/vision-language checkpoint in production, the
code instead reproduces the actual underlying mechanism from scratch
on synthetic data whose generative process is fully known and
documented, rather than fabricating numbers for a checkpoint that
can't be run here — see each topic's `STATUS.md` for the specific
reasoning.

## Topics

| # | Topic | Real result |
|---|---|---|
| 5.1 | [Emergent Behaviors and Scaling Hypotheses](01-Emergent-Behaviors-and-Scaling-Hypotheses/) | Reproduced genuine "grokking": a 2-layer Transformer hits 100% train accuracy almost immediately on mod-13 addition, but validation accuracy stays near chance until a sharp transition around step 4,000-6,000, then jumps to ~100% — a real, controlled instance of the sudden-capability-jump phenomenon behind the "emergent abilities" debate. |
| 5.2 | [Multilingual and Cross-lingual Capabilities](02-Multilingual-and-Cross-lingual-Capabilities/) | Two independently-trained synthetic embedding spaces ("languages") are aligned via orthogonal Procrustes using only 18 seed translation pairs: held-out word-translation accuracy on 42 unseen words goes from 0% (no alignment) to 100% (aligned) — the real mechanism behind cross-lingual embedding transfer. |
| 5.3 | [LLMs in Low-Resource Settings](03-LLMs-in-Low-Resource-Settings/) | A toy encoder-decoder Transformer trained on a fixed compute budget at ten different training-set sizes shows a sharp data-efficiency "knee": 0-5% exact-match accuracy at 5-15 examples, >97% by 30 examples — the same shape of curve that motivates transfer learning for genuinely low-resource languages/tasks. |
| 5.4 | [Alignment and Controllability of LLMs](04-Alignment-and-Controllability-of-LLMs/) | A literal implementation of the DPO loss, swept across 5 values of the KL-strength hyperparameter beta, shows both a real controllability effect (preference-token frequency roughly triples-to-quadruples over the unaligned base at every beta) and a real, beta-controlled "alignment tax" (base-capability retention is worst at the smallest beta, best at the largest). |
| 5.5 | [Beyond Text: Robotics, Vision-Language, Decision-Making](05-Beyond-Text-Robotics-Vision-Language-Decision-Making/) | A small vision+language grid-navigation policy reaches 98.0% closed-loop success across 500 held-out episodes; an architecturally-identical vision-only ablation (denied the instruction) manages only 55.0% — direct evidence that language conditioning is doing necessary work, not just adding parameters, once the visual scene is multi-goal-ambiguous. |

## What "real" means in this phase

Every `proof.png` in this phase is generated from a `results.json` or
`history.json` written by that topic's own `implementation.py` during
an actual execution — training runs, sampling, and evaluation all
happen when the script runs; no numbers are hand-set or estimated.
Each topic's `explanation.md` includes the exact recorded numbers from
the run that produced its chart, and each `STATUS.md` states plainly
what ran, on what hardware, and for how long.
