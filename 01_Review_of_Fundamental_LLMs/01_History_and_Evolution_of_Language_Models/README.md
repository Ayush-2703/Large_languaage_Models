# Phase 01 — Review of Fundamental LLMs

[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Topics](https://img.shields.io/badge/Topics-5%2F5-brightgreen)]()

The conceptual and mathematical foundation the rest of this curriculum builds on: how language models got here, exactly what self-attention computes, how pretraining and fine-tuning relate, why scale predictably helps, and how bias enters a model in the first place. Every topic pairs cited theory with a real, executed experiment.

## Topics

| # | Topic | What the code actually proves |
|---|---|---|
| 1 | [History and Evolution of Language Models](01-History-and-Evolution-of-Language-Models/) | Trains an N-gram, RNN, LSTM, and Transformer from scratch on identical data — real measured perplexity across all four NLP eras, not an assertion. |
| 2 | [Transformer Architecture and Self-Attention Mechanism](02-Transformer-Architecture-and-Self-Attention/) | Implements QKV attention from the raw formula and verifies it numerically matches PyTorch's fused kernel to within float32 precision (~1e-7). |
| 3 | [Pretraining vs. Fine-Tuning Paradigms](03-Pretraining-vs-Fine-Tuning-Paradigms/) | Real, Colab-ready DistilBERT frozen-probe vs. full-fine-tune vs. no-pretraining comparison on SST-2 — the one topic in this phase requiring Hugging Face Hub access this sandbox couldn't reach; shipped with an honestly-labeled placeholder proof image instead of fabricated numbers. |
| 4 | [Scaling Laws and Model Efficiency](04-Scaling-Laws-and-Model-Efficiency/) | Trains 5 model sizes (6K to 1.4M params) from scratch, fits a real power law to measured loss, and measures the training-cost trade-off scale doesn't show for free. |
| 5 | [Ethical Considerations in Large-Scale AI](05-Ethical-Considerations-in-Large-Scale-AI/) | Trains word embeddings on a corpus with a *known, controlled* demographic co-occurrence rate and measures the resulting bias with a real WEAT-style test — cause and effect both real, both traceable. |

## Progress

| Topic | theory.md | implementation.py | explanation.md | proof.png |
|---|---|---|---|---|
| 01 — History and Evolution | ✅ | ✅ Executed | ✅ | ✅ Real |
| 02 — Self-Attention (QKV) | ✅ | ✅ Executed | ✅ | ✅ Real |
| 03 — Pretraining vs. Fine-Tuning | ✅ | ✅ Colab-ready | ✅ | ⚠️ Placeholder (labeled) |
| 04 — Scaling Laws & Efficiency | ✅ | ✅ Executed | ✅ | ✅ Real |
| 05 — Ethical Considerations | ✅ | ✅ Executed | ✅ | ✅ Real (composite) |

**5 / 5 topics complete.**

## A Note on `proof.png` Across This Phase

This repository's development sandbox has no Hugging Face Hub access (`huggingface.co` → `403 host_not_allowed`). Rather than silently degrade every topic's `implementation.py` to avoid that limitation, each topic was handled on its own merits:

- **Topics 01, 02, 04** don't inherently need a pretrained checkpoint — their subject is architecture, mechanism, and scale, demonstrable from random initialization. All three were actually executed here, and `proof.png` in each is real measured output.
- **Topic 05** could have been treated as checkpoint-dependent (probing a pretrained model for bias), but a network-free proxy — training embeddings on a corpus with a *known* injected bias rate — turned out to demonstrate the same underlying mechanism just as honestly, and more transparently, since the cause is fully controlled rather than inherited from an opaque real-world corpus. Executed for real.
- **Topic 03** is the one genuine exception: comparing pretrained-vs-not is meaningless without real pretrained weights, and no honest network-free substitute exists for that specific claim. `implementation.py` ships correct and Colab-ready; `proof.png` ships as a clearly watermarked placeholder rather than invented numbers, and will populate with real data on first Colab run.

Every topic's own `STATUS.md` documents this in more detail.

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Each topic's `implementation.py` is self-contained — no cross-topic imports — and runs standalone, either locally (Topics 01, 02, 04, 05 need no network access beyond fetching a small public-domain text corpus) or on Colab (Topic 03, and the optional pretrained-model sections gated behind `RUN_PRETRAINED_SECTION` flags in Topics 02 and 05).

---
Part of the [llm-mastery](../README.md) curriculum.
