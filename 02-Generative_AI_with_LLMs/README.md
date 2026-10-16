# Generative AI with LLMs

> Module 02 of **Large Language Models**: from transformer internals to aligned, deployable LLM systems.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HF](https://img.shields.io/badge/Hugging%20Face-Transformers%20%7C%20PEFT%20%7C%20TRL-FFD21E?logo=huggingface&logoColor=black)
![Colab](https://img.shields.io/badge/Runs%20on-Colab%20T4-F9AB00?logo=googlecolab&logoColor=white)

---

## Overview

This module covers the full lifecycle of a generative LLM: how the model is built and pre-trained, how it is adapted to a task, how it is aligned with human preferences, and how it is wired into applications. Each section pairs the theory with a runnable experiment.

## Lifecycle at a glance

```
Pre-training ──► Task adaptation ──► Alignment ──► Application
 (transformer,    (prompting, SFT,     (RLHF / PPO,   (RAG, agents,
  scaling laws)    instruction tuning,  reward models,  evaluation,
                   LoRA / QLoRA)        safety)         serving)
```

## Contents

### 1. Foundations: Transformers and Pre-training
- Generative tasks and the LLM landscape
- Encoder-only, decoder-only and encoder-decoder architectures
- Self-attention, positional encoding, KV caching
- Pre-training objectives: causal LM vs. masked LM
- Compute-optimal training and scaling laws

### 2. Inference and Prompting
- Decoding strategies: greedy, beam, top-k, top-p, temperature
- Prompt engineering: zero-shot, few-shot, chain-of-thought
- Context window limits and their trade-offs

### 3. Adaptation and Fine-tuning
- Full fine-tuning and catastrophic forgetting
- Instruction tuning and dataset construction
- Parameter-efficient fine-tuning: **LoRA**, **QLoRA**, prompt tuning
- Evaluation: ROUGE, BLEU, perplexity, benchmark suites

### 4. Alignment
- The HHH objective: helpful, honest, harmless
- Reward modelling from human preferences
- RLHF with PPO, and KL-regularized optimization
- Preference-optimization alternatives (DPO)
- Reward hacking and mitigation

### 5. LLM-powered Applications
- Retrieval-Augmented Generation (RAG)
- Tool use and agent patterns (ReAct, program-aided reasoning)
- Hallucination, grounding and guardrails
- Optimization for deployment: quantization, distillation, pruning

## Repository structure

<!-- TODO: replace with the actual contents of this folder -->
```
02-Generative_AI_with_LLMs/
├── README.md
├── <notebook / script>      # e.g. lab for fine-tuning with LoRA
├── <notebook / script>      # e.g. lab for RLHF
└── ...
```

## Getting started

```bash
git clone https://github.com/Ayush-2703/Large_languaage_Models.git
cd Large_languaage_Models/02-Generative_AI_with_LLMs
pip install torch transformers datasets peft trl evaluate accelerate bitsandbytes
```

Notebooks are sized for a free-tier Colab T4 GPU. Heavier steps use small models or quantization so they stay reproducible.

## Key takeaways

| Stage | Core idea | Main trade-off |
|---|---|---|
| Pre-training | Learn language statistics at scale | Compute and data cost |
| Fine-tuning | Specialize to a task | Forgetting vs. performance |
| PEFT (LoRA) | Train low-rank adapters only | Capacity vs. memory |
| RLHF | Optimize for human preference | Reward hacking, instability |
| RAG | Ground outputs in external knowledge | Retrieval quality, latency |

## Related

- Parent repo: [Large_languaage_Models](https://github.com/Ayush-2703/Large_languaage_Models)

## Author

**Ayush Kumar Singh**: B.Tech AI & ML, Amity University Lucknow
[GitHub](https://github.com/Ayush-2703) · [LinkedIn](https://linkedin.com/in/ayushsingh2703)
