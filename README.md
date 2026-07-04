# LLM Mastery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E)](https://huggingface.co/docs/transformers)
[![Progress](https://img.shields.io/badge/Progress-0%2F22%20Topics-lightgrey)](#progress-tracker)

A topic-wise, hands-on curriculum for mastering Large Language Models — from the math behind self-attention to RAG, alignment, and multimodal frontiers. Every topic pairs grounded theory with runnable code and a generated artifact proving the code actually ran.

Companion repository to [`deep-learning-mastery`](https://github.com/Ayush-2703/deep-learning-mastery), built to the same standard.

---

## About

This repository covers Large Language Models end to end across **5 phases and 22 topics**: architectural fundamentals, generative fine-tuning, efficient training and alignment, applied systems like RAG, and where the field is headed next.

It is being built incrementally, phase by phase. The folder skeleton below is complete; topic content (`theory.md`, `implementation.py`, `explanation.md`, `proof.png`) ships in subsequent passes. Each folder's `STATUS.md` tracks where that topic currently stands.

## Design Philosophy

- **Runs anywhere, fast.** Every `implementation.py` is built to complete on a single free-tier Colab T4 GPU in well under an hour.
- **Small models, full understanding.** Hands-on code uses small open checkpoints — DistilBERT, BERT-base, GPT-2 / GPT-2-medium, T5-small / FLAN-T5-small, DistilBART, OPUS-MT — and small public datasets (IMDB, SST-2, CNN/DailyMail or WMT16 subsamples, SQuAD).
- **Scales conceptually.** Every `theory.md` explains how the technique extends to production-scale models, even where the runnable code stays deliberately small.
- **Grounded, not copied.** Theory draws on Denis Rothman's *Transformers for Natural Language Processing*, Tunstall, von Werra & Wolf's *Natural Language Processing with Transformers*, and primary papers — paraphrased throughout, cited by title and author, never block-quoted.

> **Scope notes:** Standalone Flash Attention kernels need Ampere-class GPUs or newer and won't run on a T4 — that topic demonstrates PyTorch's built-in scaled-dot-product-attention backend instead, with the dedicated kernel covered conceptually in `theory.md`. Likewise, full RLHF (reward model + PPO rollouts) is too heavy for a sub-hour Colab run — the hands-on code for Alignment is DPO, with RLHF/PPO covered conceptually alongside it.

## Repository Structure

```
llm-mastery/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── 01-Review-of-Fundamental-LLMs/
├── 02-Generative-AI-with-LLMs/
├── 03-Training-and-Optimization-of-LLMs/
├── 04-Applications-and-Use-Cases/
└── 05-Frontiers-and-Future-of-LLMs/
```

Each phase folder contains its own topic folders, numbered `01`, `02`, … starting fresh within that phase.

### Per-Topic File Convention

Every topic folder holds exactly four files:

```
03-Training-and-Optimization-of-LLMs/
└── 02-Efficient-Training-LoRA-QLoRA-Quantization-Pruning/
    ├── theory.md           # Math intuition, architecture breakdown, citations
    ├── implementation.py   # Runnable PyTorch + Hugging Face code
    ├── explanation.md      # Line-by-line walkthrough of the code's why and what
    └── proof.png           # Generated artifact proving the code ran end to end
```

No exceptions. For topics that are conceptual rather than experimental (History and Evolution of Language Models, Ethical Considerations in Large-Scale AI, Emergent Behaviors and Scaling Hypotheses), `proof.png` is still generated from real code — a composite figure pairing a conceptual diagram with a real comparison chart, rather than a static illustration alone.

## Curriculum

### Phase 01 — Review of Fundamental LLMs
| # | Topic | Folder |
|---|---|---|
| 1 | History and Evolution of Language Models | [`01-History-and-Evolution-of-Language-Models`](01-Review-of-Fundamental-LLMs/01-History-and-Evolution-of-Language-Models/) |
| 2 | Transformer Architecture and Self-Attention Mechanism (QKV math) | [`02-Transformer-Architecture-and-Self-Attention`](01-Review-of-Fundamental-LLMs/02-Transformer-Architecture-and-Self-Attention/) |
| 3 | Pretraining vs. Fine-Tuning Paradigms | [`03-Pretraining-vs-Fine-Tuning-Paradigms`](01-Review-of-Fundamental-LLMs/03-Pretraining-vs-Fine-Tuning-Paradigms/) |
| 4 | Scaling Laws and Model Efficiency | [`04-Scaling-Laws-and-Model-Efficiency`](01-Review-of-Fundamental-LLMs/04-Scaling-Laws-and-Model-Efficiency/) |
| 5 | Ethical Considerations in Large-Scale AI | [`05-Ethical-Considerations-in-Large-Scale-AI`](01-Review-of-Fundamental-LLMs/05-Ethical-Considerations-in-Large-Scale-AI/) |

### Phase 02 — Generative AI with LLMs
| # | Topic | Folder |
|---|---|---|
| 1 | Principles and Techniques for Fine-Tuning Pretrained Models | [`01-Fine-Tuning-Principles-and-Techniques`](02-Generative-AI-with-LLMs/01-Fine-Tuning-Principles-and-Techniques/) |
| 2 | Transfer Learning for Domain-Specific Tasks | [`02-Transfer-Learning-for-Domain-Specific-Tasks`](02-Generative-AI-with-LLMs/02-Transfer-Learning-for-Domain-Specific-Tasks/) |
| 3 | Implementing Fine-Tuning (GPT, BERT, T5) | [`03-Implementing-Fine-Tuning-GPT-BERT-T5`](02-Generative-AI-with-LLMs/03-Implementing-Fine-Tuning-GPT-BERT-T5/) |
| 4 | Case Studies: Text Summarization, Translation, and Sentiment Analysis | [`04-Case-Studies-Summarization-Translation-Sentiment`](02-Generative-AI-with-LLMs/04-Case-Studies-Summarization-Translation-Sentiment/) |

### Phase 03 — Training and Optimization of LLMs
| # | Topic | Folder |
|---|---|---|
| 1 | Data Collection and Preprocessing for LLMs | [`01-Data-Collection-and-Preprocessing`](03-Training-and-Optimization-of-LLMs/01-Data-Collection-and-Preprocessing/) |
| 2 | Efficient Training Strategies: LoRA, QLoRA, Quantization (INT8/NF4), and Pruning | [`02-Efficient-Training-LoRA-QLoRA-Quantization-Pruning`](03-Training-and-Optimization-of-LLMs/02-Efficient-Training-LoRA-QLoRA-Quantization-Pruning/) |
| 3 | Alignment Methodologies: RLHF & DPO | [`03-Alignment-RLHF-and-DPO`](03-Training-and-Optimization-of-LLMs/03-Alignment-RLHF-and-DPO/) |
| 4 | Memory and Computational Challenges (Gradient Checkpointing, Flash Attention) | [`04-Memory-and-Computational-Challenges`](03-Training-and-Optimization-of-LLMs/04-Memory-and-Computational-Challenges/) |

### Phase 04 — Applications and Use Cases
| # | Topic | Folder |
|---|---|---|
| 1 | Introduction to Retrieval-Augmented Generation (RAG) with Vector DBs | [`01-Retrieval-Augmented-Generation-RAG-Vector-DBs`](04-Applications-and-Use-Cases/01-Retrieval-Augmented-Generation-RAG-Vector-DBs/) |
| 2 | Integrating External Knowledge into LLMs | [`02-Integrating-External-Knowledge-into-LLMs`](04-Applications-and-Use-Cases/02-Integrating-External-Knowledge-into-LLMs/) |
| 3 | Chain-of-Thought (CoT) Prompting and Few-Shot Learning | [`03-Chain-of-Thought-Prompting-and-Few-Shot-Learning`](04-Applications-and-Use-Cases/03-Chain-of-Thought-Prompting-and-Few-Shot-Learning/) |
| 4 | Conversational AI, Chatbots, and Knowledge Extraction | [`04-Conversational-AI-Chatbots-Knowledge-Extraction`](04-Applications-and-Use-Cases/04-Conversational-AI-Chatbots-Knowledge-Extraction/) |

### Phase 05 — Frontiers and Future of LLMs
| # | Topic | Folder |
|---|---|---|
| 1 | Emergent Behaviors and Scaling Hypotheses | [`01-Emergent-Behaviors-and-Scaling-Hypotheses`](05-Frontiers-and-Future-of-LLMs/01-Emergent-Behaviors-and-Scaling-Hypotheses/) |
| 2 | Multilingual and Cross-lingual Capabilities | [`02-Multilingual-and-Cross-lingual-Capabilities`](05-Frontiers-and-Future-of-LLMs/02-Multilingual-and-Cross-lingual-Capabilities/) |
| 3 | LLMs in Low-Resource Settings | [`03-LLMs-in-Low-Resource-Settings`](05-Frontiers-and-Future-of-LLMs/03-LLMs-in-Low-Resource-Settings/) |
| 4 | Alignment and Controllability of LLMs | [`04-Alignment-and-Controllability-of-LLMs`](05-Frontiers-and-Future-of-LLMs/04-Alignment-and-Controllability-of-LLMs/) |
| 5 | Beyond Text: LLMs in Robotics, Vision-Language Models, and Decision-Making | [`05-Beyond-Text-Robotics-Vision-Language-Decision-Making`](05-Frontiers-and-Future-of-LLMs/05-Beyond-Text-Robotics-Vision-Language-Decision-Making/) |

## Setup

```bash
git clone https://github.com/Ayush-2703/llm-mastery.git
cd llm-mastery
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Running on Google Colab:** open a new notebook, set the runtime to a T4 GPU, then either clone this repo with `!git clone` or upload a single topic's `implementation.py` and run `!pip install -r requirements.txt` in the first cell. Every script is self-contained per topic — no cross-topic imports.

## Progress Tracker

| Phase | Topics | Completed | Progress |
|---|---|---|---|
| 01 — Review of Fundamental LLMs | 5 | 0 | 0% |
| 02 — Generative AI with LLMs | 4 | 0 | 0% |
| 03 — Training and Optimization of LLMs | 4 | 0 | 0% |
| 04 — Applications and Use Cases | 4 | 0 | 0% |
| 05 — Frontiers and Future of LLMs | 5 | 0 | 0% |
| **Total** | **22** | **0** | **0%** |

Each topic's folder-level status lives in its own `STATUS.md`, updated as that topic's four files ship.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Primary papers are cited individually, by title and author, within each topic's `theory.md`.

## Author

**Ayush Kumar Singh**

[GitHub](https://github.com/Ayush-2703) · [LinkedIn](https://www.linkedin.com/in/ayushsingh2703)

## License

Released under the [MIT License](LICENSE).
