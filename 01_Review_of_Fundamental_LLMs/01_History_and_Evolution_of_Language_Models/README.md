<div align="center">

![LLM Mastery: A Hands-On Curriculum for Large Language Models](https://capsule-render.vercel.app/api?type=waving&color=0:0F172A,100:7C3AED&height=220&section=header&text=LLM%20Mastery&fontSize=64&fontColor=FFFFFF&fontAlignY=36&animation=fadeIn&desc=A%20Hands-On%20Curriculum%20for%20Mastering%20Large%20Language%20Models&descSize=18&descAlignY=58)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://readme-typing-svg.demolab.com/?font=Fira+Code&size=20&duration=2800&pause=900&color=C4B5FD&center=true&vCenter=true&width=600&lines=22+Topics,+5+Phases,+One+Repo;Theory,+Code,+Explanation,+Proof;Every+Script+Runs+on+a+Free+T4;Small+Models.+Full+Understanding." />
  <img alt="Typing SVG" src="https://readme-typing-svg.demolab.com/?font=Fira+Code&size=20&duration=2800&pause=900&color=6D28D9&center=true&vCenter=true&width=600&lines=22+Topics,+5+Phases,+One+Repo;Theory,+Code,+Explanation,+Proof;Every+Script+Runs+on+a+Free+T4;Small+Models.+Full+Understanding." />
</picture>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=000000)](https://huggingface.co/docs/transformers)
[![Colab Ready](https://img.shields.io/badge/Colab-Ready-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

[![Progress](https://img.shields.io/badge/Progress-0%2F22%20Topics-lightgrey)](#progress-tracker)
[![Status](https://img.shields.io/badge/Status-Building%20in%20Public-orange)](#progress-tracker)
[![Last Commit](https://img.shields.io/github/last-commit/Ayush-2703/llm-mastery)](https://github.com/Ayush-2703/llm-mastery/commits/main)

[![Stars](https://img.shields.io/github/stars/Ayush-2703/llm-mastery?style=social)](https://github.com/Ayush-2703/llm-mastery/stargazers)
[![Forks](https://img.shields.io/github/forks/Ayush-2703/llm-mastery?style=social)](https://github.com/Ayush-2703/llm-mastery/network/members)


# Phase 01 — Review of Fundamental LLMs

[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Topics](https://img.shields.io/badge/Topics-5%2F5-brightgreen)]()

</div>

The conceptual and mathematical foundation the rest of this curriculum builds on: how language models got here, exactly what self-attention computes, how pretraining and fine-tuning relate, why scale predictably helps, and how bias enters a model in the first place. Every topic pairs cited theory with a real, executed experiment.

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
## 👤 Author

<div align="center">

### Ayush Kumar Singh

*Researcher in Adversarial ML, Geospatial AI, and LLM/NLP Systems*

[![GitHub](https://img.shields.io/badge/GitHub-Ayush%20Kumar%20Singh-181717?style=for-the-badge&logo=github)](https://github.com/Ayush-2703)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush%20Kumar%20Singh-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/ayushsingh2703)

</div>

---

<div align="center">

**If this repository helped you, please consider giving it a ⭐**  
*It takes 2 seconds and helps others discover it.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>

</div>
