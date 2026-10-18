<div align="center">

![Phase 02: Generative AI with LLMs](https://capsule-render.vercel.app/api?type=waving&color=0:18181B,100:3F3F46&height=220&section=header&text=Phase%2002%20%C2%B7%20Generative%20AI%20with%20LLMs&fontSize=34&fontColor=FFFFFF&fontAlignY=38&animation=fadeIn&desc=TODO%3A%20one-line%20tagline%20(keep%20under%2090%20characters)&descSize=15&descAlignY=62)

*Phase 02 of [`llm-mastery`](../README.md) — a four-file (theory, code, explanation, Image), Colab-first LLM curriculum*

</div>

---

## Table of Contents

- [Overview](#overview)
- [The Arc of This Phase](#the-arc-of-this-phase)
- [Topics at a Glance](#topics-at-a-glance)
- [Folder Structure](#folder-structure)
- [Datasets and Models Used in This Phase](#datasets-and-models-used-in-this-phase)
- [Results Snapshot](#results-snapshot)
- [Highlights Gallery](#highlights-gallery)
- [Getting Started](#getting-started)
- [Known Issues](#known-issues)
- [Key References](#key-references)
- [Navigate](#navigate)
- [License](#license)
- [Author](#author)

---

## Overview

<!-- 2 short paragraphs, max 3 sentences each. -->

**Paragraph 1 — the through-line.** TODO: the single idea every topic in this phase shares (what Phase 01 handed over, and what this phase builds on it). End with the question the phase answers.

**Paragraph 2 — the standard it holds itself to.** TODO: how many topics were actually executed, what real numbers are in the result images, and which topic (if any) could not be run and why. State this plainly, e.g. "N of M topics executed; Topic X needs Y and is not yet run."

## The Arc of This Phase

<!-- One short paragraph per topic. Each: the question it asks, the method, and a link to its row in Results Snapshot. Do NOT restate numbers here. -->

**2.1 — TODO title.** The question this topic asks → how it is answered (method, data, model) → [result](#results-snapshot).

**2.2 — TODO title.** Question → method → [result](#results-snapshot).

**2.3 — TODO title.** Question → method → [result](#results-snapshot).

**2.4 — TODO title.** Question → method → [result](#results-snapshot).

**2.5 — TODO title.** Question → method → [result](#results-snapshot). Close the loop to a Phase 01 topic if one exists.

## Topics at a Glance

| # | Topic | Folder | What the code actually proves |
|---|-------|--------|--------------------------------|
| 1 | TODO | [`01-TODO_Folder`](01-TODO_Folder) | One-line, evidence-style claim |
| 2 | TODO | [`02-TODO_Folder`](02-TODO_Folder) | One-line, evidence-style claim |
| 3 | TODO | [`03-TODO_Folder`](03-TODO_Folder) | One-line, evidence-style claim |
| 4 | TODO | [`04-TODO_Folder`](04-TODO_Folder) | One-line, evidence-style claim |
| 5 | TODO | [`05-TODO_Folder`](05-TODO_Folder) | One-line, evidence-style claim |

Every topic follows the repository's four-file structure: `README.md` (theory), `implementation.py`, `explanation.md`, and a result image under `Image/`. See [Known Issues](#known-issues) for any topic where a piece is missing.

## Folder Structure

```
02-Generative_AI_with_LLMs/
├── README.md                          (this file)
│
├── 01-TODO_Folder/
│   ├── README.md              — theory: TODO
│   ├── implementation.py      — TODO: what it trains / runs
│   ├── explanation.md         — TODO: line-by-line walkthrough
│   └── Image/
│       └── TODO.png
│
├── 02-TODO_Folder/
│   ├── README.md              — theory: TODO
│   ├── implementation.py      — TODO
│   ├── explanation.md
│   └── Image/
│       └── TODO.png
│
├── 03-TODO_Folder/
│   └── ...
├── 04-TODO_Folder/
│   └── ...
└── 05-TODO_Folder/
    └── ...
```

## Datasets and Models Used in This Phase

| Topic | Data / model | Notes |
|-------|--------------|-------|
| 2.1 | TODO dataset (size, split) · TODO model (params, checkpoint) | One line: hardware, runtime, anything unusual |
| 2.2 | TODO | TODO |
| 2.3 | TODO | TODO |
| 2.4 | TODO | TODO |
| 2.5 | TODO | TODO |

## Results Snapshot

<!-- Single canonical results table. Numbers copied from the committed result image or explanation.md, with units. Mark unexecuted topics honestly. -->

| Topic | Headline result |
|-------|------------------|
| 2.1 | **Metric = value** (n params, t seconds, hardware). One sentence on what it means. |
| 2.2 | **Metric = value**. One sentence on what it means. |
| 2.3 | No results yet — `implementation.py` written, not yet executed (reason). See [Known Issues](#known-issues). |
| 2.4 | **Metric = value**. Flag any anomaly honestly. |
| 2.5 | **Metric = value**. One sentence on what it means. |

## Highlights Gallery

<div align="center">

<table>
<tr>
<td width="50%"><img src="01-TODO_Folder/Image/TODO.png" alt="TODO alt text"/><br/><sub><b>2.1</b> — one-line caption</sub></td>
<td width="50%"><img src="02-TODO_Folder/Image/TODO.png" alt="TODO alt text"/><br/><sub><b>2.2</b> — one-line caption</sub></td>
</tr>
<tr>
<td width="50%"><img src="04-TODO_Folder/Image/TODO.png" alt="TODO alt text"/><br/><sub><b>2.4</b> — one-line caption</sub></td>
<td width="50%"><img src="05-TODO_Folder/Image/TODO.png" alt="TODO alt text"/><br/><sub><b>2.5</b> — one-line caption</sub></td>
</tr>
</table>

</div>

> Spaces in image filenames must be URL-encoded (`%20`) in `src`.

## Getting Started

Repo-wide setup, from the repository root:

```bash
git clone https://github.com/Ayush-2703/Large_languaage_Models.git
cd Large_languaage_Models

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
cd 02-Generative_AI_with_LLMs
```

Each topic's `implementation.py` is self-contained and runs standalone:

```bash
cd 01-TODO_Folder && python implementation.py && cd ..    # local, CPU is fine
cd 02-TODO_Folder && python implementation.py && cd ..    # Colab T4 GPU recommended
```

Topics needing a GPU or Hugging Face Hub access (TODO: list them):

```bash
# Google Colab → Runtime → Change runtime type → T4 GPU
cd 0X-TODO_Folder && python implementation.py
```

Optional flags near the top of `implementation.py`:

```python
RUN_PRETRAINED_SECTION = False   # set True on Colab with internet access
```

## Known Issues

<!-- Only phase-specific issues. Repo-wide bugs (stale root links, progress tracker) go in the root README or an issue. -->

- **TODO issue title.** What is wrong, what is affected, and what fixes it.
- **TODO issue title.** What is wrong, what is affected, and what fixes it.

## Key References

**Core texts** (paraphrased throughout this phase's theory docs, cited by title and author, never block-quoted)

1. TODO, Author. *Title.*
2. TODO, Author. *Title.*

**Topic 2.1 — TODO**

3. Author, et al. "Paper Title." *Venue*, Year.

**Topic 2.2 — TODO**

4. Author, et al. "Paper Title." *Venue*, Year.

**Topic 2.3 — TODO**

5. Author, et al. "Paper Title." *Venue*, Year.

**Topic 2.4 — TODO**

6. Author, et al. "Paper Title." *Venue*, Year.

**Topic 2.5 — TODO**

7. Author, et al. "Paper Title." *Venue*, Year.

## Navigate

⬅ [Phase 01 — Review of Fundamental LLMs](../01-Fundamental_LLMs) · [Repository root](../README.md) · ➡ [Phase 03 — TODO](../03-TODO)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:18181B,100:3F3F46&height=100&section=footer" width="100%"/>

</div>
