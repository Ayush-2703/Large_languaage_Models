"""
Topic 01.05 — Ethical Considerations in Large-Scale AI
==========================================================
Demonstrates, with a real trained model rather than an assertion, the
central mechanism behind representational bias in language models: a model
trained on text learns *whatever statistical associations are present in
that text*, including demographic ones the designer never explicitly
specified.

Part A (network-free, executed in this repository): train small word
embeddings from scratch via skip-gram with negative sampling on a synthetic
corpus with a KNOWN, controlled gender/occupation co-occurrence pattern
built in deliberately. Then run a WEAT-style (Word Embedding Association
Test, Caliskan et al., 2017) association measurement on the resulting
embeddings — because the co-occurrence pattern is fully known and
controlled, any measured association can be attributed directly to the data
statistics, not to a black-box pretrained model's opaque history.

Part B (Colab-only, requires Hugging Face Hub): the same style of probe,
run against a real pretrained DistilBERT via its fill-mask head, to connect
the controlled toy demonstration to a production-scale model.

No download required for Part A. Runs in seconds on CPU.
"""

import math
import random
import itertools
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

torch.manual_seed(0)
random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. A synthetic corpus with a deliberately controlled co-occurrence bias.
#    Male-coded and female-coded terms are seeded into "career" and "family"
#    sentence templates at DIFFERENT rates, chosen here, by design, so the
#    resulting bias is fully attributable to a known cause rather than
#    inherited unknowably from some real-world scraped corpus.
# ---------------------------------------------------------------------------
MALE_TERMS = ["he", "man", "father", "king", "brother", "son"]
FEMALE_TERMS = ["she", "woman", "mother", "queen", "sister", "daughter"]

CAREER_WORDS = ["engineer", "executive", "scientist", "president", "doctor", "programmer"]
FAMILY_WORDS = ["caregiver", "homemaker", "nurturer", "cook", "housekeeper", "nanny"]

CAREER_TEMPLATES = ["the {p} is a skilled {c}", "the {p} works as a {c} downtown",
                     "as a {c}, the {p} leads the team", "the {p} became a respected {c}"]
FAMILY_TEMPLATES = ["the {p} is a devoted {f}", "the {p} works as a {f} at home",
                     "as a {f}, the {p} cares for the family", "the {p} became a patient {f}"]

def build_corpus(n_sentences: int = 4000, male_career_rate: float = 0.85) -> list:
    """male_career_rate controls how often MALE_TERMS appear in career (vs
    family) sentences; female terms get the complementary rate. This single
    number IS the injected bias — it is reported alongside the measured
    result below so the cause and the measured effect sit side by side."""
    sentences = []
    for _ in range(n_sentences):
        is_career = random.random() < 0.5
        if is_career:
            template = random.choice(CAREER_TEMPLATES)
            use_male = random.random() < male_career_rate
            person = random.choice(MALE_TERMS if use_male else FEMALE_TERMS)
            sentence = template.format(p=person, c=random.choice(CAREER_WORDS))
        else:
            template = random.choice(FAMILY_TEMPLATES)
            use_male = random.random() < (1 - male_career_rate)
            person = random.choice(MALE_TERMS if use_male else FEMALE_TERMS)
            sentence = template.format(p=person, f=random.choice(FAMILY_WORDS))
        sentences.append(sentence.split())
    return sentences

MALE_CAREER_RATE = 0.85
corpus = build_corpus(male_career_rate=MALE_CAREER_RATE)

vocab = sorted(set(w for sent in corpus for w in sent))
stoi = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(f"Synthetic corpus: {len(corpus)} sentences, {vocab_size} unique words")
print(f"Injected bias: male-coded terms appear in career sentences {MALE_CAREER_RATE:.0%} of the time")
print(f"               (female-coded terms fill the complementary {1-MALE_CAREER_RATE:.0%})\n")

# ---------------------------------------------------------------------------
# 2. Skip-gram training pairs with negative sampling
# ---------------------------------------------------------------------------
WINDOW = 3

def make_skipgram_pairs(corpus):
    pairs = []
    for sent in corpus:
        ids = [stoi[w] for w in sent]
        for i, center in enumerate(ids):
            for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
                if i != j:
                    pairs.append((center, ids[j]))
    return pairs

pairs = make_skipgram_pairs(corpus)
word_freq = Counter(w for sent in corpus for w in sent)
unigram_probs = np.array([word_freq[w] for w in vocab], dtype=np.float64) ** 0.75
unigram_probs /= unigram_probs.sum()

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.in_embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center, context, negatives):
        v_c = self.in_embed(center)                       # (B, D)
        v_o = self.out_embed(context)                      # (B, D)
        v_neg = self.out_embed(negatives)                   # (B, K, D)

        pos_score = torch.sum(v_c * v_o, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)   # (B, K)
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)

        return -(pos_loss + neg_loss).mean()

model = SkipGramNS(vocab_size, embed_dim=32).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)

K_NEGATIVES = 5
BATCH_SIZE = 128
EPOCHS = 8

pairs_arr = np.array(pairs)
n_batches_per_epoch = len(pairs_arr) // BATCH_SIZE

for epoch in range(EPOCHS):
    np.random.shuffle(pairs_arr)
    total_loss = 0.0
    for b in range(n_batches_per_epoch):
        batch = pairs_arr[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        centers = torch.tensor(batch[:, 0], dtype=torch.long, device=DEVICE)
        contexts = torch.tensor(batch[:, 1], dtype=torch.long, device=DEVICE)
        neg_ids = np.random.choice(vocab_size, size=(len(batch), K_NEGATIVES), p=unigram_probs)
        negatives = torch.tensor(neg_ids, dtype=torch.long, device=DEVICE)

        loss = model(centers, contexts, negatives)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"  epoch {epoch+1}/{EPOCHS}  avg_loss={total_loss/n_batches_per_epoch:.4f}")

embeddings = model.in_embed.weight.detach().cpu().numpy()

# ---------------------------------------------------------------------------
# 3. WEAT-style association measurement (Caliskan et al., 2017)
# ---------------------------------------------------------------------------
def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def mean_cos_sim(word, word_set):
    v = embeddings[stoi[word]]
    return np.mean([cos_sim(v, embeddings[stoi[w]]) for w in word_set if w in stoi])

def association(word, male_set, female_set):
    """s(w, A, B) = mean_sim(w, A) - mean_sim(w, B) — the core WEAT statistic
    per target word, here A=male-coded terms, B=female-coded terms."""
    return mean_cos_sim(word, male_set) - mean_cos_sim(word, female_set)

career_assoc = [association(w, MALE_TERMS, FEMALE_TERMS) for w in CAREER_WORDS if w in stoi]
family_assoc = [association(w, MALE_TERMS, FEMALE_TERMS) for w in FAMILY_WORDS if w in stoi]

mean_career = float(np.mean(career_assoc))
mean_family = float(np.mean(family_assoc))
pooled_std = float(np.std(career_assoc + family_assoc))
effect_size = (mean_career - mean_family) / (pooled_std + 1e-8)   # Cohen's-d-style WEAT effect size

print(f"\nWEAT-style measured association (male-coded minus female-coded cosine similarity):")
print(f"  career words:  {mean_career:+.4f}  (positive = more associated with male-coded terms)")
print(f"  family words:  {mean_family:+.4f}")
print(f"  effect size (d): {effect_size:+.3f}")
print(f"\nThis association was NOT hand-coded into the model. It was learned purely from")
print(f"the {MALE_CAREER_RATE:.0%}/{1-MALE_CAREER_RATE:.0%} co-occurrence rate built into the synthetic corpus —")
print(f"the exact same mechanism (statistical co-occurrence -> learned embedding geometry)")
print(f"by which real word embeddings trained on real, uncontrolled text have been shown")
print(f"to encode real-world demographic biases (Bolukbasi et al., 2016; Caliskan et al., 2017).")

# ---------------------------------------------------------------------------
# 4. proof.png — composite: conceptual pipeline diagram (top) + real
#    measured association chart (bottom). This topic is conceptual, so per
#    this repository's convention (README.md, Repository Structure) the
#    proof image pairs a diagram with a real generated chart rather than
#    using either alone.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.35)

# --- Top: conceptual diagram — where bias enters the ML pipeline ---
ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(0, 10); ax0.set_ylim(0, 3); ax0.axis("off")
ax0.set_title("Where Bias Enters the ML Pipeline (conceptual)", fontsize=13, fontweight="bold", loc="left")

stages = [
    (1.2, "Data\nCollection", "#8d99ae", "Real-world text over-\nrepresents some groups\nand contexts"),
    (3.7, "Preprocessing /\nTokenization", "#4361ee", "Filtering and normalization\ncan erase or distort\ndialects, names"),
    (6.2, "Training\n(this demo)", "#3a0ca3", "Model learns whatever\nco-occurrence statistics\nare present, verbatim"),
    (8.8, "Deployment", "#f72585", "Biased outputs affect\nreal decisions at\nreal-world scale"),
]
for x, label, color, note in stages:
    box = mpatches.FancyBboxPatch((x - 0.9, 1.5), 1.8, 0.9, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax0.add_patch(box)
    ax0.text(x, 1.95, label, ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
    ax0.text(x, 1.15, note, ha="center", va="top", fontsize=7.8, color="#333333")

for i in range(len(stages) - 1):
    x0 = stages[i][0] + 0.9
    x1 = stages[i+1][0] - 0.9
    ax0.add_patch(FancyArrowPatch((x0, 1.95), (x1, 1.95), arrowstyle="-|>", mutation_scale=15, color="#333333"))

ax0.text(6.2, 0.55, "This topic's real experiment isolates the TRAINING stage: a synthetic\n"
                     "corpus with a fully known, controlled co-occurrence rate is the ONLY input —\n"
                     "any bias measured below is therefore directly attributable to that rate.",
         ha="center", fontsize=8.3, style="italic", color="#444444")

# --- Bottom: real measured WEAT-style association ---
ax1 = fig.add_subplot(gs[1])
x_pos = np.arange(2)
means = [mean_career, mean_family]
colors = ["#4361ee", "#f72585"]
bars = ax1.bar(x_pos, means, color=colors, edgecolor="black", width=0.5)
ax1.axhline(0, color="black", linewidth=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"Career words\n{CAREER_WORDS}", f"Family words\n{FAMILY_WORDS}"], fontsize=7.5)
ax1.set_ylabel("mean cosine-similarity association\n(male-coded terms − female-coded terms)", fontsize=9.5)
ax1.set_title(
    f"Real Measured Bias: WEAT-Style Association on Embeddings Trained From Scratch\n"
    f"Injected co-occurrence rate: {MALE_CAREER_RATE:.0%} male-coded in career sentences  ·  "
    f"Measured effect size d = {effect_size:+.2f}",
    fontsize=10,
)
for bar, m in zip(bars, means):
    ax1.text(bar.get_x() + bar.get_width()/2, m + (0.01 if m >= 0 else -0.02),
              f"{m:+.3f}", ha="center", fontsize=9, va="bottom" if m >= 0 else "top")
ax1.grid(axis="y", alpha=0.3)

fig.suptitle("Topic 01.05 — Ethical Considerations: Bias Is Learned, Not Assumed", fontsize=12, y=0.99)
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("\nSaved proof.png")

# ---------------------------------------------------------------------------
# 5. Real-world probe on a pretrained model — requires internet / Colab
# ---------------------------------------------------------------------------
RUN_PRETRAINED_SECTION = False   # set True on Colab with internet access

if RUN_PRETRAINED_SECTION:
    from transformers import pipeline

    unmasker = pipeline("fill-mask", model="distilbert-base-uncased")
    prompts = ["The engineer picked up [MASK] tools.", "The nurse picked up [MASK] tools."]
    for prompt in prompts:
        preds = unmasker(prompt)
        print(f"\n{prompt}")
        for p in preds[:5]:
            print(f"  {p['token_str']:12s}  {p['score']:.4f}")
else:
    print(
        "\n[Section 5 skipped locally: requires Hugging Face Hub access.]\n"
        "On Colab, set RUN_PRETRAINED_SECTION = True to run the same style of\n"
        "association probe against real pretrained DistilBERT via fill-mask,\n"
        "connecting this toy, fully-controlled demonstration to a production model."
    )
