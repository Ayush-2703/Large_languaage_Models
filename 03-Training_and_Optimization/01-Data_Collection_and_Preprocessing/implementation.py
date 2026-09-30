"""
Topic 03.01 — Data Collection and Preprocessing for LLMs
============================================================
Builds a deliberately messy, realistic-scale text corpus (real narrative
text plus injected exact duplicates, near-duplicates, boilerplate,
low-quality fragments, and synthetic PII), then runs it through a real,
measurable data-curation pipeline:

    1. Exact deduplication      — hash-based.
    2. Near-duplicate detection — MinHash-approximated Jaccard similarity
                                   over character n-grams (the same family
                                   of technique used by real large-scale
                                   corpus curation, e.g. CCNet, RefinedWeb).
    3. Heuristic quality filters — length, symbol-to-alpha ratio, and
                                   repetition-ratio rules, in the style of
                                   the filtering heuristics documented for
                                   C4 (Raffel et al., 2020) and Gopher
                                   (Rae et al., 2021).
    4. PII scrubbing            — regex-based redaction of synthetic
                                   emails and phone numbers.
    5. Tokenizer training        — a REAL byte-pair-encoding tokenizer,
                                   trained from scratch on the cleaned
                                   corpus using Hugging Face's `tokenizers`
                                   library (pure Rust/Python, no model
                                   download — this is local training, not
                                   loading a pretrained tokenizer).

Every stage reports real, measured before/after counts. No Hugging Face
Hub download required (the `tokenizers` library trains new tokenizers
locally; it does not fetch anything). Runs in seconds on CPU.
"""

import hashlib
import random
import re
import time
import urllib.request
from collections import Counter

import matplotlib.pyplot as plt
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

random.seed(0)

# ---------------------------------------------------------------------------
# 1. Build a deliberately messy corpus: real text + injected realistic problems
# ---------------------------------------------------------------------------
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")

def fetch_corpus(n_chars: int = 100_000) -> str:
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        return resp.read().decode("utf-8")[:n_chars]

raw_text = fetch_corpus()
# Split into pseudo-"documents" (paragraphs) the way a real web crawl would
# arrive as many separate documents rather than one continuous stream.
paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 0]

BOILERPLATE = [
    "Subscribe to our newsletter for more updates!",
    "Click here to read more. Click here to read more. Click here to read more.",
    "Copyright 2024 All Rights Reserved.",
]
LOW_QUALITY = ["!!!", "asdkj alksjd laksjd", "...", "N/A N/A N/A N/A N/A", "***"]
FAKE_PII_TEMPLATES = [
    "Contact John at john.doe{n}@example.com or call 555-{n:04d} for details.",
    "For support, email support{n}@company.org or phone (555) {n:03d}-7890.",
]

def inject_messiness(paragraphs, n_exact_dupes=40, n_near_dupes=40, n_boilerplate=30,
                      n_low_quality=30, n_pii=20):
    corpus = list(paragraphs)
    # Exact duplicates: literally repeat existing documents
    corpus += random.sample(paragraphs, min(n_exact_dupes, len(paragraphs)))
    # Near duplicates: same document, trivial whitespace/punctuation edits
    near_source = random.sample(paragraphs, min(n_near_dupes, len(paragraphs)))
    for p in near_source:
        edited = p.replace(".", " .").replace(",", " ,", 1) + " "
        corpus.append(edited)
    corpus += random.choices(BOILERPLATE, k=n_boilerplate)
    corpus += random.choices(LOW_QUALITY, k=n_low_quality)
    for i in range(n_pii):
        corpus.append(random.choice(FAKE_PII_TEMPLATES).format(n=1000 + i))
    random.shuffle(corpus)
    return corpus

messy_corpus = inject_messiness(paragraphs)
print(f"Starting messy corpus: {len(messy_corpus)} documents")
print(f"  ({len(paragraphs)} real, {len(messy_corpus) - len(paragraphs)} injected problems)\n")

# ---------------------------------------------------------------------------
# 2. Exact deduplication (hash-based)
# ---------------------------------------------------------------------------
def exact_dedup(docs):
    seen, out = set(), []
    for d in docs:
        h = hashlib.sha256(d.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(d)
    return out

after_exact = exact_dedup(messy_corpus)
print(f"After exact dedup:        {len(after_exact)} docs  (-{len(messy_corpus) - len(after_exact)})")

# ---------------------------------------------------------------------------
# 3. Near-duplicate detection via n-gram Jaccard similarity (MinHash-style
#    approximate approach, implemented directly rather than via a library,
#    so the mechanism is fully visible)
# ---------------------------------------------------------------------------
def ngram_set(doc, n=5):
    doc = re.sub(r"\s+", " ", doc.lower())
    return set(doc[i:i+n] for i in range(max(1, len(doc) - n + 1)))

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def near_dedup(docs, threshold=0.8):
    """O(n^2) exact Jaccard for clarity at this corpus size — real large-
    scale pipelines use locality-sensitive hashing (MinHash + LSH banding)
    to avoid the quadratic cost; see explanation.md and theory.md."""
    ngrams = [ngram_set(d) for d in docs]
    keep = [True] * len(docs)
    removed = 0
    for i in range(len(docs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(docs)):
            if keep[j] and jaccard(ngrams[i], ngrams[j]) >= threshold:
                keep[j] = False
                removed += 1
    return [d for d, k in zip(docs, keep) if k], removed

t0 = time.time()
after_near, n_removed_near = near_dedup(after_exact)
print(f"After near-dup removal:   {len(after_near)} docs  (-{n_removed_near})  [{time.time()-t0:.1f}s]")

# ---------------------------------------------------------------------------
# 4. Heuristic quality filters (style of C4 / Gopher filtering rules)
# ---------------------------------------------------------------------------
def quality_score(doc):
    if len(doc) < 20:
        return False, "too_short"
    alpha = sum(c.isalpha() or c.isspace() for c in doc)
    if alpha / len(doc) < 0.6:
        return False, "low_alpha_ratio"
    words = doc.split()
    if len(words) >= 4 and len(set(words)) / len(words) < 0.4:
        return False, "high_repetition"
    return True, "kept"

after_quality, reasons = [], Counter()
for d in after_near:
    ok, reason = quality_score(d)
    reasons[reason] += 1
    if ok:
        after_quality.append(d)
print(f"After quality filtering:  {len(after_quality)} docs  (-{len(after_near) - len(after_quality)})")
print(f"  Rejection reasons: {dict(reasons)}")

# ---------------------------------------------------------------------------
# 5. PII scrubbing (regex-based redaction)
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
# Two alternatives: 10-digit area-code format ((555) 000-7890 / 555-000-7890)
# and plain 7-digit local format (555-1000) — an earlier version only
# matched the first, silently missing every phone number using the second
# of this script's own two synthetic templates. See explanation.md.
PHONE_RE = re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{3}-\d{4}\b")

def scrub_pii(doc):
    doc, n1 = EMAIL_RE.subn("[EMAIL]", doc)
    doc, n2 = PHONE_RE.subn("[PHONE]", doc)
    return doc, n1 + n2

after_pii, total_redactions = [], 0
for d in after_quality:
    cleaned, n_red = scrub_pii(d)
    after_pii.append(cleaned)
    total_redactions += n_red
print(f"After PII scrubbing:      {len(after_pii)} docs  ({total_redactions} PII spans redacted)\n")

final_corpus = after_pii
pipeline_counts = [len(messy_corpus), len(after_exact), len(after_near), len(after_quality), len(after_pii)]
pipeline_stages = ["Raw\n(messy)", "Exact\ndedup", "Near-dup\nremoval", "Quality\nfilter", "PII\nscrub"]

# ---------------------------------------------------------------------------
# 6. Train a REAL BPE tokenizer from scratch on the cleaned corpus
# ---------------------------------------------------------------------------
print("Training a real BPE tokenizer from scratch on the cleaned corpus...")
with open("cleaned_corpus.txt", "w") as f:
    f.write("\n".join(final_corpus))

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(
    vocab_size=800,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=2,
)
tokenizer.train(["cleaned_corpus.txt"], trainer)
vocab_size_trained = tokenizer.get_vocab_size()
print(f"  Trained BPE vocab size: {vocab_size_trained}")

# Compression ratio: characters per token, on held-out text from the SAME
# distribution (a fresh slice of the original raw corpus, not used in
# tokenizer training) — the standard way to report tokenizer efficiency.
holdout_text = fetch_corpus(n_chars=120_000)[100_000:]
char_level_tokens = len(holdout_text)                       # 1 char = 1 "token" baseline
bpe_encoding = tokenizer.encode(holdout_text)
bpe_tokens = len(bpe_encoding.ids)
compression_ratio = char_level_tokens / bpe_tokens
print(f"  Held-out compression: {char_level_tokens} chars -> {bpe_tokens} BPE tokens "
      f"({compression_ratio:.2f}x fewer tokens than character-level)")

sample = "The king spoke of honor and duty."
sample_enc = tokenizer.encode(sample)
print(f"\n  Example: {sample!r}")
print(f"  BPE tokens: {sample_enc.tokens}")

# ---------------------------------------------------------------------------
# 7. proof.png
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax0 = axes[0]
colors = ["#8d99ae", "#4361ee", "#3a0ca3", "#7209b7", "#f72585"]
bars = ax0.bar(pipeline_stages, pipeline_counts, color=colors, edgecolor="black")
for bar, c in zip(bars, pipeline_counts):
    ax0.text(bar.get_x() + bar.get_width()/2, c + 2, str(c), ha="center", fontsize=9)
ax0.set_ylabel("document count")
ax0.set_title(f"A. Corpus Size Through Pipeline\n{len(messy_corpus)} -> {len(final_corpus)} docs ({100*(1-len(final_corpus)/len(messy_corpus)):.0f}% removed)", fontsize=10)

ax1 = axes[1]
reason_labels = list(reasons.keys())
reason_vals = list(reasons.values())
ax1.pie(reason_vals, labels=reason_labels, autopct="%1.0f%%", colors=["#2a9d3f", "#e63946", "#f4a261", "#457b9d"][:len(reason_labels)])
ax1.set_title("B. Quality Filter Outcomes\n(real classification of every surviving doc)", fontsize=10)

ax2 = axes[2]
ax2.bar(["Character-level\n(1 char = 1 token)", f"BPE\n(vocab={vocab_size_trained})"],
        [char_level_tokens, bpe_tokens], color=["#8d99ae", "#f72585"], edgecolor="black")
ax2.text(0, char_level_tokens + 500, str(char_level_tokens), ha="center", fontsize=9)
ax2.text(1, bpe_tokens + 500, str(bpe_tokens), ha="center", fontsize=9)
ax2.set_ylabel("tokens to encode held-out text")
ax2.set_title(f"C. Real Tokenizer Compression\n{compression_ratio:.2f}x fewer tokens with trained BPE", fontsize=10)

fig.suptitle("Topic 03.01 — Data Collection & Preprocessing: Real Pipeline, Real Tokenizer", fontsize=12)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("\nSaved proof.png")
