"""
Topic 04.02 — Integrating External Knowledge into LLMs
===========================================================
Two real, complementary mechanisms for bringing external knowledge into a
model's answers, beyond the dense-retrieval RAG pipeline in 04.01:

  (A) SPARSE vs. DENSE RETRIEVAL — a real BM25 implementation (Robertson &
      Zaparyaniuk et al.'s probabilistic ranking function, the standard
      classical IR baseline) compared against 04.01's dense dual-encoder,
      on the SAME held-out-entity generalization test. BM25 needs no
      training at all — it computes term-statistics directly over whatever
      corpus it's given — so it is expected to generalize to entirely new
      entities in a way a small trained neural retriever, per 04.01's
      finding, does not. This is measured directly, not assumed.

  (B) TOOL USE / FUNCTION CALLING — a model trained to recognize when a
      query needs an external lookup (emitting a [CALL_LOOKUP] token)
      versus when it can answer directly from general knowledge, rather
      than always answering from parametric memory (which risks
      hallucination) or always retrieving (which is wasteful for queries
      that don't need it).

No Hugging Face Hub download required. Runs in about a minute on CPU.
"""

import math
import random
import re
import time
import urllib.request
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import faiss

torch.manual_seed(0)
random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Knowledge base — same design principles as 04.01: shared category
#    labels (not 1-to-1 with entities) and a genuine held-out split, applied
#    from the start this time rather than discovered through debugging.
# ---------------------------------------------------------------------------
FACTS = [
    ("the Plimwick device", "The Plimwick device converts sound waves into stored energy for later use.", 0),
    ("the Kestrel Protocol", "The Kestrel Protocol is a data encryption standard used by orbital stations.", 0),
    ("the Corvane Filter", "The Corvane Filter removes microplastics from seawater using magnetic resonance.", 0),
    ("the Wrenfield Loom", "The Wrenfield Loom weaves conductive fabric for wearable electronics.", 0),
    ("the Thistledown Engine", "The Thistledown Engine reduces fuel consumption by recycling exhaust heat.", 0),
    ("the Anchorwave Beacon", "The Anchorwave Beacon guides autonomous vessels through storm conditions.", 0),
    ("Dr. Elena Vasquez", "Dr. Elena Vasquez pioneered research in quantum biology at Mirenna Institute.", 1),
    ("Professor Tamsin Reyk", "Professor Tamsin Reyk discovered the Halvorin particle in twenty forty one.", 1),
    ("Engineer Priya Solan", "Engineer Priya Solan redesigned the cooling systems used in orbital greenhouses.", 1),
    ("Historian Wendell Ashby", "Historian Wendell Ashby catalogued lost records from the early colonial period.", 1),
    ("Dr. Marcus Feld", "Dr. Marcus Feld developed a vaccine platform used across three colonies.", 1),
    ("Analyst Roran Kade", "Analyst Roran Kade identified trade patterns linking four separate colonies.", 1),
    ("Blornak Industries", "Blornak Industries manufactures self repairing hull plating for starships.", 2),
    ("Novask Biotech", "Novask Biotech develops synthetic coral for reef restoration projects.", 2),
    ("the Tessender Group", "The Tessender Group supplies water purification systems to remote colonies.", 2),
    ("Marrow Fabrication", "Marrow Fabrication produces lightweight alloys for spacecraft frames.", 2),
    ("Ovren Dynamics", "Ovren Dynamics builds modular habitats for asteroid mining crews.", 2),
    ("Hollowreach Logistics", "Hollowreach Logistics manages cargo transport between outer settlements.", 2),
    ("the Vindalor Accord", "The Vindalor Accord established trade rules between six lunar colonies.", 3),
    ("the Ashgrove Treaty", "The Ashgrove Treaty banned autonomous weapons research across the federation.", 3),
    ("the Meridian Compact", "The Meridian Compact sets shared safety standards for mining operations.", 3),
    ("the Osprey Directive", "The Osprey Directive requires environmental review before new settlements.", 3),
    ("the Halcyon Registry", "The Halcyon Registry tracks ownership records for deep space vessels.", 3),
    ("the Fenwick Charter", "The Fenwick Charter guarantees minimum resource shares for colony workers.", 3),
    ("Zorvath Corp", "Zorvath Corp was founded in 2087 on the orbital station Kestrel Nine.", 4),
    ("the Ferrowing Array", "The Ferrowing Array is a telescope network that maps deep space anomalies.", 4),
    ("the Solmere Station", "The Solmere Station serves as a refueling waypoint between inner colonies.", 4),
    ("the Brackwater Vessel", "The Brackwater Vessel was the first ship to cross the outer debris belt.", 4),
    ("Captain Idris Malen", "Captain Idris Malen commanded the first crewed mission past the Kuiper belt.", 4),
    ("Commander Yusuf Vantrel", "Commander Yusuf Vantrel led the evacuation of the Halden outpost.", 4),
]
N_FACTS = len(FACTS)
CATEGORY_NAMES = ["TECH", "PERSON", "ORG", "LAW", "SPACE"]
SEEN_FACTS = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21, 24, 25, 26, 27]
HELD_OUT_FACTS = [4, 5, 10, 11, 16, 17, 22, 23, 28, 29]
TEMPLATES = ["Tell me about {e}.", "What do you know about {e}?", "Can you describe {e}?",
             "Explain {e} to me.", "Give me information on {e}.", "What can you say about {e}?"]

CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")

def load_distractors(n_chars=60_000, chunk_len=90):
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        text = resp.read().decode("utf-8")[:n_chars]
    chunks = [text[i:i+chunk_len].replace("\n", " ").strip() for i in range(0, len(text), chunk_len)]
    return [c for c in chunks if len(c) > 40][:200]

distractors = load_distractors()
documents = [f[1] for f in FACTS] + distractors
print(f"Knowledge base: {N_FACTS} facts ({len(SEEN_FACTS)} seen + {len(HELD_OUT_FACTS)} held-out) "
      f"+ {len(distractors)} distractors = {len(documents)} total\n")

def make_query(fact_row):
    return random.choice(TEMPLATES).format(e=FACTS[fact_row][0])

# ===========================================================================
# (A) BM25 — a real, from-scratch implementation, NO training required
# ===========================================================================
print("=" * 70)
print("(A) BM25 (sparse, statistical, non-learned) vs. Dense Retrieval")
print("=" * 70)

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

class BM25:
    """Robertson & Walker's BM25 ranking function, computed directly from
    term frequencies — no training, no learned parameters. k1 and b are the
    standard tunable constants controlling term-frequency saturation and
    document-length normalization, not learned from data."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tokenized = [tokenize(d) for d in docs]
        self.doc_lens = [len(d) for d in self.tokenized]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)
        self.N = len(docs)
        df = Counter()
        for doc in self.tokenized:
            for term in set(doc):
                df[term] += 1
        self.idf = {term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()}

    def score(self, query, doc_idx):
        q_terms = tokenize(query)
        doc_terms = self.tokenized[doc_idx]
        tf = Counter(doc_terms)
        dl = self.doc_lens[doc_idx]
        total = 0.0
        for term in q_terms:
            if term not in self.idf:
                continue
            freq = tf.get(term, 0)
            numer = freq * (self.k1 + 1)
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += self.idf[term] * (numer / denom if denom > 0 else 0)
        return total

    def search(self, query, k=3):
        scores = [self.score(query, i) for i in range(self.N)]
        top_k = np.argsort(scores)[::-1][:k]
        return top_k, [scores[i] for i in top_k]

bm25 = BM25(documents)

def bm25_eval(fact_rows):
    top1, top3 = 0, 0
    for f in fact_rows:
        query = make_query(f)
        top_idx, _ = bm25.search(query, k=3)
        top1 += top_idx[0] == f
        top3 += f in top_idx
    return top1 / len(fact_rows), top3 / len(fact_rows)

bm25_seen_top1, bm25_seen_top3 = bm25_eval(SEEN_FACTS)
bm25_ho_top1, bm25_ho_top3 = bm25_eval(HELD_OUT_FACTS)
print(f"BM25 (no training at all):")
print(f"  Seen entities:      top-1={bm25_seen_top1:.3f}  top-3={bm25_seen_top3:.3f}")
print(f"  Held-out entities:  top-1={bm25_ho_top1:.3f}  top-3={bm25_ho_top3:.3f}\n")

# --- Dense retriever: same architecture/training recipe as 04.01 ---
all_text = " ".join(documents) + " ".join(TEMPLATES)
chars = sorted(set(all_text))
SPECIALS = ["[PAD]", "[CLS]"]
vocab = SPECIALS + chars
stoi = {c: i for i, c in enumerate(vocab)}
VOCAB = len(vocab)
PAD, CLS = stoi["[PAD]"], stoi["[CLS]"]
BLOCK = 96

def encode(s, max_len=BLOCK):
    ids = [CLS] + [stoi.get(c, PAD) for c in s[:max_len-1]]
    return torch.tensor(ids + [PAD]*(max_len-len(ids)), dtype=torch.long)

D_MODEL, N_HEADS, N_LAYERS = 64, 4, 2

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_heads
        q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, hd).transpose(1, 2)
        attn_mask = pad_mask[:, None, None, :].expand(B, self.n_heads, T, T) if pad_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        return x + self.mlp(self.ln2(x))

class Encoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, d_model)
        self.pos_embed = nn.Embedding(BLOCK, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, ids, pad_mask=None):
        B, T = ids.shape
        h = self.tok_embed(ids) + self.pos_embed(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            h = blk(h, pad_mask)
        return self.ln_f(h)[:, 0, :]

print("Training dense dual-encoder retriever (same recipe as Topic 04.01)...")
retriever = Encoder().to(DEVICE)
opt = torch.optim.AdamW(retriever.parameters(), lr=1e-3)
for step in range(600):
    batch_facts = random.sample(SEEN_FACTS, 16)
    q_ids = torch.stack([encode(make_query(f)) for f in batch_facts]).to(DEVICE)
    d_ids = torch.stack([encode(FACTS[f][1]) for f in batch_facts]).to(DEVICE)
    q_emb = F.normalize(retriever(q_ids, (q_ids != PAD).float()), dim=-1)
    d_emb = F.normalize(retriever(d_ids, (d_ids != PAD).float()), dim=-1)
    sim = q_emb @ d_emb.T * 10.0
    loss = F.cross_entropy(sim, torch.arange(len(batch_facts), device=DEVICE))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

retriever.eval()
with torch.no_grad():
    doc_ids = torch.stack([encode(d) for d in documents]).to(DEVICE)
    doc_emb = F.normalize(retriever(doc_ids, (doc_ids != PAD).float()), dim=-1).cpu().numpy().astype("float32")
index = faiss.IndexFlatIP(D_MODEL)
index.add(doc_emb)

def dense_eval(fact_rows):
    top1, top3 = 0, 0
    with torch.no_grad():
        for f in fact_rows:
            q_emb = F.normalize(retriever(encode(make_query(f)).unsqueeze(0).to(DEVICE)), dim=-1).cpu().numpy().astype("float32")
            _, idx = index.search(q_emb, 3)
            top1 += idx[0][0] == f
            top3 += f in idx[0]
    return top1 / len(fact_rows), top3 / len(fact_rows)

dense_seen_top1, dense_seen_top3 = dense_eval(SEEN_FACTS)
dense_ho_top1, dense_ho_top3 = dense_eval(HELD_OUT_FACTS)
print(f"Dense dual-encoder (trained on SEEN facts only):")
print(f"  Seen entities:      top-1={dense_seen_top1:.3f}  top-3={dense_seen_top3:.3f}")
print(f"  Held-out entities:  top-1={dense_ho_top1:.3f}  top-3={dense_ho_top3:.3f}\n")

# ===========================================================================
# (B) TOOL USE — learn WHEN to call an external lookup vs. answer directly
# ===========================================================================
print("=" * 70)
print("(B) Tool Use: learning when a query needs external lookup")
print("=" * 70)

GENERAL_TEMPLATES = ["What is two plus two?", "Say hello to me.", "What color is the sky?",
                     "Count from one to three.", "What is the opposite of hot?", "Repeat the word yes."]

def build_tool_examples(n=300):
    examples = []
    for _ in range(n):
        if random.random() < 0.5:
            f = random.choice(SEEN_FACTS + HELD_OUT_FACTS)
            examples.append((make_query(f), 1))     # 1 = NEEDS lookup
        else:
            examples.append((random.choice(GENERAL_TEMPLATES), 0))  # 0 = answer directly
    return examples

tool_examples = build_tool_examples()
tool_train, tool_val = tool_examples[:240], tool_examples[240:]

class ToolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(D_MODEL, 2)

    def forward(self, ids, pad_mask=None):
        return self.head(self.encoder(ids, pad_mask))

tool_model = ToolClassifier().to(DEVICE)
opt = torch.optim.AdamW(tool_model.parameters(), lr=1e-3)
for step in range(200):
    batch = random.sample(tool_train, 16)
    ids = torch.stack([encode(q) for q, _ in batch]).to(DEVICE)
    pad_mask = (ids != PAD).float().to(DEVICE)
    y = torch.tensor([lbl for _, lbl in batch], device=DEVICE)
    loss = F.cross_entropy(tool_model(ids, pad_mask), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

tool_model.eval()
with torch.no_grad():
    ids = torch.stack([encode(q) for q, _ in tool_val]).to(DEVICE)
    pad_mask = (ids != PAD).float().to(DEVICE)
    y = torch.tensor([lbl for _, lbl in tool_val], device=DEVICE)
    preds = tool_model(ids, pad_mask).argmax(-1)
    tool_acc = (preds == y).float().mean().item()
    tp = int(((preds == 1) & (y == 1)).sum()); fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum()); fp = int(((preds == 1) & (y == 0)).sum())
print(f"Tool-use routing accuracy: {tool_acc:.3f}  (TP={tp} TN={tn} FP={fp} FN={fn})")
print("(FP = called lookup unnecessarily; FN = should have looked up but tried to answer directly)\n")

# ===========================================================================
# proof.png
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax0 = axes[0]
x = np.arange(2); w = 0.35
ax0.bar(x - w/2, [bm25_seen_top1, bm25_ho_top1], w, label="BM25 (sparse)", color="#3a0ca3", edgecolor="black")
ax0.bar(x + w/2, [dense_seen_top1, dense_ho_top1], w, label="Dense (neural)", color="#f72585", edgecolor="black")
ax0.set_xticks(x); ax0.set_xticklabels(["Seen\nentities", "Held-out\nentities"])
for i, v in enumerate([bm25_seen_top1, bm25_ho_top1]):
    ax0.text(i - w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate([dense_seen_top1, dense_ho_top1]):
    ax0.text(i + w/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax0.set_ylim(0, 1.05); ax0.legend(fontsize=8)
ax0.set_ylabel("top-1 retrieval accuracy")
ax0.set_title("A. Sparse (BM25) vs. Dense Retrieval\n(BM25 needs no training at all)", fontsize=10)

ax1 = axes[1]
ax1.bar(["Top-1", "Top-3"], [bm25_ho_top1, bm25_ho_top3], color="#3a0ca3", edgecolor="black", alpha=0.6, label="BM25")
ax1.bar(["Top-1", "Top-3"], [dense_ho_top1, dense_ho_top3], color="#f72585", edgecolor="black", alpha=0.6, label="Dense", width=0.5)
ax1.set_ylim(0, 1.05); ax1.legend(fontsize=8)
ax1.set_ylabel("held-out-entity accuracy")
ax1.set_title("B. Held-Out Generalization, Zoomed In\n(BM25's non-learned scoring transfers for free)", fontsize=10)

ax2 = axes[2]
cm = np.array([[tn, fp], [fn, tp]])
im = ax2.imshow(cm, cmap="Purples")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                  color="white" if cm[i, j] > cm.max()/2 else "black")
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Pred: Direct", "Pred: Lookup"])
ax2.set_yticks([0, 1]); ax2.set_yticklabels(["True: Direct", "True: Lookup"])
ax2.set_title(f"C. Tool-Use Routing (real)\naccuracy={tool_acc:.3f}", fontsize=10)

fig.suptitle("Topic 04.02 — Integrating External Knowledge: Sparse vs. Dense Retrieval + Real Tool-Use Routing", fontsize=11)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("Saved proof.png")
