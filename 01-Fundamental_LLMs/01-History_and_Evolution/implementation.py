"""
Topic 01.01 — History and Evolution of Language Models
=========================================================
Trains four language models that represent four eras of NLP history on the
*same* tiny character-level corpus, then compares them on held-out
perplexity. This turns "language models got better over time" into a single
reproducible number per architecture rather than a claim you have to take on
faith.

Eras represented:
    1. N-gram (count-based, pre-2000s statistical NLP)
    2. RNN (Elman, 1990s-2010s)
    3. LSTM (Hochreiter & Schmidhuber, 1997 — dominant 2014-2017)
    4. Transformer, decoder-only (Vaswani et al., 2017 — dominant today)

Runs in well under a minute on CPU; well under a minute on a Colab T4.
No Hugging Face Hub download is required — the point of this topic is the
*architectures themselves*, trained from identical random init on identical
data, so every checkpoint here is "pretrained" by this script, live.

Data: tiny-Shakespeare (Karpathy's char-rnn corpus), ~1.1MB of public-domain
text, fetched at runtime. Standard toy corpus for character-level LM demos.
"""

import math
import time
import urllib.request
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

def load_corpus(n_chars: int = 200_000) -> str:
    """Fetch tiny-Shakespeare and truncate to n_chars for a fast demo run."""
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        text = resp.read().decode("utf-8")
    return text[:n_chars]

text = load_corpus()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

BLOCK_SIZE = 48   # context length in characters
BATCH_SIZE = 32

def get_batch(source: torch.Tensor):
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ---------------------------------------------------------------------------
# 2. Four architectures, one per NLP era
# ---------------------------------------------------------------------------

class NGramLM:
    """Order-3 count-based language model with add-1 (Laplace) smoothing.
    No gradient descent — this is frequency counting, exactly how language
    modeling worked before neural nets took over."""

    def __init__(self, order: int = 3):
        self.order = order
        self.counts = defaultdict(lambda: defaultdict(int))

    def fit(self, ids: torch.Tensor):
        seq = ids.tolist()
        for i in range(len(seq) - self.order):
            ctx = tuple(seq[i:i + self.order - 1])
            nxt = seq[i + self.order - 1]
            self.counts[ctx][nxt] += 1

    @torch.no_grad()
    def eval_loss(self, ids: torch.Tensor) -> float:
        seq = ids.tolist()
        total_nll, n = 0.0, 0
        for i in range(len(seq) - self.order):
            ctx = tuple(seq[i:i + self.order - 1])
            nxt = seq[i + self.order - 1]
            row = self.counts.get(ctx, {})
            denom = sum(row.values()) + vocab_size          # add-1 smoothing
            numer = row.get(nxt, 0) + 1
            total_nll += -math.log(numer / denom)
            n += 1
        return total_nll / n


class RNNLM(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.RNN(hidden, hidden, batch_first=True, nonlinearity="tanh")
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        h, _ = self.rnn(self.embed(x))
        return self.head(h)


class LSTMLM(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        h, _ = self.lstm(self.embed(x))
        return self.head(h)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_heads
        q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, hd).transpose(1, 2)
        # causal=True enforces the same masked QK^T/sqrt(d) softmax V used in
        # GPT-style decoders (see Topic 01.02 for the manual derivation).
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, d_model: int = 96, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


# ---------------------------------------------------------------------------
# 3. Train the three neural models identically; fit the N-gram model directly
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_neural(model: nn.Module, steps: int = 800, lr: float = 3e-3) -> float:
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        xb, yb = get_batch(train_data)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(20):
            xb, yb = get_batch(val_data)
            logits = model(xb)
            losses.append(F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)).item())
    return sum(losses) / len(losses)


results = {}
t0 = time.time()

ngram = NGramLM(order=3)
ngram.fit(train_data)
results["N-gram\n(order-3)"] = {
    "loss": ngram.eval_loss(val_data[:5000]),
    "params": len(ngram.counts),
    "era": "Pre-2003, statistical NLP",
}

rnn = RNNLM()
results["RNN\n(Elman)"] = {
    "loss": train_neural(rnn),
    "params": count_params(rnn),
    "era": "1990s-2013",
}

lstm = LSTMLM()
results["LSTM"] = {
    "loss": train_neural(lstm),
    "params": count_params(lstm),
    "era": "1997 / dominant 2014-2017",
}

transformer = TransformerLM()
results["Transformer\n(decoder-only)"] = {
    "loss": train_neural(transformer),
    "params": count_params(transformer),
    "era": "2017-present",
}

elapsed = time.time() - t0
for name, r in results.items():
    r["ppl"] = math.exp(r["loss"])

print(f"Trained all 4 models in {elapsed:.1f}s on {DEVICE}")
for name, r in results.items():
    print(f"  {name.replace(chr(10), ' '):28s} loss={r['loss']:.3f}  ppl={r['ppl']:.2f}  params={r['params']:,}")

# ---------------------------------------------------------------------------
# 4. proof.png — conceptual timeline (top) + real perplexity comparison (bottom)
# ---------------------------------------------------------------------------
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1, 1.4]}
)

# --- Top panel: conceptual era timeline ---
ax_top.set_xlim(0, 10)
ax_top.set_ylim(0, 2)
ax_top.axis("off")
ax_top.set_title("Language Modeling: Four Eras", fontsize=14, fontweight="bold", loc="left")
milestones = [
    (0.6, "N-gram\ncounting\n(pre-2000s)", "#8d99ae"),
    (3.2, "RNN\n(Elman, 1990)", "#4361ee"),
    (5.6, "LSTM\n(Hochreiter &\nSchmidhuber, 1997)", "#3a0ca3"),
    (8.2, "Transformer\n(Vaswani et al., 2017)", "#f72585"),
]
ax_top.annotate(
    "", xy=(9.6, 1), xytext=(0.2, 1),
    arrowprops=dict(arrowstyle="->", lw=2, color="#333333"),
)
for x, label, color in milestones:
    ax_top.scatter([x], [1], s=260, color=color, zorder=3, edgecolor="white", linewidth=1.5)
    ax_top.text(x, 1.35, label, ha="center", va="bottom", fontsize=9.5)

# --- Bottom panel: real measured perplexity ---
names = list(results.keys())
ppls = [results[n]["ppl"] for n in names]
colors = ["#8d99ae", "#4361ee", "#3a0ca3", "#f72585"]
bars = ax_bot.bar(names, ppls, color=colors, edgecolor="black", linewidth=0.6)
ax_bot.set_ylabel("Validation Perplexity (lower = better)", fontsize=11)
ax_bot.set_title(
    f"Measured on held-out tiny-Shakespeare text · {sum(r['params'] for r in results.values() if isinstance(r['params'], int)):,} total params across models · {elapsed:.0f}s on {str(DEVICE).upper()}",
    fontsize=9.5, color="#444444",
)
for bar, name in zip(bars, names):
    h = bar.get_height()
    r = results[name]
    ax_bot.text(
        bar.get_x() + bar.get_width() / 2, h + max(ppls) * 0.02,
        f"{h:.1f}\n({r['params']:,} params)",
        ha="center", va="bottom", fontsize=9,
    )
ax_bot.spines[["top", "right"]].set_visible(False)

fig.suptitle("Topic 01.01 — History and Evolution of Language Models: Real Comparison", fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("Saved proof.png")
