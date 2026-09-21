"""
Topic 01.04 — Scaling Laws and Model Efficiency
===================================================
Trains five decoder-only Transformers of increasing size, identical
architecture family, identical data, identical training-step budget, and
measures how validation loss falls as parameter count grows — a small,
fast, from-scratch reproduction of the shape of finding behind Kaplan et
al.'s and Hoffmann et al.'s neural scaling laws: loss vs. parameters follows
an approximately power-law (straight-line-in-log-log) relationship.

Also measures wall-clock training cost per model size, since "Model
Efficiency" in this topic's title is a distinct question from "does bigger
help" — bigger models cost more compute per step even when the accuracy
trend is favorable.

No Hugging Face Hub download required — every model is trained from random
initialization on the same tiny-Shakespeare corpus used in
01-History-and-Evolution-of-Language-Models. Runs in well under a minute
on CPU; seconds on a Colab T4.
"""

import math
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Data (same corpus and split logic as Topic 01.01, kept self-contained
#    here per this repository's one-topic-one-script convention)
# ---------------------------------------------------------------------------
CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

def load_corpus(n_chars: int = 200_000) -> str:
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        text = resp.read().decode("utf-8")
    return text[:n_chars]

text = load_corpus()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

BLOCK_SIZE = 48
BATCH_SIZE = 32

def get_batch(source: torch.Tensor):
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ---------------------------------------------------------------------------
# 2. A configurable decoder-only Transformer (same architecture family used
#    throughout this phase; only width/depth change between runs)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ScalableTransformerLM(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 3. Train five model sizes, identical steps/data/optimizer, only width/depth
#    change. This isolates parameter count as the variable under study.
# ---------------------------------------------------------------------------
CONFIGS = [
    dict(d_model=16,  n_heads=2, n_layers=1, label="Tiny"),
    dict(d_model=32,  n_heads=2, n_layers=1, label="Small"),
    dict(d_model=64,  n_heads=4, n_layers=2, label="Medium"),
    dict(d_model=128, n_heads=4, n_layers=2, label="Large"),
    dict(d_model=192, n_heads=4, n_layers=3, label="XL"),
]
TRAIN_STEPS = 400   # fixed budget across all sizes — see explanation.md

results = []
for cfg in CONFIGS:
    model = ScalableTransformerLM(cfg["d_model"], cfg["n_heads"], cfg["n_layers"]).to(DEVICE)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    t0 = time.time()
    for step in range(TRAIN_STEPS):
        xb, yb = get_batch(train_data)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        val_losses = []
        for _ in range(20):
            xb, yb = get_batch(val_data)
            logits = model(xb)
            val_losses.append(F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)).item())
    val_loss = sum(val_losses) / len(val_losses)

    # Approximate training FLOPs using the standard 6ND heuristic (Kaplan et
    # al., 2020): ~6 FLOPs per parameter per token, forward + backward.
    tokens_processed = TRAIN_STEPS * BATCH_SIZE * BLOCK_SIZE
    approx_flops = 6 * n_params * tokens_processed

    results.append(dict(
        label=cfg["label"], params=n_params, val_loss=val_loss,
        train_time=train_time, approx_flops=approx_flops,
    ))
    print(f"{cfg['label']:8s}  params={n_params:8,}  val_loss={val_loss:.4f}  "
          f"time={train_time:.1f}s  approx_FLOPs={approx_flops:.2e}")

# ---------------------------------------------------------------------------
# 4. Fit a power law: loss ≈ exp(b) * N^a  <=>  log(loss) = a*log(N) + b
# ---------------------------------------------------------------------------
log_n = np.log([r["params"] for r in results])
log_loss = np.log([r["val_loss"] for r in results])
a, b = np.polyfit(log_n, log_loss, 1)
print(f"\nFitted power law: loss ≈ {math.exp(b):.3f} * N^({a:.4f})")
print("(Negative exponent = loss falls as parameter count grows, the qualitative")
print(" signature of a neural scaling law — see theory.md for why the exact")
print(" exponent here should NOT be compared to published large-scale values.)")

# ---------------------------------------------------------------------------
# 5. proof.png — loss-vs-params scaling curve + training-cost efficiency panel
# ---------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

ns = [r["params"] for r in results]
losses = [r["val_loss"] for r in results]
ax0.scatter(ns, losses, s=90, color="#f72585", zorder=3, edgecolor="black", label="measured")
fit_n = np.logspace(math.log10(min(ns)), math.log10(max(ns)), 50)
fit_loss = np.exp(b) * fit_n ** a
ax0.plot(fit_n, fit_loss, "--", color="#4361ee", label=f"power-law fit (exponent={a:.3f})")
for r in results:
    ax0.annotate(r["label"], (r["params"], r["val_loss"]), textcoords="offset points",
                 xytext=(6, 6), fontsize=8)
ax0.set_xscale("log"); ax0.set_yscale("log")
ax0.set_xlabel("parameters (N)"); ax0.set_ylabel("validation loss")
ax0.set_title("A. Loss vs. Model Size (measured)", fontsize=10)
ax0.legend(fontsize=8); ax0.grid(True, which="both", alpha=0.3)

times = [r["train_time"] for r in results]
ax1.plot(ns, times, marker="o", color="#3a0ca3")
for r in results:
    ax1.annotate(r["label"], (r["params"], r["train_time"]), textcoords="offset points",
                 xytext=(6, 6), fontsize=8)
ax1.set_xscale("log")
ax1.set_xlabel("parameters (N)"); ax1.set_ylabel(f"wall-clock time for {TRAIN_STEPS} steps (s)")
ax1.set_title(f"B. Training Cost vs. Model Size ({str(DEVICE).upper()})", fontsize=10)
ax1.grid(alpha=0.3)

fig.suptitle("Topic 01.04 — Scaling Laws and Model Efficiency: 5 Real Trained Models, Fixed Data & Steps", fontsize=11)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("\nSaved proof.png")
