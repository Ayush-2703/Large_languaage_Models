"""
Topic 03.02 — Efficient Training: LoRA, QLoRA, Quantization (INT8/NF4), Pruning
====================================================================================
Four efficient-training techniques, each implemented from its actual
mathematical definition and applied to a real pretrained-then-fine-tuned
model — not simulated, not described only in prose:

  (A) Full fine-tuning        — baseline: every parameter trainable.
  (B) LoRA                    — freeze the base weights; inject trainable
                                 low-rank update matrices (Hu et al., 2021)
                                 into the Q and V attention projections.
  (C) Post-training quantization — INT8 (uniform affine) and NF4 (the
                                 QLoRA paper's quantile-based 4-bit scheme,
                                 Dettmers et al., 2023) applied to the
                                 fully fine-tuned model's weights.
  (D) QLoRA                   — LoRA fine-tuning on top of an NF4-quantized
                                 FROZEN base (dequantized on-the-fly in the
                                 forward pass) — the combination the QLoRA
                                 paper is named for.
  (E) Magnitude pruning        — unstructured, at five sparsity levels,
                                 applied to the fully fine-tuned model.

None of this requires a Hugging Face Hub download: every technique here is
an algorithm applied to a model's own weights, not a property of a specific
pretrained checkpoint. All numbers below are real and measured. Runs in a
few minutes on CPU.
"""

import copy
import math
import random
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import norm

torch.manual_seed(0)
random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Data + shared encoder architecture (same pattern as Phase 02)
# ---------------------------------------------------------------------------
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")

def load_corpus(n_chars=150_000):
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        return resp.read().decode("utf-8")[:n_chars]

POS_ADJ = ["brilliant", "wonderful", "captivating", "outstanding", "delightful", "masterful", "engaging", "superb"]
NEG_ADJ = ["terrible", "boring", "disappointing", "awful", "tedious", "clumsy", "forgettable", "dreadful"]
TEMPLATES = ["the movie was {adj}", "a {adj} film with strong direction", "critics called it {adj}",
             "audiences found it {adj}", "the plot was {adj} throughout", "the acting was {adj}",
             "overall a {adj} experience", "this film is {adj} from start to finish"]

def build_corpus(n):
    sents, labels = [], []
    for _ in range(n):
        is_pos = random.random() < 0.5
        adj = random.choice(POS_ADJ if is_pos else NEG_ADJ)
        sents.append(random.choice(TEMPLATES).format(adj=adj))
        labels.append(1 if is_pos else 0)
    return sents, labels

pretrain_text = load_corpus()
sents, labels = build_corpus(1200)
all_text = pretrain_text + " ".join(sents)
chars = sorted(set(all_text))
SPECIALS = ["[PAD]", "[MASK]", "[CLS]"]
vocab = SPECIALS + chars
stoi = {c: i for i, c in enumerate(vocab)}
VOCAB = len(vocab)
PAD, MASK, CLS = stoi["[PAD]"], stoi["[MASK]"], stoi["[CLS]"]
BLOCK = 40

def encode(s, max_len=BLOCK):
    ids = [CLS] + [stoi.get(c, PAD) for c in s[:max_len-1]]
    return torch.tensor(ids + [PAD]*(max_len-len(ids)), dtype=torch.long)

pretrain_ids = torch.tensor([stoi[c] for c in pretrain_text], dtype=torch.long)
n = len(sents); perm = list(range(n)); random.shuffle(perm)
train_idx, val_idx = perm[:900], perm[900:]

D_MODEL, N_HEADS, N_LAYERS = 64, 4, 3

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        hd = C // self.n_heads
        q = self.q_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
        attn_mask = pad_mask[:, None, None, :].expand(B, self.n_heads, T, T) if pad_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        return x + self.mlp(self.ln2(x))

class MiniBERT(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, d_model)
        self.pos_embed = nn.Embedding(BLOCK, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, VOCAB)

    def encode(self, ids, pad_mask=None):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.tok_embed(ids) + self.pos_embed(pos)
        for blk in self.blocks:
            h = blk(h, pad_mask)
        return self.ln_f(h)

    def forward_mlm(self, ids, pad_mask=None):
        return self.mlm_head(self.encode(ids, pad_mask))

class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(D_MODEL, 2)

    def forward(self, ids, pad_mask=None):
        return self.head(self.encoder.encode(ids, pad_mask)[:, 0, :])

def count_params(m, trainable_only=False):
    ps = (p for p in m.parameters() if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in ps)

def model_bytes(m, bits_per_param=32):
    return count_params(m) * bits_per_param / 8

def prep_batch(idx):
    ids = torch.stack([encode(sents[i]) for i in idx]).to(DEVICE)
    pad_mask = (ids != PAD).float().to(DEVICE)
    y = torch.tensor([labels[i] for i in idx], device=DEVICE)
    return ids, pad_mask, y

def evaluate(model):
    model.eval()
    with torch.no_grad():
        ids, pad_mask, y = prep_batch(val_idx)
        acc = (model(ids, pad_mask).argmax(-1) == y).float().mean().item()
    model.train()
    return acc

# ---------------------------------------------------------------------------
# 2. Pretrain the shared base encoder (CLS-aware from the start)
# ---------------------------------------------------------------------------
print("Pretraining base encoder via MLM...")
base = MiniBERT().to(DEVICE)
opt = torch.optim.AdamW(base.parameters(), lr=3e-3)
for step in range(500):
    ix = torch.randint(len(pretrain_ids) - (BLOCK - 1), (32,))
    body = torch.stack([pretrain_ids[i:i+BLOCK-1] for i in ix])
    xb = torch.cat([torch.full((32, 1), CLS, dtype=torch.long), body], dim=1).clone()
    yb = xb.clone()
    maskable = torch.rand(xb.shape) < 0.15
    maskable[:, 0] = False
    yb[~maskable] = -100
    xb[maskable] = MASK
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    loss = F.cross_entropy(base.forward_mlm(xb).view(-1, VOCAB), yb.view(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
base_state = {k: v.clone() for k, v in base.state_dict().items()}
print(f"  done, params={count_params(base):,}\n")

# ===========================================================================
# (A) FULL FINE-TUNING — baseline
# ===========================================================================
print("=== (A) Full fine-tuning ===")
enc_a = MiniBERT().to(DEVICE); enc_a.load_state_dict(base_state)
model_a = Classifier(enc_a).to(DEVICE)
opt = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
t0 = time.time()
for step in range(250):
    idx = random.sample(train_idx, 32)
    ids, pad_mask, y = prep_batch(idx)
    loss = F.cross_entropy(model_a(ids, pad_mask), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
acc_a = evaluate(model_a)
trainable_a = count_params(model_a, trainable_only=True)
print(f"  trainable={trainable_a:,}  acc={acc_a:.3f}  time={time.time()-t0:.1f}s\n")

# ===========================================================================
# (B) LoRA — freeze base, inject low-rank adapters into Q and V projections
# ===========================================================================
class LoRALinear(nn.Module):
    """W' x = Wx + (alpha/r) * B(Ax) — Hu et al., 2021. B initialized to
    zero so the adapted layer is IDENTICAL to the frozen base at step 0;
    all learning happens through A and B from there."""
    def __init__(self, base_linear: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

def apply_lora(encoder, r=4, alpha=8):
    for blk in encoder.blocks:
        blk.attn.q_proj = LoRALinear(blk.attn.q_proj, r, alpha)
        blk.attn.v_proj = LoRALinear(blk.attn.v_proj, r, alpha)
    return encoder

print("=== (B) LoRA fine-tuning (r=4, Q+V projections only) ===")
enc_b = MiniBERT().to(DEVICE); enc_b.load_state_dict(base_state)
for p in enc_b.parameters():
    p.requires_grad = False               # freeze EVERYTHING in the base encoder
apply_lora(enc_b, r=4, alpha=8)
model_b = Classifier(enc_b).to(DEVICE)    # head is fresh -> trainable by default
opt = torch.optim.AdamW([p for p in model_b.parameters() if p.requires_grad], lr=3e-3)
t0 = time.time()
for step in range(250):
    idx = random.sample(train_idx, 32)
    ids, pad_mask, y = prep_batch(idx)
    loss = F.cross_entropy(model_b(ids, pad_mask), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
acc_b = evaluate(model_b)
trainable_b = count_params(model_b, trainable_only=True)
total_b = count_params(model_b, trainable_only=False)
print(f"  trainable={trainable_b:,} / {total_b:,} total ({100*trainable_b/total_b:.1f}%)  acc={acc_b:.3f}  time={time.time()-t0:.1f}s\n")

# ===========================================================================
# (C) POST-TRAINING QUANTIZATION — INT8 and NF4, applied to model (A)'s weights
# ===========================================================================
print("=== (C) Post-training quantization of the fully fine-tuned model (A) ===")

def quantize_int8(w: torch.Tensor):
    scale = w.abs().max() / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    dequant = q.float() * scale
    return dequant, scale

def compute_nf4_levels():
    """16 quantile-based levels of N(0,1), normalized to [-1, 1] — the core
    idea behind NF4 (Dettmers et al., 2023): equal-probability-mass bins are
    information-theoretically well-matched to the roughly-normal
    distribution of trained neural network weights. This reproduces that
    core quantile-based mechanism; see theory.md for how it differs from
    the paper's exact (asymmetric, zero-preserving) construction."""
    quantiles = norm.ppf((np.arange(16) + 0.5) / 16)
    return torch.tensor(quantiles / np.abs(quantiles).max(), dtype=torch.float32)

NF4_LEVELS = compute_nf4_levels()

def quantize_nf4(w: torch.Tensor):
    scale = w.abs().max()
    w_norm = w / scale
    flat = w_norm.reshape(-1, 1)
    dists = (flat - NF4_LEVELS.view(1, -1)).abs()
    nearest = dists.argmin(dim=1)
    dequant = NF4_LEVELS[nearest].reshape(w.shape) * scale
    return dequant, scale

def quantize_model(model, method):
    qmodel = copy.deepcopy(model)
    total_err, total_n = 0.0, 0
    with torch.no_grad():
        for module in qmodel.modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                dequant, _ = (quantize_int8(w) if method == "int8" else quantize_nf4(w))
                total_err += ((dequant - w) ** 2).sum().item()
                total_n += w.numel()
                module.weight.data.copy_(dequant)
    return qmodel, math.sqrt(total_err / total_n)

acc_fp32 = acc_a
model_int8, rmse_int8 = quantize_model(model_a, "int8")
acc_int8 = evaluate(model_int8)
model_nf4, rmse_nf4 = quantize_model(model_a, "nf4")
acc_nf4 = evaluate(model_nf4)

size_fp32 = model_bytes(model_a, 32)
size_int8 = model_bytes(model_a, 8)
size_nf4 = model_bytes(model_a, 4)
print(f"  FP32:  size={size_fp32/1024:.1f}KB  acc={acc_fp32:.3f}  (reference)")
print(f"  INT8:  size={size_int8/1024:.1f}KB  acc={acc_int8:.3f}  weight RMSE={rmse_int8:.5f}  ({size_fp32/size_int8:.1f}x smaller)")
print(f"  NF4:   size={size_nf4/1024:.1f}KB  acc={acc_nf4:.3f}  weight RMSE={rmse_nf4:.5f}  ({size_fp32/size_nf4:.1f}x smaller)\n")

# ===========================================================================
# (D) QLoRA — LoRA fine-tuning on top of an NF4-QUANTIZED frozen base
# ===========================================================================
print("=== (D) QLoRA: LoRA adapters on top of an NF4-quantized frozen base ===")

class QLoRALinear(nn.Module):
    """Same adapter math as LoRALinear, but the frozen base weight is stored
    quantized (NF4) and dequantized on-the-fly every forward pass — exactly
    QLoRA's key idea: the large frozen base lives in low precision; only the
    small trainable adapters need full precision."""
    def __init__(self, base_linear: nn.Linear, r=4, alpha=8):
        super().__init__()
        w_dequant, self.scale = quantize_nf4(base_linear.weight.data)
        nearest_idx = ((w_dequant / self.scale).reshape(-1, 1) - NF4_LEVELS.view(1, -1)).abs().argmin(dim=1)
        self.register_buffer("q_indices", nearest_idx.reshape(base_linear.weight.shape))
        self.bias = nn.Parameter(base_linear.bias.data.clone(), requires_grad=False) if base_linear.bias is not None else None
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scaling = alpha / r

    def dequantized_weight(self):
        return NF4_LEVELS.to(self.q_indices.device)[self.q_indices] * self.scale

    def forward(self, x):
        base_out = F.linear(x, self.dequantized_weight(), self.bias)
        return base_out + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

def apply_qlora(encoder, r=4, alpha=8):
    for blk in encoder.blocks:
        blk.attn.q_proj = QLoRALinear(blk.attn.q_proj, r, alpha)
        blk.attn.v_proj = QLoRALinear(blk.attn.v_proj, r, alpha)
    return encoder

enc_d = MiniBERT().to(DEVICE); enc_d.load_state_dict(base_state)
for p in enc_d.parameters():
    p.requires_grad = False
apply_qlora(enc_d, r=4, alpha=8)
model_d = Classifier(enc_d).to(DEVICE)
opt = torch.optim.AdamW([p for p in model_d.parameters() if p.requires_grad], lr=3e-3)
t0 = time.time()
for step in range(250):
    idx = random.sample(train_idx, 32)
    ids, pad_mask, y = prep_batch(idx)
    loss = F.cross_entropy(model_d(ids, pad_mask), y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
acc_d = evaluate(model_d)
trainable_d = count_params(model_d, trainable_only=True)
print(f"  trainable={trainable_d:,}  acc={acc_d:.3f}  time={time.time()-t0:.1f}s")
print(f"  (compare to (B) full-precision LoRA: acc={acc_b:.3f}, same trainable-param count)\n")

# ===========================================================================
# (E) MAGNITUDE PRUNING — unstructured, at 5 sparsity levels
# ===========================================================================
print("=== (E) Magnitude pruning of the fully fine-tuned model (A) ===")

def prune_model(model, sparsity):
    pmodel = copy.deepcopy(model)
    with torch.no_grad():
        for module in pmodel.modules():
            if isinstance(module, nn.Linear):
                w = module.weight.data
                k = int(w.numel() * sparsity)
                if k == 0:
                    continue
                threshold = w.abs().flatten().kthvalue(k).values
                mask = w.abs() > threshold
                module.weight.data.mul_(mask)
    return pmodel

sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]
pruning_results = []
for s in sparsity_levels:
    pmodel = prune_model(model_a, s)
    acc = evaluate(pmodel)
    pruning_results.append(acc)
    print(f"  sparsity={s:.0%}  acc={acc:.3f}")

# ===========================================================================
# proof.png
# ===========================================================================
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(gs[0, 0])
methods = ["(A) Full FT", "(B) LoRA r=4", "(D) QLoRA r=4"]
trainables = [trainable_a, trainable_b, trainable_d]
accs = [acc_a, acc_b, acc_d]
colors = ["#8d99ae", "#4361ee", "#7209b7"]
bars = ax0.bar(methods, trainables, color=colors, edgecolor="black")
for bar, t, a in zip(bars, trainables, accs):
    ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02, f"{t:,}\nacc={a:.3f}", ha="center", fontsize=8)
ax0.set_yscale("log")
ax0.set_ylabel("trainable parameters (log scale)")
ax0.set_title("A. LoRA/QLoRA: Trainable Params vs. Full Fine-Tune\n(accuracy annotated per bar)", fontsize=10)

ax1 = fig.add_subplot(gs[0, 1])
qmethods = ["FP32\n(reference)", "INT8", "NF4"]
qsizes = [size_fp32/1024, size_int8/1024, size_nf4/1024]
qaccs = [acc_fp32, acc_int8, acc_nf4]
ax1b = ax1.twinx()
bars2 = ax1.bar(qmethods, qsizes, color=["#8d99ae", "#f72585", "#3a0ca3"], edgecolor="black", alpha=0.85)
ax1b.plot(qmethods, qaccs, color="#2a9d3f", marker="o", markersize=10, linewidth=2, label="accuracy")
for bar, s in zip(bars2, qsizes):
    ax1.text(bar.get_x()+bar.get_width()/2, s*1.02, f"{s:.1f}KB", ha="center", fontsize=8)
ax1.set_ylabel("model size (KB)"); ax1b.set_ylabel("accuracy", color="#2a9d3f")
ax1b.set_ylim(0, 1.05)
ax1.set_title("B. Quantization: Size Reduction vs. Accuracy Retained", fontsize=10)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot([s*100 for s in sparsity_levels], pruning_results, marker="o", color="#f72585", linewidth=2)
ax2.set_xlabel("sparsity (% weights zeroed)"); ax2.set_ylabel("validation accuracy")
ax2.set_title("C. Magnitude Pruning: Accuracy vs. Sparsity", fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1.05)

ax3 = fig.add_subplot(gs[1, 1])
ax3.axis("off")
summary = (
    f"Base encoder: {count_params(base):,} total params\n\n"
    f"(A) Full FT:      {trainable_a:,} trainable  ->  acc {acc_a:.3f}\n"
    f"(B) LoRA r=4:     {trainable_b:,} trainable ({100*trainable_b/total_b:.1f}%)  ->  acc {acc_b:.3f}\n"
    f"(D) QLoRA r=4:    {trainable_d:,} trainable, NF4-frozen base  ->  acc {acc_d:.3f}\n\n"
    f"(C) INT8 quant:   {size_fp32/size_int8:.1f}x smaller  ->  acc {acc_int8:.3f}\n"
    f"(C) NF4 quant:    {size_fp32/size_nf4:.1f}x smaller  ->  acc {acc_nf4:.3f}\n\n"
    f"(E) Pruning 70%:  acc {pruning_results[sparsity_levels.index(0.7)]:.3f}\n"
    f"(E) Pruning 90%:  acc {pruning_results[sparsity_levels.index(0.9)]:.3f}"
)
ax3.text(0.05, 0.95, summary, fontsize=10.5, family="monospace", va="top", transform=ax3.transAxes)
ax3.set_title("D. Summary — All Numbers Measured, Same Task/Model", fontsize=10)

fig.suptitle("Topic 03.02 — Efficient Training: LoRA, Quantization (INT8/NF4), QLoRA, Pruning", fontsize=12)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("\nSaved proof.png")
