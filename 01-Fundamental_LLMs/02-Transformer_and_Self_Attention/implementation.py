"""
Topic 01.02 — Transformer Architecture and Self-Attention Mechanism
=======================================================================
Implements scaled dot-product and multi-head attention entirely from the
QKV formula, then proves correctness by checking it numerically against
PyTorch's own fused kernel (`F.scaled_dot_product_attention`) rather than
just asserting it "should" be right. Also benchmarks the O(n^2) growth in
compute that self-attention's full pairwise comparison implies, and
visualizes real attention weights on a real pretrained model.

Section 1-3 (the QKV math, multi-head attention, correctness verification,
complexity benchmark) run anywhere — pure PyTorch, no network required.

Section 4 (real attention weights from a pretrained DistilBERT) requires
Hugging Face Hub access and is intended for Colab; it is written to run
identically there.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Scaled dot-product attention, derived directly from the QKV formula
# ---------------------------------------------------------------------------
def manual_scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Shapes: q, k, v are (B, H, T, d_k). Returns (output, attention_weights)
    so the weights can be inspected/visualized — the fused kernel below
    does not expose them, which is exactly why this manual version exists.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)     # (B, H, T, T)

    if causal:
        T = scores.size(-1)
        causal_mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


class MultiHeadAttention(nn.Module):
    """Splits d_model into n_heads parallel attention computations, each
    operating on a d_model/n_heads-dimensional slice, then concatenates and
    projects back — the mechanism behind every 'attention head' visualization
    you've seen in a Transformer paper."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)

    def forward(self, x: torch.Tensor, causal: bool = False, use_builtin: bool = False):
        q, k, v = self.split_heads(self.w_q(x)), self.split_heads(self.w_k(x)), self.split_heads(self.w_v(x))

        if use_builtin:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            weights = None
        else:
            out, weights = manual_scaled_dot_product_attention(q, k, v, causal=causal)

        B, H, T, d_k = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, H * d_k)
        return self.w_o(out), weights


# ---------------------------------------------------------------------------
# 2. Sinusoidal positional encoding (Vaswani et al., 2017, Section 3.5)
# ---------------------------------------------------------------------------
def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
       PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Fixed, not learned — chosen so relative positions correspond to a linear
    transformation, letting the model generalize to sequence lengths it did
    not see during training."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---------------------------------------------------------------------------
# 3. Correctness verification: manual implementation vs. PyTorch's fused kernel
# ---------------------------------------------------------------------------
D_MODEL, N_HEADS, SEQ_LEN, BATCH = 64, 4, 16, 4

mha = MultiHeadAttention(D_MODEL, N_HEADS).to(DEVICE)
x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)

manual_out, attn_weights = mha(x, causal=True, use_builtin=False)
builtin_out, _ = mha(x, causal=True, use_builtin=True)

max_abs_diff = (manual_out - builtin_out).abs().max().item()
mean_abs_diff = (manual_out - builtin_out).abs().mean().item()
print(f"Manual vs. built-in scaled_dot_product_attention:")
print(f"  max  |diff| = {max_abs_diff:.3e}")
print(f"  mean |diff| = {mean_abs_diff:.3e}")
assert max_abs_diff < 1e-4, "Manual attention does not match PyTorch's built-in kernel!"
print("  PASS: manual QKV implementation matches PyTorch's fused kernel within float32 tolerance.\n")

# ---------------------------------------------------------------------------
# 4. Complexity benchmark: attention cost grows O(T^2) in sequence length
# ---------------------------------------------------------------------------
seq_lengths = [16, 32, 64, 128, 256, 512]
timings = []
for T in seq_lengths:
    xb = torch.randn(2, T, D_MODEL, device=DEVICE)
    # warmup
    for _ in range(3):
        mha(xb, causal=True, use_builtin=False)
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        mha(xb, causal=True, use_builtin=False)
    timings.append((time.perf_counter() - t0) / reps)

print("Sequence length vs. wall-clock time per forward pass (manual attention):")
for T, t in zip(seq_lengths, timings):
    print(f"  T={T:4d}  {t*1000:.3f} ms")

# ---------------------------------------------------------------------------
# 5. Attention heatmap on a toy sequence
# ---------------------------------------------------------------------------
toy_tokens = ["The", "cat", "sat", "on", "the", "warm", "mat", "."]
T = len(toy_tokens)
torch.manual_seed(1)
toy_embed = nn.Embedding(len(toy_tokens) + 1, D_MODEL).to(DEVICE)
toy_ids = torch.arange(T, device=DEVICE).unsqueeze(0)
toy_x = toy_embed(toy_ids) + sinusoidal_positional_encoding(T, D_MODEL).unsqueeze(0).to(DEVICE)

_, toy_weights = mha(toy_x, causal=False, use_builtin=False)
head0_weights = toy_weights[0, 0].detach().cpu().numpy()   # (T, T), head 0

# ---------------------------------------------------------------------------
# 6. proof.png — verification, heatmap, and complexity growth
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 1.3])

# Panel A: numerical verification
ax0 = fig.add_subplot(gs[0])
diff = (manual_out - builtin_out).abs().detach().cpu().numpy().flatten()
ax0.hist(diff, bins=30, color="#4361ee", edgecolor="black", linewidth=0.4)
ax0.set_title(f"A. Manual vs. Built-In Attention\nmax|diff|={max_abs_diff:.1e}  (float32 noise floor)", fontsize=10)
ax0.set_xlabel("|manual_output - builtin_output|")
ax0.set_ylabel("count")
ax0.axvline(max_abs_diff, color="crimson", linestyle="--", linewidth=1, label="max diff")
ax0.legend(fontsize=8)

# Panel B: attention heatmap
ax1 = fig.add_subplot(gs[1])
im = ax1.imshow(head0_weights, cmap="viridis", vmin=0, vmax=head0_weights.max())
ax1.set_xticks(range(T)); ax1.set_xticklabels(toy_tokens, rotation=45, ha="right", fontsize=8)
ax1.set_yticks(range(T)); ax1.set_yticklabels(toy_tokens, fontsize=8)
ax1.set_title("B. Real Attention Weights\n(untrained weights, head 0, toy sentence)", fontsize=10)
ax1.set_xlabel("key position (attended TO)")
ax1.set_ylabel("query position (attending FROM)")
plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

# Panel C: complexity growth
ax2 = fig.add_subplot(gs[2])
ax2.plot(seq_lengths, [t * 1000 for t in timings], marker="o", color="#f72585", label="measured")
# reference quadratic curve, scaled to match the first non-trivial point
ref = [(t / seq_lengths[2] ** 2) * seq_lengths[2] ** 2 for t in [timings[2]]]
quad_ref = [timings[2] * 1000 * (T / seq_lengths[2]) ** 2 for T in seq_lengths]
ax2.plot(seq_lengths, quad_ref, linestyle="--", color="gray", label="O(T²) reference")
ax2.set_xscale("log", base=2); ax2.set_yscale("log")
ax2.set_xlabel("sequence length (T)")
ax2.set_ylabel("time per forward pass (ms)")
ax2.set_title("C. Attention Cost Grows O(T²)", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, which="both", alpha=0.3)

fig.suptitle("Topic 01.02 — Self-Attention: Correctness, Weights, and Complexity (all measured, not illustrative)", fontsize=11, y=1.03)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("\nSaved proof.png")


# ---------------------------------------------------------------------------
# 7. Real-world attention on a pretrained model — requires internet / Colab
# ---------------------------------------------------------------------------
RUN_PRETRAINED_SECTION = False   # set True on Colab with internet access

if RUN_PRETRAINED_SECTION:
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    hf_model = AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
    hf_model.eval()

    sentence = "The cat sat on the warm mat."
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = hf_model(**enc)

    # out.attentions: tuple of (n_layers,) tensors, each (batch, n_heads, T, T)
    last_layer_attn = out.attentions[-1][0, 0].numpy()   # last layer, head 0
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])

    plt.figure(figsize=(6, 5))
    plt.imshow(last_layer_attn, cmap="viridis")
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.yticks(range(len(tokens)), tokens)
    plt.title("DistilBERT real attention weights (last layer, head 0)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("pretrained_attention_example.png", dpi=150)
    print("Saved pretrained_attention_example.png (Colab-only section)")
else:
    print(
        "\n[Section 7 skipped locally: requires Hugging Face Hub access.]\n"
        "On Colab, set RUN_PRETRAINED_SECTION = True to load distilbert-base-uncased\n"
        "and visualize its real, trained attention weights on the same mechanism\n"
        "verified from scratch above."
    )
