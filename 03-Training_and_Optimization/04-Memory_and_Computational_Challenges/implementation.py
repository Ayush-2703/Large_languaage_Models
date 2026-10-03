"""
Topic 03.04 — Memory and Computational Challenges (Gradient Checkpointing, Flash Attention)
================================================================================================
Two real, measured memory/compute trade-offs:

  (A) GRADIENT CHECKPOINTING — trains a deliberately deep Transformer with
      and without `torch.utils.checkpoint`, measuring REAL peak process
      memory (via `resource.getrusage(...).ru_maxrss`) and wall-clock time
      for each. Because peak RSS only ever increases within one running
      process, each condition is measured in its OWN fresh subprocess —
      measuring both in the same process would silently contaminate
      whichever ran second with the first condition's already-reached peak.

  (B) FLASH ATTENTION — the standalone `flash-attn` package requires
      Ampere-class GPUs or newer (this was flagged in this repository's
      very first planning message, before any code was written) and this
      sandbox has no GPU at all, so the real fused kernel cannot run here
      under any circumstances. What IS measured: naive attention (which
      materializes the full T x T score matrix explicitly, exactly like
      01-Review-of-Fundamental-LLMs/02-Transformer-Architecture-and-Self-
      Attention's manual implementation) versus PyTorch's built-in
      `scaled_dot_product_attention`, across sequence lengths, on both
      timing AND memory. This demonstrates the real, general principle
      Flash Attention exploits (avoiding materializing the full attention
      matrix saves memory) without claiming to reproduce Flash Attention's
      specific GPU-memory-hierarchy (SRAM vs. HBM) mechanism, which has no
      CPU analogue — theory.md is explicit about this boundary.

No Hugging Face Hub download required. Runs in a few minutes on CPU.
"""

import subprocess
import sys
import time
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# (A) GRADIENT CHECKPOINTING — isolated subprocess measurement
# ===========================================================================
print("=" * 70)
print("(A) GRADIENT CHECKPOINTING — real peak-memory measurement")
print("=" * 70)

MEMORY_TEST_SCRIPT = textwrap.dedent("""
    import torch, torch.nn as nn, torch.nn.functional as F, resource, time, sys
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(0)
    USE_CHECKPOINT = sys.argv[1] == "1"
    D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, BATCH = 128, 8, {n_layers}, 256, 16

    class Block(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.ln1 = nn.LayerNorm(d); self.attn_qkv = nn.Linear(d, 3*d); self.attn_out = nn.Linear(d, d)
            self.n_heads = h
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

        def attn(self, x):
            B, T, C = x.shape
            q, k, v = self.attn_qkv(x).split(C, dim=2)
            hd = C // self.n_heads
            q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
            k = k.view(B, T, self.n_heads, hd).transpose(1, 2)
            v = v.view(B, T, self.n_heads, hd).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            return self.attn_out(out.transpose(1, 2).contiguous().view(B, T, C))

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class DeepTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(200, D_MODEL)
            self.blocks = nn.ModuleList([Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
            self.head = nn.Linear(D_MODEL, 200)

        def forward(self, x):
            h = self.embed(x)
            for block in self.blocks:
                if USE_CHECKPOINT:
                    h = checkpoint(block, h, use_reentrant=False)
                else:
                    h = block(h)
            return self.head(h)

    model = DeepTransformer()
    x = torch.randint(0, 200, (BATCH, SEQ_LEN))
    y = torch.randint(0, 200, (BATCH, SEQ_LEN))

    t0 = time.time()
    for _ in range(5):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 200), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
    elapsed = time.time() - t0

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"RESULT peak_kb={{peak_kb}} elapsed={{elapsed:.3f}} params={{sum(p.numel() for p in model.parameters())}}")
""")

N_LAYERS_DEEP = 24

def run_memory_subprocess(use_checkpoint: bool, n_layers: int):
    script = MEMORY_TEST_SCRIPT.format(n_layers=n_layers)
    result = subprocess.run(
        [sys.executable, "-c", script, "1" if use_checkpoint else "0"],
        capture_output=True, text=True, timeout=120,
    )
    line = [l for l in result.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError("Subprocess did not report a result")
    parts = dict(kv.split("=") for kv in line[0].replace("RESULT ", "").split())
    return int(parts["peak_kb"]), float(parts["elapsed"]), int(parts["params"])

print(f"Running WITHOUT gradient checkpointing ({N_LAYERS_DEEP}-layer Transformer, isolated subprocess)...")
peak_no_ckpt, time_no_ckpt, n_params = run_memory_subprocess(False, N_LAYERS_DEEP)
print(f"  Peak RSS: {peak_no_ckpt/1024:.1f} MB   Time (5 fwd+bwd): {time_no_ckpt:.2f}s")

print(f"Running WITH gradient checkpointing ({N_LAYERS_DEEP}-layer Transformer, isolated subprocess)...")
peak_ckpt, time_ckpt, _ = run_memory_subprocess(True, N_LAYERS_DEEP)
print(f"  Peak RSS: {peak_ckpt/1024:.1f} MB   Time (5 fwd+bwd): {time_ckpt:.2f}s")

mem_reduction_pct = 100 * (1 - peak_ckpt / peak_no_ckpt)
time_overhead_pct = 100 * (time_ckpt / time_no_ckpt - 1)
print(f"\n  Memory REDUCTION with checkpointing: {mem_reduction_pct:.1f}%  "
      f"({peak_no_ckpt/1024:.0f}MB -> {peak_ckpt/1024:.0f}MB)")
print(f"  Time OVERHEAD with checkpointing:    {time_overhead_pct:.1f}%  "
      f"({time_no_ckpt:.1f}s -> {time_ckpt:.1f}s)")
print(f"  ({n_params:,} params, {N_LAYERS_DEEP} layers, seq_len=256, batch=16)\n")

# ===========================================================================
# (B) ATTENTION MEMORY/TIME: naive (materialized T x T matrix) vs. PyTorch SDPA
# ===========================================================================
print("=" * 70)
print("(B) Attention: naive materialized matrix vs. PyTorch SDPA")
print("=" * 70)

ATTN_TEST_SCRIPT = textwrap.dedent("""
    import torch, torch.nn.functional as F, resource, time, math, sys

    torch.manual_seed(0)
    METHOD = sys.argv[1]   # "naive" or "sdpa"
    SEQ_LEN = int(sys.argv[2])
    D_MODEL, N_HEADS, BATCH = 128, 8, 8
    hd = D_MODEL // N_HEADS

    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, hd, requires_grad=True)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, hd, requires_grad=True)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, hd, requires_grad=True)

    def naive_attention(q, k, v):
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)   # materializes full (T,T) matrix
        weights = F.softmax(scores, dim=-1)
        return weights @ v

    t0 = time.time()
    for _ in range(10):
        if METHOD == "naive":
            out = naive_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out.sum().backward()
        q.grad = k.grad = v.grad = None
    elapsed = (time.time() - t0) / 10

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"RESULT peak_kb={peak_kb} elapsed={elapsed:.4f}")
""")

def run_attn_subprocess(method: str, seq_len: int):
    result = subprocess.run(
        [sys.executable, "-c", ATTN_TEST_SCRIPT, method, str(seq_len)],
        capture_output=True, text=True, timeout=60,
    )
    line = [l for l in result.stdout.splitlines() if l.startswith("RESULT")]
    if not line:
        print("STDOUT:", result.stdout); print("STDERR:", result.stderr)
        raise RuntimeError("Subprocess did not report a result")
    parts = dict(kv.split("=") for kv in line[0].replace("RESULT ", "").split())
    return int(parts["peak_kb"]), float(parts["elapsed"])

seq_lengths = [128, 256, 512, 1024]
naive_mem, naive_time, sdpa_mem, sdpa_time = [], [], [], []
for T in seq_lengths:
    pk_n, t_n = run_attn_subprocess("naive", T)
    pk_s, t_s = run_attn_subprocess("sdpa", T)
    naive_mem.append(pk_n); naive_time.append(t_n)
    sdpa_mem.append(pk_s); sdpa_time.append(t_s)
    print(f"  T={T:5d}   naive: {pk_n/1024:.1f}MB / {t_n*1000:.2f}ms   "
          f"sdpa: {pk_s/1024:.1f}MB / {t_s*1000:.2f}ms")

print("\n[Flash Attention's standalone fused CUDA kernel requires Ampere-class GPUs")
print(" or newer and CANNOT run on this CPU-only sandbox under any configuration.")
print(" The comparison above shows the general principle (avoiding materializing")
print(" the full attention matrix saves memory) via PyTorch's SDPA backend, NOT a")
print(" reproduction of Flash Attention's specific GPU SRAM/HBM mechanism. See")
print(" theory.md for why these are related but distinct claims.]\n")

# ===========================================================================
# proof.png
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax0 = axes[0, 0]
labels = ["No Checkpointing", "Gradient\nCheckpointing"]
mems = [peak_no_ckpt/1024, peak_ckpt/1024]
bars = ax0.bar(labels, mems, color=["#8d99ae", "#4361ee"], edgecolor="black")
for bar, m in zip(bars, mems):
    ax0.text(bar.get_x()+bar.get_width()/2, m*1.01, f"{m:.0f}MB", ha="center", fontsize=9)
ax0.set_ylabel("peak process memory (MB)")
ax0.set_title(f"A. Gradient Checkpointing: Real Peak Memory\n{N_LAYERS_DEEP}-layer Transformer, {mem_reduction_pct:.1f}% memory REDUCTION", fontsize=10)

ax1 = axes[0, 1]
times = [time_no_ckpt, time_ckpt]
bars = ax1.bar(labels, times, color=["#8d99ae", "#4361ee"], edgecolor="black")
for bar, t in zip(bars, times):
    ax1.text(bar.get_x()+bar.get_width()/2, t*1.01, f"{t:.2f}s", ha="center", fontsize=9)
ax1.set_ylabel("wall-clock time, 5x fwd+bwd (s)")
ax1.set_title(f"B. Gradient Checkpointing: Real Time Cost\n{time_overhead_pct:.1f}% time OVERHEAD (recomputation cost)", fontsize=10)

ax2 = axes[1, 0]
ax2.plot(seq_lengths, [m/1024 for m in naive_mem], marker="o", label="naive (materializes T×T)", color="#f72585")
ax2.plot(seq_lengths, [m/1024 for m in sdpa_mem], marker="s", label="PyTorch SDPA", color="#3a0ca3")
ax2.set_xlabel("sequence length"); ax2.set_ylabel("peak process memory (MB)")
ax2.set_title("C. Attention Memory vs. Sequence Length", fontsize=10)
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

ax3 = axes[1, 1]
ax3.plot(seq_lengths, [t*1000 for t in naive_time], marker="o", label="naive", color="#f72585")
ax3.plot(seq_lengths, [t*1000 for t in sdpa_time], marker="s", label="PyTorch SDPA", color="#3a0ca3")
ax3.set_xlabel("sequence length"); ax3.set_ylabel("time per fwd+bwd (ms)")
ax3.set_yscale("log")
ax3.set_title("D. Attention Time vs. Sequence Length (log scale)", fontsize=10)
ax3.legend(fontsize=8); ax3.grid(alpha=0.3, which="both")

fig.suptitle("Topic 03.04 — Memory & Computational Challenges: Real Measurements (CPU)", fontsize=12)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("Saved proof.png")
