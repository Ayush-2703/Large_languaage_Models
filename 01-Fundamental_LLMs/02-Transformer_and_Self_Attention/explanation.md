# Explanation: `implementation.py`

## `manual_scaled_dot_product_attention`

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
```

**What:** Computes the `T × T` similarity matrix and scales it, in one line — the first two steps of the QKV formula in `theory.md` §2.

**Why `k.transpose(-2, -1)` and not `k.T`:** `q` and `k` here are 4D tensors shaped `(B, H, T, d_k)` — batch, heads, sequence, head-dimension. `.T` would reverse *all* dimensions, which is wrong; `.transpose(-2, -1)` swaps only the last two (sequence and head-dim), which is the `Kᵀ` the formula actually calls for while leaving batch and head dimensions untouched.

```python
if causal:
    causal_mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float("-inf"))
```

**What:** Builds an upper-triangular boolean mask (`diagonal=1` excludes the diagonal itself, so a position *can* attend to itself) and sets every masked score to `-∞` before softmax.

**Why build the mask fresh inside the function rather than precomputing it once:** Simplicity and correctness over micro-optimization — this function is meant to be read and verified, not to be the fastest possible implementation (that's what the fused kernel in Section 7 exists for). A production implementation would cache this mask; a from-scratch teaching implementation should be as easy as possible to check against the formula.

```python
weights = F.softmax(scores, dim=-1)
output = weights @ v
return output, weights
```

**Why `dim=-1`:** Softmax must normalize *across keys* for a fixed query — i.e., across the last dimension of a `(B, H, T_query, T_key)` tensor — so each query position's attention weights sum to 1 over all the positions it's allowed to see.

**Why return `weights` at all:** PyTorch's fused `scaled_dot_product_attention` deliberately does *not* expose the intermediate attention matrix (it's fused into a single kernel for speed, precisely by avoiding materializing that matrix — see `theory.md` §8 on Flash Attention). This manual version keeps `weights` because Section 6 of this script needs it to draw the heatmap; it's the direct trade-off between "fast" and "inspectable."

## `MultiHeadAttention.forward`

```python
if use_builtin:
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    weights = None
else:
    out, weights = manual_scaled_dot_product_attention(q, k, v, causal=causal)
```

**What:** A single module that can run either the from-scratch attention or PyTorch's built-in kernel on identical Q/K/V, selected by a flag.

**Why one class with a switch, rather than two separate classes:** This is what makes the correctness check in Section 3 airtight. `w_q`, `w_k`, `w_v`, `w_o` are the *same* learned (here, randomly initialized but fixed) weight matrices in both code paths — only the attention computation itself differs. If outputs match, the difference genuinely isolates "does my manual math match PyTorch's," with no confound from different random initialization.

## Verification (Section 3)

```python
manual_out, attn_weights = mha(x, causal=True, use_builtin=False)
builtin_out, _ = mha(x, causal=True, use_builtin=True)
max_abs_diff = (manual_out - builtin_out).abs().max().item()
assert max_abs_diff < 1e-4, "..."
```

**What:** Runs the exact same input through both code paths and asserts the outputs are numerically indistinguishable.

**Why `1e-4` as the tolerance, not exact equality:** Floating-point matrix multiplication is not perfectly associative — PyTorch's fused kernel may accumulate sums in a different order than the naive `@` operator, producing tiny (`~10⁻⁷`, per the printed output) differences that are numerical noise, not a real disagreement in the underlying math. `1e-4` is generous relative to that noise floor while still being tight enough that a genuine implementation bug (e.g., forgetting the `√d_k` scale, or masking the wrong triangle) would fail it immediately — those bugs produce differences of order 1, not order `10⁻⁷`.

## Complexity Benchmark (Section 4)

```python
for T in seq_lengths:
    xb = torch.randn(2, T, D_MODEL, device=DEVICE)
    for _ in range(3):
        mha(xb, causal=True, use_builtin=False)   # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        mha(xb, causal=True, use_builtin=False)
    timings.append((time.perf_counter() - t0) / reps)
```

**What:** Times `manual_scaled_dot_product_attention` (not the built-in kernel) across increasing sequence lengths, averaging over 10 repetitions per length.

**Why a warmup loop before timing:** The first few calls into any PyTorch op pay a one-time cost (memory allocation, backend dispatch setup) unrelated to the actual computation. Discarding 3 untimed warmup calls before starting the clock keeps the measurement focused on steady-state cost.

**Why the manual implementation specifically, not the fused kernel:** The fused kernel is optimized precisely to avoid the naive `O(T²)` memory pattern this benchmark is trying to demonstrate — timing it would undersell the actual point. The manual version materializes the full `T × T` score matrix exactly as the formula describes, so its timing curve is the one that should visibly bend toward quadratic as `T` grows, which `theory.md` §6 discusses.

## Toy Attention Heatmap (Section 5)

```python
toy_embed = nn.Embedding(len(toy_tokens) + 1, D_MODEL).to(DEVICE)
toy_x = toy_embed(toy_ids) + sinusoidal_positional_encoding(T, D_MODEL).unsqueeze(0).to(DEVICE)
_, toy_weights = mha(toy_x, causal=False, use_builtin=False)
```

**What:** Embeds a short toy sentence, adds positional encoding, and runs it through the same `MultiHeadAttention` module (non-causal, so it's readable as a bidirectional/BERT-style attention pattern rather than a masked triangle).

**Why the embedding and attention weights are untrained (random init) rather than loaded from a real model:** This section's purpose is to show *the mechanism produces a real, valid probability distribution over positions* — every row of the heatmap genuinely sums to 1, computed by real softmax, not mocked. It deliberately does not claim the pattern is *linguistically meaningful* (an untrained model has no reason to attend to the word a real one would). Section 7 exists specifically to show what a *trained* model's attention looks like on the same sentence, once real pretrained weights are available on Colab.

## `proof.png` Generation

**Panel A** plots a histogram of `|manual_output - builtin_output|` across every element — visually, this should be (and is) a single sharp spike near zero, which is a more honest way to show "these match" than just printing one number, since it shows the *distribution* of differences has no outliers.

**Panel B** is the raw attention heatmap from Section 5 — head 0's `T × T` weight matrix, with rows and columns labeled by the actual toy tokens so query/key relationships are directly readable.

**Panel C** plots measured time-per-forward-pass against a dashed `O(T²)` reference curve anchored at the T=64 measurement, making the sub-quadratic-then-quadratic transition described in `theory.md` §6 visually explicit rather than something the reader has to compute from the printed numbers themselves.

## Section 7 — Why It's Gated Behind a Flag

```python
RUN_PRETRAINED_SECTION = False   # set True on Colab with internet access
```

**Why a flag instead of just letting the script fail on `AutoModel.from_pretrained(...)` when there's no network:** A script that crashes with a raw connection-error traceback halfway through looks broken, even though the failure has nothing to do with this topic's code being wrong — it's an environment limitation (this repo's development sandbox has no Hugging Face Hub access, documented in `STATUS.md`). Gating the section behind an explicit flag means the script always completes and always produces `proof.png` from the fully-verified, network-free sections above, and clearly tells the reader exactly what to flip on to get the additional, real-pretrained-model visualization when running somewhere with internet access — which is the normal Colab environment this whole repository targets.
