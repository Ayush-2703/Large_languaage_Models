# Explanation: `implementation.py`

## `LoRALinear`: Freezing the Base at Construction Time

```python
def __init__(self, base_linear: nn.Linear, r=4, alpha=8):
    super().__init__()
    self.base = base_linear
    for p in self.base.parameters():
        p.requires_grad = False
```

**What:** Wraps an *existing* `nn.Linear` module (taken directly from the already-pretrained encoder) and immediately disables gradients on its parameters.

**Why wrap the existing layer object rather than construct a new one with copied weights:** Wrapping the live object guarantees the frozen "base" really is the exact pretrained weight tensor, with no risk of a copy subtly diverging (e.g., through a `.clone()` that isn't perfectly bit-identical, or an accidental double-conversion). `apply_lora` then simply reassigns `blk.attn.q_proj = LoRALinear(blk.attn.q_proj, ...)`, replacing the attribute in place — the encoder's forward pass calls whatever object currently sits at that attribute, so nothing elsewhere in `Attention.forward` needs to change to accommodate LoRA at all.

```python
self.lora_A = nn.Parameter(torch.randn(r, in_f) * (1.0 / r))
self.lora_B = nn.Parameter(torch.zeros(out_f, r))
```

**Why scale `lora_A`'s random initialization by `1/r` rather than using PyTorch's default `nn.Linear` initialization:** This follows the original LoRA paper's initialization scheme directly — `A` is initialized small (scaled down as rank `r` grows) specifically so that early training steps, once `B` starts moving away from zero, don't produce an outsized initial update relative to the frozen base's existing scale. Combined with `B` starting at exactly zero (`theory.md` §2), this keeps the adapter's very first few gradient steps well-behaved rather than needing a separate warmup phase just to stabilize LoRA's own initialization.

## Post-Training Quantization: Working on a Deep Copy

```python
def quantize_model(model, method):
    qmodel = copy.deepcopy(model)
    ...
    return qmodel, math.sqrt(total_err / total_n)
```

**Why `copy.deepcopy` before quantizing, rather than quantizing `model_a`'s weights in place:** `model_a` (the full-fine-tuned baseline) needs to remain unmodified so it can still serve as the FP32 reference point in the final comparison table — quantizing in place would destroy the very baseline the comparison is measuring against. Deep-copying first means INT8 quantization, NF4 quantization, and (later) pruning can all be applied independently to fresh copies of the *same* starting weights, keeping every row of `theory.md`'s comparison table anchored to the identical original model.

```python
total_err += ((dequant - w) ** 2).sum().item()
total_n += w.numel()
...
return qmodel, math.sqrt(total_err / total_n)
```

**What:** Accumulates squared reconstruction error across *every* `nn.Linear` layer in the model (not just one), then reports a single RMSE (root-mean-squared-error) figure at the end.

**Why aggregate across all layers into one number rather than reporting per-layer error:** A single summary RMSE is what makes the INT8-vs-NF4 comparison in `theory.md` §3's table a clean one-line-per-method comparison. Per-layer error would be more granular but would require a reader to mentally aggregate it anyway to answer the practical question this topic cares about — "which method reconstructs the trained weights better overall" — so the aggregation is done once, in code, rather than left as an exercise.

## NF4: Nearest-Level Lookup via Broadcasting

```python
flat = w_norm.reshape(-1, 1)
dists = (flat - NF4_LEVELS.view(1, -1)).abs()
nearest = dists.argmin(dim=1)
```

**What:** Reshapes the (normalized) weight tensor to a column vector and the 16 NF4 levels to a row vector, so subtracting them broadcasts into a full `(num_weights, 16)` distance matrix in one vectorized operation, then picks each weight's nearest level via `argmin`.

**Why broadcast rather than loop over weights (or over levels) in Python:** With potentially tens of thousands of weights per layer, a Python-level loop would be needlessly slow for an operation PyTorch can vectorize directly. This is also a deliberate example of writing the *quantization* step itself efficiently even though `near_dedup` in the previous topic (`01-Data-Collection-and-Preprocessing`) deliberately left an `O(n²)` loop unoptimized for clarity — the trade-off there was between exposing a *definition* clearly (near-duplication) versus here, where the nearest-level lookup's definition is already fully clear from the one-line `dists.argmin(dim=1)`, so there's no clarity cost to writing it efficiently.

## `QLoRALinear`: Storing Indices, Not Floats

```python
nearest_idx = ((w_dequant / self.scale).reshape(-1, 1) - NF4_LEVELS.view(1, -1)).abs().argmin(dim=1)
self.register_buffer("q_indices", nearest_idx.reshape(base_linear.weight.shape))
```

**What:** Rather than storing the dequantized (already-back-to-float32) weight values, `QLoRALinear` stores only the integer *index* (0–15) of which NF4 level each weight maps to, plus one scalar `scale`.

**Why store indices instead of the dequantized floats directly, given `quantize_nf4` already computes the dequantized tensor:** This is the actual point of QLoRA's memory savings — an index into 16 buckets needs only 4 bits to represent, while a dequantized float32 value needs 32 bits regardless of how it was derived. Storing the dequantized floats (as `quantize_model` does for the standalone quantization comparison in §3) is appropriate there, since that section's whole point is measuring *reconstruction error*, which requires the actual dequantized values to compare against the originals. `QLoRALinear` instead stores only what a real memory-constrained deployment would keep on disk or in memory — the compact 4-bit indices — and reconstructs the float values only transiently, inside `forward`, exactly matching `theory.md` §4's "dequantized on-the-fly" description.

**Why `register_buffer` rather than a plain attribute or an `nn.Parameter`:** `register_buffer` tells PyTorch this tensor is part of the module's state (so it moves correctly with `.to(device)`, appears in `state_dict()`, etc.) without being a trainable parameter — exactly right for `q_indices`, which is fixed once at construction and never updated by the optimizer, unlike `lora_A`/`lora_B`, which are `nn.Parameter`s specifically so they *do* receive gradients.

## Pruning: `kthvalue` for an Exact Percentile Threshold

```python
k = int(w.numel() * sparsity)
threshold = w.abs().flatten().kthvalue(k).values
mask = w.abs() > threshold
```

**What:** Finds the exact value below which the smallest `k` weights (by absolute value) fall, then zeros out everything at or below that threshold.

**Why `kthvalue` rather than sorting the whole tensor:** `kthvalue` finds the k-th smallest element without fully sorting the tensor, which is the more efficient operation for "give me the threshold that separates the smallest `k` from the rest" — sorting the entire weight tensor to read off one order statistic does unnecessary extra work. `theory.md` §5's exact sparsity-level results depend on this threshold being computed per-layer, independently for every `nn.Linear` module — a global threshold across the whole model would prune layers unevenly relative to their own weight-magnitude distributions, which is a different (and less standard) pruning strategy than the one being demonstrated here.

## `proof.png`: A Four-Panel Layout for Four Distinct Comparisons

Each of this topic's four techniques answers a differently-shaped question — LoRA/QLoRA is "how few trainable parameters for how much accuracy" (Panel A, log-scale bar chart), quantization is "how much smaller for how much accuracy, with a *second* metric — reconstruction error — mentioned in the panel title rather than plotted directly" (Panel B, dual-axis bars-plus-line), pruning is "accuracy as a continuous function of one swept parameter" (Panel C, a genuine line plot, the only technique here naturally suited to one). Panel D's plain-text summary exists specifically because a reader comparing across all four techniques — not just within one — needs every number in one place; no single chart type could show LoRA's parameter count, quantization's size-in-KB, and pruning's sparsity percentage on shared, comparable axes without badly distorting at least one of them.
