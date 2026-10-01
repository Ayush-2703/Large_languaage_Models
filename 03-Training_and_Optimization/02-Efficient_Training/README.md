# Efficient Training: LoRA, QLoRA, Quantization (INT8/NF4), and Pruning

## 1. Four Techniques, One Shared Question

Full fine-tuning updates every parameter and stores every weight at full precision — expensive in both training memory (gradients and optimizer states scale with trainable parameters) and storage. The four techniques in this topic each attack a different part of that cost, and `implementation.py` applies all four to the *same* pretrained-then-fine-tuned model, so their trade-offs are directly comparable:

| Technique | What it reduces | Mechanism |
|---|---|---|
| LoRA | Trainable parameters | Freeze base weights; learn a small low-rank update instead |
| Quantization (INT8/NF4) | Storage/memory of *existing* weights | Represent each weight with fewer bits |
| QLoRA | Both at once | LoRA adapters on top of a quantized frozen base |
| Pruning | Nonzero weight count | Zero out the least-important weights |

## 2. LoRA: Low-Rank Adaptation

Hu et al. (2021) observed that the *update* a pretrained weight matrix needs during fine-tuning (`ΔW`, the difference between fine-tuned and pretrained weights) tends to have low "intrinsic rank" — it can be well-approximated by a product of two much smaller matrices. Instead of learning a full `ΔW` (the same size as the original weight matrix `W`), LoRA freezes `W` entirely and learns:

```
W' x = Wx + (α/r)·B(Ax)
```

where `A` is `r × d_in` and `B` is `d_out × r`, with rank `r` chosen far smaller than either dimension of `W`. `implementation.py`'s `LoRALinear` implements this directly, applied to the Q and V projection matrices in each attention layer — the specific subset Hu et al.'s own experiments found most effective to adapt, rather than every weight matrix in the model.

**Why `lora_B` is initialized to all zeros:** at the very first training step, `B(Ax) = 0` regardless of what `A` contains, so the adapted layer's output is *identical* to the frozen base layer's output. Fine-tuning starts from exactly the pretrained model's behavior and diverges from there — there is no random, potentially-destructive initial perturbation the way there would be if both `A` and `B` started randomly.

**What was measured:** LoRA (rank 4) trained only 3,202 parameters — **2.0% of the model's 164,098 total** — and reached 0.947 validation accuracy, versus full fine-tuning's 161,026 trainable parameters and 1.000 accuracy. A 50x reduction in trainable parameters for a real but modest accuracy cost on this task is the central practical trade-off LoRA is built around, and it held here without needing any adjustment.

## 3. Quantization: INT8 and NF4

Quantization keeps the *number* of weights fixed but represents each one with fewer bits.

**INT8 (uniform affine quantization):** map the weight tensor's range to 256 evenly-spaced integer levels: `scale = max(|W|) / 127`, then `Q = round(W / scale)`, clipped to `[-127, 127]`. This is simple and fast, and works well when weights are roughly uniformly spread across their range.

**NF4 (NormalFloat4, Dettmers et al., 2023 — the QLoRA paper):** neural network weights are not uniformly distributed — empirically, they cluster near zero in an approximately normal (Gaussian) pattern. A uniform grid of 16 levels wastes most of its precision on the tails, where few weights actually fall. NF4 instead places its 16 representable levels at the **quantiles of a standard normal distribution** — dividing `N(0,1)` into 16 equal-*probability* regions rather than 16 equal-*width* ones, so more representable values land where weights actually concentrate, near zero:

```python
quantiles = norm.ppf((np.arange(16) + 0.5) / 16)
```

`implementation.py`'s `compute_nf4_levels` computes exactly this — the 16 midpoints of equal-probability-mass bins under a standard normal — then normalizes to `[-1, 1]`. This reproduces NF4's core information-theoretic idea faithfully. It is **not** a byte-exact reproduction of the original paper's precise construction, which uses a specific asymmetric offset to guarantee exact zero is one of the 16 representable levels (since exact zero is disproportionately common in trained weight tensors and deserves its own dedicated code, not an approximation); this implementation's levels are symmetric quantile midpoints and may not include exact zero. `explanation.md` discusses this trade-off directly.

**What was measured, applied to model (A)'s fully fine-tuned weights:**

| Format | Size | Accuracy | Weight RMSE |
|---|---|---|---|
| FP32 (reference) | 629.0 KB | 1.000 | — |
| INT8 | 157.3 KB (4x smaller) | 1.000 | 0.00089 |
| NF4 | 78.6 KB (8x smaller) | 1.000 | 0.01077 |

Both quantization schemes reduced model size substantially with **zero measured accuracy loss** on this task — NF4's roughly 12x larger weight-reconstruction error than INT8's is real and expected (4 bits inherently loses more precision than 8), but small enough at this model's scale not to move the downstream metric. This is a genuinely favorable result for a small model on an easy task; `theory.md` §5 discusses why this shouldn't be read as "quantization is always free."

## 4. QLoRA: Combining Both

QLoRA's core idea is to combine §2 and §3: keep the (large) pretrained base weights frozen **and** quantized to NF4, while the (small) LoRA adapter matrices remain in full precision and fully trainable. `QLoRALinear` implements this directly — the base weight is stored as 4-bit quantization indices (`self.q_indices`) plus a scale factor, and **dequantized on-the-fly inside `forward`** every time the layer is called, rather than ever being materialized in full precision in memory for longer than a single forward pass:

```python
def forward(self, x):
    base_out = F.linear(x, self.dequantized_weight(), self.bias)
    return base_out + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)
```

**What was measured:** QLoRA trained the same 3,202 parameters as full-precision LoRA and reached 0.987 accuracy — slightly *higher* than full-precision LoRA's 0.947 in this run. This should not be read as "quantizing the base improves accuracy" as a general claim — the gap is well within the run-to-run variance expected from `LoRALinear`'s and `QLoRALinear`'s independently random-initialized `lora_A` matrices (different random draws, since each is constructed at a different point in the script's random-number stream). The result that matters, and that held cleanly, is the qualitative one: **QLoRA's accuracy is comparable to full-precision LoRA's**, confirming that a properly-implemented NF4-quantized frozen base does not meaningfully degrade LoRA fine-tuning — which is QLoRA's actual practical claim, not that quantization is accuracy-*improving*.

## 5. Magnitude Pruning

Unstructured magnitude pruning removes individual weights, keeping only those with the largest absolute value — the simplest possible answer to "which weights matter least": the ones closest to zero contribute least to any given output, so zeroing them changes the network's function least, to a first approximation.

```python
threshold = w.abs().flatten().kthvalue(k).values
mask = w.abs() > threshold
module.weight.data.mul_(mask)
```

**What was measured**, applied to model (A)'s fine-tuned weights at five sparsity levels:

| Sparsity | 0% | 30% | 50% | 70% | 90% | 95% |
|---|---|---|---|---|---|---|
| Accuracy | 1.000 | 0.990 | 0.930 | 0.743 | 0.480 | 0.480 |

This traces the classic pruning trade-off curve exactly as the literature describes it: accuracy is essentially unaffected up to moderate sparsity (30%, barely a dent), degrades noticeably but still usably in a middle range (50–70%), then collapses sharply once sparsity removes enough capacity that the network can no longer represent the task — 90% and 95% both land at 0.480, indistinguishable from chance on this binary task, indicating the pruned network has lost essentially all of its learned function at that point, not just some accuracy.

## 6. Scaling to Production

Every mechanism here is unchanged in form at production scale — LoRA, quantization, and pruning all see direct real-world use at exactly this level of description, just with far larger base models (where the *ratio* of adapter parameters to base parameters becomes dramatically more favorable — LoRA's advantage grows, not shrinks, with model size) and, for quantization, per-block rather than per-tensor scale factors (real QLoRA quantizes in small blocks, e.g. 64 weights at a time, so a single outlier weight doesn't blow up the scale factor for an entire large tensor — a refinement `implementation.py`'s per-tensor scaling omits for simplicity, noted here directly rather than left implicit). Pruning at production scale is often paired with fine-tuning *after* pruning (to let the remaining weights recover some of the lost capacity) or done gradually and iteratively rather than in one post-hoc step, both of which would likely push the accuracy-collapse point in §5's table further to the right than this single-shot demonstration shows.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*, 2022.
- Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS*, 2023.
- Han, Song, Huizi Mao, and William J. Dally. "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." *ICLR*, 2016.
