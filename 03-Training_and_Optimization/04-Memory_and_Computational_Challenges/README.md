# Memory and Computational Challenges (Gradient Checkpointing, Flash Attention)

## 1. Two Different Memory Problems

Training a large Transformer runs into memory limits from two largely independent directions: **activation memory** (every intermediate value computed during the forward pass must be kept around for the backward pass's gradient computation, and this grows with model depth) and **attention memory** (self-attention's `T × T` score matrix, per `01-Review-of-Fundamental-LLMs/02-Transformer-Architecture-and-Self-Attention` §6, grows quadratically with sequence length). Gradient checkpointing addresses the first; Flash Attention addresses the second. `implementation.py` measures both directly, with real peak-memory numbers rather than asserted ones.

## 2. Gradient Checkpointing

Backpropagation needs every layer's forward-pass activations to compute gradients — naively, a `24`-layer network must keep all `24` layers' worth of intermediate activations in memory simultaneously until the backward pass has worked through all of them. **Gradient checkpointing** (`torch.utils.checkpoint`) trades this away: during the forward pass, it *discards* activations for checkpointed sections as soon as they're no longer immediately needed, then **recomputes** them from scratch during the backward pass, exactly when each is required. Memory drops because far fewer activations are held at once; computation increases because part of the forward pass effectively runs twice — once for real, once again during the backward recomputation.

```python
if USE_CHECKPOINT:
    h = checkpoint(block, h, use_reentrant=False)
else:
    h = block(h)
```

`implementation.py` applies this per-layer to a deliberately deep, 24-layer, ~4.8-million-parameter Transformer — deep enough that the cumulative activation memory across all layers is large relative to the model's own parameter memory, which is what makes the trade-off clearly measurable.

## 3. Why Peak Memory Requires a Fresh Subprocess Per Measurement

`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` reports a process's peak resident-set size **since that process started, and this number never decreases** — it is a high-water mark, not a current reading. Measuring "no checkpointing" and then "with checkpointing" sequentially in the same running process would mean the second measurement's reported peak reflects whichever of the two conditions used *more* memory, contaminating whichever one ran second regardless of its own true peak. `implementation.py`'s `run_memory_subprocess` avoids this entirely by launching each condition as a **completely separate Python subprocess**, each starting from a fresh memory baseline and reporting its own independent peak — the only way to get two genuinely comparable numbers from a metric that only ever goes up.

## 4. What Was Actually Measured

| | Without Checkpointing | With Checkpointing |
|---|---|---|
| Peak memory | 1,630 MB | 1,060 MB |
| Time (5 fwd+bwd passes) | 10.6s | 12.6s |

**Memory reduction: 35.0%. Time overhead: 18.5%.** Both directions of the trade-off `theory.md` §2 predicts held cleanly and simultaneously in one real measurement — checkpointing is not "free": every megabyte of memory saved here corresponds to on-the-order-of forward-pass recomputation happening a second time during backward, which is exactly the mechanism, not a side effect.

## 5. Flash Attention: What This Sandbox Can and Cannot Show

This repository's very first planning exchange, before any code was written, flagged this directly: the standalone `flash-attn` package's fused CUDA kernels require Ampere-class GPUs or newer, and this sandbox has no GPU at all — there is no configuration under which the actual Flash Attention kernel can execute here. This is stated plainly rather than worked around.

What Flash Attention (Dao et al., 2022) actually does, conceptually: standard attention computes `softmax(QKᵀ/√d)V` by first materializing the full `T × T` score matrix in GPU memory, then applying softmax, then multiplying by `V` — three separate passes over an increasingly large intermediate object. Flash Attention restructures this into a single fused, tiled computation that never materializes the full `T × T` matrix in slow GPU memory (HBM) at all, instead operating on small tiles that fit in fast on-chip memory (SRAM), exploiting the specific memory hierarchy GPUs have. **This SRAM/HBM distinction has no meaningful CPU analogue** — CPU memory is comparatively flat by comparison, without the same order-of-magnitude bandwidth gap between "on-chip" and "main" memory that makes Flash Attention's specific optimization so valuable on GPUs.

What `implementation.py` measures instead is the **related but distinct** general principle that materializing a large intermediate object costs real memory and time, regardless of hardware: a "naive" attention implementation (explicitly computing and storing the full `T × T` score matrix, exactly like the manual implementation verified in `01-Review-of-Fundamental-LLMs/02-Transformer-Architecture-and-Self-Attention`) is compared against PyTorch's built-in `scaled_dot_product_attention`, which can select more memory-conscious computation paths internally even on CPU.

| Seq. Length | Naive memory | SDPA memory | Naive time | SDPA time |
|---|---|---|---|---|
| 128 | 524.5 MB | 520.1 MB | 9.3 ms | 5.2 ms |
| 256 | 586.6 MB | 520.1 MB | 32.6 ms | 15.9 ms |
| 512 | 709.2 MB | 525.6 MB | 228.8 ms | 56.1 ms |
| 1024 | 1,299.4 MB | 545.7 MB | 1,126.6 ms | 206.4 ms |

At the shortest sequence length, both implementations use nearly identical memory (the ~520 MB baseline is mostly fixed Python/PyTorch process overhead, unrelated to attention itself) and SDPA is already meaningfully faster. As sequence length grows, the gap widens sharply in **both** memory and time: by `T=1024`, naive attention uses roughly 2.4x SDPA's memory and takes roughly 5.5x as long. This is the real, measured shape of "avoiding the full materialized attention matrix matters more and more as sequences get longer" — the same qualitative principle Flash Attention exploits, demonstrated honestly within what CPU hardware can actually show, without overclaiming that this reproduces Flash Attention's specific GPU kernel-level mechanism.

## 6. Scaling to Production

Gradient checkpointing is applied routinely to production-scale models — often to entire transformer blocks, exactly as demonstrated here, letting substantially deeper or larger-batch training runs fit within fixed GPU memory at the cost of roughly 20–30% more compute, consistent with the range measured in §4. Flash Attention (and its successors, Flash Attention 2 and 3) are close to universal in production Transformer training and inference on supported GPU hardware, precisely because the SRAM/HBM gap §5 describes only grows more consequential as sequence lengths scale into the thousands or tens of thousands of tokens — exactly the regime where this topic's own CPU measurements already show naive attention's cost diverging sharply from a more memory-conscious implementation, even without GPU-specific kernel optimization.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Chen, Tianqi, et al. "Training Deep Nets with Sublinear Memory Cost." 2016. (Gradient checkpointing.)
- Dao, Tri, et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*, 2022.

