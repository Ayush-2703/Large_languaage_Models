# Transformer Architecture and Self-Attention Mechanism

## 1. The QKV Formula

Every attention mechanism in every Transformer traces back to one equation, introduced by Vaswani et al. in "Attention Is All You Need" (2017):

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

Three learned linear projections of the same input `x` produce Q (query), K (key), and V (value):

```
Q = x W_Q      K = x W_K      V = x W_V
```

The intuition Rothman develops in *Transformers for Natural Language Processing* — and that Tunstall, von Werra, and Wolf reinforce with worked examples in *Natural Language Processing with Transformers* — is a retrieval metaphor: **Query** is "what this position is looking for," **Key** is "what each position advertises about itself," and **Value** is "what each position actually contributes if it's a good match." The dot product `Q · K` measures similarity between a query and every key; a high score means "this key is relevant to what I'm looking for."

## 2. Term by Term

**Step 1 — Similarity scores: `Q Kᵀ`.** For a sequence of length T, this produces a `T × T` matrix where entry `(i, j)` is how strongly position `i`'s query matches position `j`'s key — a full pairwise comparison, computed in one matrix multiplication rather than T sequential steps.

**Step 2 — Scaling: `/ √d_k`.** Without this term, dot products over a `d_k`-dimensional space grow in magnitude as `d_k` grows (the variance of a dot product of two random `d_k`-dimensional vectors scales with `d_k`). Large-magnitude inputs push softmax into a regime where its gradient is nearly zero almost everywhere except at one sharp peak, which stalls learning. Dividing by `√d_k` keeps the pre-softmax scores in a range where gradients stay well-behaved regardless of head dimension — this is precisely the fix Vaswani et al. describe as their motivation for including the scaling term at all.

**Step 3 — Normalization: `softmax(...)`.** Converts each row of similarity scores into a probability distribution over the sequence — how much of position `i`'s output should be a weighted blend of every other position's value vector. This is the number `implementation.py` visualizes directly as the attention heatmap.

**Step 4 — Weighted sum: `... V`.** Each output position becomes a convex combination of all value vectors, weighted by how relevant the corresponding key was to this position's query.

## 3. Causal Masking

A decoder-only model (GPT-style) must not let position `i` see positions `> i` — otherwise "predicting" the next token would just be copying it. This is enforced *before* the softmax, by setting disallowed entries in the score matrix to `-∞`:

```python
scores = scores.masked_fill(causal_mask, float("-inf"))
```

`softmax(-∞) = 0`, so those positions receive exactly zero attention weight after normalization — not "very small," genuinely zero. `implementation.py`'s `manual_scaled_dot_product_attention` implements this mask explicitly, using the same upper-triangular pattern PyTorch's fused kernel applies internally when `is_causal=True`.

## 4. Why Multiple Heads

A single attention computation forces every position to blend context through one shared similarity function. Multi-head attention instead splits the `d_model`-dimensional Q, K, V into `n_heads` smaller, independent attention computations (each of dimension `d_model / n_heads`), run in parallel, then concatenates and projects the results:

```
head_i = Attention(Q_i, K_i, V_i)
MultiHead(x) = Concat(head_1, ..., head_h) W_O
```

Different heads are free to specialize — empirically, some heads in trained Transformers attend primarily to adjacent tokens, others to syntactically related tokens (e.g., a verb attending to its subject), others to the previous occurrence of the same token. No head is told what to specialize in; this division of labor emerges purely from gradient descent optimizing the shared training objective.

## 5. Positional Encoding

Self-attention has no inherent sense of order: swap the rows of the input sequence and `Q Kᵀ` produces the same pairwise similarities, just permuted — attention is fundamentally a set operation. This is the trade-off for removing recurrence: an RNN's sequential processing bakes in order automatically, but a Transformer must inject position explicitly.

The original paper's solution, implemented directly in `implementation.py`'s `sinusoidal_positional_encoding`, is fixed (non-learned) sinusoids of varying frequency:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

added elementwise to the token embedding before the first attention layer. The choice of sinusoids (rather than, say, just concatenating a raw position index) matters for two reasons: values stay bounded in `[-1, 1]` regardless of sequence length, and — because `sin`/`cos` of a sum can be written as a linear function of `sin`/`cos` of the parts — the relative offset between any two positions corresponds to a fixed linear transformation the model can, in principle, learn to exploit, rather than needing to memorize every absolute position pair separately. Many modern LLMs (e.g., LLaMA-family models) instead use rotary position embeddings (RoPE), which encode relative position directly into the attention computation rather than adding a separate vector; the sinusoidal scheme here is the original, and still the clearest, entry point into why positional information needs injecting at all.

## 6. Computational Complexity

The `Q Kᵀ` step is a `(T × d_k) × (d_k × T)` matrix multiplication — its cost scales as `O(T² · d_k)` per head. This is the central efficiency trade-off of self-attention: doubling sequence length quadruples the compute (and the `T × T` score matrix's memory) for that step, in exchange for the constant path length between any two positions described in the previous topic's history.

`implementation.py`'s Panel C measures this directly by timing the manual attention implementation across sequence lengths from 16 to 512. The measured growth is *sub*-quadratic at small T (fixed Python/kernel-launch overhead dominates when the actual math is tiny) and converges toward the theoretical `O(T²)` curve as T grows — from T=256 to T=512, measured time roughly quadruples, matching theory closely once the fixed-cost regime is left behind. This distinction — asymptotic complexity vs. small-scale measured behavior — is exactly why efficient-attention techniques (Flash Attention, sparse/linear attention variants) only pay off decisively at longer sequence lengths, covered in Phase 03's `04-Memory-and-Computational-Challenges`.

## 7. What `implementation.py` Actually Proves

Most explanations of attention stop at the formula. This topic's code verifies it: `manual_scaled_dot_product_attention` is implemented from the four steps above with no shortcuts, then its output is compared element-by-element against PyTorch's own fused `F.scaled_dot_product_attention` kernel on identical inputs. The measured maximum absolute difference (`~10⁻⁷`, printed at runtime and shown in proof.png Panel A) is at the float32 numerical-precision noise floor — meaningfully zero, not "close." That is the actual proof this topic's `proof.png` is named for: the QKV formula above is not merely descriptive of what production libraries do, it *is*, bit-for-bit, what they do.

## 8. Scaling to Production

The formula, the masking, and the multi-head split shown here are unchanged at production scale — GPT-style and BERT-style models at any parameter count use exactly this mechanism per layer, stacked dozens of times with a wider `d_model` and more heads. What changes with scale is entirely engineering around the `O(T²)` cost identified in §6:

- **Flash Attention** restructures the same mathematical computation to avoid materializing the full `T × T` score matrix in slow GPU memory, fusing the softmax and matmul steps — same output, far less memory traffic. Covered in `03-Training-and-Optimization-of-LLMs/04-Memory-and-Computational-Challenges`.
- **KV-caching** during autoregressive generation reuses previously computed key/value vectors instead of recomputing them for the whole sequence at every new token.
- **Multi-Query / Grouped-Query Attention** shares K and V projections across multiple query heads, trading a small amount of modeling capacity for a large reduction in the memory bandwidth needed at inference time.

None of these change the QKV formula itself — they change how efficiently it is computed at the sequence lengths and batch sizes production systems require.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Vaswani, Ashish, et al. "Attention Is All You Need." *NeurIPS*, 2017.
- Su, Jianlin, et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.
- Dao, Tri, et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*, 2022.
