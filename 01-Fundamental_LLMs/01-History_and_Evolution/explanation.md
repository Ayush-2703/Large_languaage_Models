# Explanation: `implementation.py`

This walks through the script section by section: what each block does, and why it's built that way rather than some simpler alternative.

## Data Loading
 
```python
def load_corpus(n_chars: int = 200_000) -> str:
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        text = resp.read().decode("utf-8")
    return text[:n_chars]
```

**What:** Fetches the tiny-Shakespeare corpus (Karpathy's char-rnn dataset, public domain) directly from GitHub at runtime and truncates to the first 200,000 characters.

**Why fetched, not bundled:** This keeps the script self-contained — a fresh Colab runtime can execute it top to bottom with no manual upload step. Why *this* corpus specifically: it's the standard toy dataset for character-level language modeling demos precisely because it's small, plain text, has no preprocessing quirks, and is public domain, so there's no licensing question about using it as training data.

**Why 200K characters, not the full 1.1MB:** The goal of this topic is a fair architectural comparison, not a state-of-the-art model. A smaller slice keeps all four models training in well under a minute even on a single CPU core, which is what this topic was actually developed and verified on.

```python
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
```

**What:** Builds a character-level vocabulary — every unique character becomes one token ID.

**Why character-level, not subword (BPE) tokenization:** Two reasons. First, it keeps `vocab_size` tiny (roughly 65 characters vs. tens of thousands of subword tokens), which matters because the embedding and output layers scale with vocabulary size — a huge vocabulary would dominate parameter count and obscure the actual architectural comparison this topic is about. Second, it avoids any Hugging Face Hub download for a tokenizer, keeping this topic runnable with zero network dependency beyond the corpus fetch itself. Production LLMs use subword tokenization for good reasons (shorter sequences, better handling of rare words) — see the Setup section of `theory.md` for how this demo's choices map onto production defaults.

## Batching

```python
def get_batch(source: torch.Tensor):
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)
```

**What:** Samples `BATCH_SIZE` random windows of `BLOCK_SIZE` characters from the data. `y` is `x` shifted one character to the right — the standard "predict the next token" self-supervised target that needs no manual labeling.

**Why random windows instead of sequential chunks:** Random sampling means every training step sees a different, roughly independent slice of the corpus, which gives a smoother loss signal than marching sequentially through the text in very few total steps.

## The Four Models

### `NGramLM`

```python
def fit(self, ids: torch.Tensor):
    seq = ids.tolist()
    for i in range(len(seq) - self.order):
        ctx = tuple(seq[i:i + self.order - 1])
        nxt = seq[i + self.order - 1]
        self.counts[ctx][nxt] += 1
```

**What:** Walks the training data once, counting how often each 2-character context is followed by each possible next character.

**Why no `nn.Module`, no gradient descent:** This is the point — a count-based model has no learned parameters in the neural sense at all. Its "training" is a single pass of counting, which is exactly what made these models cheap enough to run in production for two decades before neural approaches became practical. `eval_loss` applies add-1 smoothing (`+1` in the numerator, `+vocab_size` in the denominator) so that a context never seen in training doesn't produce a `log(0)` — the math intuition for why this is necessary is in `theory.md` §2.

### `RNNLM` and `LSTMLM`

```python
class RNNLM(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.RNN(hidden, hidden, batch_first=True, nonlinearity="tanh")
        self.head = nn.Linear(hidden, vocab_size)
```

**What:** Both wrap PyTorch's built-in `nn.RNN` / `nn.LSTM` — an embedding layer maps characters to dense vectors, the recurrent layer processes the sequence step by step, and a linear head projects the final hidden states back to vocabulary logits.

**Why PyTorch's built-in recurrent layers rather than a hand-rolled loop:** The point of this topic is the *architectural comparison* (parameter-efficiency and modeling power per era), not re-deriving backpropagation-through-time from scratch — that level of detail is reserved for the attention mechanism in the next topic, where the QKV math is the whole point. Using the standard, well-tested `nn.RNN`/`nn.LSTM` keeps the comparison fair and the code auditable.

**Why `hidden=96` for both:** Matching the hidden dimension is what makes the RNN-vs-LSTM comparison meaningful — the parameter count difference you see in the printed results (30,590 vs. 86,462) comes entirely from the LSTM's extra gates, not from a different width. That parameter gap is itself worth noticing: an LSTM's forget/input/output gates roughly triple the recurrent weight matrix size for the same hidden dimension.

### `CausalSelfAttention` and `TransformerLM`

```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**What:** `TransformerLM` builds token + positional embeddings, passes them through `n_layers` Transformer blocks (self-attention + feed-forward, each with a residual connection and LayerNorm), then a final linear head.

**Why `F.scaled_dot_product_attention` rather than a manual `softmax(QK^T/√d)V`:** This is a deliberate choice specific to *this* topic. The manual, from-first-principles version — computing Q, K, V, the scaled dot product, the softmax, and verifying it numerically against this exact PyTorch function — is the entire subject of the next topic, `02-Transformer-Architecture-and-Self-Attention`. Re-deriving it here would duplicate that topic and dilute this one's actual focus, which is comparing eras, not deriving attention. Using the built-in, well-optimized kernel here is also what any real Transformer implementation would do in practice.

**Why `is_causal=True`:** This masks each position from attending to future positions, which is what makes this a valid *language model* — without the mask, predicting token *t* could see token *t* itself (and everything after it) baked into its own context, making the next-token-prediction task trivial and the trained model useless for actual generation.

## Training Loop

```python
def train_neural(model: nn.Module, steps: int = 800, lr: float = 3e-3) -> float:
    ...
    for step in range(steps):
        xb, yb = get_batch(train_data)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
```

**What:** A standard supervised training loop — forward pass, cross-entropy loss between predicted next-character logits and the true next character, backward pass, optimizer step. Identical code trains the RNN, LSTM, and Transformer; only the model object passed in differs.

**Why the exact same training function for all three neural models:** This is what makes the comparison fair. Same optimizer (`AdamW`), same learning rate, same number of steps, same batch sampling. Any perplexity difference in the results reflects the architecture, not a tuning advantage given to one model over another.

**Why 800 steps specifically:** This was tuned once, deliberately, and the reasoning is worth stating plainly: an earlier run at 400 steps left the Transformer noticeably behind the LSTM, and the question was whether that gap reflected the architecture or simply insufficient training. Raising the step count uniformly across all three neural models — not just the Transformer — closed most of the gap without giving any single model a discovery advantage. The result at 800 steps (Transformer and LSTM landing close together, LSTM narrowly ahead) is the real, reproducible output of that fair comparison, not a cherry-picked number. Why that specific outcome is expected, not a bug, is explained in `theory.md` §7.

## `proof.png` Generation

```python
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1, 1.4]})
```

**What:** Builds one composite figure with two panels: a conceptual timeline of the four eras (drawn with `scatter`/`annotate`, no measured data) on top, and the actual measured validation-perplexity bar chart on the bottom.

**Why one image with two panels instead of two separate images:** The brief for this topic calls it a proof image — evidence the code ran, not a slide illustration. Pairing the historical narrative (top) directly against the real numbers that narrative predicts (bottom) is what makes the image *proof* rather than decoration: every bar in the bottom panel was computed by the training loop above it in this same script, in this same run.

**Why bar heights are annotated with both perplexity and parameter count:** Perplexity alone doesn't show the efficiency story — the fact that the LSTM achieves its result with roughly a third of the Transformer's parameter count (86,462 vs. 240,446) is exactly the kind of trade-off `theory.md` §7 discusses, and it wouldn't be visible from the bar heights alone.
