# Explanation: `implementation.py`

## One `Attention` Class for Three Architectures

```python
def forward(self, x_q, x_kv=None, causal=False, pad_mask=None):
    if x_kv is None:
        x_kv = x_q
    ...
```

**What:** A single attention implementation where the query source (`x_q`) and key/value source (`x_kv`) are separate arguments, defaulting to self-attention (`x_kv=x_q`) when only one is provided.

**Why one shared class instead of separate self-attention and cross-attention implementations:** Self-attention and cross-attention are the *same* mathematical operation — `softmax(QKᵀ/√d)V` — differing only in where K and V come from. Writing them as one class with an optional second input makes that equivalence explicit in the code, not just in prose, and guarantees `MiniGPT`'s self-attention, `MiniBERT`'s self-attention, and `MiniT5`'s cross-attention are all provably running identical math, differing only in their call site's arguments.

## `DecoderBlock`'s Optional Cross-Attention

```python
def __init__(self, d_model, n_heads, use_cross_attn: bool):
    ...
    if use_cross_attn:
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = Attention(d_model, n_heads)
```

**What:** One `DecoderBlock` class serves two roles: with `use_cross_attn=False`, it's a plain causal-self-attention block — exactly what `MiniGPT` needs. With `use_cross_attn=True`, it gains a second attention sub-layer that cross-attends to an encoder's output — what `MiniT5`'s decoder needs.

**Why one class with a flag rather than two separate `GPTDecoderBlock` / `T5DecoderBlock` classes:** Beyond avoiding duplicated code for the (identical) self-attention and feed-forward sub-layers, this makes the *only* structural difference between "GPT-style decoder block" and "T5-style decoder block" — the presence of a cross-attention sub-layer — visible as a single boolean, directly in the constructor signature, rather than buried in two independently-maintained class bodies that a reader would have to diff by hand to find the actual difference.

## `GPTClassifier`'s Last-Token Pooling

```python
lengths = (ids != PAD).sum(dim=1) - 1
lengths = lengths.clamp(min=0)
last_hidden = h[torch.arange(h.size(0)), lengths]
```

**What:** For each sequence in the batch, counts non-pad tokens to find the index of the last *real* token (not a padding position), then gathers each sequence's hidden state at exactly that index.

**Why not just always index position `BLOCK_SIZE - 1`:** Sentences in this dataset have varying lengths, and shorter ones are padded. If pooling always read the final position in the fixed-length tensor, it would read a `[PAD]` token's hidden state for every sentence shorter than `BLOCK_SIZE` — a position that, under causal attention, has seen nothing but padding since its own position, and carries no useful signal about the actual sentence. Computing each sequence's *true* last-token index individually, then gathering per-row, is what correctly implements "the position that has seen the whole real sequence" per `theory.md` §2, for sequences of any length within the batch.

## MLM Pretraining, CLS-Aware From the Start

```python
ix = torch.randint(len(pretrain_ids) - (BLOCK_SIZE - 1), (32,))
body = torch.stack([pretrain_ids[i:i + BLOCK_SIZE - 1] for i in ix])
cls_col = torch.full((32, 1), CLS, dtype=torch.long)
xb = torch.cat([cls_col, body], dim=1).clone()
```

**Why this matches `02-Transfer-Learning-for-Domain-Specific-Tasks`'s *fixed* version, not its original buggy one:** That topic's `explanation.md` documents a real bug where MLM pretraining batches never included a `[CLS]` token, leaving the classifier's readout position untrained until fine-tuning began. Building `MiniBERT`'s pretraining loop with `[CLS]` present from the very first line here — rather than needing to discover and fix the same bug a second time — is a direct, deliberate application of that earlier lesson. It's also a big part of why this topic's results (§7 of `theory.md`) came out clean on the first run, unlike the previous topic's.

## Constructing `MiniT5` From an Already-Trained `MiniBERT`

```python
class MiniT5(nn.Module):
    def __init__(self, pretrained_encoder: MiniBERT, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.encoder = pretrained_encoder
```

**What:** `MiniT5.__init__` takes an *already-constructed, already-trained* `MiniBERT` instance directly as an argument — it does not build or initialize an encoder of its own.

**Why pass the live object rather than, e.g., a saved state dict to load:** Passing the object directly means `MiniT5`'s encoder and the standalone `bert` object used by `BERTClassifier` are, after construction, the literal same underlying parameters in memory (`t5 = MiniT5(pretrained_encoder=bert)`) — not two separately-initialized copies that happen to match. This matters for the comparison in `theory.md` §6: it guarantees any accuracy difference between MiniBERT's and MiniT5's fine-tuned results reflects only the difference in fine-tuning *method* applied on top of that shared encoder, with zero risk that two nominally-identical encoders drifted apart from separate training runs.

**Consequence worth being explicit about:** because `t5.encoder` and `bert` share the same parameter tensors, further fine-tuning `t5` also updates the weights `bert_clf` (from the previous section) was built from — but by the time `MiniT5` is fine-tuned, `bert_clf`'s own fine-tuning has already completed and its accuracy already measured and recorded, so this ordering (BERT fine-tuned and evaluated *before* T5 fine-tuning begins) is what keeps the two results independent and valid despite the shared underlying object.

## Single-Token Generation as Classification

```python
dec_in = torch.full((BATCH, 1), BOS, dtype=torch.long, device=DEVICE)
target = torch.tensor([POS_TOK if review_labels[i] == 1 else NEG_TOK for i in idx], device=DEVICE)
logits = t5(enc_ids, dec_in, enc_pad)[:, -1, :]
loss = F.cross_entropy(logits, target)
```

**What:** The decoder's input is always just one token, `[BOS]`. The model produces one position's worth of output logits; cross-entropy compares those logits against whichever label token (`[POS]` or `[NEG]`) is correct for that example.

**Why a single generated token, rather than generating a full word like real T5's "positive"/"negative":** This is a deliberate simplification specific to this character-level vocabulary, stated plainly rather than left implicit. A multi-character target ("positive") would require a multi-step autoregressive generation loop during both training (teacher forcing across multiple positions) and evaluation (actually decoding several steps and checking whether the full generated string matches). Reserving single-character `[POS]`/`[NEG]` tokens captures the same core idea — classification framed as next-token generation, conditioned on the full input via cross-attention — without that additional decoding-loop complexity, which would add real implementation surface area without teaching anything additional about the *mechanism* this topic is focused on. `theory.md` §4 states this trade-off directly rather than presenting the single-token version as if it were the only possible one.

## `evaluate_t5`'s Greedy Check

```python
logits = model(enc_ids, dec_in, enc_pad)[:, -1, :]
preds = logits.argmax(-1)
acc = (preds == target).float().mean().item()
```

**What:** Takes the logits at the single decoded position, greedily picks the highest-scoring vocabulary entry, and checks whether it matches the expected label token exactly.

**Why this is a legitimate accuracy measurement despite operating over the *entire* vocabulary, not just the two label tokens:** The model is never told at inference time to restrict its output to `{[POS], [NEG]}` — it has to assign the highest probability to the correct label token out of every token in the vocabulary, including every ordinary character. That the model does this correctly (reaching 1.000 accuracy, per `theory.md` §7) is a meaningfully stronger result than it would be if evaluation only compared relative probabilities between the two label tokens specifically; it demonstrates the model has genuinely learned that, in this context (right after a `[BOS]` token, conditioned via cross-attention on a review's content), a label token is what belongs there at all.
