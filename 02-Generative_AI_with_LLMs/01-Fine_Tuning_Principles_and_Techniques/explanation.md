# Explanation: `implementation.py`

## Two Corpora, Two Roles

```python
pretrain_text = load_pretrain_corpus()          # tiny-Shakespeare — general narrative text
review_sents, review_labels = build_review_corpus()   # synthetic sentiment — the downstream task
```

**Why deliberately different domains for pretraining vs. fine-tuning, rather than pretraining directly on review-like text:** A real pretrain/fine-tune split always crosses a domain gap — models are pretrained on broad web text, then fine-tuned on narrow, task-specific labeled data. Using the same tiny-Shakespeare corpus from Phase 01 for pretraining (general, unlabeled, narrative) and a fresh synthetic review corpus for fine-tuning (narrow, labeled, task-specific) reproduces that same gap honestly, rather than giving every technique an unrealistically easy time by pretraining on data that already looks like the target task.

## Shared Vocabulary, Special Tokens

```python
SPECIALS = ["[PAD]", "[MASK]", "[CLS]"]
vocab = SPECIALS + chars
```

**What:** Builds one character vocabulary spanning both corpora, with three reserved special-token IDs prepended.

**Why build the vocabulary from *both* corpora combined, not just the pretraining corpus:** If the review corpus contained characters absent from Shakespeare's text (it doesn't, in practice, since both are plain English text, but the code doesn't rely on that coincidence), an incomplete vocabulary would silently break fine-tuning. Building from the union guarantees every character the fine-tuning stage will ever see already has a valid embedding row, even one that started from a MLM-pretrained (not random) initialization.

## The `[CLS]` Position Fix

```python
def encode(s: str, max_len: int = BLOCK_SIZE) -> torch.Tensor:
    ids = [CLS] + [stoi.get(c, PAD) for c in s[:max_len - 1]]
    ids = ids + [PAD] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)
```

**What:** Every encoded sequence begins with the `[CLS]` token's integer ID at position 0, followed by the sentence's characters.

**Why this needed fixing during development:** An earlier version of this function tried to prepend the literal substring `"[CLS]"` to the sentence text before character-encoding it — which crashed immediately, because characters like `[` and `]` were never added to the vocabulary (they don't appear in either source corpus). The fix reserves position 0 for the special token's *integer ID* directly, never treating `[CLS]` as literal text to be character-encoded at all. This is worth stating plainly rather than silently: it's exactly the kind of off-by-one-concept bug that's easy to make when adapting a "prepend a special token" pattern from subword tokenization (where `[CLS]` really is one token) to a character-level vocabulary (where it very much is not).

## MLM Pretraining Batch Construction

```python
mask_prob = torch.rand(x.shape)
maskable = mask_prob < 0.15
y[~maskable] = -100
x[maskable] = MASK
```

**What:** For every character position, with 15% probability, replace it with the `[MASK]` token in the input and record the *original* character as that position's target; every other position's target is set to `-100`.

**Why `-100` specifically:** `-100` is `nn.CrossEntropyLoss`'s default `ignore_index` — positions with this label contribute zero loss and zero gradient. This is what makes the MLM objective correctly "predict only the masked 15%, ignore the rest," rather than requiring a separate manual masking step in the loss computation.

**Why 15%:** This is Devlin et al.'s original BERT masking rate, carried through essentially unchanged in most MLM-pretrained models since — included here for methodological fidelity to the technique being demonstrated, not derived from first principles for this toy setting.

## Loading Four Identical Copies of One Checkpoint

```python
pretrained_state = {k: v.clone() for k, v in base_model.state_dict().items()}
...
def fresh_model():
    m = MiniEncoder().to(DEVICE)
    m.load_state_dict(pretrained_state)
    return ClassifierHead(m).to(DEVICE)
```

**What:** Saves the pretrained encoder's weights once, then `fresh_model()` is called four times — once per technique — each time constructing a brand-new `MiniEncoder`, immediately overwriting its random initialization with the identical saved pretrained weights, and wrapping it in a fresh (randomly initialized) classification head.

**Why `.clone()` when saving the state dict:** Without cloning, `pretrained_state` would hold references to the *same tensors* `base_model` itself uses. If `base_model` (or anything sharing its memory) were modified later, `pretrained_state` would silently change too — cloning guarantees a true, independent snapshot, which is what makes "all four techniques start from byte-identical weights" a checkable claim rather than an assumption.

**Why the classification head is freshly randomly initialized for each technique, rather than also being loaded from a shared saved state:** The head never existed during pretraining — MLM pretraining only trains `mlm_head`, not a 2-class classifier. Sharing initialization across the four `ClassifierHead`s (fixing a shared random seed for head initialization only, separate from data shuffling) would be a defensible additional control, but was traded off here against implementation simplicity, since head initialization is a comparatively minor variance source relative to the fine-tuning technique itself.

## Discriminative Learning Rate Groups

```python
param_groups.append({"params": block.parameters(), "lr": lr_i})
...
opt_b = torch.optim.AdamW(param_groups)
```

**Why PyTorch parameter groups rather than manually scaling gradients:** `torch.optim.Optimizer` natively supports per-group learning rates via the `param_groups` list passed at construction — this is the standard, correct mechanism for discriminative LR in PyTorch, and using it directly (rather than manually multiplying `.grad` tensors by a per-layer factor) keeps the optimizer's internal state (Adam's moment estimates) correctly scoped per parameter, which manual gradient scaling would not guarantee.

## Gradual Unfreezing's Optimizer Rebuild

```python
for step, idx in enumerate(...):
    n_unfrozen = min(n_layers, step // UNFREEZE_EVERY)
    for i, block in enumerate(model_d.encoder.blocks):
        want_trainable = i >= (n_layers - n_unfrozen)
        for p in block.parameters():
            p.requires_grad = want_trainable
    opt_d = torch.optim.AdamW([p for p in model_d.parameters() if p.requires_grad], lr=1e-3)
```

**Why rebuild the optimizer every single step, rather than only when the unfreeze schedule actually changes:** This is a simplicity-over-efficiency trade-off, stated plainly: rebuilding every step is correct (a freshly constructed `AdamW` always has the exact currently-trainable parameter set, so nothing can go stale) but wasteful, since `AdamW`'s momentum state resets whenever it's rebuilt, and rebuilding only matters on the ~5 steps where `n_unfrozen` actually changes. A production implementation would guard the rebuild behind an `if n_unfrozen != previous_n_unfrozen:` check to preserve momentum across steps where the trainable set doesn't change; that refinement was traded off here for a shorter, more obviously-correct loop, since this topic's subject is the unfreezing *schedule* itself, not optimizer-state bookkeeping efficiency.

## `proof.png` Generation

**Panel A** plots validation accuracy, evaluated every 20 steps, for all four techniques on one shared axis — the *shape* of each curve (how quickly it rises, whether it plateaus early) is the direct visual evidence for `theory.md` §6's claim that discriminative LR and gradual unfreezing trade convergence speed for caution, not just their different final endpoints.

**Panel B** isolates the final-accuracy comparison as a bar chart, annotated with the shared encoder's parameter count and step budget — making explicit that every bar represents the *same* model size and *same* compute budget, so any accuracy gap is attributable purely to technique.
