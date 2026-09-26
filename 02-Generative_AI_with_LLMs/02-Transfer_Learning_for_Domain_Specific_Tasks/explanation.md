# Explanation: `implementation.py`

## Two Downstream-Facing Corpora, One Purpose Split in Two

```python
dapt_sents, _ = build_tech_corpus(1500)      # unlabeled — DAPT only reads the text
dapt_text = " ".join(dapt_sents)
task_sents, task_labels = build_tech_corpus(1200)   # labeled — fine-tuning only
```

**Why generate two separate tech-review pools rather than reusing one set of sentences for both DAPT and fine-tuning:** Using the same exact sentences for both would let the model "see" the fine-tuning task's precise examples during the unsupervised DAPT stage too, inflating (B)'s apparent advantage for a reason that has nothing to do with domain adaptation as a general technique — it would be closer to leakage than transfer. Generating two independently sampled pools (both from the same template/vocabulary distribution, but not overlapping) keeps DAPT honestly unsupervised relative to the labeled task it's later evaluated on.

**Why the DAPT pool discards the labels (`_`) even though `build_tech_corpus` returns them:** This is the entire point of DAPT as a technique — it's a continued *unsupervised* pretraining pass, using only raw text, specifically so it can be applied even when in-domain labels are scarce or unavailable. Keeping the labels around and unused would be misleading about what information the DAPT stage actually has access to.

## Three Checkpoints From One Shared Origin

```python
general_state = {k: v.clone() for k, v in general_model.state_dict().items()}
...
dapt_model.load_state_dict(general_state)     # start from the SAME general checkpoint
dapt_loss = mlm_train(dapt_model, dapt_ids, steps=200, lr=3e-4)
dapt_state = {k: v.clone() for k, v in dapt_model.state_dict().items()}
```

**What:** `general_state` is pretrained once. Arm (A) fine-tunes an encoder loaded directly from it. Arm (B) first loads the *same* `general_state` into a separate model, continues training it on tech-review text, and only *then* saves a second checkpoint (`dapt_state`) that fine-tuning actually starts from.

**Why arm (B) needs its own separate `dapt_model` object rather than continuing to train `general_model` in place:** `general_state` must remain an untouched, reusable snapshot for arm (A) to load from later. If Stage 2 continued training `general_model` directly, the general (non-domain-adapted) checkpoint would no longer exist anywhere once Stage 2 finished — arm (A) would then be accidentally comparing against a partially-DAPT'd model, defeating the entire point of having an unadapted baseline.

**Why the continued-pretraining learning rate (`3e-4`) is lower than the original pretraining rate (`3e-3`), a factor of 10 gentler:** This was a deliberate correction made during development, not the original choice. An earlier run used the same aggressive rate for both stages, and the resulting DAPT checkpoint performed *worse* than expected relative to the no-adaptation baseline — consistent with continued pretraining at too high a rate disrupting already-useful general representations faster than 200 steps could productively reshape them for the new domain. Continued pretraining is conventionally gentler than initial pretraining precisely because there's already a meaningful starting point worth not overwriting too quickly; `theory.md` §3 discusses why this distinction mattered here specifically.

## The `[CLS]`-in-Pretraining Fix

```python
def mlm_batch(source_ids_tensor, batch_size=32, block_size=BLOCK_SIZE):
    ix = torch.randint(len(source_ids_tensor) - (block_size - 1), (batch_size,))
    body = torch.stack([source_ids_tensor[i:i + block_size - 1] for i in ix]).clone()
    cls_col = torch.full((batch_size, 1), CLS, dtype=torch.long)
    x = torch.cat([cls_col, body], dim=1)
    ...
    maskable[:, 0] = False   # never mask/predict the CLS token itself
```

**What changed and why it mattered:** The original version of this function sliced raw windows directly out of the corpus tensor — never inserting a `[CLS]` token, since plain corpus text naturally doesn't contain one. But `ClassifierHead.forward` (below) always reads its prediction from position 0, which `encode()` always fills with the `[CLS]` token ID during fine-tuning. That mismatch meant the `[CLS]` row of the token-embedding table, and every layer's learned handling of "what's normally at position 0," had literally never been exposed to a training gradient before fine-tuning began for arms (A) and (B) — while arm (C), training everything jointly from a fresh random initialization, faced no such gap. Rebuilding pretraining batches to structurally match fine-tuning's `[CLS]`-first format (masking every position except the CLS slot itself, which isn't a content token there is anything meaningful to "predict") closes that gap. `theory.md` §3 discusses exactly how this changed the measured results — this is the kind of fix that's worth explaining in detail rather than applying silently, since it's the difference between measuring the intended comparison and measuring an artifact of an inconsistent evaluation protocol.

```python
maskable[:, 0] = False
```

**Why exclude the CLS position from masking, rather than letting it occasionally get masked and predicted like everything else:** There is no "correct answer" to predict at the CLS position — it isn't standing in for a real corpus character the way every other position is. Masking it would ask the model to reconstruct an essentially arbitrary target, adding pure noise to the loss rather than a meaningful training signal.

## Data-Scarce Fine-Tuning Regime

```python
train_idx, val_idx = perm[:150], perm[150:450]
```

**Why only 150 labeled training examples, when 1,200 are available:** `theory.md` §4 explains this directly — pretraining's advantage is best demonstrated, and most commonly motivated in practice, when labeled data is scarce. An earlier version of this experiment used 900 training examples; every arm (including random-init) saturated to near-100% accuracy, which made the comparison uninformative — not wrong, just uninteresting, since a ceiling effect hides whatever real gap exists underneath it. Shrinking the labeled set is a direct, principled response to that observation, not an attempt to engineer a particular outcome.

## `proof.png` Generation

**Why both panels are needed, not just the final-accuracy bar chart:** The line chart (Panel A) shows *how* each arm gets to its final number — whether a lead was consistent throughout training or only emerged at the very end, which the single endpoint in Panel B can't distinguish. Given this topic's honestly mixed result (§3/§4 of `theory.md`), showing the full trajectory is part of being transparent about what was actually measured, not just the headline comparison.
