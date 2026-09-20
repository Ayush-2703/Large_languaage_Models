# Explanation: `implementation.py`

## Data

```python
train_raw = raw["train"].shuffle(seed=0).select(range(2000))
val_raw = raw["validation"].shuffle(seed=0).select(range(400))
```

**What:** Loads GLUE SST-2 via `datasets.load_dataset`, shuffles deterministically (`seed=0`), and takes small fixed-size slices — 2,000 training examples, 400 validation examples.

**Why subsample at all, when SST-2's full training set (~67K examples) would still fit the feasibility constraint on a T4:** Two reasons, both deliberate. First, speed: three full training runs (frozen probe, full fine-tune, random-init) on the full training set would still fit "under an hour" but with little margin — subsampling keeps the whole three-way comparison comfortably fast. Second, and more important pedagogically: `theory.md` §5 explains that pretraining's advantage is clearest in a data-scarce regime. A small training set is not just a speed compromise here, it's the setting where this experiment's comparison is most informative.

```python
def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=64)
```

**Why `max_length=64` and not the tokenizer's default (512 for BERT-family models):** SST-2 sentences are short movie-review snippets; 64 subword tokens comfortably covers the vast majority without truncating real content, while keeping every batch's compute cost down — attention cost grows quadratically with sequence length, per `theory.md` §6 of the previous topic, so this is a meaningful speed lever, not an arbitrary number.

## `ProbeOrFineTuneModel`

```python
if pretrained:
    self.backbone = AutoModel.from_pretrained(CHECKPOINT)
else:
    config = AutoConfig.from_pretrained(CHECKPOINT)
    self.backbone = AutoModel.from_config(config)
```

**What:** Loads either the real pretrained DistilBERT weights, or a DistilBERT with the *identical architecture* but randomly initialized parameters.

**Why `AutoModel.from_config(config)` rather than just a different, simpler architecture for the "no pretraining" baseline:** This is the single most important design decision in this script. If the random-init baseline used a different (e.g., smaller or simpler) architecture, a worse result would be ambiguous — is it worse because it wasn't pretrained, or because it's a weaker architecture? Using the exact same config (same depth, same hidden size, same number of heads) and changing *only* the initialization isolates pretraining as the sole variable, which is what `theory.md` §1's three-way comparison table depends on.

```python
if freeze_backbone:
    for p in self.backbone.parameters():
        p.requires_grad = False
```

**What:** Sets `requires_grad = False` on every backbone parameter, so `.backward()` computes no gradients for them and the optimizer (which only receives `trainable` parameters, below) never updates them.

```python
cls_repr = out.last_hidden_state[:, 0, :]
```

**Why the token at index 0 specifically:** BERT-family tokenizers prepend a special `[CLS]` token to every input. Through pretraining, this position's final-layer representation is trained (via the next-sentence-prediction-style objectives BERT-family models use) to aggregate information about the whole sequence — it's the standard place to read a single sentence-level representation from for classification, rather than e.g. averaging every token's representation.

## Training Loop

```python
opt = torch.optim.AdamW(trainable, lr=lr if not freeze_backbone else 1e-3)
```

**Why a different learning rate for the frozen-probe variant:** A frozen backbone's only trainable parameters are one linear layer — a much smaller, simpler optimization problem than fine-tuning tens of millions of backbone parameters, and linear layers on already-good features typically tolerate (and benefit from) a higher learning rate than full-model fine-tuning does. Using the *same* small learning rate (`2e-5`, standard for full Transformer fine-tuning) for the frozen probe would undertrain it relative to what it's actually capable of, unfairly disadvantaging that variant in the comparison.

```python
n_trainable = sum(p.numel() for p in trainable)
n_total = sum(p.numel() for p in model.parameters())
```

**Why track and report trainable-parameter count alongside accuracy:** Accuracy alone hides the efficiency story. If the frozen probe reaches, say, 80% of full-fine-tune's accuracy while training under 1% of the parameters, that ratio — not just the raw accuracy numbers — is the practically useful takeaway, and it's exactly the trade-off `theory.md` §6 connects forward to LoRA/QLoRA at production scale.

## `proof.png` Generation

**Why two panels (accuracy-over-epochs, and final-accuracy bar chart) rather than one:** The line chart shows *how quickly* each variant reaches its final performance — a frozen probe with only one trainable layer should converge in fewer effective epochs than full fine-tuning, which the curve shape (not just the endpoint) makes visible. The bar chart isolates the *endpoint* comparison and annotates each bar with its trainable-parameter count, which is the efficiency-vs-accuracy trade-off `theory.md` §4 is built around.

## Why This Script Was Not Executed During Development

Every other design choice above was verified the same way the rest of this repository's topics were — by running the code and inspecting real output. This script could not be: `AutoModel.from_pretrained("distilbert-base-uncased")` and `load_dataset("glue", "sst2")` both require Hugging Face Hub access, and this repository's development sandbox has none (`huggingface.co` resolves to a `403 host_not_allowed` from that environment specifically — a constraint of the development sandbox, not of Colab, where this script is designed to run). Rather than weaken the experiment to something network-free but less meaningful — for instance, "pretraining" a small model from scratch on a tiny local corpus, which would no longer be testing the same phenomenon this topic is named for — `implementation.py` stays the correct, real, Hugging-Face-based comparison. `proof.png` in this topic's folder is consequently a placeholder, watermarked and explicitly labeled as such, rather than a chart with any invented numbers. It will populate with real measured accuracy the first time this script runs anywhere with internet access.
