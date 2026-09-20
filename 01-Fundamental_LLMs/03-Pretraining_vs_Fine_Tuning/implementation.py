"""
Topic 01.03 — Pretraining vs. Fine-Tuning Paradigms
======================================================
Compares three ways of using DistilBERT on a downstream task, isolating
exactly what "pretraining" buys you:

    (A) Frozen probe   — pretrained DistilBERT backbone frozen; train only
                          a linear classification head on top of [CLS].
    (B) Full fine-tune — same pretrained backbone, but every parameter is
                          updated on the downstream task.
    (C) Random-init     — same architecture as (A)/(B), but with randomly
                          initialized weights instead of pretrained ones,
                          then fully trained. This isolates "did pretraining
                          help at all," not just "is fine-tuning better than
                          freezing."

Dataset: GLUE SST-2 (binary sentiment), subsampled for a fast run.
Checkpoint: distilbert-base-uncased.

REQUIRES INTERNET / HUGGING FACE HUB ACCESS to download the pretrained
checkpoint and dataset — this is the one topic in this repository that
cannot be verified in a network-restricted sandbox, because the entire
point of the comparison is "does a genuinely pretrained backbone help,"
which is meaningless without genuinely pretrained weights. See STATUS.md
for details. Runs in ~10-15 minutes on a Colab T4.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "distilbert-base-uncased"

# ---------------------------------------------------------------------------
# 1. Data — SST-2, subsampled for a fast demonstration run
# ---------------------------------------------------------------------------
raw = load_dataset("glue", "sst2")
train_raw = raw["train"].shuffle(seed=0).select(range(2000))
val_raw = raw["validation"].shuffle(seed=0).select(range(400))

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=64)

train_ds = train_raw.map(tokenize, batched=True)
val_ds = val_raw.map(tokenize, batched=True)
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


# ---------------------------------------------------------------------------
# 2. Model wrapper: a DistilBERT backbone + linear head, with backbone
#    optionally frozen and optionally pretrained.
# ---------------------------------------------------------------------------
class ProbeOrFineTuneModel(nn.Module):
    def __init__(self, pretrained: bool, freeze_backbone: bool):
        super().__init__()
        if pretrained:
            self.backbone = AutoModel.from_pretrained(CHECKPOINT)
        else:
            config = AutoConfig.from_pretrained(CHECKPOINT)
            self.backbone = AutoModel.from_config(config)   # same architecture, random init

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.last_hidden_state[:, 0, :]   # [CLS] token representation
        return self.head(cls_repr)


# ---------------------------------------------------------------------------
# 3. Train / evaluate loop, shared across all three variants
# ---------------------------------------------------------------------------
def run_variant(name: str, pretrained: bool, freeze_backbone: bool, epochs: int = 3, lr: float = 2e-5):
    model = ProbeOrFineTuneModel(pretrained, freeze_backbone).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr if not freeze_backbone else 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids, attn)
            loss = loss_fn(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # validation accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attn = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                preds = model(input_ids, attn).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        history.append(acc)
        print(f"  [{name}] epoch {epoch+1}/{epochs}  val_acc={acc:.4f}")

    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    return {"history": history, "trainable_params": n_trainable, "total_params": n_total}


# ---------------------------------------------------------------------------
# 4. Run all three variants
# ---------------------------------------------------------------------------
results = {}
t0 = time.time()

print("Running (A) Frozen probe — pretrained backbone, linear head only...")
results["Frozen Probe\n(pretrained, frozen)"] = run_variant("frozen-probe", pretrained=True, freeze_backbone=True)

print("Running (B) Full fine-tune — pretrained backbone, everything trained...")
results["Full Fine-Tune\n(pretrained, unfrozen)"] = run_variant("full-fine-tune", pretrained=True, freeze_backbone=False)

print("Running (C) Random-init baseline — same architecture, no pretraining...")
results["Random Init\n(no pretraining)"] = run_variant("random-init", pretrained=False, freeze_backbone=False)

elapsed = time.time() - t0
print(f"\nAll three variants completed in {elapsed/60:.1f} min on {DEVICE}")

# ---------------------------------------------------------------------------
# 5. proof.png — accuracy curves + final-accuracy comparison
# ---------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

colors = {"Frozen Probe\n(pretrained, frozen)": "#4361ee",
          "Full Fine-Tune\n(pretrained, unfrozen)": "#f72585",
          "Random Init\n(no pretraining)": "#8d99ae"}

for name, r in results.items():
    ax0.plot(range(1, len(r["history"]) + 1), r["history"], marker="o", label=name.replace("\n", " "), color=colors[name])
ax0.set_xlabel("epoch"); ax0.set_ylabel("validation accuracy")
ax0.set_title("SST-2 Validation Accuracy by Training Regime")
ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

names = list(results.keys())
finals = [results[n]["history"][-1] for n in names]
bars = ax1.bar(names, finals, color=[colors[n] for n in names], edgecolor="black")
for bar, r in zip(bars, results.values()):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
              f"{bar.get_height():.3f}\n{r['trainable_params']:,} trainable",
              ha="center", fontsize=8)
ax1.set_ylabel("final validation accuracy")
ax1.set_title(f"Final Accuracy · {elapsed/60:.1f} min total on {str(DEVICE).upper()}")
ax1.set_ylim(0, 1)

fig.suptitle("Topic 01.03 — Pretraining vs. Fine-Tuning: Frozen Probe vs. Full Fine-Tune vs. No Pretraining", fontsize=11)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("Saved proof.png")
