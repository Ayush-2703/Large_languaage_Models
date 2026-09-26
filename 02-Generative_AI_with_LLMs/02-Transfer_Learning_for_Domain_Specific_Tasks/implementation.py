"""
Topic 02.02 — Transfer Learning for Domain-Specific Tasks
=============================================================
Demonstrates domain-adaptive pretraining (DAPT — Gururangan et al., "Don't
Stop Pretraining," 2020): does a brief, unlabeled continued-pretraining pass
on IN-DOMAIN text, inserted between general pretraining and task fine-tuning,
improve downstream performance versus fine-tuning directly on an
out-of-domain checkpoint?

Three arms, same downstream task, same fine-tuning budget:
    (A) No adaptation  — general-domain (Shakespeare) pretrained checkpoint,
                          fine-tuned directly on the target domain's task.
    (B) DAPT            — the SAME starting checkpoint, first continued-
                          pretrained (MLM) on unlabeled target-domain text,
                          THEN fine-tuned on the target domain's task.
    (C) No pretraining  — random initialization, trained directly on the
                          target domain's task. Isolates how much of any
                          gap is "pretraining helps at all" vs. "domain-
                          matched pretraining helps more."

Source domain: general narrative text (tiny-Shakespeare).
Target domain: synthetic tech-product reviews — a deliberately different
vocabulary and register from both Shakespeare and the movie-review corpus
used in 01-Fine-Tuning-Principles-and-Techniques.

No Hugging Face Hub download required. Runs in about a minute on CPU.
"""

import math
import random
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Source domain (general) and target domain (tech reviews) corpora
# ---------------------------------------------------------------------------
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")

def load_source_corpus(n_chars: int = 150_000) -> str:
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        return resp.read().decode("utf-8")[:n_chars]

TECH_POS = ["snappy", "responsive", "sleek", "reliable", "impressive", "seamless", "sturdy", "efficient"]
TECH_NEG = ["sluggish", "buggy", "flimsy", "unreliable", "frustrating", "clunky", "overpriced", "glitchy"]
TECH_TEMPLATES = ["the battery life is {adj}", "the interface feels {adj}", "this laptop is {adj} for daily use",
                   "the build quality is {adj}", "software updates are {adj}", "the camera performance is {adj}",
                   "the touchpad is {adj} to use", "for the price this device is {adj}"]

def build_tech_corpus(n: int):
    sentences, labels = [], []
    for _ in range(n):
        is_pos = random.random() < 0.5
        adj = random.choice(TECH_POS if is_pos else TECH_NEG)
        sentences.append(random.choice(TECH_TEMPLATES).format(adj=adj))
        labels.append(1 if is_pos else 0)
    return sentences, labels

source_text = load_source_corpus()
# Unlabeled pool for DAPT continued-pretraining (larger, no labels needed/used)
dapt_sents, _ = build_tech_corpus(1500)
dapt_text = " ".join(dapt_sents)
# Separate, labeled pool for the actual downstream fine-tuning task
task_sents, task_labels = build_tech_corpus(1200)

all_text = source_text + dapt_text + " ".join(task_sents)
chars = sorted(set(all_text))
SPECIALS = ["[PAD]", "[MASK]", "[CLS]"]
vocab = SPECIALS + chars
stoi = {c: i for i, c in enumerate(vocab)}
vocab_size = len(vocab)
PAD, MASK, CLS = stoi["[PAD]"], stoi["[MASK]"], stoi["[CLS]"]
BLOCK_SIZE = 44

def encode(s: str, max_len: int = BLOCK_SIZE) -> torch.Tensor:
    ids = [CLS] + [stoi.get(c, PAD) for c in s[:max_len - 1]]
    ids = ids + [PAD] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

source_ids = torch.tensor([stoi[c] for c in source_text], dtype=torch.long)
dapt_ids = torch.tensor([stoi[c] for c in dapt_text], dtype=torch.long)

# ---------------------------------------------------------------------------
# 2. Same small bidirectional encoder architecture as 02.01
# ---------------------------------------------------------------------------
D_MODEL, N_HEADS, N_LAYERS = 48, 4, 3

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_heads
        q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, hd).transpose(1, 2)
        attn_mask = None
        if pad_mask is not None:
            attn_mask = pad_mask[:, None, None, :].expand(B, self.n_heads, T, T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class MiniEncoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def encode(self, ids, pad_mask=None):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.tok_embed(ids) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h, pad_mask)
        return self.ln_f(h)

    def forward_mlm(self, ids, pad_mask=None):
        return self.mlm_head(self.encode(ids, pad_mask))

class ClassifierHead(nn.Module):
    def __init__(self, encoder: MiniEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(D_MODEL, 2)

    def forward(self, ids, pad_mask=None):
        return self.head(self.encoder.encode(ids, pad_mask)[:, 0, :])

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def mlm_batch(source_ids_tensor, batch_size=32, block_size=BLOCK_SIZE):
    """Structurally matches encode()'s [CLS, ...text...] format so the CLS
    token's embedding and the encoder's handling of a CLS-first sequence
    are actually exercised during pretraining — not just during fine-tuning,
    where it would otherwise start untouched since MLM here draws directly
    from a mid-corpus window, which naturally contains no [CLS] token."""
    ix = torch.randint(len(source_ids_tensor) - (block_size - 1), (batch_size,))
    body = torch.stack([source_ids_tensor[i:i + block_size - 1] for i in ix]).clone()
    cls_col = torch.full((batch_size, 1), CLS, dtype=torch.long)
    x = torch.cat([cls_col, body], dim=1)
    y = x.clone()
    maskable = torch.rand(x.shape) < 0.15
    maskable[:, 0] = False   # never mask/predict the CLS token itself
    y[~maskable] = -100
    x[maskable] = MASK
    return x.to(DEVICE), y.to(DEVICE)

def mlm_train(model, ids_tensor, steps, lr=3e-3, block_size=BLOCK_SIZE):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        xb, yb = mlm_batch(ids_tensor, block_size=block_size)
        logits = model.forward_mlm(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return loss.item()

# ---------------------------------------------------------------------------
# 3. Stage 1 — general-domain pretraining (shared starting point for A and B)
# ---------------------------------------------------------------------------
print("Stage 1: general-domain (Shakespeare) pretraining...")
t0 = time.time()
general_model = MiniEncoder().to(DEVICE)
general_loss = mlm_train(general_model, source_ids, steps=500)
general_state = {k: v.clone() for k, v in general_model.state_dict().items()}
print(f"  done in {time.time()-t0:.1f}s, final MLM loss={general_loss:.3f}, params={count_params(general_model):,}\n")

# ---------------------------------------------------------------------------
# 4. Stage 2 (arm B only) — domain-adaptive continued pretraining on
#    UNLABELED target-domain (tech review) text
# ---------------------------------------------------------------------------
print("Stage 2 (arm B): domain-adaptive continued pretraining on tech-review text...")
t0 = time.time()
dapt_model = MiniEncoder().to(DEVICE)
dapt_model.load_state_dict(general_state)     # start from the SAME general checkpoint
dapt_loss = mlm_train(dapt_model, dapt_ids, steps=200, lr=3e-4)
dapt_state = {k: v.clone() for k, v in dapt_model.state_dict().items()}
print(f"  done in {time.time()-t0:.1f}s, final MLM loss={dapt_loss:.3f}\n")

# ---------------------------------------------------------------------------
# 5. Downstream task data + fine-tuning harness (shared across all 3 arms)
# ---------------------------------------------------------------------------
n = len(task_sents)
perm = list(range(n)); random.shuffle(perm)
# Deliberately data-scarce fine-tuning regime: transfer's benefit is clearest
# with limited labeled data (see 01-Review-of-Fundamental-LLMs/
# 03-Pretraining-vs-Fine-Tuning-Paradigms, theory.md §5) — with 900 labeled
# examples this easy a task, every arm saturates near 100% and the
# comparison stops being informative.
train_idx, val_idx = perm[:150], perm[150:450]
TOTAL_STEPS, BATCH = 250, 16

def prep_batch(idx):
    ids = torch.stack([encode(task_sents[i]) for i in idx]).to(DEVICE)
    pad_mask = (ids != PAD).float().to(DEVICE)
    y = torch.tensor([task_labels[i] for i in idx], dtype=torch.long, device=DEVICE)
    return ids, pad_mask, y

def evaluate(model):
    model.eval()
    with torch.no_grad():
        ids, pad_mask, y = prep_batch(val_idx)
        acc = (model(ids, pad_mask).argmax(-1) == y).float().mean().item()
    model.train()
    return acc

def finetune(model, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = []
    for step in range(TOTAL_STEPS):
        idx = random.sample(train_idx, BATCH)
        ids, pad_mask, y = prep_batch(idx)
        loss = F.cross_entropy(model(ids, pad_mask), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 20 == 0:
            hist.append(evaluate(model))
    return hist

print("Stage 3: fine-tuning all three arms on the SAME labeled target-domain task...")
enc_a = MiniEncoder().to(DEVICE); enc_a.load_state_dict(general_state)
model_a = ClassifierHead(enc_a).to(DEVICE)
hist_a = finetune(model_a)
print(f"  (A) No adaptation      final={hist_a[-1]:.3f}")

enc_b = MiniEncoder().to(DEVICE); enc_b.load_state_dict(dapt_state)
model_b = ClassifierHead(enc_b).to(DEVICE)
hist_b = finetune(model_b)
print(f"  (B) DAPT               final={hist_b[-1]:.3f}")

enc_c = MiniEncoder().to(DEVICE)   # random init, no pretraining at all
model_c = ClassifierHead(enc_c).to(DEVICE)
hist_c = finetune(model_c)
print(f"  (C) No pretraining     final={hist_c[-1]:.3f}")

# ---------------------------------------------------------------------------
# 6. proof.png
# ---------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
histories = {"(A) No Adaptation\n(general ckpt -> task)": hist_a,
             "(B) DAPT\n(general -> tech text -> task)": hist_b,
             "(C) No Pretraining\n(random init -> task)": hist_c}
colors = ["#8d99ae", "#f72585", "#4361ee"]
xs = list(range(0, TOTAL_STEPS, 20))
for (name, h), c in zip(histories.items(), colors):
    ax0.plot(xs[:len(h)], h, marker="o", markersize=3, label=name.split("\n")[0], color=c)
ax0.set_xlabel("fine-tuning step"); ax0.set_ylabel("validation accuracy")
ax0.set_title("A. Convergence on Target-Domain Task", fontsize=10)
ax0.legend(fontsize=7.5); ax0.grid(alpha=0.3)

names = list(histories.keys())
finals = [h[-1] for h in histories.values()]
bars = ax1.bar(range(len(names)), finals, color=colors, edgecolor="black")
ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, fontsize=7.5)
for bar, v in zip(bars, finals):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax1.set_ylabel("final validation accuracy"); ax1.set_ylim(0, 1)
ax1.set_title(f"B. Final Accuracy · same {TOTAL_STEPS}-step fine-tune budget for all 3", fontsize=10)

fig.suptitle("Topic 02.02 — Transfer Learning: Does Domain-Adaptive Pretraining Help?", fontsize=11)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("\nSaved proof.png")
