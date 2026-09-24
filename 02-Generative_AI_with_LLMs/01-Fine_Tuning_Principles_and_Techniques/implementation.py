"""
Topic 02.01 — Principles and Techniques for Fine-Tuning Pretrained Models
============================================================================= 
Pretrains one small bidirectional Transformer encoder via masked-language-
modeling (MLM) on general narrative text, then loads four IDENTICAL copies
of that same pretrained checkpoint and fine-tunes each with a different
technique on a downstream sentiment-classification task:

    (A) Naive full fine-tune    — every parameter, one flat learning rate.
    (B) Discriminative LR       — lower LR for early (more general) layers,
                                   higher LR for later layers + head
                                   (Howard & Ruder, ULMFiT, 2018).
    (C) Warmup + cosine decay   — LR ramps up then decays smoothly, the
                                   schedule used by nearly every modern
                                   Transformer fine-tuning recipe.
    (D) Gradual unfreezing      — start with only the classification head
                                   trainable; unfreeze one more encoder
                                   layer every few hundred steps
                                   (Howard & Ruder, 2018).

All four start from byte-identical pretrained weights and train for the
same total step budget, isolating the fine-tuning TECHNIQUE as the only
variable — same logic used for the architecture comparisons in Phase 01.

No Hugging Face Hub download required: this topic is about technique
mechanics, which a small self-pretrained checkpoint demonstrates just as
validly as a production one. Runs in well under a minute on CPU.
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
# 1. Pretraining corpus (general narrative text) and downstream task corpus
#    (synthetic movie-review sentiment) — deliberately different domains,
#    exactly like a real pretrain-on-web-text / fine-tune-on-labeled-task split.
# ---------------------------------------------------------------------------
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")

def load_pretrain_corpus(n_chars: int = 150_000) -> str:
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        return resp.read().decode("utf-8")[:n_chars]

POSITIVE_WORDS = ["brilliant", "wonderful", "captivating", "outstanding", "delightful", "masterful", "engaging", "superb"]
NEGATIVE_WORDS = ["terrible", "boring", "disappointing", "awful", "tedious", "clumsy", "forgettable", "dreadful"]
REVIEW_TEMPLATES = ["the movie was {adj}", "a {adj} film with strong direction", "critics called it {adj}",
                     "audiences found it {adj}", "the plot was {adj} throughout", "the acting was {adj}",
                     "overall a {adj} experience", "this film is {adj} from start to finish"]

def build_review_corpus(n: int = 1200):
    sentences, labels = [], []
    for _ in range(n):
        is_pos = random.random() < 0.5
        adj = random.choice(POSITIVE_WORDS if is_pos else NEGATIVE_WORDS)
        sentences.append(random.choice(REVIEW_TEMPLATES).format(adj=adj))
        labels.append(1 if is_pos else 0)
    return sentences, labels

pretrain_text = load_pretrain_corpus()
review_sents, review_labels = build_review_corpus()

# Shared character vocabulary spans both corpora plus special tokens.
all_text = pretrain_text + " ".join(review_sents)
chars = sorted(set(all_text))
SPECIALS = ["[PAD]", "[MASK]", "[CLS]"]
vocab = SPECIALS + chars
stoi = {c: i for i, c in enumerate(vocab)}
vocab_size = len(vocab)
PAD, MASK, CLS = stoi["[PAD]"], stoi["[MASK]"], stoi["[CLS]"]

BLOCK_SIZE = 40   # downstream sentences are short; keep pretraining context matched

def encode(s: str, max_len: int = BLOCK_SIZE) -> torch.Tensor:
    """Position 0 is always the [CLS] token id; the sentence fills positions
    1..max_len-1, padded with [PAD] if shorter."""
    ids = [CLS] + [stoi.get(c, PAD) for c in s[:max_len - 1]]
    ids = ids + [PAD] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

pretrain_ids = torch.tensor([stoi[c] for c in pretrain_text], dtype=torch.long)

# ---------------------------------------------------------------------------
# 2. A small bidirectional Transformer encoder (BERT-style: full, non-causal
#    self-attention — appropriate here since this topic studies FINE-TUNING
#    technique, not architecture family; that comparison is Topic 02.03)
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# ---------------------------------------------------------------------------
# 3. Pretrain ONE encoder via MLM — this becomes the shared starting
#    checkpoint every fine-tuning technique below loads a fresh copy of.
# ---------------------------------------------------------------------------
def get_pretrain_batch(batch_size=32):
    ix = torch.randint(len(pretrain_ids) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([pretrain_ids[i:i + BLOCK_SIZE] for i in ix]).clone()
    y = x.clone()
    mask_prob = torch.rand(x.shape)
    maskable = mask_prob < 0.15
    y[~maskable] = -100                      # ignore_index: only compute loss on masked positions
    x[maskable] = MASK
    return x.to(DEVICE), y.to(DEVICE)

print("Pretraining shared encoder via MLM...")
base_model = MiniEncoder().to(DEVICE)
opt = torch.optim.AdamW(base_model.parameters(), lr=3e-3)
t0 = time.time()
for step in range(500):
    xb, yb = get_pretrain_batch()
    logits = base_model.forward_mlm(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
pretrain_time = time.time() - t0
pretrained_state = {k: v.clone() for k, v in base_model.state_dict().items()}
print(f"  Pretraining done in {pretrain_time:.1f}s, final MLM loss={loss.item():.3f}, "
      f"params={count_params(base_model):,}\n")

# ---------------------------------------------------------------------------
# 4. Downstream classification model wrapper + shared data split
# ---------------------------------------------------------------------------
class ClassifierHead(nn.Module):
    def __init__(self, encoder: MiniEncoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(D_MODEL, 2)

    def forward(self, ids, pad_mask=None):
        h = self.encoder.encode(ids, pad_mask)
        cls_repr = h[:, 0, :]
        return self.head(cls_repr)

def prep_batch(sentences, labels, idx):
    ids = torch.stack([encode(sentences[i]) for i in idx]).to(DEVICE)
    pad_mask = (ids != PAD).float().to(DEVICE)
    y = torch.tensor([labels[i] for i in idx], dtype=torch.long, device=DEVICE)
    return ids, pad_mask, y

n = len(review_sents)
perm = list(range(n)); random.shuffle(perm)
train_idx, val_idx = perm[:900], perm[900:]

TOTAL_STEPS = 300
BATCH = 32

def iterate_steps(total_steps, batch_size):
    for _ in range(total_steps):
        idx = random.sample(train_idx, batch_size)
        yield idx

def evaluate(model):
    model.eval()
    with torch.no_grad():
        ids, pad_mask, y = prep_batch(review_sents, review_labels, val_idx)
        preds = model(ids, pad_mask).argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    model.train()
    return acc

def fresh_model():
    m = MiniEncoder().to(DEVICE)
    m.load_state_dict(pretrained_state)
    return ClassifierHead(m).to(DEVICE)

# ---------------------------------------------------------------------------
# 5. Four fine-tuning techniques, identical steps, identical starting weights
# ---------------------------------------------------------------------------
histories = {}

# (A) Naive: flat LR, full fine-tune
model_a = fresh_model()
opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
hist_a = []
for step, idx in enumerate(iterate_steps(TOTAL_STEPS, BATCH)):
    ids, pad_mask, y = prep_batch(review_sents, review_labels, idx)
    loss = F.cross_entropy(model_a(ids, pad_mask), y)
    opt_a.zero_grad(set_to_none=True); loss.backward(); opt_a.step()
    if step % 20 == 0:
        hist_a.append(evaluate(model_a))
histories["(A) Naive Flat-LR"] = hist_a

# (B) Discriminative LR: lower LR for earlier layers, higher for later + head
model_b = fresh_model()
param_groups = []
n_layers = len(model_b.encoder.blocks)
for i, block in enumerate(model_b.encoder.blocks):
    lr_i = 1e-4 * (3 ** i)              # each later layer gets 3x the previous layer's LR
    param_groups.append({"params": block.parameters(), "lr": lr_i})
param_groups.append({"params": model_b.encoder.tok_embed.parameters(), "lr": 1e-4})
param_groups.append({"params": model_b.encoder.pos_embed.parameters(), "lr": 1e-4})
param_groups.append({"params": model_b.encoder.ln_f.parameters(), "lr": 5e-4})
param_groups.append({"params": model_b.encoder.mlm_head.parameters(), "lr": 5e-4})
param_groups.append({"params": model_b.head.parameters(), "lr": 3e-3})   # head trains fastest
opt_b = torch.optim.AdamW(param_groups)
hist_b = []
for step, idx in enumerate(iterate_steps(TOTAL_STEPS, BATCH)):
    ids, pad_mask, y = prep_batch(review_sents, review_labels, idx)
    loss = F.cross_entropy(model_b(ids, pad_mask), y)
    opt_b.zero_grad(set_to_none=True); loss.backward(); opt_b.step()
    if step % 20 == 0:
        hist_b.append(evaluate(model_b))
histories["(B) Discriminative LR"] = hist_b

# (C) Warmup + cosine decay, full fine-tune
model_c = fresh_model()
opt_c = torch.optim.AdamW(model_c.parameters(), lr=1e-3)
warmup_steps = 30
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, TOTAL_STEPS - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
sched_c = torch.optim.lr_scheduler.LambdaLR(opt_c, lr_lambda)
hist_c = []
for step, idx in enumerate(iterate_steps(TOTAL_STEPS, BATCH)):
    ids, pad_mask, y = prep_batch(review_sents, review_labels, idx)
    loss = F.cross_entropy(model_c(ids, pad_mask), y)
    opt_c.zero_grad(set_to_none=True); loss.backward(); opt_c.step(); sched_c.step()
    if step % 20 == 0:
        hist_c.append(evaluate(model_c))
histories["(C) Warmup+Cosine"] = hist_c

# (D) Gradual unfreezing: head-only first, unfreeze one more layer every 60 steps
model_d = fresh_model()
for p in model_d.encoder.parameters():
    p.requires_grad = False
UNFREEZE_EVERY = 60
opt_d = torch.optim.AdamW(model_d.parameters(), lr=1e-3)
hist_d = []
for step, idx in enumerate(iterate_steps(TOTAL_STEPS, BATCH)):
    n_unfrozen = min(n_layers, step // UNFREEZE_EVERY)
    for i, block in enumerate(model_d.encoder.blocks):
        want_trainable = i >= (n_layers - n_unfrozen)
        for p in block.parameters():
            p.requires_grad = want_trainable
    opt_d = torch.optim.AdamW([p for p in model_d.parameters() if p.requires_grad], lr=1e-3)
    ids, pad_mask, y = prep_batch(review_sents, review_labels, idx)
    loss = F.cross_entropy(model_d(ids, pad_mask), y)
    opt_d.zero_grad(set_to_none=True); loss.backward(); opt_d.step()
    if step % 20 == 0:
        hist_d.append(evaluate(model_d))
histories["(D) Gradual Unfreeze"] = hist_d

print("Final validation accuracy by technique:")
for name, h in histories.items():
    print(f"  {name:24s} final={h[-1]:.3f}  peak={max(h):.3f}")

# ---------------------------------------------------------------------------
# 6. proof.png
# ---------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
colors = {"(A) Naive Flat-LR": "#8d99ae", "(B) Discriminative LR": "#4361ee",
          "(C) Warmup+Cosine": "#3a0ca3", "(D) Gradual Unfreeze": "#f72585"}
xs = list(range(0, TOTAL_STEPS, 20))
for name, h in histories.items():
    ax0.plot(xs[:len(h)], h, marker="o", markersize=3, label=name, color=colors[name])
ax0.set_xlabel("fine-tuning step"); ax0.set_ylabel("validation accuracy")
ax0.set_title("A. Convergence by Fine-Tuning Technique\n(identical pretrained start, identical step budget)", fontsize=10)
ax0.legend(fontsize=7.5); ax0.grid(alpha=0.3)

names = list(histories.keys())
finals = [histories[n][-1] for n in names]
bars = ax1.bar(names, finals, color=[colors[n] for n in names], edgecolor="black")
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
for bar, v in zip(bars, finals):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax1.set_ylabel("final validation accuracy")
ax1.set_title(f"B. Final Accuracy · {count_params(base_model):,}-param encoder · {TOTAL_STEPS} steps each", fontsize=10)
ax1.set_ylim(0, 1)

fig.suptitle("Topic 02.01 — Fine-Tuning Techniques: Same Pretrained Weights, Same Budget, Different Strategy", fontsize=11)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("\nSaved proof.png")
