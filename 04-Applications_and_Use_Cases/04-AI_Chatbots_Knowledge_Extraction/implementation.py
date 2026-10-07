"""
Topic 04.04 — Conversational AI, Chatbots, and Knowledge Extraction
=======================================================================
Two real, from-scratch experiments:

  (A) MULTI-TURN DIALOGUE CONTEXT RETENTION — a decoder trained on
      synthetic multi-turn conversations where the user states a fact in
      an early turn and asks about it in a later turn. Evaluated on
      HELD-OUT fact combinations never seen during training, to test
      genuine context-tracking across turns rather than memorized
      question-answer pairs (the same held-out-generalization discipline
      established in 04.01 and 04.02, applied here to dialogue).

  (B) KNOWLEDGE EXTRACTION VIA NER — a bidirectional encoder trained for
      token-level BIO-tagged named-entity recognition (person / organization
      / location), the standard formulation for extracting structured
      entities from unstructured text, evaluated with real token-level and
      entity-level metrics.

No Hugging Face Hub download required. Runs in a couple of minutes on CPU.
"""

import random
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)
random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# (A) MULTI-TURN DIALOGUE CONTEXT RETENTION
# ===========================================================================
print("=" * 70)
print("(A) Multi-Turn Dialogue Context Retention")
print("=" * 70)

NAMES = ["Alex", "Priya", "Sam", "Noor", "Kim", "Leo", "Mia", "Ravi"]
COLORS = ["blue", "green", "red", "purple", "yellow", "orange"]
JOBS = ["a teacher", "an engineer", "a doctor", "a chef", "a pilot", "a writer"]

def make_dialogue(name, color, job):
    """3-turn dialogue: user states two facts, then asks about ONE of them
    in a later turn — the model must have retained the earlier statement
    across the intervening turn to answer correctly."""
    turn1 = f"User: Hi, I am {name} and my favorite color is {color}. Assistant: Nice to meet you, {name}."
    turn2 = f"User: I work as {job}. Assistant: That sounds like a great job."
    turn3 = f"User: What is my favorite color? Assistant: Your favorite color is {color}."
    return turn1 + " " + turn2 + " " + turn3

all_combos = [(n, c, j) for n in NAMES for c in COLORS for j in JOBS]
random.shuffle(all_combos)
HELD_OUT_COMBOS = all_combos[:20]
SEEN_COMBOS = all_combos[20:]

CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")

def load_corpus(n_chars=60_000):
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        return resp.read().decode("utf-8")[:n_chars]

general_text = load_corpus()
actually_trained_combos = random.choices(SEEN_COMBOS, k=400)
dialogue_examples = [make_dialogue(n, c, j) for n, c, j in actually_trained_combos]
dialogue_text = "\n".join(dialogue_examples)
full_text = general_text + "\n" + dialogue_text

chars = sorted(set(full_text))
PAD = "\x00"
vocab = [PAD] + chars
stoi = {c: i for i, c in enumerate(vocab)}
VOCAB = len(vocab)
PAD_ID = stoi[PAD]
BLOCK = 256

max_dialogue_len = max(len(make_dialogue(n, c, j)) for n, c, j in HELD_OUT_COMBOS[:5])
print(f"Corpus: {len(full_text):,} chars, vocab={VOCAB}, max sample dialogue length={max_dialogue_len} "
      f"(BLOCK={BLOCK} confirmed sufficient)\n")
assert max_dialogue_len < BLOCK, "BLOCK too small for a full dialogue — checked before training, not after"

pretrain_ids = torch.tensor([stoi[c] for c in full_text], dtype=torch.long)

D_MODEL, N_HEADS, N_LAYERS = 96, 6, 3

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_heads
        q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class DialogueGPT(nn.Module):
    def __init__(self, block_size=BLOCK):
        super().__init__()
        self.block_size = block_size
        self.tok_embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos_embed = nn.Embedding(block_size, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.tok_embed(ids) + self.pos_embed(pos)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))

    @torch.no_grad()
    def generate(self, ids, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.block_size:])[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

def count_params(m): return sum(p.numel() for p in m.parameters())

print("Pretraining dialogue model...")
dlg_model = DialogueGPT().to(DEVICE)
opt = torch.optim.AdamW(dlg_model.parameters(), lr=3e-3)
t0 = time.time()
for step in range(1000):
    ix = torch.randint(len(pretrain_ids) - BLOCK - 1, (16,))
    xb = torch.stack([pretrain_ids[i:i+BLOCK] for i in ix]).to(DEVICE)
    yb = torch.stack([pretrain_ids[i+1:i+BLOCK+1] for i in ix]).to(DEVICE)
    loss = F.cross_entropy(dlg_model(xb).view(-1, VOCAB), yb.view(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(f"  done in {time.time()-t0:.1f}s, params={count_params(dlg_model):,}, final loss={loss.item():.3f}\n")

def encode_str(s):
    return torch.tensor([stoi.get(c, PAD_ID) for c in s], dtype=torch.long)

def eval_dialogue(combos, n_new_tokens=20):
    dlg_model.eval()
    correct = 0
    examples = []
    for name, color, job in combos:
        prompt = (f"User: Hi, I am {name} and my favorite color is {color}. Assistant: Nice to meet you, {name}. "
                  f"User: I work as {job}. Assistant: That sounds like a great job. "
                  f"User: What is my favorite color? Assistant: Your favorite color is")
        ids = encode_str(prompt).unsqueeze(0).to(DEVICE)
        out = dlg_model.generate(ids, n_new_tokens)
        generated = "".join(vocab[i] for i in out[0, ids.shape[1]:].tolist())
        said_color = generated.strip().split(".")[0].split(",")[0].strip().split(" ")[0] if generated.strip() else ""
        is_correct = said_color == color
        correct += is_correct
        examples.append((name, color, job, generated[:20], is_correct))
    dlg_model.train()
    return correct / len(combos), examples

seen_acc, seen_examples = eval_dialogue(random.sample(sorted(set(actually_trained_combos)), 20))
ho_acc, ho_examples = eval_dialogue(HELD_OUT_COMBOS)
print(f"Context-retention accuracy — SEEN name/color/job combos:      {seen_acc:.3f}")
print(f"Context-retention accuracy — HELD-OUT name/color/job combos:  {ho_acc:.3f}")
for name, color, job, gen, ok in ho_examples[:4]:
    print(f"  [{'correct' if ok else 'WRONG'}] {name}/{color}/{job} -> generated: {gen!r}")
print()

# ===========================================================================
# (B) KNOWLEDGE EXTRACTION VIA NER (BIO tagging)
# ===========================================================================
print("=" * 70)
print("(B) Knowledge Extraction: Named Entity Recognition (BIO tagging)")
print("=" * 70)

PERSONS = ["Elena Cruz", "Marcus Lee", "Fatima Noor", "David Chen", "Ingrid Voss", "Omar Said"]
ORGS = ["Bright Robotics", "Solace Health", "Vantage Freight", "Cedar Analytics", "Northlight Bank", "Quiver Media"]
LOCS = ["Denver", "Lisbon", "Nairobi", "Osaka", "Toronto", "Cairo"]
NER_TEMPLATES = [
    "{person} joined {org} in {loc} last spring.",
    "{org} announced a new office in {loc}, led by {person}.",
    "{person} met with representatives from {org} while visiting {loc}.",
    "Reports say {org} is expanding operations near {loc} under {person}.",
]

def make_ner_example(person, org, loc, template):
    text = template.format(person=person, org=org, loc=loc)
    words = text.replace(".", " .").replace(",", " ,").split()
    labels = []
    p_words, o_words, l_words = person.split(), org.split(), loc.split()
    i = 0
    while i < len(words):
        if words[i:i+len(p_words)] == p_words:
            labels += ["B-PER"] + ["I-PER"] * (len(p_words) - 1); i += len(p_words)
        elif words[i:i+len(o_words)] == o_words:
            labels += ["B-ORG"] + ["I-ORG"] * (len(o_words) - 1); i += len(o_words)
        elif words[i:i+len(l_words)] == l_words:
            labels += ["B-LOC"] + ["I-LOC"] * (len(l_words) - 1); i += len(l_words)
        else:
            labels.append("O"); i += 1
    return words, labels

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label2id = {l: i for i, l in enumerate(NER_LABELS)}

ner_examples = []
for _ in range(500):
    p, o, l, t = random.choice(PERSONS), random.choice(ORGS), random.choice(LOCS), random.choice(NER_TEMPLATES)
    ner_examples.append(make_ner_example(p, o, l, t))

n = len(ner_examples)
perm = list(range(n)); random.shuffle(perm)
ner_train, ner_val = perm[:400], perm[400:]

ner_vocab = sorted(set(w for words, _ in ner_examples for w in words))
NER_PAD = "<pad>"
w2id = {NER_PAD: 0, **{w: i+1 for i, w in enumerate(ner_vocab)}}
NER_VOCAB = len(w2id)
NER_BLOCK = max(len(words) for words, _ in ner_examples)

def encode_ner(words, labels):
    ids = [w2id.get(w, 0) for w in words] + [0] * (NER_BLOCK - len(words))
    lbl = [label2id[l] for l in labels] + [-100] * (NER_BLOCK - len(labels))
    return torch.tensor(ids), torch.tensor(lbl)

class NERAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_heads
        q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, hd).transpose(1, 2)
        attn_mask = pad_mask[:, None, None, :].expand(B, self.n_heads, T, T) if pad_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class NERBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.attn = NERAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        return x + self.mlp(self.ln2(x))

class NERTagger(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.tok_embed = nn.Embedding(NER_VOCAB, d_model)
        self.pos_embed = nn.Embedding(NER_BLOCK, d_model)
        self.blocks = nn.ModuleList([NERBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.tag_head = nn.Linear(d_model, len(NER_LABELS))

    def forward(self, ids, pad_mask=None):
        B, T = ids.shape
        h = self.tok_embed(ids) + self.pos_embed(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            h = blk(h, pad_mask)
        return self.tag_head(self.ln_f(h))

ner_model = NERTagger().to(DEVICE)
opt = torch.optim.AdamW(ner_model.parameters(), lr=1e-3)
print(f"Training NER tagger ({count_params(ner_model):,} params, {len(NER_LABELS)} BIO labels)...")
t0 = time.time()
for step in range(300):
    idx = random.sample(ner_train, 32)
    batch = [encode_ner(*ner_examples[i]) for i in idx]
    ids = torch.stack([b[0] for b in batch]).to(DEVICE)
    lbl = torch.stack([b[1] for b in batch]).to(DEVICE)
    pad_mask = (ids != 0).float().to(DEVICE)
    loss = F.cross_entropy(ner_model(ids, pad_mask).view(-1, len(NER_LABELS)), lbl.view(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(f"  done in {time.time()-t0:.1f}s, final loss={loss.item():.3f}\n")

def entity_spans(labels):
    spans, cur = [], None
    for i, lbl in enumerate(labels):
        if lbl.startswith("B-"):
            if cur: spans.append(cur)
            cur = [lbl[2:], i, i]
        elif lbl.startswith("I-") and cur and cur[0] == lbl[2:]:
            cur[2] = i
        else:
            if cur: spans.append(cur)
            cur = None
    if cur: spans.append(cur)
    return set(tuple(s) for s in spans)

ner_model.eval()
token_correct, token_total = 0, 0
entity_tp, entity_fp, entity_fn = 0, 0, 0
with torch.no_grad():
    for i in ner_val:
        words, true_labels = ner_examples[i]
        ids, lbl = encode_ner(words, true_labels)
        ids_b = ids.unsqueeze(0).to(DEVICE)
        pad_mask = (ids_b != 0).float().to(DEVICE)
        preds = ner_model(ids_b, pad_mask).argmax(-1)[0, :len(words)].cpu().tolist()
        pred_labels = [NER_LABELS[p] for p in preds]
        token_correct += sum(p == t for p, t in zip(pred_labels, true_labels))
        token_total += len(true_labels)
        true_spans = entity_spans(true_labels)
        pred_spans = entity_spans(pred_labels)
        entity_tp += len(true_spans & pred_spans)
        entity_fp += len(pred_spans - true_spans)
        entity_fn += len(true_spans - pred_spans)

token_acc = token_correct / token_total
precision = entity_tp / (entity_tp + entity_fp + 1e-8)
recall = entity_tp / (entity_tp + entity_fn + 1e-8)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
print(f"Token-level accuracy:  {token_acc:.3f}")
print(f"Entity-level:  precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
print(f"  (TP={entity_tp} FP={entity_fp} FN={entity_fn} exact-span entity matches)\n")

# ===========================================================================
# proof.png
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax0 = axes[0]
bars = ax0.bar(["Seen combos", "Held-out combos"], [seen_acc, ho_acc], color=["#4361ee", "#f72585"], edgecolor="black")
for bar, v in zip(bars, [seen_acc, ho_acc]):
    ax0.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=10)
ax0.set_ylim(0, 1.05)
ax0.set_ylabel("context-retention accuracy")
ax0.set_title("A. Multi-Turn Dialogue: Recalling an\nEarlier-Stated Fact 2 Turns Later", fontsize=10)

ax1 = axes[1]
ax1.bar(["Token\naccuracy", "Entity\nprecision", "Entity\nrecall", "Entity\nF1"],
        [token_acc, precision, recall, f1], color=["#3a0ca3", "#4361ee", "#7209b7", "#f72585"], edgecolor="black")
for i, v in enumerate([token_acc, precision, recall, f1]):
    ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ax1.set_ylim(0, 1.05)
ax1.set_title("B. NER Knowledge Extraction\n(real BIO-tag token classifier)", fontsize=10)

ax2 = axes[2]
ax2.axis("off")
ex_text = "HELD-OUT DIALOGUE EXAMPLES:\n\n"
for name, color, job, gen, ok in ho_examples[:5]:
    mark = "correct" if ok else "WRONG"
    ex_text += f"[{mark}] {name} said fav color={color}\n   generated: {gen.strip()[:24]!r}\n"
ax2.text(0.02, 0.98, ex_text, fontsize=8.5, family="monospace", va="top", transform=ax2.transAxes)
ax2.set_title("C. Real Generation Examples (held-out)", fontsize=10)

fig.suptitle("Topic 04.04 — Conversational AI & Knowledge Extraction: Real Dialogue Context + Real NER", fontsize=11)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("Saved proof.png")
