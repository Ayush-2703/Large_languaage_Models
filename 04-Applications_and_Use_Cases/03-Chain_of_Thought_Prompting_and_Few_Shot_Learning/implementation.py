"""
Topic 04.03 — Chain-of-Thought (CoT) Prompting and Few-Shot Learning
========================================================================
A real in-context-learning experiment: NO weight updates happen between
conditions below. A single pretrained decoder is given prompts with 0, 1,
2, or 3 worked examples, then must complete a NEW problem correctly purely
from patterns in the prompt — genuine few-shot / in-context learning, not
fine-tuning.

Two things are measured, both for real:
  (A) The few-shot LEARNING CURVE — does accuracy improve as more examples
      are shown in-context (0-shot -> 1-shot -> 2-shot -> 3-shot)?
  (B) CHAIN-OF-THOUGHT vs. DIRECT-ANSWER prompting — do few-shot examples
      that show worked reasoning steps improve accuracy over few-shot
      examples that show only the final answer?

IMPORTANT EXPECTATION, stated before running anything: Wei et al. ("Chain-
of-Thought Prompting Elicits Reasoning in Large Language Models," 2022)
found that CoT's benefit is itself an emergent, scale-dependent property —
it shows negligible or even negative effect on small models and only
becomes reliably positive above a certain model scale. The model trained
here (a few hundred thousand parameters) is far below that regime. If CoT
does not clearly help below, that is not a failed experiment — it would be
a small-scale replication consistent with the published literature, and is
reported as such either way. See theory.md for the full discussion.

No Hugging Face Hub download required. Runs in a couple of minutes on CPU.
"""

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
# 1. Synthetic arithmetic word-problem generator, with a DIRECT-ANSWER format
#    and a CHAIN-OF-THOUGHT format for the same underlying problems.
# ---------------------------------------------------------------------------
NAMES = ["Sam", "Ravi", "Mia", "Leo", "Zara", "Tom", "Ana", "Kai"]
ITEMS = ["apples", "coins", "marbles", "pencils"]

def make_problem():
    name = random.choice(NAMES)
    item = random.choice(ITEMS)
    start = random.randint(2, 9)
    gain = random.randint(1, 5)
    lose = random.randint(1, min(4, start + gain - 1))
    answer = start + gain - lose
    problem = f"{name} had {start} {item}, gained {gain}, lost {lose}. Total?"
    direct = f"Answer: {answer}"
    cot = f"Reasoning: {start}+{gain}={start+gain}, {start+gain}-{lose}={answer}. Answer: {answer}"
    return problem, answer, direct, cot

def sample_fresh_problem(seen_signatures):
    while True:
        problem, answer, direct, cot = make_problem()
        if problem not in seen_signatures:
            return problem, answer, direct, cot

# ---------------------------------------------------------------------------
# 2. Pretraining corpus: general narrative text + many worked arithmetic
#    examples (both direct-answer and CoT-style), giving the model broad
#    exposure to the TASK FORMAT without memorizing specific test problems.
# ---------------------------------------------------------------------------
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")

def load_corpus(n_chars=100_000):
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        return resp.read().decode("utf-8")[:n_chars]

general_text = load_corpus()

pretrain_problems = []
seen_signatures = set()
for _ in range(1500):
    problem, answer, direct, cot = make_problem()
    seen_signatures.add(problem)
    style = random.choice(["direct", "cot"])
    completion = direct if style == "direct" else cot
    pretrain_problems.append(f"Problem: {problem} {completion}")

arithmetic_text = "\n".join(pretrain_problems)
full_pretrain_text = general_text + "\n" + arithmetic_text

chars = sorted(set(full_pretrain_text))
PAD = "\x00"
vocab = [PAD] + chars
stoi = {c: i for i, c in enumerate(vocab)}
VOCAB = len(vocab)
PAD_ID = stoi[PAD]
BLOCK = 384

pretrain_ids = torch.tensor([stoi[c] for c in full_pretrain_text], dtype=torch.long)
print(f"Pretraining corpus: {len(full_pretrain_text):,} chars "
      f"({len(general_text):,} general + {len(arithmetic_text):,} arithmetic, "
      f"{len(pretrain_problems)} worked examples), vocab={VOCAB}\n")

# ---------------------------------------------------------------------------
# 3. A modestly larger decoder — this topic is specifically testing an
#    emergent-style capability, so it gets more capacity than this
#    repository's typical smallest demo models, though still far below the
#    scale Wei et al. find necessary for reliable CoT benefit.
# ---------------------------------------------------------------------------
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

class GPT(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, block_size=BLOCK):
        super().__init__()
        self.block_size = block_size
        self.tok_embed = nn.Embedding(VOCAB, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB)

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
            ids_cond = ids[:, -self.block_size:]
            logits = self(ids_cond)[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

def count_params(m): return sum(p.numel() for p in m.parameters())

print("Pretraining decoder on general text + worked arithmetic examples...")
model = GPT().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
t0 = time.time()
for step in range(500):
    ix = torch.randint(len(pretrain_ids) - BLOCK - 1, (8,))
    xb = torch.stack([pretrain_ids[i:i+BLOCK] for i in ix]).to(DEVICE)
    yb = torch.stack([pretrain_ids[i+1:i+BLOCK+1] for i in ix]).to(DEVICE)
    loss = F.cross_entropy(model(xb).view(-1, VOCAB), yb.view(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(f"  done in {time.time()-t0:.1f}s, params={count_params(model):,}, final loss={loss.item():.3f}\n")

# ---------------------------------------------------------------------------
# 4. Few-shot / in-context evaluation — NO weight updates from here on.
#    Fresh test problems are sampled to be DISTINCT from every pretraining
#    problem (checked via `seen_signatures`), so correct answers cannot be
#    memorized verbatim from pretraining.
# ---------------------------------------------------------------------------
def encode_str(s):
    return torch.tensor([stoi.get(c, PAD_ID) for c in s], dtype=torch.long)

def build_prompt(k_shot, style, test_problem):
    lines = []
    for _ in range(k_shot):
        p, a, direct, cot = sample_fresh_problem(seen_signatures)
        completion = direct if style == "direct" else cot
        lines.append(f"Problem: {p} {completion}")
    prefix = "\n".join(lines)
    tail = f"Problem: {test_problem} " + ("Reasoning:" if style == "cot" else "Answer:")
    return (prefix + "\n" + tail) if prefix else tail

def extract_answer(generated_text, style):
    """For 'direct' style, the prompt itself already ends in 'Answer:', so
    the model's generated continuation should START with the number
    directly — there is no 'Answer:' substring to search for WITHIN the
    generated text itself. For 'cot' style, the model must generate its own
    reasoning AND its own trailing 'Answer: <n>', so searching for the last
    'Answer:' within the generated text is correct there. An earlier version
    of this function used the cot-style search logic for BOTH styles,
    which meant direct-style trials always returned None (0% accuracy)
    regardless of what the model actually generated — a real extraction
    bug, not a finding about the model's arithmetic ability. See
    explanation.md."""
    if style == "direct":
        digits = ""
        for ch in generated_text.lstrip():
            if ch.isdigit():
                digits += ch
            elif digits:
                break
            else:
                break
        return int(digits) if digits else None
    else:
        if "Answer:" not in generated_text:
            return None
        tail = generated_text.rsplit("Answer:", 1)[1]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            elif digits:
                break
            elif ch not in " ":
                break
        return int(digits) if digits else None

def evaluate(k_shot, style, n_trials=15):
    max_new_tokens = 6 if style == "direct" else 40
    model.eval()
    correct = 0
    for _ in range(n_trials):
        test_problem, true_answer, _, _ = sample_fresh_problem(seen_signatures)
        prompt = build_prompt(k_shot, style, test_problem)
        ids = encode_str(prompt).unsqueeze(0).to(DEVICE)
        out = model.generate(ids, max_new_tokens=max_new_tokens)
        generated = "".join(vocab[i] for i in out[0, ids.shape[1]:].tolist())
        pred = extract_answer(generated, style)
        if pred == true_answer:
            correct += 1
    model.train()
    return correct / n_trials

print("=== (A) Few-shot learning curve (direct-answer style) ===")
k_values = [0, 1, 2, 3]
direct_accs = []
for k in k_values:
    acc = evaluate(k, "direct")
    direct_accs.append(acc)
    print(f"  {k}-shot: accuracy = {acc:.3f}")

print("\n=== (B) Chain-of-Thought vs. Direct-Answer, at each shot count ===")
cot_accs = []
for k in k_values:
    acc = evaluate(k, "cot")
    cot_accs.append(acc)
    print(f"  {k}-shot CoT: accuracy = {acc:.3f}   (direct was {direct_accs[k_values.index(k)]:.3f})")

print("\n[Interpreting these numbers: see theory.md for why a flat or unclear CoT")
print(" benefit at this model scale is consistent with, not contrary to, Wei et")
print(" al. (2022)'s finding that CoT's benefit is itself scale-emergent.]\n")

# ---------------------------------------------------------------------------
# proof.png
# ---------------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

ax0.plot(k_values, direct_accs, marker="o", color="#4361ee", label="Direct-answer few-shot")
ax0.plot(k_values, cot_accs, marker="s", color="#f72585", label="Chain-of-Thought few-shot")
ax0.set_xlabel("number of in-context examples (k-shot)")
ax0.set_ylabel("accuracy on fresh, unseen problems")
ax0.set_title(f"A. Few-Shot Learning Curve\n({count_params(model):,}-param model, NO weight updates between points)", fontsize=10)
ax0.set_ylim(0, 1.05); ax0.legend(fontsize=8); ax0.grid(alpha=0.3)
ax0.set_xticks(k_values)

x = range(len(k_values))
w = 0.35
ax1.bar([i - w/2 for i in x], direct_accs, w, label="Direct-answer", color="#4361ee", edgecolor="black")
ax1.bar([i + w/2 for i in x], cot_accs, w, label="Chain-of-Thought", color="#f72585", edgecolor="black")
ax1.set_xticks(list(x)); ax1.set_xticklabels([f"{k}-shot" for k in k_values])
ax1.set_ylabel("accuracy"); ax1.set_ylim(0, 1.05); ax1.legend(fontsize=8)
ax1.set_title("B. CoT vs. Direct-Answer at Each Shot Count\n(does showing reasoning steps help THIS model?)", fontsize=10)

fig.suptitle("Topic 04.03 — Few-Shot Learning & CoT: Real In-Context Evaluation, No Fine-Tuning", fontsize=11)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("Saved proof.png")
