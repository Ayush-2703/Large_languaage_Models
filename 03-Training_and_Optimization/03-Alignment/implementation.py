"""
Topic 03.03 — Alignment Methodologies: RLHF and DPO
=======================================================
Builds a synthetic (prompt, chosen, rejected) preference dataset with a
KNOWN, controlled quality signal (chosen = helpful/polite response styles,
rejected = curt/dismissive styles) and implements two real, trained
alignment techniques on it:

  (A) REWARD MODEL — a scalar-output encoder trained via the Bradley-Terry
      pairwise preference loss (Christiano et al., 2017): the same
      mechanism a full RLHF pipeline uses to turn human preference labels
      into a scalar reward signal.

  (B) DPO (Direct Preference Optimization, Rafailov et al., 2023) — trains
      a policy DIRECTLY on preference pairs, using a frozen reference copy
      of the pre-alignment policy, with no separate reward model and no RL
      rollouts at all.

RLHF's remaining piece — PPO-based policy optimization against the reward
model from (A), with KL penalty and advantage estimation — is deliberately
NOT implemented here. This was scoped explicitly at the start of this
repository's build: correctly implementing PPO's rollout/advantage/clipping
machinery carries real risk of subtle bugs that would produce code that
"runs" without actually demonstrating correct RL optimization, which would
be worse than a rigorous conceptual treatment. theory.md covers the full
algorithm, its math, and why DPO emerged as a popular simpler alternative.

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
# 1. Synthetic preference data with a known, controlled quality signal
# ---------------------------------------------------------------------------
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")

def load_corpus(n_chars=120_000):
    with urllib.request.urlopen(CORPUS_URL, timeout=15) as resp:
        return resp.read().decode("utf-8")[:n_chars]

PROMPTS = [
    "how do I fix this bug", "what should I cook tonight", "can you explain photosynthesis",
    "how do I improve my resume", "what is the capital of France", "how do I learn to code",
    "can you help me plan a trip", "what is a good book to read", "how does the internet work",
    "what should I study today",
]
CHOSEN_TEMPLATES = [
    "Sure, here is a clear answer for you.",
    "Of course, let me help you with that.",
    "Great question, glad to assist you now.",
    "Absolutely, here is some good guidance.",
]
REJECTED_TEMPLATES = [
    "Not my problem, go figure it out.",
    "I really do not know or care much.",
    "That is boring, ask someone else now.",
    "Whatever, just go look it up there.",
]

def build_preference_data(n_pairs=400):
    triples = []
    for _ in range(n_pairs):
        prompt = random.choice(PROMPTS)
        chosen = random.choice(CHOSEN_TEMPLATES)
        rejected = random.choice(REJECTED_TEMPLATES)
        triples.append((prompt, chosen, rejected))
    return triples

pretrain_text = load_corpus()
pref_data = build_preference_data(400)

all_text = (pretrain_text + " ".join(PROMPTS) + " ".join(CHOSEN_TEMPLATES) + " ".join(REJECTED_TEMPLATES))
chars = sorted(set(all_text))
SPECIALS = ["[PAD]", "[CLS]", "[SEP]"]
vocab = SPECIALS + chars
stoi = {c: i for i, c in enumerate(vocab)}
VOCAB = len(vocab)
PAD, CLS, SEP = stoi["[PAD]"], stoi["[CLS]"], stoi["[SEP]"]
BLOCK = 96

def encode_for_reward(prompt, response, max_len=BLOCK):
    """[CLS] prompt [SEP] response -> encoder input for the reward model."""
    text = prompt + " [SEP] " + response
    ids = [CLS]
    for tok in text.split(" "):
        if tok == "[SEP]":
            ids.append(SEP)
        else:
            ids += [stoi.get(c, PAD) for c in tok] + [stoi[" "]]
    ids = ids[:max_len]
    return torch.tensor(ids + [PAD] * (max_len - len(ids)), dtype=torch.long)

def encode_plain(text, max_len=BLOCK):
    ids = [stoi.get(c, PAD) for c in text[:max_len]]
    return torch.tensor(ids + [PAD] * (max_len - len(ids)), dtype=torch.long)

pretrain_ids = torch.tensor([stoi[c] for c in pretrain_text], dtype=torch.long)
perm = list(range(len(pref_data))); random.shuffle(perm)
train_idx, val_idx = perm[:320], perm[320:]

D_MODEL, N_HEADS, N_LAYERS = 64, 4, 2

# ---------------------------------------------------------------------------
# 2. Shared architecture pieces (bidirectional encoder for the reward model;
#    causal decoder for the DPO policy/reference)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False, pad_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_heads
        q = q.view(B, T, self.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, hd).transpose(1, 2)
        attn_mask = pad_mask[:, None, None, :].expand(B, self.n_heads, T, T) if pad_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, causal):
        super().__init__()
        self.causal = causal
        self.ln1 = nn.LayerNorm(d_model); self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), causal=self.causal, pad_mask=None if self.causal else pad_mask)
        return x + self.mlp(self.ln2(x))

def count_params(m): return sum(p.numel() for p in m.parameters())

# ===========================================================================
# (A) REWARD MODEL — bidirectional encoder + scalar head, Bradley-Terry loss
# ===========================================================================
print("=" * 70)
print("(A) REWARD MODEL")
print("=" * 70)

class RewardModel(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, d_model)
        self.pos_embed = nn.Embedding(BLOCK, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, causal=False) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, ids, pad_mask=None):
        B, T = ids.shape
        h = self.tok_embed(ids) + self.pos_embed(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            h = blk(h, pad_mask)
        h = self.ln_f(h)
        return self.reward_head(h[:, 0, :]).squeeze(-1)   # scalar reward per (prompt, response)

rm = RewardModel().to(DEVICE)
opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
print(f"Reward model params: {count_params(rm):,}")

def rm_batch(indices):
    chosen_ids, rejected_ids, pad_c, pad_r = [], [], [], []
    for i in indices:
        prompt, chosen, rejected = pref_data[i]
        c = encode_for_reward(prompt, chosen); r = encode_for_reward(prompt, rejected)
        chosen_ids.append(c); rejected_ids.append(r)
    chosen_ids = torch.stack(chosen_ids).to(DEVICE)
    rejected_ids = torch.stack(rejected_ids).to(DEVICE)
    pad_c = (chosen_ids != PAD).float(); pad_r = (rejected_ids != PAD).float()
    return chosen_ids, rejected_ids, pad_c, pad_r

def rm_eval():
    rm.eval()
    with torch.no_grad():
        c_ids, r_ids, pad_c, pad_r = rm_batch(val_idx)
        r_chosen = rm(c_ids, pad_c); r_rejected = rm(r_ids, pad_r)
        acc = (r_chosen > r_rejected).float().mean().item()
        margin = (r_chosen - r_rejected).mean().item()
    rm.train()
    return acc, margin

acc0, margin0 = rm_eval()
print(f"Before training: pairwise accuracy={acc0:.3f}  mean reward margin={margin0:+.3f}")

t0 = time.time()
rm_history = []
for step in range(300):
    idx = random.sample(train_idx, 32)
    c_ids, r_ids, pad_c, pad_r = rm_batch(idx)
    r_chosen = rm(c_ids, pad_c); r_rejected = rm(r_ids, pad_r)
    # Bradley-Terry pairwise loss: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 20 == 0:
        acc, margin = rm_eval()
        rm_history.append((step, acc, margin))
rm_time = time.time() - t0
acc_final, margin_final = rm_eval()
print(f"After training ({rm_time:.1f}s): pairwise accuracy={acc_final:.3f}  mean reward margin={margin_final:+.3f}\n")

# ===========================================================================
# (B) DPO — direct preference optimization on a causal policy model
# ===========================================================================
print("=" * 70)
print("(B) DPO (Direct Preference Optimization)")
print("=" * 70)

class CausalLM(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, d_model)
        self.pos_embed = nn.Embedding(BLOCK, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, causal=True) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, VOCAB)

    def forward(self, ids):
        B, T = ids.shape
        h = self.tok_embed(ids) + self.pos_embed(torch.arange(T, device=ids.device))
        for blk in self.blocks:
            h = blk(h)
        return self.lm_head(self.ln_f(h))

print("Pretraining/SFT-ing the policy backbone (next-token prediction)...")
policy = CausalLM().to(DEVICE)
opt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
for step in range(500):
    ix = torch.randint(len(pretrain_ids) - BLOCK - 1, (32,))
    xb = torch.stack([pretrain_ids[i:i+BLOCK] for i in ix]).to(DEVICE)
    yb = torch.stack([pretrain_ids[i+1:i+BLOCK+1] for i in ix]).to(DEVICE)
    loss = F.cross_entropy(policy(xb).view(-1, VOCAB), yb.view(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print(f"  done, params={count_params(policy):,}")

# Reference model: a FROZEN copy of the policy at this exact point — DPO's
# implicit KL penalty is measured against this snapshot, never updated again.
reference = CausalLM().to(DEVICE)
reference.load_state_dict(policy.state_dict())
for p in reference.parameters():
    p.requires_grad = False
reference.eval()

def sequence_logprob(model, prompt, response, requires_grad=True):
    """log pi(response | prompt) = sum of log-probs the model assigns to
    each response token, conditioned (via preceding context) on the prompt."""
    full = prompt + " " + response
    ids = encode_plain(full).unsqueeze(0).to(DEVICE)
    ctx = ids[:, :-1]
    targets = ids[:, 1:]
    prompt_len = min(len(prompt), BLOCK - 1)
    ctx_manager = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx_manager:
        logits = model(ctx)
        logprobs = F.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        response_mask = torch.zeros_like(targets, dtype=torch.bool)
        response_mask[:, prompt_len:] = True
        response_mask &= (targets != PAD)
        seq_logprob = (token_logprobs * response_mask).sum()
    return seq_logprob

BETA = 0.1

def dpo_eval(idx_list):
    correct, total_margin = 0, 0.0
    for i in idx_list:
        prompt, chosen, rejected = pref_data[i]
        with torch.no_grad():
            lp_c_policy = sequence_logprob(policy, prompt, chosen, requires_grad=False)
            lp_r_policy = sequence_logprob(policy, prompt, rejected, requires_grad=False)
            lp_c_ref = sequence_logprob(reference, prompt, chosen, requires_grad=False)
            lp_r_ref = sequence_logprob(reference, prompt, rejected, requires_grad=False)
        implicit_reward_margin = (lp_c_policy - lp_c_ref) - (lp_r_policy - lp_r_ref)
        total_margin += implicit_reward_margin.item()
        if lp_c_policy > lp_r_policy:
            correct += 1
    return correct / len(idx_list), total_margin / len(idx_list)

acc0_dpo, margin0_dpo = dpo_eval(val_idx[:40])
print(f"Before DPO: policy prefers chosen {acc0_dpo:.3f} of the time  implicit reward margin={margin0_dpo:+.3f}")

opt = torch.optim.AdamW(policy.parameters(), lr=5e-5)
t0 = time.time()
dpo_history = []
DPO_STEPS = 150
for step in range(DPO_STEPS):
    idx = random.sample(train_idx, 8)
    total_loss = 0.0
    opt.zero_grad(set_to_none=True)
    for i in idx:
        prompt, chosen, rejected = pref_data[i]
        lp_c_policy = sequence_logprob(policy, prompt, chosen, requires_grad=True)
        lp_r_policy = sequence_logprob(policy, prompt, rejected, requires_grad=True)
        with torch.no_grad():
            lp_c_ref = sequence_logprob(reference, prompt, chosen, requires_grad=False)
            lp_r_ref = sequence_logprob(reference, prompt, rejected, requires_grad=False)
        logits_dpo = BETA * ((lp_c_policy - lp_c_ref) - (lp_r_policy - lp_r_ref))
        loss = -F.logsigmoid(logits_dpo) / len(idx)
        loss.backward()
        total_loss += loss.item()
    opt.step()
    if step % 15 == 0:
        acc, margin = dpo_eval(val_idx[:40])
        dpo_history.append((step, acc, margin))
        print(f"  step {step:3d}  loss={total_loss:.4f}  policy-prefers-chosen={acc:.3f}  implicit_margin={margin:+.3f}")
dpo_time = time.time() - t0

acc_final_dpo, margin_final_dpo = dpo_eval(val_idx[:40])
print(f"\nAfter DPO ({dpo_time:.1f}s, {DPO_STEPS} steps): "
      f"policy prefers chosen {acc_final_dpo:.3f} of the time  implicit reward margin={margin_final_dpo:+.3f}\n")

# ===========================================================================
# proof.png
# ===========================================================================
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

steps_rm = [s for s, a, m in rm_history]
acc_rm = [a for s, a, m in rm_history]
margin_rm = [m for s, a, m in rm_history]
ax0b = ax0.twinx()
ax0.plot(steps_rm, acc_rm, marker="o", color="#4361ee", label="pairwise accuracy")
ax0b.plot(steps_rm, margin_rm, marker="s", color="#f72585", label="reward margin")
ax0.set_xlabel("training step"); ax0.set_ylabel("pairwise accuracy", color="#4361ee")
ax0b.set_ylabel("reward margin (chosen - rejected)", color="#f72585")
ax0.set_title(f"A. Reward Model: Real Training\n{acc0:.3f}->{acc_final:.3f} accuracy, {margin0:+.3f}->{margin_final:+.3f} margin", fontsize=10)
ax0.set_ylim(0, 1.05)

steps_dpo = [s for s, a, m in dpo_history]
acc_dpo = [a for s, a, m in dpo_history]
margin_dpo = [m for s, a, m in dpo_history]
ax1b = ax1.twinx()
ax1.plot(steps_dpo, acc_dpo, marker="o", color="#3a0ca3", label="policy prefers chosen")
ax1b.plot(steps_dpo, margin_dpo, marker="s", color="#2a9d3f", label="implicit reward margin")
ax1.set_xlabel("DPO training step"); ax1.set_ylabel("P(policy prefers chosen)", color="#3a0ca3")
ax1b.set_ylabel("implicit reward margin", color="#2a9d3f")
ax1.set_title(f"B. DPO: Real Training (no reward model, no RL)\n{acc0_dpo:.3f}->{acc_final_dpo:.3f} preference-accuracy", fontsize=10)
ax1.set_ylim(0, 1.05)

fig.suptitle("Topic 03.03 — Alignment: Reward Model (Bradley-Terry) + DPO, Both Real and Trained", fontsize=11)
plt.tight_layout()
plt.savefig("proof.png", dpi=150, bbox_inches="tight")
print("Saved proof.png")

print("\n[RLHF's PPO-based policy optimization step is NOT implemented here —")
print(" deliberately, per this repository's original scope. See theory.md for")
print(" the full algorithm, its math, and why DPO is a common simpler alternative.]")
