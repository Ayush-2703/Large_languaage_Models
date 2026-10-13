"""
Topic 5.4 — Alignment and Controllability of LLMs
====================================================

GOAL
----
Reproduce, in miniature, the actual optimization mechanism behind
Direct Preference Optimization (DPO) (Rafailov et al., 2023, "Direct
Preference Optimization: Your Language Model is Secretly a Reward
Model") and use it to demonstrate two real, measured phenomena central
to LLM alignment:

1. **Controllability**: a language model's behavior can be
   deliberately shifted toward a specified preference using only
   pairwise preference data, without ever training an explicit reward
   model or running reinforcement learning.
2. **The alignment/capability trade-off ("alignment tax")**: pushing
   behavior toward the preference too aggressively degrades the
   model's other, unrelated capabilities -- and the strength of this
   trade-off is a direct, controllable function of the KL-regularization
   coefficient (beta) in the DPO objective.

SETUP
-----
- A small causal Transformer is pretrained from scratch (real
  cross-entropy training, no downloaded weights) on sequences from a
  fixed, structured synthetic "grammar" -- a Markov chain with a
  strong forward-cycling bias. This gives a base policy with a real,
  non-trivial capability (predicting the grammar) that alignment
  training can either preserve or damage.
- A synthetic **preference signal** is defined as "sequences containing
  more occurrences of a designated token" -- a stand-in for any
  scalar human preference (e.g., helpfulness, politeness) that is
  largely orthogonal to grammaticality.
- An offline preference dataset is built by sampling several
  completions per prompt from the frozen base (reference) policy and
  labeling the highest- and lowest-reward samples as chosen/rejected --
  exactly how a real preference dataset is built from a base model's
  own outputs, without any external reward model.
- The trainable policy (initialized from the same pretrained weights)
  is optimized directly on this preference data using the closed-form
  DPO loss, at several values of the KL-strength hyperparameter beta.

WHY THIS IS A FAIR STAND-IN FOR REAL RLHF/DPO
------------------------------------------------
This is not a simplified metaphor for DPO -- it is the literal DPO
loss, `-log sigmoid(beta * ((logpi(chosen)-logref(chosen)) -
(logpi(rejected)-logref(rejected))))`, applied to a real (if tiny)
autoregressive Transformer and a real preference dataset built from
that model's own sampled completions. The only thing that is
synthetic is the size of the model and the definition of "preference"
(a token-frequency count rather than a human/AI judgment) -- the
optimization mechanics are identical to production DPO pipelines.

SCALING TO PRODUCTION LLMS
---------------------------
On a Colab T4, the direct real-checkpoint analogue fine-tunes a small
GPT-2 checkpoint with DPO on a small preference dataset (e.g., a
downsampled subset of Anthropic's HH-RLHF-style pairs or a synthetic
instruction-following preference set), using the HF `trl` library's
`DPOTrainer`. The beta-sweep result reproduced here at toy scale --
smaller beta => bigger preference-satisfaction gain but bigger
degradation elsewhere -- is the same trade-off documented in
production RLHF/DPO literature on reward over-optimization
(Gao, Schulman, Hilton, 2023, "Scaling Laws for Reward Model
Overoptimization").
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
V = 8                 # vocabulary size for the toy "grammar"
T = 10                # sequence length
D_MODEL, N_HEADS, N_LAYERS = 32, 2, 2
REWARD_TOKEN = 0       # frequency of this token defines the synthetic preference
PRETRAIN_STEPS = 800
N_PROMPTS, K_SAMPLES = 300, 4   # offline preference-dataset construction
DPO_EPOCHS, DPO_BATCH = 60, 32
BETAS = [0.05, 0.2, 0.5, 1.0, 3.0]   # KL-strength sweep

# A strongly-structured Markov chain: mostly cycles forward by 1 -- a real,
# learnable sequential grammar (not close to uniform-random).
transition = torch.full((V, V), 0.15 / (V - 1))
for i in range(V):
    transition[i, (i + 1) % V] = 0.85
transition = transition / transition.sum(dim=1, keepdim=True)


def sample_markov_seq(batch, length, gen):
    seqs = torch.zeros(batch, length, dtype=torch.long)
    seqs[:, 0] = torch.randint(0, V, (batch,), generator=gen)
    for t in range(1, length):
        probs = transition[seqs[:, t - 1]]
        seqs[:, t] = torch.multinomial(probs, 1, generator=gen).squeeze(1)
    return seqs


class CausalLM(nn.Module):
    """Small causal (decoder-only) Transformer -- the same architecture
    family as GPT-2, scaled down for CPU-only, from-scratch training."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, D_MODEL)
        self.pos_emb = nn.Embedding(T, D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, dim_feedforward=128, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Linear(D_MODEL, V)

    def forward(self, x):
        pos = torch.arange(x.size(1)).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        h = self.enc(h, mask=mask, is_causal=True)
        return self.head(h)


def pretrain(model, steps, batch=64):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        seqs = sample_markov_seq(batch, T, gen)
        logits = model(seqs[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), seqs[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, loss.item()


@torch.no_grad()
def sample_from_model(model, batch, length, gen, temperature=1.0):
    seqs = torch.zeros(batch, length, dtype=torch.long)
    seqs[:, 0] = torch.randint(0, V, (batch,), generator=gen)
    for t in range(1, length):
        logits = model(seqs[:, :t])[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        seqs[:, t] = torch.multinomial(probs, 1, generator=gen).squeeze(1)
    return seqs


@torch.no_grad()
def markov_loglik(seqs):
    """Average log-likelihood of generated sequences under the TRUE grammar
    -- our proxy for 'has the model retained its original capability?'"""
    ll = 0.0
    for t in range(1, seqs.size(1)):
        p = transition[seqs[:, t - 1], seqs[:, t]]
        ll += torch.log(p + 1e-9).mean().item()
    return ll


def reward(seqs):
    return (seqs == REWARD_TOKEN).float().sum(dim=1)


def seq_logprob(model, seqs):
    logits = model(seqs[:, :-1])
    logprobs = F.log_softmax(logits, dim=-1)
    tgt = seqs[:, 1:].unsqueeze(-1)
    picked = torch.gather(logprobs, 2, tgt).squeeze(-1)
    return picked.sum(dim=1)


def build_preference_dataset(ref_model, n_prompts, k_samples, seed=2024):
    gen = torch.Generator().manual_seed(seed)
    chosen_list, rejected_list = [], []
    for _ in range(n_prompts):
        cands = sample_from_model(ref_model, k_samples, T, gen, temperature=1.2)
        r = reward(cands)
        hi, lo = torch.argmax(r).item(), torch.argmin(r).item()
        if hi == lo:
            continue
        chosen_list.append(cands[hi])
        rejected_list.append(cands[lo])
    return torch.stack(chosen_list), torch.stack(rejected_list)


def train_dpo(base_model, ref_model, chosen, rejected, beta, epochs=DPO_EPOCHS, batch=DPO_BATCH):
    policy_model = copy.deepcopy(base_model)
    opt = torch.optim.Adam(policy_model.parameters(), lr=1e-3)
    n = chosen.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            c, rj = chosen[idx], rejected[idx]
            with torch.no_grad():
                lref_c = seq_logprob(ref_model, c)
                lref_r = seq_logprob(ref_model, rj)
            lpi_c = seq_logprob(policy_model, c)
            lpi_r = seq_logprob(policy_model, rj)
            dpo_logits = beta * ((lpi_c - lref_c) - (lpi_r - lref_r))
            loss = -F.logsigmoid(dpo_logits).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy_model


def run():
    t0 = time.time()
    base, pretrain_loss = pretrain(CausalLM(), steps=PRETRAIN_STEPS)
    ref_model = copy.deepcopy(base)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    chosen, rejected = build_preference_dataset(ref_model, N_PROMPTS, K_SAMPLES)

    gen_eval = torch.Generator().manual_seed(777)
    base_samples = sample_from_model(ref_model, 300, T, gen_eval)
    base_reward = reward(base_samples).float().mean().item()
    base_loglik = markov_loglik(base_samples)

    results = {
        "pretrain_loss": pretrain_loss,
        "n_preference_pairs": chosen.size(0),
        "base_reward": base_reward,
        "base_markov_loglik": base_loglik,
        "betas": [],
        "dpo_reward": [],
        "dpo_markov_loglik": [],
    }
    print(f"BASE (reference, no alignment): avg_reward={base_reward:.3f} markov_loglik={base_loglik:.2f}")

    for beta in BETAS:
        policy = train_dpo(base, ref_model, chosen, rejected, beta=beta)
        gen_eval_b = torch.Generator().manual_seed(777)
        samples = sample_from_model(policy, 300, T, gen_eval_b)
        r, ll = reward(samples).float().mean().item(), markov_loglik(samples)
        results["betas"].append(beta)
        results["dpo_reward"].append(r)
        results["dpo_markov_loglik"].append(ll)
        print(f"beta={beta:5.2f} | avg_reward={r:.3f} | markov_loglik={ll:.2f}")

    print(f"Total time: {time.time() - t0:.1f}s on 1 CPU core")
    return results


if __name__ == "__main__":
    run()
