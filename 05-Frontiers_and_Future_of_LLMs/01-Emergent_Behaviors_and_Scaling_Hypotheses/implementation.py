"""
Topic 5.1 — Emergent Behaviors and Scaling Hypotheses
======================================================

GOAL
----
Give a *real, measurable* example of a sudden capability transition — the
phenomenon researchers point to when they debate whether "emergent abilities"
in LLMs are a genuine phase transition or an artifact of the metric used
(Wei et al., 2022, "Emergent Abilities of Large Language Models";
Schaeffer et al., 2023, "Are Emergent Abilities of Large Language Models a
Mirage?"). We reproduce the cleanest controlled instance of this debate that
exists in the literature: **grokking** (Power et al., 2022, "Grokking:
Generalization Beyond Overfitting on Small Algorithmic Datasets").

TASK
----
A 2-layer Transformer encoder is trained to compute modular addition,
(a + b) mod P, on half of all possible (a, b) pairs for a small P. This is
small enough to train from scratch on a single CPU core in under two
minutes, while still reproducing the qualitative phenomenon at full-size
LLM scale: the model reaches 100% *training* accuracy almost immediately
(it has memorized the training pairs), but *validation* accuracy stays
near chance for thousands of further optimization steps before suddenly
jumping to ~100% — i.e., it "groks" the underlying mod-P addition rule
long after it looked done.

WHY THIS MATTERS FOR THE TOPIC
-------------------------------
This is exactly the shape of curve that fuels the "emergent abilities"
debate at LLM scale: a capability that looks flat/near-zero on a
coarse (exact-match) metric for a long time, then rises sharply once
some internal representation crystallizes — versus the alternative
explanation that the same underlying improvement was continuous all
along and only *looks* sudden through a nonlinear or thresholded metric.
Because our setup is small enough to fully control, we can watch the
underlying mechanism (weight decay pressure on redundant memorization
circuits) directly, which is the closest thing to "ground truth" on this
debate available at 1-CPU-core scale.

SCALING TO PRODUCTION LLMS
---------------------------
At GPT-3/4 scale, the same qualitative pattern is proposed (not proven)
to explain step-function-looking jumps in benchmark accuracy as model
size or training tokens cross a threshold (e.g. 3-digit arithmetic,
multi-step reasoning). The mechanism argued for grokking — the optimizer
slowly moving from a "memorizing" solution to a lower-norm "generalizing"
solution under weight decay / implicit regularization — is one candidate
explanation researchers extend (informally) to why some LLM capabilities
appear to switch on abruptly with scale, though this remains an open
research question rather than a settled result.
"""

import random
import time

import torch
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
P = 13                      # modulus -> P*P = 169 total (a, b) pairs
TRAIN_FRACTION = 0.5         # classic grokking uses a held-out split, not all data
D_MODEL, N_HEADS, N_LAYERS = 32, 2, 2
LR, WEIGHT_DECAY = 1e-3, 2.0  # weight decay is the key ingredient that induces grokking
TOTAL_STEPS = 10_000
EVAL_EVERY = 100
VOCAB = P + 1                 # tokens 0..P-1 are the numbers, token P is the '=' token


class TinyGrokTransformer(nn.Module):
    """Minimal decoder-style Transformer encoder for the 3-token sequence [a, b, '=']."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(3, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.head = nn.Linear(D_MODEL, P)

    def forward(self, x):
        pos = torch.arange(x.size(1)).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return self.head(h[:, -1, :])  # prediction is read off the '=' position


def build_dataset():
    pairs = [(a, b) for a in range(P) for b in range(P)]
    random.shuffle(pairs)
    split = int(TRAIN_FRACTION * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    def to_tensors(pairs):
        x = torch.tensor([[a, b, P] for a, b in pairs])
        y = torch.tensor([(a + b) % P for a, b in pairs])
        return x, y

    return to_tensors(train_pairs), to_tensors(val_pairs)


def train():
    (x_train, y_train), (x_val, y_val) = build_dataset()
    model = TinyGrokTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    history = {"step": [], "train_acc": [], "val_acc": [], "train_loss": []}
    t0 = time.time()
    for step in range(TOTAL_STEPS):
        model.train()
        opt.zero_grad()
        out = model(x_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0 or step == TOTAL_STEPS - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (model(x_train).argmax(-1) == y_train).float().mean().item()
                val_acc = (model(x_val).argmax(-1) == y_val).float().mean().item()
            history["step"].append(step)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_loss"].append(loss.item())
            print(f"step {step:5d} | loss {loss.item():.4f} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

    print(f"Total training time: {time.time() - t0:.1f}s on 1 CPU core")
    return history


if __name__ == "__main__":
    train()
