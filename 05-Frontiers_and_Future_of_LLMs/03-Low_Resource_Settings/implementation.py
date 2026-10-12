"""
Topic 5.3 — LLMs in Low-Resource Settings
============================================

GOAL
----
Give a real, measured demonstration of the central fact that makes
"low-resource" a distinct research problem in the first place: model
quality is a steep function of training-example count in the small-data
regime, with a comparatively sharp transition from "fails almost
completely" to "solves the task reliably" — not a smooth, forgiving
decline. This is the mechanistic reason low-resource languages,
domains, and tasks need dedicated techniques (transfer learning, data
augmentation, cross-lingual transfer as in Topic 5.2, synthetic data
generation) rather than "just use less data and hope."

TASK
----
A tiny encoder-decoder Transformer is trained on a toy machine
translation problem: translate a 6-token synthetic "source language"
sequence into a synthetic "target language" sequence via a fixed,
unknown-to-the-model rule combining (a) a lexical substitution table
and (b) word-order reversal (a stand-in for the real fact that language
pairs differ both lexically and in word order, e.g. SVO vs. SOV). The
model must learn *both* the substitution and the reordering purely from
examples.

The experiment trains an independent model from scratch at each of
several training-set sizes (5, 10, 15, 20, 30, 50, 75, 100, 300, 1000
examples), using a **fixed compute budget** (600 minibatch steps) so
that training-set size — not training time — is the only thing that
changes between runs. Every model is evaluated on the *same* fixed,
held-out 500-example test set, using exact sequence-match accuracy.

WHY THIS IS A FAIR STAND-IN FOR "LOW-RESOURCE"
------------------------------------------------
Real low-resource machine translation research (e.g. the WMT
low-resource shared tasks, work on languages like Swahili, Nepali, or
Sinhala) faces exactly this shape of problem: a systematic
transformation (translation) that must be learned from a training set
that may only have hundreds to low-thousands of parallel sentence
pairs, orders of magnitude less than the millions available for
high-resource pairs like English-German. The toy task here is smaller
and cleaner, but the underlying phenomenon — a data-efficiency curve
with a sharp knee — is the same one that motivates the field.

SCALING TO PRODUCTION LLMS
---------------------------
On a Colab T4, the direct real-checkpoint analogue is fine-tuning a
small opus-mt (MarianMT) or mT5-small model on a deliberately
downsampled WMT16/FLORES language-pair subsample (e.g. 50 / 500 / 5000
sentence pairs) and plotting BLEU/chrF against training-set size — the
same experiment design used here, just with a real pretrained
checkpoint and a real parallel corpus in place of the synthetic
substitution-and-reversal rule. The reason production low-resource NLP
leans so heavily on pretrained checkpoints and cross-lingual transfer
(rather than training from scratch, as this toy experiment does) is
precisely to start the data-efficiency curve further to the left,
so a few hundred examples land past the "knee" instead of before it.
"""

import random
import time

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRC_VOCAB, TGT_VOCAB = 14, 14   # 12 content tokens + PAD + BOS
SEQ_LEN = 6
D_MODEL, N_HEADS, N_LAYERS = 32, 2, 2
PAD, BOS = 12, 13
STEPS = 600          # fixed compute budget: identical for every training-set size
BATCH = 32
TEST_SIZE = 500      # fixed held-out test set, same for every run
TRAIN_SIZES = [5, 10, 15, 20, 30, 50, 75, 100, 300, 1000]

# fixed, unknown-to-the-model translation rule: lexical substitution + reversal
_rng = random.Random(123)
LEXICON = list(range(12))
_rng.shuffle(LEXICON)


def make_example(rng):
    src = [rng.randint(0, 11) for _ in range(SEQ_LEN)]
    tgt = [LEXICON[t] for t in reversed(src)]  # word-order reversal + lexical substitution
    return src, tgt


def make_dataset(n, seed):
    rng = random.Random(seed)
    return [make_example(rng) for _ in range(n)]


class ToyTranslationTransformer(nn.Module):
    """Small encoder-decoder Transformer -- the standard seq2seq architecture,
    scaled down so ten independent training runs fit in a couple of minutes."""

    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(SRC_VOCAB, D_MODEL)
        self.tgt_emb = nn.Embedding(TGT_VOCAB, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN + 1, D_MODEL)
        self.transformer = nn.Transformer(
            d_model=D_MODEL, nhead=N_HEADS,
            num_encoder_layers=N_LAYERS, num_decoder_layers=N_LAYERS,
            dim_feedforward=128, batch_first=True,
        )
        self.out = nn.Linear(D_MODEL, TGT_VOCAB)

    def forward(self, src, tgt_in):
        sp = torch.arange(src.size(1)).unsqueeze(0)
        tp = torch.arange(tgt_in.size(1)).unsqueeze(0)
        s = self.src_emb(src) + self.pos(sp)
        t = self.tgt_emb(tgt_in) + self.pos(tp)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1))
        h = self.transformer(s, t, tgt_mask=tgt_mask)
        return self.out(h)


def batchify(data):
    src = torch.tensor([d[0] for d in data])
    tgt = torch.tensor([d[1] for d in data])
    tgt_in = torch.cat([torch.full((tgt.size(0), 1), BOS), tgt[:, :-1]], dim=1)
    return src, tgt_in, tgt


def train_and_eval(n_train, seed, xs_test, xtin_test, xt_test):
    train_data = make_dataset(n_train, seed=seed)
    xs, xtin, xt = batchify(train_data)
    n = xs.size(0)

    model = ToyTranslationTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    t0 = time.time()
    for _ in range(STEPS):
        idx = torch.randint(0, n, (min(BATCH, n),), generator=gen)
        model.train()
        opt.zero_grad()
        out = model(xs[idx], xtin[idx])
        loss = loss_fn(out.reshape(-1, TGT_VOCAB), xt[idx].reshape(-1))
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(xs_test, xtin_test)
        pred = out.argmax(-1)
        exact_match_acc = (pred == xt_test).all(dim=1).float().mean().item()
    return exact_match_acc, time.time() - t0


def run():
    test_data = make_dataset(TEST_SIZE, seed=999)  # never used for training, fixed across all runs
    xs_test, xtin_test, xt_test = batchify(test_data)

    results = {"train_sizes": [], "accuracy": [], "seconds": []}
    for n in TRAIN_SIZES:
        acc, secs = train_and_eval(n, seed=n, xs_test=xs_test, xtin_test=xtin_test, xt_test=xt_test)
        results["train_sizes"].append(n)
        results["accuracy"].append(acc)
        results["seconds"].append(secs)
        print(f"n_train={n:5d} | exact-match accuracy={acc:.3f} | {secs:.1f}s (fixed {STEPS} steps)")
    return results


if __name__ == "__main__":
    run()
