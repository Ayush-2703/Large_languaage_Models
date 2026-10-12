# explanation.md — Topic 5.3 walkthrough

## What the script does, line by line

**The translation rule (`LEXICON`, `make_example`)**
A fixed random bijection `LEXICON` maps each of 12 "source language"
tokens to a "target language" token. `make_example` builds one
training pair by sampling a random 6-token source sequence and
producing the target as the *reversed* sequence of substituted tokens
— so the model has to learn two things at once: a per-token
substitution and a global word-order change, similar in spirit to how
real language pairs differ both lexically and structurally (e.g. verb
position).

**`make_dataset(n, seed)`**
Generates `n` examples from a given random seed. This is the only
thing that changes between runs — the *same* fixed rule generates
every dataset, at every size.

**`ToyTranslationTransformer`**
A standard `nn.Transformer` encoder-decoder: the encoder reads the
source sequence, the decoder generates the target sequence one
position at a time during training via teacher forcing, using
`generate_square_subsequent_mask` so the decoder can't "cheat" by
looking at future target tokens.

**`batchify`**
Converts a list of `(src, tgt)` pairs into padded tensors, and builds
`tgt_in` — the decoder's input — by prepending a `BOS` token and
dropping the last target token (the standard teacher-forcing shift).

**`train_and_eval(n_train, seed, ...)`**
Trains one model from scratch on `n_train` examples for a **fixed 600
minibatch steps**, regardless of how large or small `n_train` is —
this is the key experimental control. If we let bigger datasets train
for proportionally more steps, we couldn't tell whether an accuracy
improvement came from more data or from more compute; fixing the step
count isolates the effect of dataset size. Each minibatch is sampled
with replacement from the (possibly tiny) training set.

**`run()`**
Builds one fixed 500-example held-out test set (generated once, seed
`999`, identical for every training-set size), then loops over
`TRAIN_SIZES = [5, 10, 15, 20, 30, 50, 75, 100, 300, 1000]`, training
an independent model at each size and recording its exact-sequence-
match accuracy on that same fixed test set.

## What actually happened when it ran (real numbers, this run)

| training examples | exact-match accuracy |
|---:|---:|
| 5 | 0.000 |
| 10 | 0.000 |
| 15 | 0.050 |
| 20 | 0.678 |
| 30 | 0.974 |
| 50 | 0.998 |
| 75 | 1.000 |
| 100 | 0.982 |
| 300 | 1.000 |
| 1000 | 1.000 |

The transition is sharp: accuracy is essentially zero through 10
examples, crosses 50% between 15 and 20 examples, and is
functionally solved (>97%) by 30 examples — a roughly 2x change in
dataset size separates "the model has learned almost nothing
generalizable" from "the model has fully learned the rule." Total
runtime for all ten independent training runs: under 2.5 minutes on a
single CPU core.

## Why the chart is proof, not illustration

`proof.png` plots the `accuracy` list from `results.json` directly
against `train_sizes` — both written by `run()` in `implementation.py`
during the actual training runs. The "knee" highlighted on the chart
is the observed 15-30 example range in *this specific run's* recorded
numbers, not a generic textbook curve shape.
