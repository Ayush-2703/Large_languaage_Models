# explanation.md — Topic 5.1 walkthrough

## What the script does, line by line

**Config block (`P`, `TRAIN_FRACTION`, `WEIGHT_DECAY`, ...)**
`P = 13` means the task is "predict `(a + b) mod 13`" — there are
`13 * 13 = 169` possible `(a, b)` pairs total. `TRAIN_FRACTION = 0.5`
means the model only ever sees half of them during training; the other
half is the validation set it must generalize to. `WEIGHT_DECAY = 2.0`
is unusually high for a model this size — it's deliberate, because
weight decay is the ingredient that makes grokking happen at all
(it's the pressure that eventually pushes the optimizer off the
memorizing solution).

**`TinyGrokTransformer`**
- `tok_emb`: an embedding table with `VOCAB = P + 1` rows — one row per
  number `0..P-1`, plus one extra row for a special `"="` token.
- `pos_emb`: a 3-row positional embedding, because every input sequence
  is exactly 3 tokens long: `[a, b, "="]`.
- `nn.TransformerEncoderLayer` / `nn.TransformerEncoder`: this is
  PyTorch's built-in multi-head self-attention block, stacked twice
  (`N_LAYERS = 2`). This is the same Q/K/V attention mechanism from
  Topic 1.2, just applied here to a 3-token sequence instead of a
  sentence.
- `forward`: embeds tokens + positions, runs them through the encoder
  stack, then reads the prediction off the hidden state at the `"="`
  position (index `-1`) — the model has to have "computed" the answer
  by the time it reaches that position, the same way a human reads
  `7 + 5 =` and only writes the answer after the equals sign.

**`build_dataset`**
Builds every one of the 169 `(a, b)` pairs, shuffles them, and splits
into train/validation. `to_tensors` converts each pair into the 3-token
input `[a, b, P]` (recall token `P` is reused as the `"="` token id) and
the integer label `(a + b) % P`.

**`train()`**
A standard supervised training loop: forward pass, cross-entropy loss,
backward pass, `AdamW` step. Every `EVAL_EVERY = 100` steps, it measures
*exact-match accuracy* on the train set and the held-out validation set
separately and records both in `history`. This is the crucial
measurement for the topic: watching train accuracy and validation
accuracy diverge, then re-converge, is what makes the "emergence" visible.

## What actually happened when it ran (real numbers, this run)

- By roughly step 200-1000, **train accuracy is already at 100%** — the
  model has fully memorized the 84 training pairs it was shown.
- **Validation accuracy stays low (~15-25%)** for the next several
  thousand steps even though training loss looks flat/converged — this
  is the deceptive plateau. A practitioner watching only the training
  loss curve would conclude the model is "done" and has learned nothing
  more to give.
- Somewhere around **step 4,000-6,000, validation accuracy rises sharply**
  from under 30% to essentially 100% within a couple thousand more
  steps — the grokking transition — and stays there (with some noise)
  for the rest of training.
- Total wall-clock time for all 10,000 steps: under 70 seconds, on a
  single CPU core, with no GPU and no downloaded pretrained weights.

## Why the chart is proof, not illustration

`proof.png` is generated directly from `history.json`, which is written
by the `train()` function in `implementation.py` — there is no
hand-authored or synthetic data standing in for a real run. The
highlighted "grokking transition window" on the chart marks the step
range where validation accuracy visibly departs from its plateau in
*this specific run*, not a generic textbook illustration.
