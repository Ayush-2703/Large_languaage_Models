# Explanation: `implementation.py`

## Model Configurations

```python
CONFIGS = [
    dict(d_model=16,  n_heads=2, n_layers=1, label="Tiny"),
    dict(d_model=32,  n_heads=2, n_layers=1, label="Small"),
    dict(d_model=64,  n_heads=4, n_layers=2, label="Medium"),
    dict(d_model=128, n_heads=4, n_layers=2, label="Large"),
    dict(d_model=192, n_heads=4, n_layers=3, label="XL"),
]
```

**What:** Five configurations of the same `ScalableTransformerLM` class, growing in both width (`d_model`) and depth (`n_layers`).

**Why vary both width and depth rather than just one:** Growing only width or only depth would test one specific scaling axis rather than "scale" broadly. Growing both together (roughly following how production model families actually scale — e.g., successive GPT generations grow in both dimensions) keeps this closer in spirit to what a real scaling study varies, even at drastically smaller absolute size.

**Why these specific numbers, spanning roughly 223x in parameter count:** Scaling laws are a claim about the *shape* of a curve across a wide range, not a two-point comparison. Five sizes spread across more than two orders of magnitude give `np.polyfit` enough spread to fit a meaningful line — two or three sizes clustered close together would make the fitted exponent far noisier.

## Fixed Training Budget

```python
TRAIN_STEPS = 400   # fixed budget across all sizes
```

**What:** Every model size trains for exactly the same number of optimizer steps, on batches drawn the same way from the same data.

**Why fixed steps rather than fixed wall-clock time or training-to-convergence:** This isolates parameter count as the variable under study, the same logic `explanation.md` in `01-History-and-Evolution-of-Language-Models` used for comparing RNN/LSTM/Transformer fairly. Fixed wall-clock time would let smaller (faster) models take more optimizer steps than larger ones, confounding "smaller model" with "more training" — the opposite comparison from what this topic is about. Training every model fully to convergence would be more representative of a real scaling-law study, but would cost far more compute than this repository's feasibility constraint allows; a fixed, deliberately short step budget is the explicit trade-off made here, stated plainly rather than hidden, and is exactly why `theory.md` §5 cautions against comparing the fitted exponent numerically to published values.

## FLOPs Estimation

```python
tokens_processed = TRAIN_STEPS * BATCH_SIZE * BLOCK_SIZE
approx_flops = 6 * n_params * tokens_processed
```

**What:** Applies the standard `6ND` approximation (`theory.md` §4) using this model's actual parameter count and the total tokens every model in this experiment processes (constant, since steps/batch/block are fixed across all five runs).

**Why report this alongside wall-clock time rather than instead of it:** FLOPs is a hardware-independent measure of computational work — useful for comparing model sizes on paper — but wall-clock time is what a person training a model actually experiences, and the two can diverge (per `theory.md` §5's note on the Tiny/Small timing anomaly) when fixed overhead matters more than raw FLOPs at very small scale. Reporting both keeps the script honest about that distinction rather than presenting FLOPs as if it were the same thing as measured cost.

## The Power-Law Fit

```python
log_n = np.log([r["params"] for r in results])
log_loss = np.log([r["val_loss"] for r in results])
a, b = np.polyfit(log_n, log_loss, 1)
```

**What:** A first-degree (linear) least-squares fit of `log(loss)` against `log(params)`. `a` is the fitted slope — the power-law exponent; `b` is the intercept.

**Why `np.polyfit` on logged values rather than fitting `loss = c * N^a` directly with a nonlinear solver:** Taking logs turns a power-law fit into an ordinary linear regression, which is both simpler to implement correctly and exactly what makes a scaling law visually identifiable as "a straight line in log-log space" in the first place — the same transformation `theory.md` §1 walks through algebraically. This is the standard approach used in the scaling-laws literature itself, not a simplification specific to this toy demo.

## `proof.png` Generation

**Panel A** plots measured `(params, val_loss)` points on log-log axes, overlaid with the fitted power-law line — visually, points landing close to the dashed line is what "this data actually follows a power law" looks like, rather than something you have to take on faith from the fitted exponent number alone.

**Panel B** plots wall-clock training time against parameter count on a log x-axis, annotated with the same size labels as Panel A. This panel exists specifically to keep "bigger models have lower loss" (Panel A) and "bigger models cost more to train" (Panel B) visually separate — conflating the two into one chart would obscure the efficiency question `theory.md` §4 raises as distinct from the pure scaling question.

**Why every data point is labeled with its config name (Tiny/Small/Medium/Large/XL) directly on the chart:** So the reader can cross-reference a specific point back to `CONFIGS` above without needing to infer which marker corresponds to which model size from position alone — useful given Panel B's Tiny/Small ordering doesn't follow the trend everywhere else in the chart does, which is exactly the anomaly `theory.md` §5 discusses rather than hides.
