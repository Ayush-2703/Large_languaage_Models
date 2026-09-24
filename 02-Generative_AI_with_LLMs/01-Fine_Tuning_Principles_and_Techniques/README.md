# Principles and Techniques for Fine-Tuning Pretrained Models

## 1. Fine-Tuning Is a Family of Techniques, Not One Procedure

"Fine-tune the model" understates how many real decisions sit inside that phrase: which parameters update, at what rate, on what schedule, and in what order. `implementation.py` isolates exactly this choice by loading four byte-identical copies of the same pretrained checkpoint and applying a different technique to each, with everything else — data, step budget, architecture — held fixed. Any difference in the resulting convergence curves is attributable to the technique alone.

## 2. Naive Full Fine-Tuning

The simplest approach: every parameter, one learning rate, standard optimizer. It's the baseline every more sophisticated technique below is implicitly compared against, and it is often a perfectly reasonable choice — its main risk is **catastrophic forgetting**, introduced in `01-Review-of-Fundamental-LLMs/03-Pretraining-vs-Fine-Tuning-Paradigms`: with no protection on earlier, more general layers, aggressive updates driven by a narrow downstream task can overwrite representations that were useful more broadly.

## 3. Discriminative Learning Rates

Howard & Ruder's ULMFiT (2018) proposed assigning **different learning rates to different layers**, motivated by an intuition borrowed from computer vision's layer-hierarchy findings: earlier layers tend to capture more general, broadly-reusable structure (in NLP: something closer to syntax, morphology, orthography), while later layers capture more task-specific structure. If that's true, aggressively updating early layers risks discarding generality that a downstream task didn't need to discard, while later layers can safely absorb larger updates.

`implementation.py` implements this directly as PyTorch parameter groups, each with its own learning rate:

```python
for i, block in enumerate(model_b.encoder.blocks):
    lr_i = 1e-4 * (3 ** i)     # layer 0 gets 1e-4, layer 1 gets 3e-4, layer 2 gets 9e-4
```

Layer 0 (earliest) trains at `1e-4`; each subsequent layer's rate triples; the freshly-initialized classification head — which has no pretrained knowledge to protect — trains fastest of all, at `3e-3`.

## 4. Learning Rate Warmup and Decay

A large learning rate applied immediately to a pretrained model can produce a destructively large first update, before the optimizer's internal statistics (e.g., Adam's moment estimates) have had a chance to calibrate to the current loss landscape. **Warmup** — linearly ramping the learning rate up from zero over the first several hundred steps — avoids this. Pairing warmup with a smooth **decay** schedule for the remainder of training (here, cosine decay, following the SGDR schedule popularized by Loshchilov & Hutter, 2017) lets training start cautiously, spend most of its budget near peak learning rate, and settle gently rather than overshooting near convergence:

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (TOTAL_STEPS - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
```

This exact warmup-then-decay shape, or close variants of it, is the default schedule behind nearly every modern Transformer training recipe — including the original Transformer paper's own learning rate schedule and BERT's fine-tuning recipe (Devlin et al., 2019).

## 5. Gradual Unfreezing

ULMFiT's second technique: rather than choosing one learning rate per layer and training everything simultaneously, freeze the entire pretrained backbone except the new task head, train briefly, then unfreeze one additional layer (starting from the output side, working backward toward the input) at fixed intervals, until the whole network is trainable:

```python
n_unfrozen = min(n_layers, step // UNFREEZE_EVERY)
want_trainable = i >= (n_layers - n_unfrozen)
```

The intuition is protective: the task head, trained on frozen features, has to become reasonably competent *before* the encoder underneath it is allowed to move — reducing the chance that early, noisy gradients from a randomly-initialized head cause large, poorly-informed updates to the pretrained encoder.

## 6. What `implementation.py` Actually Measures — and an Honest Surprise

All four techniques start from one shared MLM-pretrained checkpoint and train for an identical 300-step budget on a synthetic sentiment task. The measured result is not the clean "sophisticated techniques win" story a reader might expect:

| Technique | Final Val. Accuracy |
|---|---|
| (A) Naive Flat-LR | 0.993 |
| (B) Discriminative LR | 0.827 |
| (C) Warmup + Cosine | 0.987 |
| (D) Gradual Unfreeze | 0.810 |

The naive full fine-tune and the warmup+cosine schedule converge to near-perfect accuracy; discriminative LR and gradual unfreezing land meaningfully lower — on this run, not close.

This is a real, explainable trade-off, not a flaw in the demonstration. Both discriminative LR and gradual unfreezing are, by design, **conservative** — they deliberately slow down how quickly early layers (and, for gradual unfreezing, most of the network for the first several hundred steps) are allowed to change, specifically to protect against catastrophic forgetting. That protection has a cost: within a short, fixed step budget, "protected" parameters simply don't get to adapt as far as they would under a less cautious schedule. On an easy task — this synthetic sentiment task, built from a small, clean vocabulary of eight positive and eight negative adjectives, is easy — naive fine-tuning's willingness to update everything aggressively from step one is an advantage, not a liability, precisely because there is little risk of forgetting anything the task needs. Discriminative LR and gradual unfreezing are techniques whose benefit shows up on **harder tasks, with more steps, or where the pretrained representation is doing more real work that's worth protecting** — none of which describes this deliberately simple, fast demonstration. Reporting the numbers as measured, rather than adjusting hyperparameters until the "expected" ordering appears, is the more honest — and, properly explained, more informative — choice.

## 7. Scaling to Production

Every technique here is used, largely unchanged in form, at production LLM fine-tuning scale. Discriminative learning rates and gradual unfreezing see continued use particularly in settings with limited labeled data and real forgetting risk (e.g., continual fine-tuning of an already-instruction-tuned model on a narrow new domain). Warmup-then-decay schedules are close to universal — essentially every large model training or fine-tuning run in current practice uses some variant. What changes at scale is mostly a matter of *values* — production fine-tuning runs might warm up over thousands of steps rather than 30, and layer-count-dependent LR ratios are tuned per architecture — not the underlying mechanisms this topic's code implements directly.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Howard, Jeremy, and Sebastian Ruder. "Universal Language Model Fine-tuning for Text Classification." *ACL*, 2018.
- Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019.
- Loshchilov, Ilya, and Frank Hutter. "SGDR: Stochastic Gradient Descent with Warm Restarts." *ICLR*, 2017.
- Vaswani, Ashish, et al. "Attention Is All You Need." *NeurIPS*, 2017.
