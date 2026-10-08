# 5.1 — Emergent Behaviors and Scaling Hypotheses

## 1. What "emergent" means in the LLM literature

An ability is usually called *emergent* when it is essentially absent in
smaller models, then appears — sometimes abruptly — once model size,
training compute, or data crosses some threshold, without that ability
having been explicitly optimized for. Wei et al.'s widely-cited survey on
emergent abilities of large language models catalogued dozens of
benchmark tasks (multi-step arithmetic, certain in-context reasoning
tasks) where accuracy stays at chance level across several orders of
magnitude of scale and then rises sharply — a shape that looks like a
phase transition rather than a smooth curve.

## 2. The counter-argument: is it a mirage?

Schaeffer, Miller, and Koyejo's follow-up analysis argued that many
reported "emergent" jumps are largely an artifact of the *metric*
chosen. Discontinuous, all-or-nothing metrics (exact-match accuracy on a
multi-token answer) can turn a smooth, continuous improvement in the
model's underlying per-token probability of being correct into what
looks like a sudden jump, simply because a multi-token answer is only
scored "correct" once every token clears the threshold at once. Swap the
metric for something continuous (token-level log-likelihood, partial
credit) and, in several cases, the same underlying models show smooth,
predictable improvement instead of a cliff. This is an active,
unresolved empirical debate — the code in this module does not attempt
to settle it, only to give a fully-controlled example of the phenomenon
both sides are arguing about.

## 3. Grokking: the cleanest controlled case we have

Separately from the metric debate, Power et al. documented a genuinely
mechanistic instance of sudden generalization on small, synthetic
algorithmic tasks (like modular arithmetic): a network reaches 100%
*training* accuracy almost immediately by memorizing the training
examples, then — under continued optimization with weight decay —
*validation* accuracy stays near chance for a long plateau before rising
sharply to near-100%, long after training loss looked converged. This is
called "grokking." Because the task is small enough to fully inspect,
later mechanistic-interpretability work has been able to show *why* it
happens: weight decay slowly pushes the network's parameters away from a
high-norm, purely-memorizing solution and toward a lower-norm solution
that happens to implement the actual periodic structure of modular
arithmetic, and the switch-over in which solution dominates the model's
output is comparatively abrupt.

## 4. QKV / architecture note

The model used for the code experiment in this topic is an ordinary
Transformer encoder stack — the same scaled-dot-product multi-head
self-attention mechanism covered in Topic 1.2 (Q, K, V projections,
softmax(QK^T/√d)V), just with a tiny `d_model` and two layers, applied to
a 3-token sequence `[a, b, "="]`. Nothing about the attention math is
special-cased for this experiment; grokking is a property of the
optimization dynamics (weight decay + long training), not of any
particular architecture.

## 5. Why this matters for scaling hypotheses

Both "genuine phase transition" and "measurement artifact" explanations
are actively used to argue for or against different scaling
hypotheses — e.g., whether pushing a frontier model past some compute
threshold will reliably unlock a new capability (the "scale is all you
need" hypothesis) versus whether apparent capability jumps are mostly a
function of benchmark design and would look continuous under a better
metric. Grokking matters here because it is proof that *at least one*
real, mechanistic sudden-transition phenomenon exists in trained neural
networks — so the debate at LLM scale is about whether that same
mechanism (or one like it) explains real benchmark jumps, not about
whether sudden transitions can happen in principle.

## 6. Scaling to production-size LLMs

At GPT-3/4 scale, this experiment cannot be reproduced directly (we do
not have the compute, and the "task" a frontier LLM is grokking, if
anything, is not a single clean function like modular addition). The
transferable idea is the *diagnostic pattern*: when investigating a
claimed emergent capability in a production model, plot the metric
against a continuous/partial-credit alternative and against model scale
in fine-grained increments before concluding the ability "emerged" — the
same care this topic's controlled experiment makes possible at toy
scale.

## References

- Wei, J. et al. — *Emergent Abilities of Large Language Models*
- Schaeffer, R., Miller, B., Koyejo, S. — *Are Emergent Abilities of Large Language Models a Mirage?*
- Power, A. et al. — *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*
- Rothman, D. — *Transformers for Natural Language Processing* (attention/architecture grounding)
- Tunstall, L., von Werra, L., Wolf, T. — *Natural Language Processing with Transformers* (training dynamics grounding)
