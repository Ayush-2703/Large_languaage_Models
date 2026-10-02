# Alignment Methodologies: RLHF and DPO

## 1. The Problem Both Techniques Solve

Pretraining and even instruction-tuning optimize a model to predict likely text — not to produce text humans actually *prefer*. Alignment techniques close that gap using **preference data**: pairs of responses to the same prompt, one marked as preferred ("chosen") and one as not ("rejected"), collected from human raters (or, as in `implementation.py`, generated here with a fully known, controlled quality signal — chosen responses are drawn from helpful/polite templates, rejected from curt/dismissive ones, exactly the kind of controlled-ground-truth setup used throughout this repository, e.g. `01-Review-of-Fundamental-LLMs/05-Ethical-Considerations-in-Large-Scale-AI`).

Two different algorithms turn this same kind of data into a better-aligned model: **RLHF** (reinforcement learning from human feedback) and **DPO** (direct preference optimization). `implementation.py` implements one full component of RLHF (the reward model) plus all of DPO, both trained for real; the RL-specific remainder of RLHF is covered in this document only, for reasons §5 explains directly.

## 2. The Reward Model: Turning Preferences Into a Scalar

The first stage of a full RLHF pipeline (Christiano et al., 2017; Ouyang et al.'s InstructGPT, 2022, applies this directly to language models) trains a **reward model** — here, a bidirectional encoder with a scalar output head — to score any (prompt, response) pair with a single number, such that preferred responses score higher. The **Bradley-Terry model** of pairwise comparison provides the loss: interpreting the reward difference as log-odds of a human preferring the chosen response,

```
P(chosen ≻ rejected) = σ(r(chosen) − r(rejected))
Loss = −log σ(r(chosen) − r(rejected))
```

`implementation.py`'s reward-model training loop is a direct, one-line implementation of this: `loss = -F.logsigmoid(r_chosen - r_rejected).mean()`. Once trained, this scalar function is what a full RLHF pipeline would use as the reward signal for policy optimization — `implementation.py` trains it to convergence (reaching **1.000 pairwise accuracy** and a large positive reward margin, from `explanation.md`'s reported numbers) but does not take the next RLHF step of optimizing a separate policy against it.

## 3. DPO: Skipping the Reward Model and the RL Step Entirely

Rafailov et al.'s "Direct Preference Optimization" (2023) makes a mathematical observation: the RLHF objective (maximize expected reward under the policy, subject to a KL-divergence penalty keeping the policy close to a reference model) has a closed-form optimal solution relating the optimal policy directly to the reward function and the reference policy. Substituting this relationship back into the Bradley-Terry preference loss **eliminates the reward model entirely** — preference data can train the policy directly:

```
L_DPO = −log σ( β · [ (log π(chosen|x) − log π_ref(chosen|x)) − (log π(rejected|x) − log π_ref(rejected|x)) ] )
```

Every term here is directly computable from two models' token-level log-probabilities — no reward model, no sampling rollouts, no reinforcement learning at all. `implementation.py`'s `sequence_logprob` computes `log π(y|x)` exactly as this formula requires: summing the model's log-probability for each token of the response, conditioned on the prompt via ordinary teacher-forced next-token prediction — the same computation `01-Review-of-Fundamental-LLMs/01-History-and-Evolution-of-Language-Models` uses for perplexity, applied here to a specific labeled continuation instead of the whole corpus.

**The reference model must stay frozen.** `implementation.py` constructs `reference` as an exact copy of the policy's weights immediately after SFT-style pretraining, then disables its gradients permanently:

```python
reference.load_state_dict(policy.state_dict())
for p in reference.parameters():
    p.requires_grad = False
```

Without a frozen anchor, nothing would stop the policy from drifting arbitrarily far from sensible language just to satisfy the preference loss — the `(log π − log π_ref)` term is precisely what keeps that drift bounded, playing the same role RLHF's explicit KL penalty plays, but built directly into the loss rather than added as a separate term.

## 4. What Was Actually Measured

Both techniques converged cleanly to strong final results:

| | Before training | After training |
|---|---|---|
| Reward model — pairwise accuracy | 0.587 | **1.000** |
| Reward model — mean reward margin | +0.001 | +12.313 |
| DPO — P(policy prefers chosen) | 0.275 | **1.000** |
| DPO — implicit reward margin | +0.000 | +37.668 |

DPO's implicit reward margin grew smoothly and monotonically across every logged checkpoint (0.45 → 6.65 → 12.42 → 17.55 → 22.04 → 25.81 → … → 37.67) — exactly the behavior the loss function is designed to produce, with no instability or reversal at any point.

**A real, honestly-reported wrinkle: the pre-training starting point (0.275, not 0.5).** `sequence_logprob` sums per-token log-probability, so a shorter response accumulates fewer (necessarily negative) log-probability terms than a longer one, all else equal — a systematic bias toward preferring shorter sequences that has nothing to do with content quality. An earlier version of this experiment's `CHOSEN_TEMPLATES`/`REJECTED_TEMPLATES` had a much larger length gap (chosen responses averaging roughly 13 characters longer than rejected ones), and the pre-DPO policy preferred chosen only 5% of the time — a length artifact large enough to look, misleadingly, like the model had already learned a strong *anti*-preference. Rebalancing the templates to near-equal length (chosen averaging 38 characters, rejected 34.75) reduced this to the 0.275 reported above; the residual gap is the same phenomenon at a smaller scale, not fully eliminated. This is not a bug specific to this implementation — **length bias in log-probability-based preference signals is a real, documented phenomenon** in the RLHF/DPO literature, sometimes called length or verbosity bias, and is one motivation for length-normalized variants of preference losses in follow-up work. Reporting it plainly here, rather than only reporting the clean post-training numbers, keeps this topic's claim accurate: DPO demonstrably works and converges cleanly *despite* a real confound present in the underlying metric, not in the absence of one.

## 5. Why RLHF's PPO Step Is Not Implemented Here

A complete RLHF pipeline's second half optimizes the policy against the trained reward model using **Proximal Policy Optimization** (PPO; Schulman et al., 2017): the policy generates (samples) responses to prompts, the reward model scores them, a **value function** estimates expected future reward for advantage estimation (GAE — generalized advantage estimation), and the policy is updated via PPO's clipped surrogate objective:

```
L_PPO = E[ min( ratio_t · Â_t,  clip(ratio_t, 1−ε, 1+ε) · Â_t ) ]   where ratio_t = π_new(a_t|s_t) / π_old(a_t|s_t)
```

with an additional KL penalty term against the reference policy, exactly analogous to DPO's `(log π − log π_ref)` term but applied at the level of an RL reward signal rather than folded directly into a supervised loss.

This was scoped out of `implementation.py` deliberately, decided before this repository's build began: correctly implementing PPO's full machinery — rollout sampling, advantage estimation, value-function training, clipping-range tuning, and reward/KL balancing — carries real risk of subtle bugs (a slightly wrong advantage normalization, an unstable value-function target, a poorly-tuned KL coefficient) that would produce code that *runs* without actually demonstrating correct reinforcement learning, which would be a worse outcome than a rigorous conceptual treatment. Every other RL-specific concept above (the reward model, the KL-penalty role, the policy/reference-model relationship) has already been implemented and trained for real in §2–§4; §5's algorithm is the one piece described in equations and citations rather than code, and is exactly why DPO — mathematically equivalent in what it optimizes for, without any of this section's implementation surface area — has seen such wide practical adoption as a simpler alternative achieving similar ends.

## 6. Scaling to Production

Real RLHF and DPO pipelines apply exactly these mechanisms — the Bradley-Terry reward loss, the DPO objective, the frozen-reference-model KL anchor — to models with billions of parameters and preference datasets collected from thousands of real human raters rather than four fixed templates. The length-bias phenomenon in §4 is a genuine, active concern at that scale too; production DPO implementations sometimes apply length normalization or length-penalty terms specifically to counteract it. RLHF's PPO stage, sketched conceptually in §5, is the most computationally expensive part of a full alignment pipeline in practice (requiring the policy, reward model, value model, and reference model simultaneously in memory, plus live generation during training) — a major part of why DPO's elimination of that entire stage, while producing empirically comparable alignment quality in many published comparisons, has made it an attractive default for teams without the infrastructure RLHF's full pipeline demands.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Christiano, Paul F., et al. "Deep Reinforcement Learning from Human Preferences." *NeurIPS*, 2017.
- Schulman, John, et al. "Proximal Policy Optimization Algorithms." 2017.
- Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS*, 2022.
- Rafailov, Rafael, et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS*, 2023.
