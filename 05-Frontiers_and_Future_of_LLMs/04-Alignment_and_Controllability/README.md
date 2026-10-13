# 5.4 — Alignment and Controllability of LLMs

## 1. From RLHF to DPO

Classic RLHF (as used for InstructGPT and early ChatGPT-style models,
per Ouyang et al.'s InstructGPT paper) trains an explicit reward model
on human preference comparisons, then optimizes the policy against
that reward model using reinforcement learning (typically PPO), with a
KL penalty back to a reference (usually the supervised-fine-tuned)
policy to keep the model from drifting too far and degenerating.
Rafailov et al.'s Direct Preference Optimization paper showed that,
under a Bradley-Terry preference model, the RL step can be replaced
with a single closed-form classification-style loss computed directly
on `(chosen, rejected)` pairs and the reference model's own
log-probabilities -- no reward model, no RL rollout, no PPO
instability. This module's `implementation.py` implements exactly that
loss.

## 2. What beta actually controls

DPO's loss is `-log sigmoid(beta * ((logpi(chosen) - logref(chosen)) -
(logpi(rejected) - logref(rejected))))`. The `beta` term plays the same
role as the KL-penalty coefficient in RLHF: it is a knob that trades
off two competing pressures on the policy on every single gradient
step --

- **push down**: increase the log-probability gap between chosen and
  rejected completions (satisfy the preference), versus
- **push back**: how far the policy is allowed to drift from the
  reference model's own probabilities to do so.

A small beta lets the policy chase the preference aggressively with
little regard for the reference model's overall behavior; a large beta
keeps the policy close to the reference and only shifts behavior
where doing so is cheap in KL terms.

## 3. Reward over-optimization / "alignment tax" is a real, documented cost

Gao, Schulman, and Hilton's work on scaling laws for reward-model
over-optimization documented, at production LLM scale, that optimizing
a policy hard against any imperfect proxy for human preference
eventually degrades the policy on axes the proxy doesn't capture --
even as the proxy reward itself keeps climbing. This module's beta
sweep reproduces the same qualitative pattern in miniature: pushing
reward up aggressively (low beta) buys a bigger jump in the preferred
attribute (reward-token frequency) but a correspondingly bigger drop in
an unrelated, previously-learned capability (fidelity to the base
grammar) -- exactly the "tax" the term describes.

## 4. Why controllability and safety are two sides of the same knob

The same mechanism that lets a practitioner deliberately shift model
behavior toward a desired attribute (helpfulness, refusing certain
requests, adopting a house style) is what makes over-optimization a
safety-relevant risk if the proxy reward is even slightly misspecified
relative to what's actually wanted: the policy will find and exploit
any gap between the stated preference and the intended one, which is
precisely what happens here when reward-token frequency, not overall
sequence quality, is the only thing being optimized for.

## 5. QKV / architecture note

The causal Transformer used for both the reference and policy models
is architecturally identical to Topic 5.3's encoder half and Topic
1.2's attention mechanism, with a causal mask so each position can only
attend to itself and earlier positions -- the standard GPT-style
decoder-only setup DPO is normally applied to.

## 6. Scaling to production-size LLMs

On a Colab T4, the direct real-checkpoint analogue fine-tunes a small
GPT-2 checkpoint with the HF `trl` library's `DPOTrainer` on a real
preference dataset (e.g., a downsampled Anthropic HH-RLHF-style set),
sweeping beta the same way this module does, and measuring both a
preference-satisfaction metric and a held-out capability metric (e.g.,
perplexity on unrelated held-out text) to reproduce the same
controllability-vs-capability curve shown in `proof.png`.

## References

- Rafailov, R. et al. — *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- Ouyang, L. et al. — *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT)
- Gao, L., Schulman, J., Hilton, J. — *Scaling Laws for Reward Model Overoptimization*
- Tunstall, L., von Werra, L., Wolf, T. — *Natural Language Processing with Transformers* (fine-tuning/alignment grounding)
- Rothman, D. — *Transformers for Natural Language Processing* (decoder architecture grounding)
