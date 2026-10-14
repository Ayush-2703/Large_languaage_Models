# 5.5 — Beyond Text: Robotics, Vision-Language, and Decision-Making

## 1. Why "beyond text" is a distinct research frontier

Everything covered in Phases 1-4 operates on text tokens in and text
tokens (or a classification label) out. Extending LLM-style models to
robotics, vision-language understanding, and embodied decision-making
requires answering a question text-only models never face: how should
a continuous, high-dimensional perceptual stream (pixels, joint
angles, depth) be fused with language, and how should the model's
output cash out as *actions in the world* rather than more tokens?
This is the organizing question behind the vision-language-action
(VLA) model family this topic's experiment is a small, controlled
instance of.

## 2. The vision-language-action (VLA) recipe

Modern robot-control systems in this space (e.g., Google DeepMind's
RT-2 line of work, which reframes robot actions as tokens a
vision-language model can predict the same way it predicts words)
generally share a common shape: a vision encoder, a language encoder
(often a pretrained LLM backbone), a fusion mechanism between them,
and a policy/action head that outputs discretized or continuous
actions. This topic's code implements exactly this shape at toy scale:
a small CNN vision encoder, an instruction embedding standing in for a
language encoder, concatenation as the fusion step, and a 4-way action
classifier as the policy head.

## 3. Why closed-loop evaluation, not single-step accuracy, is the right test

A policy can look excellent by single-step "next action matches expert"
accuracy while still failing almost every full episode, because small
per-step errors compound over a rollout (a classic distribution-shift
problem in imitation learning -- once the policy makes one small
mistake, it finds itself in a state the expert data never covered).
This is why this topic's experiment measures **closed-loop rollout
success rate** -- the policy's own predicted actions are executed and
fed back in as new observations -- rather than accuracy on a fixed,
expert-generated trajectory.

## 4. Why the vision-only ablation is the crucial control

The chart's key comparison isn't "how good is the full model" in
isolation -- it's the *gap* between the full model and an
architecturally identical model denied the instruction. If that gap
were small, language conditioning would be decorative (the vision
encoder could already infer the task on its own). Because there are
three visually-present but differently-intended goals in every
episode, a vision-only policy is structurally incapable of knowing
which one to pursue -- its performance is bounded near chance/near
"pick a plausible default," while the full model's performance is
bounded only by the shortest-path controller it was trained to imitate.
This is exactly the ablation real household/warehouse
language-conditioned robotics papers run to justify that the language
channel is doing necessary work.

## 5. QKV / architecture note

This experiment uses concatenation-based fusion rather than
cross-attention for simplicity and speed at this scale; production VLA
systems more often use cross-attention (the language/instruction
representation as queries attending over spatial vision features, or
vice versa) or simply feed both modalities as a single token sequence
into a shared Transformer -- the same self-attention mechanism from
Topic 1.2, just applied to a sequence that mixes visual patch tokens
and language tokens.

## 6. Scaling to production-size LLMs

On a Colab T4, the direct real-checkpoint analogue replaces the small
CNN with a pretrained small vision backbone (or CLIP's image encoder)
and the instruction embedding with a small pretrained language model's
hidden states (e.g., distilbert or t5-small's encoder), fine-tuning the
fusion + action head on a real, small demonstration dataset (e.g., a
public simulated-robot dataset or a hand-collected small set of
teleoperated trajectories in a simulator). The evaluation methodology
-- closed-loop success rate with a vision-only ablation as a control --
transfers directly.

## References

- Brohan, A. et al. — *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*
- Ross, S., Gordon, G., Bagnell, D. — *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger; motivates why closed-loop evaluation matters for imitation learning)
- Radford, A. et al. — *Learning Transferable Visual Models From Natural Language Supervision* (CLIP; vision-language fusion grounding)
- Rothman, D. — *Transformers for Natural Language Processing* (attention/fusion grounding)
- Tunstall, L., von Werra, L., Wolf, T. — *Natural Language Processing with Transformers* (encoder architecture grounding)
