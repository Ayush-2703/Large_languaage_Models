# Chain-of-Thought (CoT) Prompting and Few-Shot Learning

## 1. In-Context Learning: No Weight Updates, Ever

Every technique in this repository up to this point has involved training — updating a model's weights via gradient descent. **Few-shot prompting** (Brown et al.'s GPT-3 paper, "Language Models are Few-Shot Learners," 2020) is different in kind: a handful of worked examples are placed directly in the input context, and the model completes a new instance of the pattern purely through its forward pass, with weights entirely frozen. `implementation.py`'s `evaluate` function makes this explicit — `model.eval()` and no optimizer step ever runs between conditions; every accuracy number reflects only what changing the *prompt* does to a fixed, already-pretrained model's behavior.

## 2. Chain-of-Thought Prompting

Wei et al. ("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," 2022) showed that including *worked intermediate reasoning steps* in few-shot examples — not just the final answer — can substantially improve performance on multi-step reasoning tasks, by encouraging the model to generate its own intermediate steps before committing to a final answer, rather than jumping directly to one. `implementation.py`'s `cot` format does exactly this: `"Reasoning: 9+5=14, 14-4=10. Answer: 10"` versus `direct`'s bare `"Answer: 10"`.

**The critical, explicitly stated caveat from the same paper**: Wei et al. found this benefit is itself an *emergent* property — it appears reliably only above a certain model scale, and is flat or occasionally *negative* for smaller models. This is not a minor footnote to the CoT finding; it is a central part of it, and it is the reason this topic's experiment was designed, run, and reported with that specific prediction stated in `implementation.py`'s own docstring *before* any results existed.

## 3. What Was Actually Measured

| k-shot | Direct-answer | Chain-of-Thought |
|---|---|---|
| 0 | 0.067 | 0.067 |
| 1 | 0.067 | 0.067 |
| 2 | 0.000 | 0.067 |
| 3 | 0.067 | 0.000 |

Both the few-shot learning curve (accuracy should generally rise with more examples) and the CoT-vs-direct comparison (CoT should generally outperform direct-answer prompting) are **flat and low across every condition**, with no consistent trend in either direction. With only 15 trials per condition, individual numbers carry real sampling noise (a true 10% accuracy could easily show anywhere from 0 to 3 successes by chance alone) — the specific decimal values should not be over-read. The robust, qualitative finding is that this model shows **no clear benefit from either additional few-shot examples or chain-of-thought scaffolding**, and near-zero absolute performance on the underlying arithmetic task itself.

**This is exactly what the cited literature predicts for a model this small** (387,051 parameters — smaller than even the smallest models in most published few-shot/CoT scaling studies, which typically start in the hundreds of millions of parameters and go up from there) — a small-scale replication consistent with, not contrary to, Wei et al.'s emergent-abilities finding.

## 4. Looking at *Why*: Format Mimicry Without Grounded Computation

Inspecting the model's actual generated text (`explanation.md` reproduces specific examples) reveals *why* accuracy stays low, which is more informative than the accuracy number alone. For a problem "Ana had 4 apples, gained 1, lost 4. Total?", the model's CoT-style generation was: `"8+1=10, 10-1=7. Answer: 1"`. Every piece of this is revealing:

- The numbers **4, 1, 4** from the actual problem were not copied into the reasoning at all — the model instead generated **8+1=10**, numbers that don't correspond to this problem.
- Even taken on its own terms, **8+1=10** is arithmetically wrong (8+1=9), and **10-1=7** is also wrong (10-1=9).
- The final stated answer, **1**, happens to coincidentally match the true answer for this specific problem — not because the reasoning correctly derived it, but seemingly by chance.

This is a small, stark illustration of a documented concern in the CoT literature sometimes called **unfaithful reasoning**: a model can produce text that has the *surface form* of step-by-step reasoning — plausible-looking equations, a clear "Answer:" conclusion — without that text actually, causally reflecting a correct computation grounded in the specific problem given. At this model's scale, it has learned the *format* of the arithmetic word-problem-plus-reasoning task from pretraining exposure (every generation correctly follows the "Reasoning: ... Answer: N" template) without learning the underlying *arithmetic* well enough to make the reasoning steps actually true. Whether format-without-substance is a matter of degree that improves with scale, or a categorically different failure mode entirely, is exactly the kind of question `05-Frontiers-and-Future-of-LLMs/01-Emergent-Behaviors-and-Scaling-Hypotheses` returns to.

## 5. A Real Bug Caught and Fixed Along the Way

An earlier version of this experiment's answer-extraction logic searched for the substring `"Answer:"` within the model's *generated* text for both prompting styles. For CoT-style prompts, this is correct — the model must generate its own `"Reasoning: ... Answer: N"` completion, including the word "Answer:" itself. For direct-style prompts, it is not: the prompt itself already ends in `"...Total? Answer:"`, so the model's entire generated continuation is just the number — there is no `"Answer:"` substring anywhere in the *generated* portion to find. This meant every direct-style trial returned no extracted answer at all, regardless of what number the model actually generated, making direct-style accuracy structurally zero regardless of the model's real capability. `explanation.md` documents the fix in detail — this was caught and corrected before any conclusion was drawn from the direct-vs-CoT comparison, since drawing "CoT beats direct-answer" from a broken direct-answer extraction would have been a false conclusion built on a bug, not a finding.

## 6. Scaling to Production

Both few-shot in-context learning and chain-of-thought prompting are standard, load-bearing techniques for real production LLMs — but, per Wei et al. and this topic's own small-scale replication of their finding, they are techniques whose benefit is contingent on sufficient underlying model scale and capability, not universal properties of the Transformer architecture that apply equally at every size. A production system choosing between direct prompting and CoT prompting for a given model should, per this literature, expect the answer to genuinely depend on which model is being used — the same prompting strategy is not guaranteed to help (and, per Wei et al., can occasionally hurt) below some capability threshold, exactly as demonstrated directly here.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Brown, Tom, et al. "Language Models are Few-Shot Learners." *NeurIPS*, 2020.
- Wei, Jason, et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS*, 2022.
- Wei, Jason, et al. "Emergent Abilities of Large Language Models." *TMLR*, 2022.
