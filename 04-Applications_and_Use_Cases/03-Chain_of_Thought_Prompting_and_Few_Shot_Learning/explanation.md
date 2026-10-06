# Explanation: `implementation.py`

## Two Disjoint Problem Pools

```python
seen_signatures = set()
for _ in range(1500):
    problem, answer, direct, cot = make_problem()
    seen_signatures.add(problem)
    ...

def sample_fresh_problem(seen_signatures):
    while True:
        problem, answer, direct, cot = make_problem()
        if problem not in seen_signatures:
            return problem, answer, direct, cot
```

**Why check every sampled test/few-shot-example problem against `seen_signatures`:** Pretraining exposes the model to 1,500 worked examples specifically so it can learn the *format and general mechanics* of this problem type. If evaluation could accidentally reuse one of those exact 1,500 problems, a correct answer might reflect memorization of that specific problem during pretraining rather than genuine in-context pattern completion on something new — exactly the same concern `01-Retrieval-Augmented-Generation-RAG-Vector-DBs` raised about held-out entities, applied here to problem instances instead of knowledge-base facts.

## Sizing This Model Larger Than This Repository's Smallest Demos — And Then Right-Sizing It Again for Time

```python
D_MODEL, N_HEADS, N_LAYERS = 96, 6, 3
BLOCK = 384
```

**Why this needed two rounds of adjustment before the first successful run:** An initial version used `D_MODEL=128, N_HEADS=8, N_LAYERS=4, BLOCK=512` — deliberately larger than this repository's smallest topic models, since this experiment specifically studies an emergent, scale-sensitive capability and deserved a fair chance. Two problems surfaced in sequence. First, a length check revealed a 5-shot CoT prompt runs to roughly 1,100 characters — comfortably exceeding `BLOCK=512`, meaning `model.generate`'s context-window truncation (`ids[:, -self.block_size:]`) would have silently cut off the earliest few-shot examples for exactly the longest, most information-dense condition. Second, even after shortening the problem template to fix that, a direct timing benchmark showed the original architecture cost roughly 880ms per training step — projecting to over 10 minutes for pretraining alone, before any evaluation. A second benchmark at the current, smaller configuration measured roughly 140ms per step, a manageable budget once combined with a shorter, 500-step pretraining schedule. Both adjustments are reported directly rather than silently applied, since they're a real, concrete illustration of a recurring theme in this repository: matching model/context size to actual measured compute cost, not just to what a task conceptually deserves.

## Two Different Answer-Extraction Rules for Two Different Prompt Structures

```python
def extract_answer(generated_text, style):
    if style == "direct":
        digits = ""
        for ch in generated_text.lstrip():
            if ch.isdigit():
                digits += ch
            elif digits:
                break
            else:
                break
        return int(digits) if digits else None
    else:
        if "Answer:" not in generated_text:
            return None
        ...
```

**What changed and why it mattered:** `build_prompt`'s direct-style tail is `f"Problem: {test_problem} Answer:"` — the word "Answer:" is the *last thing already in the prompt*, so `model.generate` only produces the number that follows it; `"Answer:"` itself never appears in the generated continuation being parsed. An earlier version of this function used the CoT-style search logic (look for `"Answer:"` inside the generated text) for *both* styles, which meant `extract_answer` returned `None` for every single direct-style trial, regardless of whether the model generated the exactly correct number — making direct-style accuracy structurally zero by construction, not by the model's actual performance. This was caught specifically by inspecting raw generated text directly (`" 1\nProblem: Leo had 8 apples..."` — a plausible immediate numeric answer, followed by the model continuing on to fabricate a new problem, exactly as an autoregressive model with no stop condition would) rather than trusting the accuracy numbers alone. `theory.md` §5 discusses why this distinction mattered for the validity of the whole comparison.

**Why direct-style's fix takes the *leading* digits of the generated text specifically:** Once "Answer:" is correctly recognized as already-consumed prompt text, the very next characters the model generates are, by construction, its attempt at the number itself — before it (as seen in the diagnostic examples) moves on to hallucinating a brand new "Problem:" continuation. Taking only the leading digit run, and stopping at the first non-digit character, correctly isolates just the answer before that unrelated continuation begins.

## Why Only 15 Trials Per Condition

```python
def evaluate(k_shot, style, n_trials=15):
```

**Why not more, given the concern about sampling noise raised in `theory.md` §3:** Each trial requires a full autoregressive generation loop (up to 40 sequential forward passes for CoT-style trials), and this experiment already required two rounds of size/timing adjustment to fit a reasonable compute budget at all (see above). 15 trials per condition, across 4 k-shot values and 2 styles (120 total generation trials), was the largest count that kept total evaluation time reasonable given the pretraining cost already incurred. `theory.md` is explicit that individual condition numbers should be read as noisy estimates with this sample size — the qualitative flatness of the pattern across conditions is the finding, not the precise decimal values, which is why the write-up leads with that qualitative claim rather than any single number.

## Why `proof.png` Shows Both a Line Chart and a Grouped Bar Chart of the Same Numbers

**Panel A** (line chart, k-shot on the x-axis) is the natural way to show a *learning curve* — whether accuracy trends upward as more examples are added is a shape-of-the-line question. **Panel B** (grouped bars, direct vs. CoT paired at each k) is the natural way to show a *paired comparison* at each specific k-shot level — which of the two styles wins, at each point, is a bar-height comparison question. Both charts plot the identical four numbers per style; showing both is a deliberate choice to let a reader check the "does k-shot help" question and the "does CoT help" question separately, since — as `theory.md` §3 reports — neither shows a clear trend here, and a single combined chart would make it harder to see that *both* questions independently came back flat, rather than one masking the other.
