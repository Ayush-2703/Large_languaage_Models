# Explanation: `implementation.py`

## Controlled Preference Data, Balanced by Length on Purpose

```python
CHOSEN_TEMPLATES = ["Sure, here is a clear answer for you.", ...]
REJECTED_TEMPLATES = ["Not my problem, go figure it out.", ...]
```

**Why these specific strings, chosen to be nearly equal length:** An earlier version of this file used longer, more varied chosen templates ("Of course! Let me walk you through it step by step.") against shorter rejected ones ("I do not know and I do not care."), averaging roughly a 13-character gap. Since DPO's preference signal is built from *summed* token log-probability (`theory.md` §3), and shorter sequences accumulate fewer negative log-probability terms, that length gap alone was enough to make the pre-training policy "prefer" rejected responses 95% of the time — not because it had learned anything backwards, but because the metric itself favors brevity independent of content. Rebalancing the templates to near-equal length (final gap: about 3 characters) isolates DPO's actual preference-learning mechanism as the primary signal in the reported numbers, rather than letting a length artifact dominate them. `theory.md` §4 reports the residual gap honestly rather than claiming perfect balance.

## Reward Model: Pooling From the `[CLS]` Position

```python
return self.reward_head(h[:, 0, :]).squeeze(-1)
```

**Why a single scalar output (`nn.Linear(d_model, 1)`) rather than a 2-class classifier like every other encoder in this repository:** A reward model's job is fundamentally different from classification — it needs to produce a *continuously comparable* score across arbitrarily many different responses, not assign one of a fixed number of labels. Squeezing the final dimension (`squeeze(-1)`) turns the `(batch, 1)` output into a plain `(batch,)` vector of scalars, which is what the Bradley-Terry loss (`r_chosen - r_rejected`, a plain elementwise subtraction) expects.

## `sequence_logprob`: Log-Probability of a Specific Continuation

```python
ctx = ids[:, :-1]
targets = ids[:, 1:]
...
token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
response_mask = torch.zeros_like(targets, dtype=torch.bool)
response_mask[:, prompt_len:] = True
response_mask &= (targets != PAD)
seq_logprob = (token_logprobs * response_mask).sum()
```

**What:** Standard next-token cross-entropy setup (context is everything except the last token, targets are everything except the first), but instead of returning a loss, this returns the **sum of log-probabilities the model assigned to the actual next tokens** — and only for the response portion of the sequence, masked to exclude both the prompt tokens and any padding.

**Why mask out the prompt portion specifically:** `theory.md`'s DPO formula needs `log π(response | prompt)` — the probability of the *response*, conditioned on the prompt, not the joint probability of the prompt-plus-response together. Since the prompt tokens are part of the input sequence (needed as context) but aren't themselves the thing being evaluated for preference, `response_mask[:, prompt_len:] = True` ensures only response-token log-probabilities contribute to the final sum.

**Why `.gather(-1, targets.unsqueeze(-1))` rather than indexing directly:** `logprobs` has shape `(batch, seq_len, vocab_size)` — one probability distribution per position. For each position, only the log-probability of the *actual* next token (not the whole distribution) is needed. `gather` performs exactly this "pick out one value per position, per batch element, according to an index tensor" operation in one vectorized call, which is both faster and less error-prone than a manual loop over positions.

## Why the DPO Loop Processes One Example at a Time, Not a Batch

```python
for i in idx:
    prompt, chosen, rejected = pref_data[i]
    lp_c_policy = sequence_logprob(policy, prompt, chosen, requires_grad=True)
    ...
    loss.backward()
```

**What:** Rather than stacking multiple `(prompt, chosen, rejected)` triples into a single batched tensor and calling the model once, this loop calls `sequence_logprob` separately for each example in the mini-batch, accumulating gradients via repeated `.backward()` calls before a single `opt.step()`.

**Why not batch this, given every other training loop in this repository batches aggressively:** `sequence_logprob` needs to sum log-probabilities only over each example's *own* response length — but different `(prompt, chosen)` and `(prompt, rejected)` pairs have different prompt lengths and different response lengths, meaning the position at which "response starts" (`prompt_len`) varies per example. Batching this correctly would require either padding every sequence to a common length and passing a per-example `prompt_len` tensor into a vectorized masking operation, or accepting the added complexity of ragged-length handling. Given the small preference dataset and mini-batch size (8) used here, looping directly is simpler and unambiguously correct, at a real but acceptable speed cost (`explanation.md` reports DPO's 150-step run completing in well under 15 seconds even with this less-batched approach) — a deliberate simplicity-over-throughput trade-off, stated plainly rather than presented as if it were the only reasonable implementation choice.

## Why `reference` Is Built Via `load_state_dict`, Not by Reusing `policy` Directly

```python
reference = CausalLM().to(DEVICE)
reference.load_state_dict(policy.state_dict())
for p in reference.parameters():
    p.requires_grad = False
```

**Why construct an entirely separate model object rather than, e.g., just remembering the policy's loss at initialization:** DPO's math needs `log π_ref(y|x)` re-evaluated at every training step, for every example — not just a single scalar snapshot taken once. `reference` must therefore be a fully independent, fully functional model capable of a forward pass at any point during training, permanently frozen at exactly the policy's pre-DPO state. Copying the state dict into a fresh object (rather than, say, keeping a live reference to `policy` and hoping to "undo" its updates later) guarantees `reference`'s weights literally cannot change no matter what happens to `policy` during the training loop that follows — `requires_grad = False` on every parameter is what makes this a hard guarantee rather than a convention that could accidentally be violated by a stray `.backward()` call.

## `proof.png`'s Dual-Axis Panels

**Why each panel plots two different quantities (accuracy and margin) on separate y-axes rather than two separate charts:** For both the reward model and DPO, "did the model learn to separate chosen from rejected" (accuracy) and "how confidently does it separate them" (margin) are two different, complementary claims — a model could reach 100% accuracy with a razor-thin margin (fragile, likely to flip on slightly different data) or with a large, robust margin (the pattern seen here). Plotting both on a shared x-axis (training step), even at different scales via a second y-axis, makes it visible that this topic's real training run achieves the *robust* version of convergence — margin keeps growing substantially even after accuracy has already reached its ceiling (visible directly in DPO's panel, where accuracy hits 1.000 by step 75 but margin continues climbing from 25.8 to 37.7 over the remaining steps).
