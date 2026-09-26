# Transfer Learning for Domain-Specific Tasks

## 1. Three Questions About Where Knowledge Transfers From

`01-Fine-Tuning-Principles-and-Techniques` asked *how* to fine-tune a pretrained model. This topic asks a question one level up: *how much does it matter what the model was pretrained on, relative to the domain it's being applied to?* `implementation.py` separates this into three concrete arms:

- **(A) No adaptation** — a checkpoint pretrained on general text, fine-tuned directly on an out-of-domain task.
- **(B) Domain-adaptive pretraining (DAPT)** — the same starting checkpoint, given a further, unlabeled, in-domain continued-pretraining pass before fine-tuning.
- **(C) No pretraining** — the same architecture, randomly initialized, trained directly on the task, isolating how much of any gap is "pretraining at all" versus "domain-matched pretraining specifically."

## 2. Domain-Adaptive Pretraining (DAPT)

Gururangan et al.'s "Don't Stop Pretraining" (2020) is the direct source for arm (B). Their finding: continuing a general-domain pretrained model's *unsupervised* pretraining objective on unlabeled text drawn from the target domain — before ever touching labeled task data — reliably improves downstream performance across a range of domains (biomedical, computer science, news, reviews) and tasks, and the improvement is largest when the target domain is most different from the original pretraining corpus. The mechanism is intuitive: a model pretrained on, say, general web text has never seen the specific vocabulary, register, or typical sentence structures of a specialized domain; a further unsupervised pass on in-domain text — no labels required, since it's the same masked-language-modeling objective as original pretraining, just on different data — lets the model absorb that vocabulary and register before it's asked to also learn a new labeled task on top.

`implementation.py`'s Stage 2 is a direct, minimal implementation of this: `dapt_model.load_state_dict(general_state)` starts from the identical general checkpoint arm (A) also uses, then continues MLM training — same objective, same code path as Stage 1 — on unlabeled synthetic tech-review text before Stage 3's fine-tuning ever begins.

## 3. What Was Actually Measured, Including a Bug Caught Along the Way

An earlier version of this experiment's MLM pretraining function sampled raw windows directly from the corpus, with no `[CLS]` token present. But `theory.md`'s downstream task reads out predictions from the `[CLS]` position — meaning that in the earlier version, arms (A) and (B) entered fine-tuning with a `[CLS]` embedding that had never received a single gradient update during pretraining, while arm (C)'s `[CLS]` embedding was on equal footing with every other parameter from the start, since everything in arm (C) is learned jointly from scratch. That mismatch directly undermines the exact comparison this topic is built to make, so it was fixed: `mlm_batch` now prepends a real `[CLS]` token to every pretraining window, structurally matching the format fine-tuning uses, before masking is applied to the remaining positions. `explanation.md` documents this fix in full; it's raised here because it changed the measured result in a scientifically important way, not just a cosmetic one.

With that fix in place, the measured results were:

| Arm | Final Val. Accuracy |
|---|---|
| (A) No Adaptation | 0.790 |
| (B) DAPT | 0.807 |
| (C) No Pretraining | 0.893 |

**The comparison this topic is specifically about — does matching pretraining to the target domain help, relative to leaving it mismatched — came out as the literature predicts: DAPT (B) beat no-adaptation (A).** That's the direct, positive confirmation of §2's claim, and it held after fixing the CLS bug above (before the fix, ordering between A and B was inconsistent across runs — an early sign the comparison wasn't yet measuring what it intended to).

**The broader comparison — does pretraining, adapted or not, beat no pretraining at all — did not come out as a reader might expect: random initialization (C) reached the highest accuracy of the three.** This deserves the same honest treatment given to unexpected findings elsewhere in this repository, not a quiet rerun until it goes away.

## 4. Why (C) Winning Is a Real, Explainable Result — Not a Broken Demo

Three properties of this specific toy setup make it a poor showcase for pretraining's *absolute* advantage, even though it remains a good showcase for domain-adaptation's *relative* advantage (§3's A-vs-B result):

1. **The task is deliberately simple.** Sixteen adjectives, cleanly and consistently split into two sentiment classes, filled into eight templates — a small, low-ambiguity pattern space that even 150 labeled examples covers well enough for a from-scratch model to learn directly. Pretraining's advantage is best documented on tasks with real linguistic ambiguity or long-tailed vocabulary, exactly what this deliberately clean synthetic task avoids by construction.
2. **A converged checkpoint is not automatically a better *starting point* than a well-conditioned random one for a new objective.** PyTorch's default initializations for `nn.Linear` and `nn.Embedding` are specifically designed to keep gradients well-scaled from step one for whatever objective training begins with. A model that has already converged toward an MLM-specific optimum starts fine-tuning from a different point in parameter space — not obviously closer, in gradient-descent steps, to a classification optimum, particularly over a short fine-tuning budget.
3. **Single-run, small-model measurements carry real variance.** No result in this file is averaged across multiple random seeds; a single training run's exact numbers should be read as one honest data point demonstrating the qualitative mechanism, not as a precise, noise-free estimate of any true underlying gap.

This is the same caveat `01-Review-of-Fundamental-LLMs/03-Pretraining-vs-Fine-Tuning-Paradigms` (`theory.md` §5) already raised: pretraining's advantage is most visible in a data-scarce, high-ambiguity regime, and shrinks — or, as measured directly here, can even reverse — as a task's own difficulty and scale fall short of that regime. **This does not contradict the well-established real-world finding that pretraining helps**, documented at far larger scale and on far harder tasks than this repository can run on a single CPU core; it demonstrates the boundary condition under which that finding stops being guaranteed, which is just as real and worth knowing as the headline finding itself.

## 5. Scaling to Production

At production scale, both arms of this topic's finding hold in their proper regime: DAPT-style continued pretraining is genuinely used before fine-tuning on specialized domains (legal, biomedical, financial text) precisely because those domains carry real vocabulary and structure a general checkpoint hasn't seen, and labeled data in such domains is often genuinely scarce — exactly the regime where §4 predicts pretraining's advantage should reappear clearly. The methodological lesson from §3's caught bug generalizes too: **evaluation protocol details (here, whether a readout token was itself ever pretrained) can silently determine which arm of a comparison "wins,"** a risk that scales up, not down, as real experiments grow more complex and harder to fully inspect by eye.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Gururangan, Suchin, et al. "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." *ACL*, 2020.
- Howard, Jeremy, and Sebastian Ruder. "Universal Language Model Fine-tuning for Text Classification." *ACL*, 2018.
- Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019.
