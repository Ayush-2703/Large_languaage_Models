# Pretraining vs. Fine-Tuning Paradigms

## 1. Two Separate Questions

"How good is this model?" is really two questions that are easy to conflate: **what did it learn before seeing my task**, and **how much did I let it change to fit my task**. Pretraining answers the first; fine-tuning strategy answers the second. `implementation.py` is built to isolate both, with three variants that each toggle exactly one of these questions relative to the others:

| Variant | Pretrained backbone? | Backbone trainable? | Answers |
|---|---|---|---|
| Frozen Probe | Yes | No (only linear head trains) | "How useful are pretrained features *as-is*, with zero task-specific adaptation?" |
| Full Fine-Tune | Yes | Yes | "How useful are pretrained features once every layer can adapt?" |
| Random Init | No | Yes (from scratch) | "Would training on this task alone, with no pretraining at all, get me here anyway?" |

The gap between Random-Init and Frozen-Probe isolates **the value of pretraining itself**. The gap between Frozen-Probe and Full-Fine-Tune isolates **the value of adaptation on top of pretraining**. Neither number alone answers "does pretraining matter" — the comparison across all three does.

## 2. Pretraining Objectives

**Causal (autoregressive) language modeling — GPT family.** Predict token `t` from tokens `1..t-1` only, exactly the objective trained from scratch in `01-History-and-Evolution-of-Language-Models`. Radford et al.'s GPT (2018) and its successors pretrain a decoder-only Transformer at large scale on this single objective; the resulting model is, by construction, already a text generator before any fine-tuning happens.

**Masked language modeling (MLM) — BERT family.** Devlin et al. (2019) instead mask ~15% of input tokens and train an *encoder* to reconstruct them from bidirectional context (looking both left and right, which a causal model structurally cannot do). This is why BERT-family models like DistilBERT — the checkpoint `implementation.py` uses — are typically used for *understanding* tasks (classification, extraction) rather than open-ended generation: the objective they were pretrained on never asked them to generate a coherent continuation, only to fill in blanks given full context.

**Span corruption — T5 family.** Raffel et al. (2020) generalize masking from single tokens to contiguous spans, replaced by a single sentinel token, with the target being the original span — framing every NLP task, including pretraining itself, as text-to-text.

`implementation.py` uses DistilBERT specifically, a distilled (compressed via knowledge distillation) version of BERT — same MLM-derived representations, roughly 40% fewer parameters, chosen here purely for the T4/under-an-hour feasibility constraint rather than for any conceptual reason.

## 3. Fine-Tuning Strategies

**Feature extraction / frozen probing.** Freeze every pretrained weight; train only a small task-specific head (here, a single linear layer) on top of the backbone's output representation. This tests, cheaply, whether the pretrained representation *already* linearly separates your classes — if a single `Linear` layer on frozen features gets reasonable accuracy, most of the "work" for this task was already done during pretraining.

**Full fine-tuning.** Every parameter, backbone included, receives gradients from the downstream task. This lets earlier layers reshape their representations specifically for the task at hand, at the cost of far more trainable parameters and a real risk of **catastrophic forgetting** — aggressively overwriting general-purpose pretrained knowledge in service of one narrow task, which can hurt performance on anything outside that task's distribution.

**Gradual unfreezing.** Howard & Ruder's ULMFiT (2018) proposed a middle path predating BERT: unfreeze and fine-tune one layer at a time, starting from the output layer and working backward, rather than either freezing everything or unfreezing everything at once. This reduces forgetting risk relative to full fine-tuning from the first step, at the cost of a more involved training schedule.

**Parameter-efficient fine-tuning (PEFT).** A modern middle ground — freeze the pretrained backbone entirely, but inject small trainable adapter modules (e.g., LoRA's low-rank update matrices) rather than a single linear head. This captures much of full fine-tuning's flexibility at a fraction of the trainable-parameter count. `implementation.py`'s frozen-probe variant is the simplest possible case of this family — one linear layer, zero adapter modules — with the full technique (LoRA, QLoRA) covered in `03-Training-and-Optimization-of-LLMs/02-Efficient-Training-LoRA-QLoRA-Quantization-Pruning`.

## 4. Why a Linear Probe Is a Meaningful Test at All

Reading out class predictions with a single `nn.Linear(hidden, 2)` layer on top of frozen features is not a weak or lazy baseline — it is a deliberate, standard diagnostic. A linear layer can only draw a hyperplane through the representation space; it cannot learn new features, only recombine and weight the ones already present. So probe accuracy is a direct measurement of **linear separability**: if pretrained DistilBERT's `[CLS]` representation already places positive- and negative-sentiment sentences in roughly separable regions of its embedding space (a byproduct of MLM pretraining on enormous amounts of naturally-occurring text, where sentiment-bearing words co-occur with predictable contexts), a linear probe finds a good decision boundary quickly, with very few trainable parameters. If pretrained features do *not* separate the classes linearly, no amount of head-only training will fix that — only adapting the backbone itself (full fine-tuning) can reshape the representation space to make the classes separable.

## 5. Expected Result and Why It's Not Guaranteed

The expected ordering — Random-Init worst, Frozen-Probe better, Full-Fine-Tune best — reflects the standard finding across the transfer-learning literature: pretraining on a large unlabeled corpus produces representations that transfer, and adaptation on top of that further improves task-specific performance. But it is not a mathematical certainty for every task and every dataset size. With enough labeled data, a randomly initialized model trained from scratch can eventually approach or match a pretrained one; the advantage of pretraining is most pronounced precisely when labeled data is scarce, which is why `implementation.py` deliberately subsamples SST-2 to 2,000 training examples rather than using the full training set — a data-scarce regime is where the value of pretraining is easiest to see clearly.

## 6. Scaling to Production

Every production LLM fine-tuning pipeline is a variant of the comparison in `implementation.py`, at far greater scale:

- **Instruction tuning** (e.g., turning a base GPT-style model into a chat assistant) is full or parameter-efficient fine-tuning on curated instruction-response pairs, layered on top of a next-token-pretrained backbone.
- **RLHF and DPO** (covered in `03-Training-and-Optimization-of-LLMs/03-Alignment-RLHF-and-DPO`) are best understood as a *further* fine-tuning stage on top of an already instruction-tuned model, optimizing for human preference rather than likelihood alone.
- **LoRA/QLoRA** (covered in `03-Training-and-Optimization-of-LLMs/02-Efficient-Training-LoRA-QLoRA-Quantization-Pruning`) scale the frozen-probe idea in this topic up to production-size models — freeze billions of pretrained parameters, train only a small number of injected low-rank adapter parameters, and recover most of full-fine-tuning's benefit at a fraction of the memory and compute cost.

The frozen-probe-vs-full-fine-tune-vs-no-pretraining comparison here, at DistilBERT-and-SST-2 scale, is a compressed, fast-to-run version of decisions every production LLM team makes about how much of a pretrained model to touch, and why.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019.
- Radford, Alec, et al. "Improving Language Understanding by Generative Pre-Training." OpenAI, 2018.
- Raffel, Colin, et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*, 2020.
- Howard, Jeremy, and Sebastian Ruder. "Universal Language Model Fine-tuning for Text Classification." *ACL*, 2018.
- Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." 2019.
