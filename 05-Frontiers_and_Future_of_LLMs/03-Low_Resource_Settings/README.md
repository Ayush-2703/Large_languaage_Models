# 5.3 — LLMs in Low-Resource Settings

## 1. What "low-resource" means

In NLP, "low-resource" describes a language, domain, or task for which
only a small amount of labeled or even raw text data exists — as
opposed to English news text or Wikipedia, where billions of tokens
are available. Most of the world's ~7,000 languages fall into this
category for essentially every task: parallel translation data,
labeled sentiment data, and even large raw monolingual corpora are all
scarce for the majority of languages, which is why "low-resource NLP"
is treated as a distinct research area rather than a footnote on
high-resource results.

## 2. Why data-efficiency curves have a "knee," not a straight line

A recurring empirical finding across low-resource machine translation
and low-resource fine-tuning work is that quality does not degrade
gracefully as training data shrinks — it holds up reasonably well down
to some threshold and then falls off sharply below it. Intuitively, a
model needs to see enough examples to disambiguate the *general rule*
underlying the task from the specific examples it happened to see;
below some critical count, the training set is consistent with many
different rules (including simple memorization), so held-out
performance collapses. This module's code reproduces exactly this
shape of curve on a fully controlled toy task, where "the rule" is a
combination of a fixed lexical substitution and a word-order reversal.

## 3. Why pretraining and transfer shift the curve, not just add data

The standard response to the low-resource problem is not "collect more
data" (often impossible) but transfer learning: start from a model
already pretrained on high-resource languages or related tasks, so the
model arrives at the low-resource fine-tuning stage already knowing
general linguistic structure, and only needs the scarce in-domain data
to specialize rather than to learn language from scratch. This is
equivalent to shifting the entire data-efficiency curve seen in this
topic's experiment to the left — the same accuracy is reached with
fewer target-domain examples. Topic 5.2's cross-lingual alignment
result is one concrete mechanism behind why this shift works: a
multilingual pretrained model has already discovered much of the
shared cross-lingual structure before it ever sees the low-resource
language's fine-tuning examples.

## 4. Few-shot / in-context learning as a complementary strategy

A second, more recent strategy sidesteps fine-tuning entirely: give a
large pretrained model a handful of input-output examples directly in
its prompt (in-context learning) rather than updating its weights.
This can be viewed as trading the "data-efficiency curve" of gradient-
based fine-tuning (this topic's experiment) for a different one, where
the relevant scarce resource is prompt examples rather than training
examples, and the "knowledge" the model draws on was baked in during
large-scale pretraining rather than learned from the few shown
examples.

## 5. QKV / architecture note

The encoder-decoder Transformer used in this topic's code is the same
architecture family covered in Topic 1.2, applied here to a
sequence-to-sequence setting: the decoder's self-attention is causally
masked (via `generate_square_subsequent_mask`) so it can only attend to
previously generated target tokens, while cross-attention lets each
decoder position attend over the full encoded source sequence — the
standard mechanism by which encoder-decoder Transformers condition
generation on an input sequence.

## 6. Scaling to production-size LLMs

On a Colab T4, the direct real-checkpoint version of this experiment
fine-tunes a small opus-mt or mT5-small checkpoint on progressively
downsampled subsets of a real WMT16 or FLORES language-pair corpus
(e.g., a genuinely low-resource pair), measuring BLEU or chrF instead
of exact-match accuracy. The expected qualitative result — a sharp knee
in the data-efficiency curve, shifted further left than a from-scratch
model's curve because of pretraining — is exactly the comparison real
low-resource MT papers report between training from scratch and
fine-tuning a pretrained multilingual checkpoint.

## References

- Sennrich, R., Zhang, B. — *Revisiting Low-Resource Neural Machine Translation: A Case Study* (data-efficiency / low-resource NMT framing)
- Conneau, A. et al. — *Word Translation Without Parallel Data* (transfer-learning motivation, shared with Topic 5.2)
- Brown, T. et al. — *Language Models are Few-Shot Learners* (in-context/few-shot learning framing)
- Tunstall, L., von Werra, L., Wolf, T. — *Natural Language Processing with Transformers* (fine-tuning and data-scarcity grounding)
- Rothman, D. — *Transformers for Natural Language Processing* (encoder-decoder architecture grounding)
