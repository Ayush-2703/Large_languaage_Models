# 5.2 — Multilingual and Cross-lingual Capabilities

## 1. Why cross-lingual transfer works at all

A multilingual model like mBERT or XLM-R is trained jointly on many
languages sharing a single subword vocabulary and a single set of
Transformer weights. Empirically, such models exhibit strong zero-shot
cross-lingual transfer — fine-tune on an English task, evaluate on
Hindi or Swahili, and accuracy holds up far better than chance, even
though the model saw no labeled examples in the target language.
Pires, Schlinger, and Garrette's analysis of multilingual BERT's
cross-lingual generalization was among the first to document and
probe this behavior systematically, and the strength of the effect
still surprises researchers today given no explicit cross-lingual
supervision signal is provided.

## 2. The classical explanation: shared geometry

Long before multilingual Transformers, Mikolov, Le, and Sutskever
observed that word embedding spaces trained *separately* on two
different languages' monolingual corpora have strikingly similar
internal geometry — the neighborhood structure around "cat," "dog,"
and "animal" in an English embedding space looks, up to rotation and
scale, like the neighborhood structure around the corresponding French
words in a French embedding space. This is a direct consequence of both
languages describing the same underlying world of concepts and
co-occurrence statistics. Their proposal — learn a simple linear (or
orthogonal) map from one embedding space into the other, using only a
small seed dictionary of known translations — became the seed idea
behind an entire line of "cross-lingual word embedding alignment"
research.

## 3. Orthogonal Procrustes alignment

Later work (Xing et al.; Conneau, Lample, Ranzato, Denoyer, and
Jégou's MUSE line of work) restricted the alignment map to be
*orthogonal* rather than an arbitrary linear map, which both improves
generalization and has a closed-form solution: given seed pairs
`(x_i, y_i)`, the orthogonal matrix minimizing
`sum_i ||W x_i - y_i||^2` is obtained directly from the SVD of
`X^T Y`. This is the exact routine implemented in this topic's
`implementation.py`, applied to synthetic embeddings whose true latent
structure we control end-to-end, which lets us cleanly show that a
seed dictionary covering under a third of the vocabulary is enough to
recover translations for the remaining ~70% that were never paired
during alignment.

## 4. Why this generalizes at all: the theoretical link to word2vec

Levy and Goldberg showed that skip-gram-with-negative-sampling is,
in expectation, implicitly factorizing a shifted pointwise mutual
information (PMI) matrix — which is exactly what the PPMI + truncated
SVD recipe used in this topic's code computes directly and
non-stochastically. Because both language corpora in this experiment
are generated from the same underlying bilinear similarity kernel
between latent concept vectors, their PPMI matrices converge (with
enough data) to the same matrix up to sampling noise — which is why
their SVD-derived embeddings are recoverable from one another via a
single orthogonal transform, exactly as the theory predicts.

## 5. QKV / architecture note

This topic's experiment operates purely at the embedding-table level
and does not involve self-attention; the connection to Transformer
LLMs is at the level of the token/subword embedding matrix that sits
underneath the attention stack (Topic 1.2), not the attention
mechanism itself.

## 6. Scaling to production-size LLMs

Production multilingual LLMs don't run an explicit post-hoc Procrustes
step; the alignment described here happens implicitly as a side effect
of joint pretraining on shared subword vocabulary and (for models like
mT5) parallel or comparable corpora. The practical, Colab-feasible
next step for a real checkpoint is a small MarianMT/opus-mt model
(sequence-to-sequence, trained on a real WMT16 language-pair subsample)
or a small XLM-R/mBERT-based zero-shot transfer experiment: fine-tune
on an English sentiment/NLI dataset, evaluate zero-shot on a second
language's test set, and compare against the same model fine-tuned
with a handful of target-language examples.

## References

- Mikolov, T., Le, Q., Sutskever, I. — *Exploiting Similarities among Languages for Machine Translation*
- Conneau, A., Lample, G., Ranzato, M., Denoyer, L., Jégou, H. — *Word Translation Without Parallel Data* (MUSE)
- Pires, T., Schlinger, E., Garrette, D. — *How Multilingual is Multilingual BERT?*
- Levy, O., Goldberg, Y. — *Neural Word Embedding as Implicit Matrix Factorization*
- Tunstall, L., von Werra, L., Wolf, T. — *Natural Language Processing with Transformers* (multilingual/tokenizer grounding)
- Rothman, D. — *Transformers for Natural Language Processing* (embedding-layer grounding)
