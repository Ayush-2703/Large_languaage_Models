# Case Studies: Text Summarization, Translation, and Sentiment Analysis

## 1. Three Tasks, Three Different Relationships to "Can a Toy Model Do This Honestly?" 

This topic covers three of the most common real-world LLM applications. They are not equally reproducible at toy scale, and `implementation.py` treats each on its own honest terms rather than forcing uniform treatment:

- **Sentiment analysis** is a *classification* task over a bounded, small label set. A small model, trained on a data distribution that genuinely contains the pattern to be learned, can solve it well — the task's difficulty comes from pattern complexity (e.g., negation), not from needing broad world or language knowledge.
- **Translation**, in its full real-world form, requires vast vocabulary, idiom, and grammar knowledge no toy corpus can provide — but its *core mechanism* (an encoder-decoder architecture learning to reorder and substitute tokens conditioned on source context) is fully demonstrable at small scale, using invented languages specifically to avoid any claim about real-language quality.
- **Summarization** requires *abstractive* language generation — producing fluent, coherent, meaning-preserving new text — which is precisely the capability that only emerges from substantial pretraining on real, diverse language. A from-scratch model trained in seconds has no realistic path to this capability at any corpus size this repository's compute budget allows.

## 2. Case Study 1 — Sentiment Analysis

This case study deliberately extends beyond the simpler positive/negative-adjective task used in `01-Fine-Tuning-Principles-and-Techniques` and `03-Implementing-Fine-Tuning-GPT-BERT-T5`: 25% of examples use **negation patterns** ("not a *dreadful* moment in the film" — a negative-polarity adjective, but a positive true label because of the negation wrapping it). This is a real, well-known hard case for sentiment models — a purely lexical (bag-of-adjectives) approach fails on negated examples, since the surface-level sentiment word contradicts the true label, and only a model that has learned to use the adjective's *context* (not just its identity) can get these right.

The evaluation suite matches what a real sentiment case study reports: not just accuracy, but a full **confusion matrix**, **precision**, **recall**, and **F1**. These matter separately because they answer different questions — accuracy alone can hide a lopsided error pattern (a model that always predicts "positive" scores well on an imbalanced dataset while being useless), while precision and recall expose exactly what kind of mistake a model tends to make.

## 3. Case Study 2 — Translation as a Mechanism Demonstration

Real machine translation checkpoints (e.g., the OPUS-MT family referenced in this repository's feasibility constraint) are trained on millions of real parallel sentence pairs. No toy corpus can substitute for that. What `implementation.py`'s translation experiment demonstrates instead is narrower and fully honest about its scope: **the encoder-decoder mechanism itself** — cross-attention letting a decoder generate output conditioned on an encoded source sequence — genuinely learns to perform the two things real translation requires:

- **Lexical substitution** — each source word maps to a specific, different target word (`house` → `fen`, entirely invented vocabulary chosen specifically so no claim about real language quality is implied).
- **Reordering** — Language A places adjectives before nouns ("the red house"); the constructed Language B places nouns before adjectives ("zil fen kor" — literally "the house red"), mirroring the real adjective-noun order difference between English and many Romance languages. A model that only learned word-for-word substitution, with no genuine sequence-to-sequence reasoning, could not produce correct output, because the *position* of each output word depends on a different position in the input.

Measured **exact-sequence-match accuracy of 1.000** on held-out source sentences means the model didn't just memorize training examples — it correctly generalized the substitution-plus-reordering rule to sentence combinations it had never seen during training (500 total generated example sentences, held-out validation split).

## 4. Case Study 3 — Why Summarization Is the One That Doesn't Run Here

Abstractive summarization asks a model to read a passage and produce new, shorter text that preserves its meaning — a capability that depends on genuine language understanding built from exposure to enormous amounts of real, diverse text. This is qualitatively different from §2's sentiment task (a bounded classification problem a small model can learn directly from a clean synthetic distribution) and from §3's translation demo (a structural mechanism, not requiring real-world knowledge, that invented vocabulary can demonstrate honestly). There is no honest small-scale proxy for abstractive summarization quality: a from-scratch decoder trained for seconds on a tiny corpus would produce grammatically broken, semantically empty text, and presenting that output as a "summarization case study" would actively misrepresent what the technique can do — a materially different problem than simply being a *less impressive* demo.

`implementation.py`'s summarization section is consequently written as correct, real, Colab-ready code — `sshleifer/distilbart-cnn-12-6` (a distilled BART checkpoint fine-tuned for summarization, chosen for the same T4-feasibility reasons `deep-learning-mastery`'s and this repository's other topics choose distilled checkpoints) applied to a CNN/DailyMail validation subsample, scored with **ROUGE**, the standard n-gram-overlap metric for summarization quality — but is not executed in this development sandbox, and `proof.png`'s third panel says so directly rather than showing fabricated ROUGE scores or invented example summaries.

## 5. Scaling to Production

Sentiment classification at production scale follows the exact evaluation discipline demonstrated here (confusion matrix, precision/recall/F1) on real datasets like SST-2 or IMDB, at far larger scale and with real pretrained checkpoints, per `01-Review-of-Fundamental-LLMs/03-Pretraining-vs-Fine-Tuning-Paradigms`. Production translation systems use the identical encoder-decoder-with-cross-attention mechanism demonstrated here, scaled to real subword vocabularies of tens of thousands of tokens and trained on millions of real sentence pairs rather than 500 synthetic ones. Production summarization systems are exactly what §4's gated code path runs — a pretrained encoder-decoder (BART, T5, PEGASUS-family) fine-tuned on large summarization-specific datasets, evaluated with ROUGE — at a scale this demo's from-scratch approach could never honestly approximate.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Lewis, Mike, et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *ACL*, 2020.
- Lin, Chin-Yew. "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL Workshop*, 2004.
- Tiedemann, Jörg, and Santhosh Thottingal. "OPUS-MT — Building Open Translation Services for the World." *EAMT*, 2020.
