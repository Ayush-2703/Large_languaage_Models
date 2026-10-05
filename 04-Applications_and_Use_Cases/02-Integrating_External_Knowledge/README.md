# Integrating External Knowledge into LLMs

## 1. Beyond Dense Retrieval: Two More Ways Knowledge Enters a Model's Answer

`01-Retrieval-Augmented-Generation-RAG-Vector-DBs` built a full dense-retrieval pipeline and found — honestly, after real debugging — that a small trained neural retriever generalizes well to entities it was trained on, but not to genuinely novel ones, at least not at toy scale. This topic asks two follow-up questions directly motivated by that finding: **is there a retrieval method that doesn't need training data to generalize at all** (§2), and **can a model learn when it needs external knowledge in the first place**, rather than always retrieving or never retrieving (§4)?

## 2. BM25: Statistical Retrieval With Zero Training

BM25 (Robertson & Walker, and later refinements; still the standard classical information-retrieval baseline) scores a document's relevance to a query using term-frequency statistics computed directly from the corpus — no learned parameters, no training data, no gradient descent:

```
score(q, d) = Σ_{t∈q} IDF(t) · [ f(t,d)·(k1+1) ] / [ f(t,d) + k1·(1 - b + b·|d|/avgdl) ]
```

Each term `t` in the query contributes a score based on: **IDF** (inverse document frequency — rare terms across the corpus are more informative than common ones), **term frequency** `f(t,d)` (how often `t` appears in this specific document), with two corrections built into the denominator — `k1` controls how quickly additional occurrences of a term stop adding much more score (saturation: the 10th occurrence of a word matters far less than the 1st), and `b` (weighted by document length relative to the corpus average, `avgdl`) penalizes long documents that might rack up high term frequencies simply by containing more text overall. `implementation.py`'s `BM25` class computes every term of this formula directly, with no training step at all — `IDF` is computed once from document frequencies across the corpus, and scoring a query against any document, including one never seen before construction of the corpus, is just plugging numbers into the formula above.

## 3. What Was Actually Measured — And Why the Contrast Matters

| | Seen entities (top-1) | Held-out entities (top-1) |
|---|---|---|
| BM25 (no training) | 1.000 | **1.000** |
| Dense dual-encoder (trained) | 0.700 | **0.000** |

This is a direct, clean confirmation of the mechanism behind `01-Retrieval-Augmented-Generation-RAG-Vector-DBs`'s finding, not just a repetition of it. BM25's perfect score on held-out entities isn't a stronger *retrieval* result in some general sense — it's a structural consequence of BM25 having no train/test distinction at all. Every query in this topic's synthetic facts shares distinctive vocabulary directly with its corresponding document (e.g., a query about "the Thistledown Engine" contains the words "Thistledown Engine," which appear nowhere else in the corpus), so exact term-overlap statistics alone are enough to identify the right document with certainty, regardless of whether that specific entity happened to be used during any training process — because BM25 has no training process to be excluded from.

The dense retriever's held-out failure, by contrast, reflects exactly what `theory.md` for the previous topic explains: it must *learn* a notion of semantic similarity from training examples, and with only 20 training documents, that learned notion doesn't transfer to entities never seen in any form during training. **Neither result is a flaw in its respective method — they reflect a genuine, well-known trade-off**: sparse, statistical methods like BM25 generalize immediately to new content because they don't need to learn anything about it, but they only work when a query shares literal vocabulary with its relevant document (they'd fail on a query using synonyms or paraphrases the document doesn't literally contain, a case dense retrieval is specifically better positioned to handle, given sufficient training). Production retrieval systems very often combine both — "hybrid retrieval" — precisely because they fail in different, complementary ways.

## 4. Tool Use: Learning When External Knowledge Is Needed at All

Not every query needs retrieval. A model that always retrieves wastes computation (and, in a real deployed system, real latency and API cost) on queries it could answer directly; a model that never retrieves cannot answer anything requiring knowledge outside its parameters. **Tool use / function calling** frames this as a decision the model itself makes: recognize when a query requires an external lookup versus general capability, and route accordingly — the same underlying idea behind production systems where an LLM decides whether to call a calculator, a search API, or a code interpreter mid-generation, rather than attempting everything from parametric memory alone.

`implementation.py`'s `ToolClassifier` is a direct, minimal version of this routing decision: a binary classifier trained on queries labeled "needs lookup" (questions about this topic's fictional entities) versus "answer directly" (simple general questions like arithmetic or color-naming). It reached **1.000 routing accuracy** with zero false positives or false negatives. This task's two query types are lexically quite distinct by construction (fact-queries consistently reference specific invented proper nouns; general queries don't), which makes this an easier separation than real-world tool-use routing typically is — genuine ambiguity (queries that *could* be answered either way, or that blend both needs) is a harder, more realistic version of this same problem that this clean synthetic setup doesn't test.

## 5. Scaling to Production

Hybrid retrieval (combining BM25-style sparse scoring with dense embeddings, often via a weighted combination or a re-ranking stage) is standard practice in production search and RAG systems specifically because of the complementary failure modes §3 identifies — sparse methods for exact terminology matches and zero-shot robustness to new content, dense methods for semantic/paraphrase matches once sufficiently trained. Tool-use routing at production scale (e.g., an LLM deciding whether to call a search tool, a calculator, or a code execution sandbox) uses the same underlying binary-or-multi-way decision demonstrated here, typically as one capability learned jointly with instruction-following during fine-tuning, rather than as a separately-trained classifier — but the core mechanism (query in, routing decision out) is the same.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Robertson, Stephen, and Hugo Zaragoza. "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 2009.
- Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP*, 2020.
- Schick, Timo, et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." 2023.
