# Explanation: `implementation.py`

## Applying Topic 04.01's Lessons From the Start

```python
FACTS = [
    ("the Plimwick device", "...", 0),   # category, shared many-to-one
    ...
]
SEEN_FACTS = [0, 1, 2, 3, 6, 7, ...]
HELD_OUT_FACTS = [4, 5, 10, 11, ...]
```

**Why this file's knowledge base is structured identically to the previous topic's *final*, corrected design, rather than starting simpler and discovering the same issues again:** `01-Retrieval-Augmented-Generation-RAG-Vector-DBs`'s `explanation.md` documents a three-iteration debugging process to reach a valid held-out evaluation (avoiding an impossible-classification bug, then a memorization shortcut). Re-deriving that same design from scratch here would repeat effort without adding anything — the lesson (shared category labels prevent entity-memorization shortcuts; genuinely held-out entities are required to test real generalization) transfers directly, so it's applied from the first line of this file instead.

## BM25's Vectorized-Free Implementation

```python
def score(self, query, doc_idx):
    q_terms = tokenize(query)
    doc_terms = self.tokenized[doc_idx]
    tf = Counter(doc_terms)
    ...
```

**Why compute `Counter(doc_terms)` fresh inside `score` rather than precomputing term frequencies for every document once, up front:** For a corpus this size (230 documents), the cost difference is negligible, and computing term frequency at scoring time keeps the function's relationship to the BM25 formula in `theory.md` §2 direct and easy to verify line-by-line — every symbol in the formula corresponds to one clearly-named local variable. A production BM25 implementation (e.g., inside a real search engine) would precompute an inverted index and per-document term frequencies once at corpus-build time, exactly as FAISS precomputes its index structure once rather than rescanning all documents per query — the same precompute-once-query-many-times principle discussed for FAISS in the previous topic, traded off here for code clarity at a scale where the trade-off doesn't matter.

```python
self.idf = {term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()}
```

**Why `+ 0.5` in both numerator and denominator, and `+ 1` outside the log:** This is the standard smoothed IDF formula from the BM25 literature — the `0.5` terms (Laplace-style smoothing) prevent division by zero or a negative/undefined log argument for terms appearing in every document (or in none), and the outer `+ 1` guarantees the result stays non-negative even for very common terms, rather than needing a separate special case. These constants are part of the standard formula, not tuned for this specific corpus.

## Why the Dense Retriever Is Re-Trained Identically Rather Than Reused

```python
print("Training dense dual-encoder retriever (same recipe as Topic 04.01)...")
retriever = Encoder().to(DEVICE)
...
```

**Why retrain a fresh dense retriever in this file rather than loading a saved checkpoint from the previous topic:** This repository's stated convention (`README.md`, Setup section, Phase 02's module README) is that every topic's `implementation.py` is self-contained with no cross-topic imports or shared checkpoints — the same reasoning applies here as to why `Attention`/`EncoderBlock`/`Encoder` are redefined in this file rather than imported. Retraining with an identical recipe (same architecture, same steps, same seen/held-out split) means this topic's dense-retrieval numbers are a fair, independently-reproducible comparison point against BM25, not numbers inherited from a different run under possibly-different conditions.

## Building Two Clearly-Separable Query Populations for Tool-Use Routing

```python
GENERAL_TEMPLATES = ["What is two plus two?", "Say hello to me.", "What color is the sky?", ...]

def build_tool_examples(n=300):
    for _ in range(n):
        if random.random() < 0.5:
            f = random.choice(SEEN_FACTS + HELD_OUT_FACTS)
            examples.append((make_query(f), 1))
        else:
            examples.append((random.choice(GENERAL_TEMPLATES), 0))
```

**Why draw fact-queries from *both* `SEEN_FACTS` and `HELD_OUT_FACTS` for this specific experiment, when the retrieval comparison above carefully keeps them separate:** The tool-use classifier's job is narrower than the retriever's — it only needs to recognize "this query is about one of our fictional entities" as a *category* of question, not to identify *which specific* entity or retrieve anything about it. Since that recognition doesn't require the entity itself to have been seen in training (unlike retrieval, which fundamentally must generalize its embedding of a specific new entity), mixing seen and held-out entities into the tool-use training and evaluation pools doesn't reintroduce the memorization-shortcut concern from the previous topic — there's no entity-specific label being predicted here, only "fact-shaped query" vs. "general-shaped query."

**Why report a full confusion matrix (`TP`/`TN`/`FP`/`FN`) rather than just accuracy for this classifier:** As `theory.md` §4 notes, false positives (unnecessary lookups) and false negatives (missed lookups where the model should have deferred to external knowledge but didn't) have different practical costs in a real system — wasted latency/cost versus an outright wrong or hallucinated answer. Reporting both separately, rather than collapsing them into one accuracy number, keeps that distinction visible even though this run happened to produce zero of both.

## `proof.png`: Why Panel B Exists Separately From Panel A

**Panel A** compares BM25 and dense retrieval on *both* seen and held-out entities, side by side — useful for seeing the overall pattern. **Panel B** isolates *only* the held-out condition and zooms in on top-1 versus top-3 for both methods specifically. This second, narrower view exists because the held-out comparison is this topic's central, most information-dense claim (BM25's structural generalization advantage over a small trained dense retriever) — giving it a dedicated panel, rather than requiring a reader to extract that specific comparison from Panel A's four bars, makes the topic's main point directly visible rather than merely derivable.
