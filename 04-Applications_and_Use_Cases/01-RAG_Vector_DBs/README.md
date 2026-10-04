# Introduction to Retrieval-Augmented Generation (RAG) with Vector DBs

## 1. The Core Idea

A pretrained LLM's knowledge is frozen at training time and baked into its parameters (its "parametric memory"). **Retrieval-Augmented Generation** (Lewis et al., 2020) adds a second, external memory: a searchable collection of documents the model can consult at inference time, retrieving whatever is relevant to the current query and conditioning its answer on that retrieved text. This decouples "how much the model knows" from "how large the model is" — a small model with a good retriever can answer questions about content it was never trained on, as long as that content is in its document store.

`implementation.py` builds every piece of this pipeline for real: an invented knowledge base (fictional entities that cannot exist in any pretrained corpus, so there's no ambiguity about whether "knowledge" came from training or retrieval), a trained dense retriever, a real FAISS vector index, and an answering model evaluated with and without retrieved context.

## 2. Dense Retrieval: A Dual Encoder Trained via Contrastive Learning

A **dual encoder** (also called a bi-encoder) uses the same (or a similarly-structured) network to embed both queries and documents into a shared vector space, such that a query's embedding sits close to its relevant document's embedding and far from irrelevant ones. Karpukhin et al.'s Dense Passage Retrieval (DPR, 2020) — one of the techniques that established this as standard practice for neural retrieval — trains exactly this kind of encoder via **in-batch negative contrastive learning**: for a batch of `(query_i, document_i)` pairs, every document in the batch *other than* `document_i` serves as a "negative" for `query_i`, with no need to separately mine hard negatives for small-scale training.

`implementation.py` implements this directly:

```python
sim = q_emb @ d_emb.T * 10.0    # cosine similarity matrix, every query vs. every doc in the batch
targets = torch.arange(len(batch_facts), device=DEVICE)
loss = F.cross_entropy(sim, targets)   # correct doc = the diagonal entry
```

The similarity matrix's diagonal holds each query's similarity to its *own* correct document; cross-entropy against the identity permutation directly maximizes diagonal similarity relative to every off-diagonal (wrong-document) entry in the batch — exactly the InfoNCE-style objective DPR and similar dense retrievers use.

## 3. FAISS: A Real Vector Database

Once documents are embedded, finding the nearest ones to a query embedding is a nearest-neighbor search problem — trivial to define, but naively `O(n)` per query against `n` documents, which doesn't scale to real corpora with millions of documents. **FAISS** (Facebook AI Similarity Search) is a library purpose-built for this: efficient exact and approximate nearest-neighbor search over dense vectors, at the scale actual vector databases operate at.

```python
index = faiss.IndexFlatIP(D_MODEL)   # inner product on L2-normalized vectors = cosine similarity
index.add(doc_embeddings)
...
scores, retrieved_idx = index.search(q_emb, 3)
```

`IndexFlatIP` performs *exact* nearest-neighbor search via inner product — appropriate at this topic's document-count scale (230 documents), and, since every embedding is L2-normalized (`F.normalize(..., dim=-1)`), inner product here is mathematically identical to cosine similarity. Production-scale vector databases typically use *approximate* nearest-neighbor indexes (e.g., FAISS's `IndexIVFFlat` or HNSW-based indexes) that trade a small amount of retrieval accuracy for dramatically faster search over millions or billions of vectors — the same underlying operation, at a different point on the speed/exactness trade-off curve.

## 4. What Was Actually Measured — In Full, Including the Parts That Didn't Work

This topic went through a real, multi-stage debugging process worth reporting in full rather than only showing the final numbers — `explanation.md` documents each stage's specific bug or design flaw. The final, corrected experimental design distinguishes two genuinely different questions:

**Does the mechanism work at all, in-distribution?** (Retriever and answerer both had some training exposure to the relevant entities, though evaluated on fresh question phrasings never used verbatim during training.)

| | Seen entities |
|---|---|
| Retrieval top-1 / top-3 | 0.700 / 0.900 |
| Answering (any context condition) | 1.000 |

**Does it zero-shot generalize to entities the retriever and answerer have NEVER been trained on at all?**

| | Held-out entities |
|---|---|
| Retrieval top-1 / top-3 | 0.000 / 0.200 |
| Answering — no context | 0.300 |
| Answering — oracle (true) context | 0.200 |
| Answering — FAISS-retrieved context | 0.200 |

The in-distribution numbers show the mechanism genuinely works: contrastive training pulls matching query/document pairs together in embedding space enough for FAISS to find them reliably, and the answerer correctly uses whatever signal is available. The held-out numbers do **not** show retrieval or context helping — they hover around or even slightly below the 0.200 chance baseline for 5-way category classification, and retrieval accuracy on entirely novel entities is close to zero.

## 5. Why Zero-Shot Generalization Failed Here — A Real, Expected Limitation, Not a Broken Demo

This was investigated, not simply accepted at face value. Two rounds of genuine methodological improvement were tried — expanding the knowledge base from 15 to 30 facts and increasing the contrastive batch size for more in-batch negatives — and neither meaningfully changed the held-out result. This points to a real, well-documented characteristic of dense retrieval rather than a fixable bug: **learning embeddings that generalize to genuinely novel content is a substantially harder skill than learning to fit a small, fixed training set**, and it is known to require training on a much larger and more diverse set of query-document pairs than 20–30 examples can provide. Real dense retrievers (DPR and its successors) are trained on hundreds of thousands to millions of query-passage pairs specifically because this generalization gap is real and scale-sensitive — the same qualitative lesson `01-Review-of-Fundamental-LLMs/04-Scaling-Laws-and-Model-Efficiency` demonstrates for language modeling loss applies here to retrieval quality: a 110,528-parameter encoder trained on a few dozen examples has no realistic path to learning a broadly generalizable notion of semantic similarity, only a narrow one fit to its specific training documents.

This is reported as a real finding rather than smoothed over, because it is itself informative: it demonstrates *why* production RAG systems invest so heavily in retriever pretraining at scale, rather than treating retrieval as a detail that "just works" once a vector database is wired up.

## 6. Scaling to Production

Every component demonstrated here — contrastive dual-encoder training, a real vector index, context-conditioned answering — is structurally identical to production RAG systems; what changes is scale, in exactly the dimension §5 identifies as the bottleneck. Production dense retrievers are pretrained on internet-scale query-document pairs (often initialized from a general-purpose pretrained language model, then further trained with retrieval-specific contrastive objectives) before ever being pointed at a specific downstream knowledge base, which is precisely the generalization capability this toy-scale retriever lacks. FAISS itself is unchanged between a 230-document toy index and a production billion-document one — only the index type (exact vs. approximate) and infrastructure around it (sharding, incremental updates) differ.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, 2020.
- Karpukhin, Vladimir, et al. "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP*, 2020.
- Johnson, Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity search with GPUs." (FAISS.) 2017.
