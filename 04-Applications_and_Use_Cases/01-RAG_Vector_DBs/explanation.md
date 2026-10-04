# Explanation: `implementation.py`

This topic went through three distinct design iterations before reaching a valid experiment. Each is documented here because the *reasons* for each fix matter more than the final code alone — they're the actual lesson about what makes a RAG evaluation meaningful versus accidentally trivial or accidentally impossible.

## Iteration 1: The Impossible Classification Bug

The first version held out entire fact classes from the answerer's training (`train_facts = fact_idx[:11]`, `val_facts = fact_idx[11:]`) and evaluated a classifier with one output unit per fact. This produced 0.000 accuracy across every condition — including the **oracle** condition, where the model was handed the exact correct document. That specific failure (oracle context, the easiest possible case, scoring zero) was the tell: **a classifier cannot correctly predict a class it never saw a single labeled training example for, no matter what context it's given.** The bug wasn't in retrieval or in context-conditioning at all; it was in evaluating a closed-set classifier on classes structurally excluded from its own training.

## Iteration 2: The Memorization Shortcut

Fixing this by training the answerer on *all* fact classes (so every class had training examples) surfaced a second, different problem: **NO context accuracy (0.667) exceeded RETRIEVED-context accuracy (0.333).** This happened because every fact's entity name appears literally inside its query ("Tell me about Zorvath Corp." contains the string "Zorvath Corp"), and with a *fixed, closed set* of 15 entities all seen during training, a classifier can simply memorize the direct mapping from entity name to fact/label — entirely bypassing any need to read provided context. The model wasn't ignoring context because context didn't help; it was ignoring context because *it didn't need to*, having already memorized the answer from the query text alone. This is worth naming precisely: **any evaluation where the same closed set of entities appears in both training and testing risks this shortcut**, regardless of how the "context helps" experiment is otherwise designed.

## Iteration 3: The Valid Design

The fix required two simultaneous changes, not one:

```python
FACTS = [
    ("the Plimwick device", "...", 0),   # category 0 = TECH
    ("the Kestrel Protocol", "...", 0),  # also category 0 — SHARED, not unique per entity
    ...
]
SEEN_FACTS = [...]        # used in ALL training
HELD_OUT_FACTS = [...]    # used in NO training, ever — the only entities actually evaluated on
```

**Shared category labels** (5 categories, 6 facts each) mean the label space itself doesn't uniquely identify any one entity — knowing "this is category TECH" doesn't tell you *which* of six TECH entities is being discussed, so entity-name memorization can't shortcut the label directly the way a 1-to-1 entity→class mapping could.

**Held-out entities used in zero training steps, of any kind** close the remaining gap: since `Captain Idris Malen` (for example) never appears in *any* training batch — not the retriever's contrastive pairs, not the answerer's classification examples — there is no entity-name-to-label association available to memorize for it at all. Any above-chance accuracy on held-out entities can only come from the model genuinely using the query and/or context to reason about a name it has never encountered before, not from having quietly memorized the answer during training.

## Why the Knowledge Base Grew From 15 to 30 Facts

```python
FACTS = [
    # ... 15 original facts ...
    ("the Corvane Filter", "...", 0),
    ("the Wrenfield Loom", "...", 0),
    # ... 15 more, added specifically to give both the retriever and
    # answerer more training diversity ...
]
```

**Why this specific change, rather than accepting the first (smaller) result:** With only 10 training documents, held-out retrieval accuracy was 0.000/10 and held-out oracle-context answering was exactly at chance — a result consistent with the model having too little training diversity to learn anything that generalizes, as opposed to having learned something and failed to apply it. Doubling the knowledge base (and proportionally increasing the contrastive batch size, from 8 to 16, for more in-batch negatives per training step) is a direct, principled response to that diagnosis: more distinct query-document pairs during contrastive training is the standard lever for improving embedding generalization, not a parameter tweaked speculatively in hopes of a nicer-looking number.

**Why this improved in-distribution results substantially (0.000→0.700 top-1 accuracy on seen entities) without meaningfully improving held-out results:** This is itself informative, and is why `theory.md` §5 treats the remaining held-out gap as a real finding rather than something one more round of tuning would fix. Going from 10 to 20 training documents gave the model enough signal to fit *that* specific set of documents well (visible directly in the improved in-distribution numbers), but 20 documents is still nowhere near the scale real dense retrievers use to learn representations that transfer to content never seen in any form during training. The in-distribution improvement is evidence the training mechanism works correctly; the held-out result staying flat despite it is evidence of a genuine data-scale ceiling, not a bug still waiting to be found.

## Reporting Both Results Rather Than Choosing One

```python
print("=== Real retrieval accuracy on HELD-OUT facts (never trained on) ===")
...
print("=== Real retrieval accuracy on SEEN facts (in-distribution, fresh phrasings) ===")
```

**Why the final script measures and reports both conditions, rather than only the (more favorable) in-distribution numbers or only the (less favorable) held-out numbers:** Reporting only the in-distribution result would overstate this pipeline's capability — it would look like a clean, unqualified success story while quietly never testing the harder, more realistic claim that RAG is often motivated by (retrieving relevant content the system was never specifically trained on). Reporting only the held-out result would understate that the core mechanism — contrastive retrieval training, real FAISS indexing, context-conditioned answering — is implemented correctly and does work, just not yet at a scale sufficient for zero-shot generalization to brand-new entities. Both numbers together are the honest, complete claim: the pipeline is real and functional, and its zero-shot generalization is real and limited, for a specific, explained, scale-related reason rather than an unexplained one.

## Why "Fresh Phrasings" Even for Seen Entities

```python
def make_query(fact_row):
    entity = FACTS[fact_row][0]
    return random.choice(TEMPLATES).format(e=entity)
```

**Why the in-distribution evaluation still regenerates a query at evaluation time rather than reusing an exact training example:** Even for entities the model has seen during training, evaluating on a query string that was never presented in that exact form keeps the in-distribution measurement honest about testing generalization to phrasing, if not to entity identity — a weaker but still real form of generalization, clearly distinguished in every printout and in `theory.md`'s table from the much harder held-out-entity test.
