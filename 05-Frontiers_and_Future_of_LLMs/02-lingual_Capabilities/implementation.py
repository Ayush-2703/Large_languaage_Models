"""
Topic 5.2 — Multilingual and Cross-lingual Capabilities
=========================================================

GOAL
----
Demonstrate, with a real (not simulated-by-fiat) experiment, the core
mechanism behind cross-lingual transfer in modern multilingual models:
independently-trained word/token embedding spaces for two different
"languages" that describe the same underlying concepts end up as
*linear (in fact orthogonal) transforms of one another* — so a small
seed dictionary is enough to align the two spaces and translate words
that were never seen paired together during training.

WHY A SYNTHETIC EXPERIMENT HERE, AND WHY IT'S HONEST
------------------------------------------------------
A meaningful test of real cross-lingual *transfer* (e.g. does mBERT's
English fine-tune help on Hindi) requires real, large multilingual
pretrained checkpoints (mBERT, XLM-R) and this sandbox cannot reach
Hugging Face Hub to download them (huggingface.co is blocked; only
PyPI is reachable). Rather than ship a fabricated "before/after
accuracy" number for a real checkpoint we cannot run, this topic
reproduces the actual *mechanism* researchers rely on
(Mikolov et al., 2013, "Exploiting Similarities among Languages for
Machine Translation"; Conneau et al., 2017/2018, "Word Translation
Without Parallel Data" / MUSE) on fully synthetic, from-scratch data:

1. Define a shared latent semantic space of K "concepts" (unknown to
   the embedding-training algorithm) with a real bilinear co-occurrence
   generative model — the same generative assumption that underlies
   skip-gram/word2vec and PPMI-SVD embeddings
   (Levy & Goldberg, 2014, "Neural Word Embedding as Implicit Matrix
   Factorization").
2. Independently sample two co-occurrence corpora from this same latent
   space ("language A" and "language B"), with different vocabulary
   orderings and an explicit random rotation applied to language B's
   space — modeling the well-documented empirical fact that two
   embedding spaces trained independently on different corpora recover
   the same relative geometry only up to an unknown rotation.
3. Train PPMI + truncated-SVD embeddings independently per language
   (no cross-lingual information used at training time).
4. Use a small seed dictionary (30% of concepts) to fit an **orthogonal
   Procrustes** alignment, then measure nearest-neighbor translation
   accuracy on the **held-out 70%** of concepts the aligner never saw.

Every number in `proof.png` comes from this real, executed pipeline.

SCALING TO PRODUCTION LLMS
---------------------------
Real multilingual LLMs (mBERT, XLM-R, mT5) learn cross-lingual
alignment implicitly, through joint multilingual pretraining on shared
subword vocabulary and parallel/comparable corpora, rather than through
an explicit post-hoc Procrustes step. But the underlying geometric
fact this experiment demonstrates — that semantically equivalent
content across languages occupies comparable relative geometry in
embedding space — is exactly why techniques like cross-lingual
zero-shot transfer, bitext mining, and unsupervised MT initialization
work at all. In production, opus-mt (small MarianMT checkpoints) or
mT5/XLM-R would be the small-checkpoint entry points for hands-on
cross-lingual fine-tuning on a Colab T4, using real datasets such as
small WMT16 language-pair subsamples.
"""

import numpy as np

np.random.seed(0)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K = 60              # shared "concepts" == ground-truth translation pairs
D_TRUE = 10         # true latent semantic dimensionality (unknown to the learner)
N_PAIRS = 400_000   # (center, context) co-occurrence draws per language corpus
D_EMB = 10          # embedding dimensionality used when training (matched to true rank)
SEED_FRAC = 0.3     # fraction of concepts used as the "known translation dictionary"


def build_generative_model():
    """A shared latent semantic space + bilinear co-occurrence kernel.

    This is exactly the generative assumption behind skip-gram/PMI
    embeddings: P(context=j | center=i) ~ softmax(z_i . z_j).
    """
    Z = np.random.randn(K, D_TRUE)
    logits = Z @ Z.T
    np.fill_diagonal(logits, -1e9)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return Z, probs


def sample_cooccurrence(probs_given_center, seed):
    """Draw N_PAIRS (center, context) pairs from the shared generative model."""
    rng = np.random.RandomState(seed)
    cooc = np.zeros((K, K))
    centers = rng.randint(0, K, size=N_PAIRS)
    for c in range(K):
        idx = np.where(centers == c)[0]
        if len(idx) == 0:
            continue
        ctx = rng.choice(K, size=len(idx), p=probs_given_center[c])
        for j in ctx:
            cooc[c, j] += 1
    return cooc


def embed_from_cooccurrence(cooc, dim):
    """PPMI + truncated SVD -- the classical, theoretically-grounded word
    embedding recipe (equivalent to skip-gram-with-negative-sampling in
    expectation, per Levy & Goldberg 2014)."""
    total = cooc.sum()
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((cooc * total) / (row_sum @ col_sum + 1e-12) + 1e-12)
    ppmi = np.maximum(pmi, 0)
    U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
    return U[:, :dim] * np.sqrt(S[:dim]), S


def orthogonal_procrustes(E_source_seed, E_target_seed):
    """Closed-form solution: W* = argmin_{W orthogonal} ||E_source_seed @ W - E_target_seed||_F"""
    M = E_source_seed.T @ E_target_seed
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def nn_translation_accuracy(concepts, emb_A_word, emb_B_word, perm_B, mapping):
    correct = 0
    for c in concepts:
        vec = emb_B_word[perm_B[c]].copy()
        if mapping is not None:
            vec = vec @ mapping
        dists = np.linalg.norm(emb_A_word - vec, axis=1)
        correct += int(np.argmin(dists) == c)
    return correct / len(concepts)


def run():
    Z, probs = build_generative_model()

    cooc_A = sample_cooccurrence(probs, seed=1)
    cooc_B = sample_cooccurrence(probs, seed=2)  # independent corpus = "a different language"

    emb_A_concept_order, sv_A = embed_from_cooccurrence(cooc_A, D_EMB)
    emb_B_concept_order, sv_B = embed_from_cooccurrence(cooc_B, D_EMB)

    # Independently-trained embedding spaces are only recoverable up to an
    # arbitrary rotation -- make that explicit and realistic for language B.
    rand_rotation = np.linalg.qr(np.random.RandomState(99).randn(D_EMB, D_EMB))[0]
    emb_B_concept_order = emb_B_concept_order @ rand_rotation

    # Language B's surface vocabulary is independently indexed (word id != concept id)
    perm_B = np.random.RandomState(7).permutation(K)
    inv_perm_B = np.argsort(perm_B)
    emb_A_word = emb_A_concept_order
    emb_B_word = emb_B_concept_order[inv_perm_B]

    concept_ids = np.arange(K)
    np.random.RandomState(3).shuffle(concept_ids)
    n_seed = int(SEED_FRAC * K)
    seed_concepts, test_concepts = concept_ids[:n_seed], concept_ids[n_seed:]

    E_A_seed = emb_A_word[seed_concepts]
    E_B_seed = emb_B_word[perm_B[seed_concepts]]
    W = orthogonal_procrustes(E_B_seed, E_A_seed)

    baseline_acc = nn_translation_accuracy(test_concepts, emb_A_word, emb_B_word, perm_B, None)
    aligned_acc = nn_translation_accuracy(test_concepts, emb_A_word, emb_B_word, perm_B, W)
    chance = 1 / len(test_concepts)

    results = {
        "n_concepts": K,
        "n_seed_pairs": n_seed,
        "n_test_pairs": len(test_concepts),
        "baseline_accuracy": baseline_acc,
        "procrustes_aligned_accuracy": aligned_acc,
        "chance_accuracy": chance,
        "top_singular_values_lang_A": sv_A[:12].tolist(),
        "top_singular_values_lang_B": sv_B[:12].tolist(),
    }
    print(results)
    return results


if __name__ == "__main__":
    run()
