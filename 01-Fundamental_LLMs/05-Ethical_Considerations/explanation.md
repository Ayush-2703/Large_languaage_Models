# Explanation: `implementation.py`

## Building the Synthetic Corpus

```python
def build_corpus(n_sentences: int = 4000, male_career_rate: float = 0.85) -> list:
    ...
    use_male = random.random() < male_career_rate
    person = random.choice(MALE_TERMS if use_male else FEMALE_TERMS)
```

**What:** Generates sentences from templates, choosing a male- or female-coded term for each sentence with probability controlled by a single parameter, `male_career_rate`.

**Why synthetic data with a known injected rate, rather than a real scraped corpus:** This is the central methodological choice of this topic, and it's worth stating plainly why it's not a shortcut. Real corpora (the kind Bolukbasi et al. and Caliskan et al. studied) show real bias, but with an unknown, unmeasurable "true" cause — you can observe the effect but never fully audit the cause. Building the corpus here means the cause (an 85%/15% split, chosen and printed explicitly) and the effect (the measured WEAT association, Section 3) can be placed side by side with a direct causal claim between them: *this* number produced *that* measurement, nothing hidden in between. That traceability is the whole pedagogical point — real-world bias auditing rarely gets to work backward to a clean root cause this cleanly, which `theory.md` §7 discusses directly.

**Why control the rate on the male side and let the female side be `1 - male_career_rate` rather than setting both independently:** This guarantees the two groups are perfect complements within each sentence category, so any measured asymmetry in the resulting embeddings can only be attributed to the single controlled variable — not to some other uncontrolled difference between how often male- vs. female-coded terms appear overall.

## Skip-Gram Training Pairs

```python
for i, center in enumerate(ids):
    for j in range(max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)):
        if i != j:
            pairs.append((center, ids[j]))
```

**What:** For every word in every sentence, pairs it with every other word within `WINDOW=3` positions — the standard skip-gram training-pair construction from Mikolov et al. (2013).

**Why skip-gram specifically, rather than the simpler co-occurrence-counting approach used implicitly in bigram/trigram models earlier in this phase:** Skip-gram embeddings are *dense* and *trained by gradient descent*, which is what makes cosine similarity between them a meaningful geometric quantity in the first place — a raw co-occurrence count matrix would require additional processing (e.g., PMI weighting, dimensionality reduction) to produce something WEAT's cosine-similarity math was designed for. Using skip-gram directly keeps the method identical in spirit to the real word2vec embeddings the cited literature actually studied.

## `SkipGramNS` and Negative Sampling

```python
neg_score = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)
neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)
return -(pos_loss + neg_loss).mean()
```

**What:** Implements the negative-sampling loss from Mikolov et al. (2013): maximize the predicted probability of the true context word (`pos_loss`), while minimizing the predicted probability of `K_NEGATIVES=5` randomly sampled words that were *not* actually nearby (`neg_loss`).

**Why negative sampling instead of full softmax over the vocabulary:** With only 54 unique words in this synthetic vocabulary, a full softmax would be cheap enough to use directly — negative sampling isn't necessary for speed here the way it is for real word2vec's 100,000+ word vocabularies. It's used anyway because it's the actual, standard training method behind the real embeddings the WEAT literature studied; using the same method keeps this demonstration methodologically aligned with the papers it's built to illustrate, not just conceptually similar.

```python
unigram_probs = np.array([word_freq[w] for w in vocab], dtype=np.float64) ** 0.75
unigram_probs /= unigram_probs.sum()
```

**Why raise frequencies to the 0.75 power before sampling negatives:** This is Mikolov et al.'s specific finding — sampling negatives proportional to raw frequency over-samples extremely common words and under-samples rare ones; the 0.75 exponent flattens the distribution somewhat, striking a better empirical balance. It's included here for methodological fidelity to the source technique, even though this toy vocabulary's frequency distribution is far less skewed than a real corpus's.

## The Association Measurement

```python
def association(word, male_set, female_set):
    return mean_cos_sim(word, male_set) - mean_cos_sim(word, female_set)
```

**What:** Implements the `s(w, A, B)` statistic from `theory.md` §3 directly — one line, matching the formula exactly, so the code and the math are trivially checkable against each other.

```python
effect_size = (mean_career - mean_family) / (pooled_std + 1e-8)
```

**Why report an effect size, not just the two raw mean associations:** Two numbers like `+0.149` and `-0.121` don't, by themselves, say whether the gap between them is large relative to how much individual words' associations naturally vary — a gap that looks large in absolute terms could still be unremarkable if word-to-word variance is even larger. Dividing by the pooled standard deviation puts the gap in standardized units (the same Cohen's-d convention used throughout the social sciences), which is what makes "d ≈ 2.0" interpretable as a genuinely large, not just nonzero, effect.

## `proof.png` Generation — Why Composite

```python
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.35)
```

**What:** One figure, two panels — a conceptual pipeline diagram (drawn with `FancyBboxPatch`/`FancyArrowPatch`, no measured data) on top, the real WEAT bar chart (built entirely from `career_assoc`/`family_assoc`, computed above from actually-trained embeddings) on the bottom.

**Why this topic gets a composite image rather than a pure data chart like Topics 1, 2, and 4, or a pure placeholder like Topic 3:** "Ethical Considerations" is a broader subject than any single measurement can fully cover — the pipeline diagram exists to show *where* bias can enter beyond the one mechanism this script actually measures (training-data statistics), which `theory.md` §4 discusses in full. But unlike a topic where no real experiment is possible at all, this one has a genuine, real, fully-reproducible measurement available — so the diagram is paired with real numbers rather than substituting for them. The caption directly under the diagram ("This topic's real experiment isolates the TRAINING stage...") exists specifically to prevent the two panels from being read as equally weighted illustration; the top is context, the bottom is evidence.

**Why the injected rate (85%) and the measured effect size (d) are both printed directly in the bottom panel's title:** So the cause-and-effect claim this topic is built around — a specific, known input statistic produces a specific, measured output association — is visible in the image itself, not only in the console output or this document. Someone looking at `proof.png` alone should be able to see both numbers and understand that one produced the other.

## Section 5 — Why It's Gated Behind a Flag

```python
RUN_PRETRAINED_SECTION = False   # set True on Colab with internet access
```

**Why include a real-DistilBERT fill-mask probe at all, if it can't run here:** The controlled synthetic experiment above proves the *mechanism* cleanly, precisely because its cause is fully known — but that same design choice means it doesn't, by itself, demonstrate that real production models exhibit this behavior. Section 5 exists to close that gap for anyone running this script on Colab: the same style of probe (asking a masked-language-model to fill in a blank, then inspecting which completions it favors) run against a real pretrained checkpoint connects this topic's toy demonstration to the actual, widely-documented finding it's illustrating. It's gated behind the same explicit flag pattern used in Topics 2 and 3 for the same reason: the script should always run to completion and always produce a real `proof.png` from what this sandbox *can* verify, rather than crashing partway through on a network call it can't make.
