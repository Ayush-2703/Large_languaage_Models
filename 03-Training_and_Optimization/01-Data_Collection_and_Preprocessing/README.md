# Data Collection and Preprocessing for LLMs

## 1. Why Preprocessing Is Not a Minor Preliminary Step

The scaling laws in `01-Review-of-Fundamental-LLMs/04-Scaling-Laws-and-Model-Efficiency` treat training data as a quantity — tokens processed. But *which* tokens matters enormously, and real-world text collected at web scale arrives full of exact duplicates, near-duplicates, low-quality boilerplate, and personally identifiable information (PII). `implementation.py` builds a corpus with all four of these problems deliberately injected at known rates, then runs a real, measurable five-stage cleaning pipeline over it — every count reported by the code is the actual result of running that pipeline, not an illustration of what it would do.

## 2. Why Duplicates Are a Real Problem, Not Just a Storage Inefficiency

Lee et al. ("Deduplicating Training Data Makes Language Models Better," 2022) found that large web-scraped corpora contain substantial exact and near-duplicate content, and that training on de-duplicated data measurably improves downstream model quality — not merely training efficiency. Two distinct mechanisms are commonly cited: duplicated text lets a model effectively "memorize" repeated passages rather than generalizing, and duplicated content skews the training distribution toward whatever happened to be copied or crawled multiple times, rather than reflecting a representative sample of language.

`implementation.py` separates duplication into two tiers, since they need different detection methods:

- **Exact duplicates** — byte-identical documents — are found trivially via hashing: `hashlib.sha256(d.encode()).hexdigest()`. Two documents hash identically if and only if they are exactly identical (ignoring the astronomically small probability of a SHA-256 collision), so a set of seen hashes catches every exact repeat in a single pass.
- **Near-duplicates** — documents differing only in minor edits (whitespace, punctuation, a word or two) — require a *similarity* measure, not an equality check. `implementation.py` uses **Jaccard similarity over character 5-grams**: representing each document as the set of all 5-character substrings it contains, then measuring `|A ∩ B| / |A ∪ B|` between document pairs. Two documents sharing most of their substrings — which near-duplicate text does, almost by definition — score close to 1.0.

## 3. MinHash and Locality-Sensitive Hashing: What Production Systems Use Instead

`implementation.py`'s near-duplicate detector computes exact Jaccard similarity between every pair of documents — an `O(n²)` comparison, explicitly flagged as such in the code's own docstring. This is fine at the corpus size demonstrated here (roughly 900 documents), but does not scale to a real web-crawl corpus of hundreds of millions of documents, where `n²` comparisons become computationally prohibitive. Real large-scale deduplication pipelines (e.g., CCNet, RefinedWeb) instead use **MinHash**: a fixed-size sketch (a small set of minimum hash values under several different hash functions) that *approximates* a document's n-gram set well enough that comparing two documents' MinHash sketches approximates their true Jaccard similarity, without ever materializing or comparing full n-gram sets. Combined with **Locality-Sensitive Hashing (LSH)** — bucketing documents by portions of their MinHash sketch so that only documents landing in the same bucket ever need to be compared directly — this reduces near-duplicate detection from `O(n²)` to approximately linear in corpus size. `implementation.py` computes exact Jaccard directly specifically so the *definition* of near-duplication is fully visible in the code; a production system would swap this exact computation for the MinHash+LSH approximation without changing what "near-duplicate" means, only how cheaply it's detected at scale.

## 4. Heuristic Quality Filtering

Not everything low-quality is a duplicate. `implementation.py`'s `quality_score` function applies three rules, each modeled on published large-corpus filtering heuristics (Raffel et al.'s C4 cleaning rules; Rae et al.'s Gopher paper documents a similar heuristic filter stack):

- **Length filter** — documents under 20 characters are rejected outright; too short to carry meaningful linguistic content.
- **Alpha-ratio filter** — documents where fewer than 60% of characters are alphabetic or whitespace are rejected, catching symbol-heavy junk ("!!!", "***") that passed the length filter.
- **Repetition filter** — documents with fewer than 40% unique words (for documents with at least 4 words) are rejected, catching degenerate repeated-token spam ("N/A N/A N/A N/A N/A") that could otherwise look like ordinary short text.

These are heuristics, not a learned classifier — deliberately so. `theory.md`'s own honesty about this matches the real literature's: heuristic filters are cheap, fast, and interpretable (you can always point to *which* rule rejected a document), at the cost of being cruder than a trained quality classifier would be. Both approaches see real production use, often layered together.

## 5. PII Scrubbing and Its Real Limitation

`implementation.py` redacts synthetic emails and phone numbers via regular expressions, and a real bug was caught and fixed during development: the original phone-number regex only matched a 10-digit, area-code-formatted pattern, silently missing every instance of this script's own second synthetic template (a plain 7-digit local-format number) — see `explanation.md` for the fix. That bug is itself a small, direct illustration of a real, well-documented limitation of regex-based PII detection: **phone numbers, addresses, and names appear in enough different formats that no fixed set of patterns catches all of them**, which is precisely why production PII-scrubbing pipelines increasingly rely on trained named-entity-recognition models rather than regex alone — trading interpretability for better recall on the long tail of formats a hand-written pattern will always eventually miss.

## 6. Tokenizer Training: A Real BPE Tokenizer, Not a Simulation

Every other topic in this repository up to this point has used a fixed character-level vocabulary — a deliberate simplification for architecture and mechanism demonstrations. This topic trains a **real byte-pair-encoding (BPE) tokenizer** (Sennrich et al., 2016) from scratch on the cleaned corpus, using Hugging Face's `tokenizers` library — a local training operation requiring no model download, just the library itself (installed from PyPI, unaffected by this sandbox's Hugging Face Hub restriction). BPE works by iteratively merging the most frequent adjacent symbol pair in the corpus into a new symbol, starting from individual characters and repeating until a target vocabulary size is reached — building up common subwords ("ing", "tion") and even whole common words as single tokens, while staying able to fall back to smaller pieces for rare or unseen words.

`implementation.py` measures the practical payoff directly: encoding held-out text (a fresh slice of the source corpus, never seen during tokenizer training) character-by-character takes one "token" per character, while the trained 800-token BPE vocabulary encodes the same text in roughly a third as many tokens — a **~3x compression ratio**. This matters beyond storage: since attention's computational cost scales quadratically with sequence length (`01-Review-of-Fundamental-LLMs/02-Transformer-Architecture-and-Self-Attention`, `theory.md` §6), a 3x reduction in token count is roughly a 9x reduction in attention compute for the same underlying text.

## 7. Scaling to Production

Every stage demonstrated here operates identically in kind at production scale — only the implementation of each stage changes to handle billions of documents rather than under a thousand: exact hashing scales trivially; near-duplicate detection moves from exact Jaccard to MinHash+LSH (§3); quality filtering heuristics are often supplemented with trained classifiers; PII scrubbing moves from regex to NER-based detection; and BPE tokenizer training runs on samples of the full target corpus (training on the entire multi-trillion-token corpus isn't necessary — BPE merge statistics stabilize on a representative sample). This topic's actual vocabulary size (800) is also a toy-scale choice — production tokenizers typically use 30,000–100,000+ merges, chosen large enough to represent common subwords across many languages and domains without inflating the embedding table excessively.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Sennrich, Rico, Barry Haddow, and Alexandra Birch. "Neural Machine Translation of Rare Words with Subword Units." *ACL*, 2016.
- Raffel, Colin, et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*, 2020.
- Rae, Jack W., et al. "Scaling Language Models: Methods, Analysis & Insights from Training Gopher." 2021.
- Lee, Katherine, et al. "Deduplicating Training Data Makes Language Models Better." *ACL*, 2022.
