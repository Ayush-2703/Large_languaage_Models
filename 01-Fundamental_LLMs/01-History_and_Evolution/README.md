# History and Evolution of Language Models

## 1. What a Language Model Actually Estimates

A language model assigns a probability to a sequence of tokens. Almost every architecture below is doing the same underlying job — factorizing that joint probability autoregressively:

```
P(x_1, x_2, ..., x_T) = Π_{t=1}^{T} P(x_t | x_1, ..., x_{t-1})
```

What changes across eras is **how P(x_t | context) is computed**: by counting, by a recurrent hidden state, or by attention over every prior token at once. The rest of this document — and the code in `implementation.py` — treats "language modeling progress" as a sequence of answers to that one question.

The standard way to compare models on this task is **perplexity**, the exponentiated average negative log-likelihood per token:

```
PPL = exp( -(1/N) Σ log P(x_t | context) )
```

Lower perplexity means the model was, on average, less "surprised" by the real next token. A perplexity of 1 is a perfect model; a perplexity equal to the vocabulary size is a model no better than uniform random guessing. This is the metric `implementation.py` measures for all four architectures on identical held-out text.

## 2. Era 1 — Count-Based N-gram Models (pre-2000s)

The earliest practical language models made a Markov assumption: the next token depends only on the previous *n-1* tokens, not the entire history. An order-3 (trigram) model estimates:

```
P(x_t | x_{t-2}, x_{t-1}) ≈ count(x_{t-2}, x_{t-1}, x_t) / count(x_{t-2}, x_{t-1})
```

Raw counting fails the instant a context has never been seen, so practical systems apply smoothing. `implementation.py` uses the simplest form, **add-1 (Laplace) smoothing**, which redistributes a small amount of probability mass to unseen continuations:

```
P(x_t | ctx) = (count(ctx, x_t) + 1) / (count(ctx) + |V|)
```

More sophisticated schemes (Kneser-Ney, Katz back-off) dominated production NLP — spell checkers, early machine translation, speech recognition — for roughly two decades. Their fundamental limitation is **structural, not just empirical**: a trigram model has no mechanism to represent a dependency spanning more than 2 tokens back, no matter how much data it sees. The paths taken by every subsequent architecture on this page can be read as different answers to "how do we let context length grow past the Markov horizon."

## 3. Era 2 — Neural Language Models and Recurrence

Bengio et al.'s 2003 neural probabilistic language model was an early departure: instead of counting discrete contexts, learn a dense embedding for each word and predict the next word with a small feed-forward network over those embeddings. This is the direct ancestor of every embedding layer in every model trained today, including `TransformerLM` in `implementation.py`.

The next structural shift was **recurrence**. Elman's 1990 simple recurrent network (SRN) — the architecture `RNNLM` implements — maintains a hidden state `h_t` updated at every step:

```
h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
P(x_{t+1} | x_1..x_t) = softmax(W_hy h_t)
```

Because `h_t` is a function of `h_{t-1}`, which is a function of `h_{t-2}`, and so on, an RNN can in principle condition on arbitrarily long history — the Markov ceiling is gone. In practice it is not: repeated multiplication through the recurrence during backpropagation causes gradients to vanish (or occasionally explode) over long sequences, so plain RNNs struggle to actually use context more than a handful of steps back.

## 4. Era 3 — LSTM (1997, dominant 2014–2017)

Hochreiter & Schmidhuber's Long Short-Term Memory network addresses the vanishing-gradient problem with a **gating mechanism** and a separate cell state `c_t` that information can flow through largely unchanged:

```
f_t = σ(W_f · [h_{t-1}, x_t])        # forget gate
i_t = σ(W_i · [h_{t-1}, x_t])        # input gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c · [h_{t-1}, x_t])
o_t = σ(W_o · [h_{t-1}, x_t])        # output gate
h_t = o_t ⊙ tanh(c_t)
```

The forget gate lets the network learn *what to keep*, decoupling "carry information forward" from "the same nonlinearity gets reapplied every step." This is why LSTMs (and the closely related GRU) became the dominant sequence architecture for over a decade, powering Sutskever et al.'s 2014 sequence-to-sequence framework for machine translation and, with Bahdanau et al.'s 2015 addition of an attention mechanism over encoder states, the best translation systems of the mid-2010s.

That Bahdanau attention step matters historically: it was the first widely-used mechanism letting a decoder look directly at *any* encoder position rather than relying solely on a compressed final hidden state. It is the conceptual predecessor to the mechanism in the next section — the paper that removed the recurrence entirely and kept only the attention.

## 5. Era 4 — The Transformer (2017–present)

Vaswani et al.'s 2017 paper introducing the Transformer made a deliberately provocative claim in its title: attention is all you need — no recurrence, no convolution. `implementation.py`'s `CausalSelfAttention` module implements exactly the mechanism this claim rests on:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Every position attends to every other position (masked to only *prior* positions, for a decoder) in a single matrix multiplication, rather than through T sequential recurrent steps. Two consequences follow directly from removing recurrence:

- **Parallelism.** Training no longer requires waiting for `h_{t-1}` before computing `h_t`; every position in a sequence can be processed simultaneously on a GPU. This is arguably the single biggest reason Transformers scaled to the sizes they did — LSTMs are not parallelizable across the time dimension in the same way.
- **Constant path length.** In an RNN/LSTM, information from token 1 must pass through T recurrent steps to influence token T. In self-attention, that path length is 1 — any two positions are directly connected. This is a large part of why Transformers handle long-range dependencies more reliably.

The full QKV derivation, multi-head splitting, and a from-scratch implementation verified against PyTorch's built-in kernel are the subject of the next topic (`02-Transformer-Architecture-and-Self-Attention`); this document only needs the mechanism's shape to explain the historical shift.

## 6. From Architecture to "Large" Language Model

The step from "Transformer" to "LLM" is largely a story about what happens *before* any labeled task: **pretraining** a Transformer decoder on a next-token-prediction objective over a massive unlabeled text corpus, at a scale (parameters, data, compute) that earlier architectures could not efficiently absorb. Radford et al.'s GPT line and Devlin et al.'s BERT (an encoder pretrained with masked-token prediction rather than next-token prediction) established this pretrain-then-adapt paradigm as the default; it is the direct subject of `03-Pretraining-vs-Fine-Tuning-Paradigms` later in this phase. As Rothman discusses in *Transformers for Natural Language Processing*, and as Tunstall, von Werra, and Wolf frame throughout *Natural Language Processing with Transformers*, this shift from training a model per task to adapting one pretrained backbone to many tasks is the defining practical change the Transformer architecture enabled — not just an accuracy improvement on any one benchmark.

## 7. What the Code in This Topic Actually Measures — and Why the Result Is Close

`implementation.py` trains one N-gram, one RNN, one LSTM, and one small decoder-only Transformer from identical random initialization on the same 200K-character slice of the tiny-Shakespeare corpus, then compares validation perplexity. The N-gram and RNN results land exactly where the history above predicts: the trigram model is worst by a wide margin (it structurally cannot use more than 2 characters of context), and the RNN improves markedly by conditioning on a real, if imperfect, memory of the full sequence.

The LSTM and Transformer, however, land close together, with the LSTM slightly ahead at this scale. This is not a bug in the demo — it is a faithful, well-documented property of these architectures, not an artifact to explain away: **the Transformer's advantage is not fixed, it is emergent with scale.** Vaswani et al.'s original comparisons already used far more data and compute than a single-CPU, sub-minute training run can provide here; without that scale, the Transformer has no recurrent inductive bias to lean on and needs more examples to learn what an LSTM's architecture gets closer to "for free." This is precisely the subject of `04-Scaling-Laws-and-Model-Efficiency`, where the same architecture family is trained at multiple sizes to show how the parameter-vs-loss relationship — not architecture alone — explains why today's largest models are Transformers almost without exception.

## 8. Scaling to Production

Every model in `implementation.py` is character-level, single-digit-millions of characters of training data, and trained for under a minute on one CPU core. Production LLMs differ by many orders of magnitude, not by architecture family:

| | This demo | Production scale (e.g. GPT-3 class) |
|---|---|---|
| Tokenization | Character-level | Subword (BPE / WordPiece / SentencePiece), ~50K vocab |
| Parameters | ~10²–10⁵ | ~10¹⁰–10¹¹ |
| Training tokens | ~2×10⁵ | ~10¹¹–10¹² |
| Context length | 48 characters | 2K–128K+ tokens |
| Hardware | 1 CPU core | Thousands of accelerators, weeks |

The self-attention formula, the LSTM gating equations, and the N-gram counting rule shown above are **identical in form** at both scales — nothing here is a toy approximation of the real mechanism. What changes is exclusively the data volume, parameter count, and compute budget, which is exactly the relationship Kaplan et al.'s and Hoffmann et al.'s scaling-law work quantifies, covered next in this phase.

## 9. Ethical Note (Forward Reference)

Every one of these models, including the from-scratch ones trained here, reproduces statistical patterns present in its training text without judgment about whether those patterns are desirable. `05-Ethical-Considerations-in-Large-Scale-AI`, the final topic in this phase, returns to this same "counting equals learning" mechanism — this time deliberately, to show how demographic bias enters a model exactly the way any other statistical regularity does.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Bengio, Yoshua, et al. "A Neural Probabilistic Language Model." *Journal of Machine Learning Research*, 2003.
- Elman, Jeffrey L. "Finding Structure in Time." *Cognitive Science*, 1990.
- Hochreiter, Sepp, and Jürgen Schmidhuber. "Long Short-Term Memory." *Neural Computation*, 1997.
- Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to Sequence Learning with Neural Networks." *NeurIPS*, 2014.
- Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR*, 2015.
- Vaswani, Ashish, et al. "Attention Is All You Need." *NeurIPS*, 2017.
- Radford, Alec, et al. "Improving Language Understanding by Generative Pre-Training." OpenAI, 2018.
- Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019.
