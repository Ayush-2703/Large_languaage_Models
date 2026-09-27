# Implementing Fine-Tuning (GPT, BERT, T5)

## 1. Three Architectures, Three Different Fine-Tuning Mechanics

"Fine-tune a Transformer for classification" means something structurally different depending on which architecture family you start from, because the three families differ in what information is available at which position, and in which direction attention is allowed to flow. `implementation.py` builds one small representative of each family and fine-tunes each the way its own structure demands:

| Family | Attention direction | Readout position | Fine-tuning method |
|---|---|---|---|
| GPT-style (decoder-only) | Causal (each position sees only what came before it) | Last non-pad token | Classification head on the last token's hidden state |
| BERT-style (encoder-only) | Bidirectional (every position sees the whole sequence) | `[CLS]` token | Classification head on `[CLS]`'s hidden state |
| T5-style (encoder-decoder) | Bidirectional encoder + causal decoder with cross-attention | Decoder's generated token | Text-to-text generation of a label token |

## 2. Why GPT-Style Classification Reads the LAST Token

Under a causal mask, position `t` can only attend to positions `≤ t` — `theory.md` for `01-Review-of-Fundamental-LLMs/02-Transformer-Architecture-and-Self-Attention` derives exactly why. A direct consequence: **only the final token in a sequence has had the opportunity to incorporate information from every other token.** Every earlier position's representation reflects a strict prefix of the input, not the whole thing. `GPTClassifier` in `implementation.py` therefore reads its classification logits from the last non-padding token's hidden state:

```python
lengths = (ids != PAD).sum(dim=1) - 1
last_hidden = h[torch.arange(h.size(0)), lengths]
```

This is precisely how real decoder-only models are adapted for classification in practice (e.g., Hugging Face's `GPT2ForSequenceClassification` pools the last non-padding token internally) — not a simplification specific to this toy implementation.

## 3. Why BERT-Style Classification Reads `[CLS]`

A bidirectional encoder has no such asymmetry — every position already sees the entire sequence, so in principle any position's representation could serve as a summary. BERT-style models instead dedicate a special, always-present `[CLS]` token specifically to this role, established during pretraining (Devlin et al., 2019) so that by the time fine-tuning begins, the model already has *some* notion that this position's output should aggregate whole-sequence information. `01-Fine-Tuning-Principles-and-Techniques` and `02-Transfer-Learning-for-Domain-Specific-Tasks` (this phase's earlier two topics) both build on and depend on this same convention; the latter's `explanation.md` documents a real bug caught during development, where `[CLS]` wasn't actually present during pretraining batches, undermining exactly this mechanism. `implementation.py` here applies that lesson from the start — `[CLS]` is present in every MLM pretraining batch for `MiniBERT`, not only introduced at fine-tuning time.

## 4. Why T5-Style Fine-Tuning Is a Generation Task, Not a Classification Task

T5's defining idea (Raffel et al., 2020) is that *every* task — classification, translation, summarization, regression — can be reframed as text-to-text: given input text, generate output text. A sentiment classifier, in this framing, isn't a model with a 2-unit output layer; it's a model that generates the word "positive" or "negative" (here, the single tokens `[POS]`/`[NEG]`, for character-vocabulary simplicity) one token at a time, exactly like it would generate any other text. `MiniT5`'s fine-tuning loop reflects this directly — there is no dedicated classification head at all:

```python
dec_in = torch.full((BATCH, 1), BOS, dtype=torch.long, device=DEVICE)
target = torch.tensor([POS_TOK if review_labels[i] == 1 else NEG_TOK for i in idx], device=DEVICE)
logits = t5(enc_ids, dec_in, enc_pad)[:, -1, :]
loss = F.cross_entropy(logits, target)
```

The decoder is handed a single `[BOS]` (beginning-of-sequence) token and asked to predict, via ordinary next-token cross-entropy loss — the exact same loss function `MiniGPT`'s pretraining uses — what comes next. That "next token" happens to *be* the classification label. Real T5 fine-tuning for classification works identically, just with full-word targets ("positive"/"negative") and a wordpiece vocabulary instead of single characters.

## 5. Cross-Attention: What Makes an Encoder-Decoder Different From Two Separate Models

`MiniT5`'s decoder does not only attend to its own (extremely short, here just one token) sequence — it also cross-attends to the encoder's full output:

```python
x = x + self.self_attn(self.ln1(x), causal=True)
if self.use_cross_attn:
    x = x + self.cross_attn(self.ln_cross(x), x_kv=encoder_out, causal=False, pad_mask=enc_pad_mask)
```

This is the same `Attention` module used for every other attention computation in this script — Q, K, V, scaled dot-product, softmax, exactly the formula from `02-Transformer-Architecture-and-Self-Attention` — with one change: the **queries** come from the decoder's own hidden state, while the **keys and values** come from the encoder's output. This is what lets the decoder generate output conditioned on the entire input sequence's representation at every decoding step, not just on what it has generated so far.

## 6. Reusing a Pretrained Encoder Inside a Fresh Encoder-Decoder Model

`MiniT5` does not pretrain its own encoder from scratch. It is constructed directly from the already-pretrained `MiniBERT`:

```python
t5 = MiniT5(pretrained_encoder=bert)   # encoder = the SAME pretrained MiniBERT above
```

only the decoder (and its cross-attention layers connecting to that encoder) start from random initialization, learning entirely during fine-tuning. This is not a shortcut invented for this repository — it is a small-scale version of a real, published, citable technique: Rothe, Narayan, and Severyn's "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks" (2020) shows that initializing an encoder-decoder model's encoder (and optionally decoder) from a pretrained encoder-only checkpoint like BERT is a practical, effective way to bootstrap sequence-to-sequence models without paying for a full, separate encoder-decoder pretraining run. Here, it additionally keeps the comparison in `implementation.py` scientifically clean: `MiniT5`'s encoder is *exactly* the same trained object `MiniBERT`'s classifier uses (not a separately-trained encoder that happens to be similar), so any accuracy difference between BERT-style and T5-style fine-tuning in the measured results below reflects the readout/fine-tuning *method*, not two different encoders that happened to pretrain differently.

## 7. What Was Actually Measured

All three architectures, fine-tuned on the identical downstream sentiment task with the identical step budget:

| Architecture | Parameters | Final Val. Accuracy |
|---|---|---|
| MiniGPT (last-token) | 65,157 | 1.000 |
| MiniBERT (`[CLS]`) | 65,157 | 0.990 |
| MiniT5 (text-to-text) | 147,398 | 1.000 |

All three converge to strong, comparable accuracy on this task, confirming that the three architecturally-distinct fine-tuning mechanics described in §2–§4 all genuinely work as intended — this is the primary claim this topic's code sets out to demonstrate, and it held cleanly without needing any post-hoc correction (unlike the previous topic's DAPT comparison). MiniT5's higher parameter count is structural, not incidental: it carries both the full reused encoder and an entirely separate decoder stack with cross-attention, which is an inherent property of the encoder-decoder family relative to encoder-only or decoder-only models of comparable per-layer width — the same trade-off real production T5 models make relative to same-generation BERT or GPT-family models.

## 8. Scaling to Production

Real `gpt2`, `bert-base-uncased`, and `t5-small` checkpoints implement exactly these same three fine-tuning mechanics — last-token pooling, `[CLS]` pooling, and text-to-text generation, respectively — at far greater scale, with subword tokenization in place of this demo's character-level vocabulary. Section 6 of `implementation.py`, gated behind a Colab-only flag, sketches the same three fine-tuning patterns applied to those real checkpoints on real SST-2 data, so the mapping from this topic's toy implementation to production usage is explicit in code, not just asserted here.

## References

- Rothman, Denis. *Transformers for Natural Language Processing.*
- Tunstall, Lewis, Leandro von Werra, and Thomas Wolf. *Natural Language Processing with Transformers.*
- Radford, Alec, et al. "Improving Language Understanding by Generative Pre-Training." OpenAI, 2018.
- Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019.
- Raffel, Colin, et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*, 2020.
- Rothe, Sascha, Shashi Narayan, and Aliaksei Severyn. "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks." *TACL*, 2020.
- Vaswani, Ashish, et al. "Attention Is All You Need." *NeurIPS*, 2017.
