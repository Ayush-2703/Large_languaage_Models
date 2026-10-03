# Explanation: `implementation.py`

## Why the Memory-Measurement Code Lives in a String, Not a Normal Function

```python
MEMORY_TEST_SCRIPT = textwrap.dedent("""
    import torch, torch.nn as nn, ...
    ...
""")

result = subprocess.run([sys.executable, "-c", script, "1" if use_checkpoint else "0"], ...)
```

**What:** The model definition and training loop for the memory experiment are written as a *string* containing a complete, self-sufficient Python script, then executed via `subprocess.run([sys.executable, "-c", script, ...])` — a fresh `python3 -c "..."` invocation, not a function call within the current process.

**Why this had to be a subprocess rather than an ordinary Python function:** `theory.md` §3 explains the underlying reason directly — `ru_maxrss` only ever increases within a process, so calling "run without checkpointing" and "run with checkpointing" as two ordinary function calls in the same script would mean the second call's reported peak reflects the *cumulative* high-water mark across both calls, not that call's own true peak. A subprocess is a genuinely fresh process with its own memory space and its own independent `ru_maxrss` counter starting from zero — the only way to get two numbers that are actually comparable.

**Why pass `use_checkpoint` as a command-line argument (`sys.argv[1]`) rather than an environment variable or a second script file:** Command-line arguments are the simplest way to parameterize a `-c` inline script without needing to write temporary files to disk or manage environment variable cleanup — `sys.argv[1] == "1"` inside the child script is a direct, unambiguous read of exactly one piece of configuration passed explicitly at the moment the subprocess is launched.

## Parsing the Subprocess's Own Report

```python
print(f"RESULT peak_kb={{peak_kb}} elapsed={{elapsed:.3f}} params={{sum(...)}}")
...
line = [l for l in result.stdout.splitlines() if l.startswith("RESULT")]
parts = dict(kv.split("=") for kv in line[0].replace("RESULT ", "").split())
```

**What:** The child subprocess prints one line, prefixed with the literal string `"RESULT"`, containing space-separated `key=value` pairs. The parent process filters `stdout` for that prefix, then splits the matched line into a dictionary.

**Why not just capture and parse the entire stdout, or use a structured format like JSON:** PyTorch and Python can print warnings or other incidental output to stdout that has nothing to do with the actual measurement (import-time messages, deprecation notices). Prefixing the one line that matters with a distinctive, greppable marker (`"RESULT"`) and filtering for it specifically makes parsing robust to whatever else might appear in the subprocess's output, without needing a heavier structured-serialization dependency for what is, in the end, three numbers.

## Why the Deep Transformer Uses 24 Layers, Not the 2–3 Layers Used Elsewhere

```python
D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, BATCH = 128, 8, {n_layers}, 256, 16
...
N_LAYERS_DEEP = 24
```

**Why so much deeper than every other model in this repository:** Gradient checkpointing's memory savings scale with *how many layers' worth of activations would otherwise need to be held simultaneously* — a 2-layer model, like most architecture-demonstration models elsewhere in this repository, simply doesn't hold enough simultaneous activation memory for checkpointing's effect to be clearly visible above normal measurement noise and fixed process overhead. 24 layers was chosen specifically to make the memory trade-off measurable and clear (a real 35% reduction, not a marginal few percent that could be mistaken for noise), at the cost of this being the one topic in this repository whose demonstration model isn't representative of this repository's usual "as small as the concept allows" sizing philosophy — a deliberate, stated exception rather than an unexplained inconsistency.

## Attention Memory Test: Isolating Just the Attention Computation

```python
q = torch.randn(BATCH, N_HEADS, SEQ_LEN, hd, requires_grad=True)
k = torch.randn(BATCH, N_HEADS, SEQ_LEN, hd, requires_grad=True)
v = torch.randn(BATCH, N_HEADS, SEQ_LEN, hd, requires_grad=True)
```

**Why construct raw Q/K/V tensors directly rather than running a full embedding-plus-attention model like the gradient-checkpointing test does:** This experiment's question is specifically "how does attention's own memory/time cost scale with sequence length," not "how does a full model's cost scale." Isolating Q, K, and V as standalone tensors — with no embedding layer, no feed-forward block, no surrounding Transformer structure — means every measured difference between the "naive" and "sdpa" conditions is attributable *only* to the attention computation itself, with no risk of an unrelated part of a larger model's cost diluting or obscuring the comparison `theory.md` §5's table is built to make.

```python
out.sum().backward()
```

**Why call `.backward()` on `out.sum()` rather than just running the forward pass:** A forward-only comparison would understate real training cost — in practice, attention's `T × T` score matrix (in the naive case) must be kept around for the backward pass's gradient computation, which is precisely the memory cost this experiment is trying to measure. `.sum()` is an arbitrary scalar reduction that exists purely to give `.backward()` something valid to differentiate; the actual gradient values it produces are not used or reported anywhere, only their side effect of exercising the same backward-pass memory pattern real training would.

## `proof.png`'s Four-Panel Split

**Why gradient checkpointing gets two full panels (memory, time) while attention gets two full panels (memory-vs-length, time-vs-length) rather than combining everything into fewer, denser charts:** These are genuinely four different measurements answering four different questions — "how much memory does checkpointing save" and "how much time does it cost" are a fixed before/after comparison (well suited to bar charts), while "how does naive vs. SDPA attention's memory/time scale *with sequence length*" is a trend across a swept variable (well suited to line charts). Using the chart type that fits each question, rather than forcing a uniform layout, is the same principle applied in `03-Training-and-Optimization-of-LLMs/02-Efficient-Training-LoRA-QLoRA-Quantization-Pruning`'s four-panel comparison, where LoRA's parameter count, quantization's size, and pruning's sparsity curve were kept as three distinctly-shaped charts rather than compressed into one.
