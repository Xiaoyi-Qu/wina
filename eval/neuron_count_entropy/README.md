# Activated Neuron Count Computation

This document describes how the number of activated neurons is computed in the replication of **Fig. 2** from *"Do LLMs Signal When They're Right?"* (arXiv: 2510.26277).

The goal is to measure, for each chunk of `B` tokens during generation, how many MLP neurons are meaningfully contributing to the model's output — and then correlate that count with the entropy of the output distribution.

---

## Step 1 — Capture Raw Projections via Hooks

Forward hooks are registered on each transformer layer's `gate_proj` and `up_proj` linear layers. During each forward pass, these hooks capture the **last-token output** of both projections.

In a SwiGLU-based MLP (used in Qwen, LLaMA, etc.), the feedforward block computes:

```
MLP(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
```

The hooks intercept the intermediate values `x @ W_gate` and `x @ W_up` before the element-wise product and down-projection.

---

## Step 2 — Compute Per-Neuron Contribution Scores

For each layer, the **contribution score** of neuron `i` is defined as the absolute value of its SwiGLU product:

```
contrib_i = |SiLU(gate_i) × up_i|
```

This follows the paper's **Eq. 1–2**: a neuron's "activation" is not simply a binary on/off, but reflects how much it contributes to the MLP output after the gating mechanism. A neuron with a large contribution score has a strong influence on the residual stream update at that layer.

---

## Step 3 — Build a Global Threshold via Two-Level Top-K

A dynamic, per-token threshold is constructed using a two-level top-k selection:

1. **Per-layer filtering:** Within each layer, retain only the top `TOP_K_PER_LAYER` contribution values (64 per layer in the paper, 32 in this config).

2. **Cross-layer concatenation:** Concatenate those per-layer top-k values across all layers. For a 28-layer model with top-64 per layer, this yields ~1,792 candidate values.

3. **Global top-k:** From the concatenated candidates, select the top `TOP_K_NEURONS` values (500 in both the paper and this config).

4. **Threshold extraction:** The threshold is the **smallest value** in the global top-500 set.

This threshold adapts per token. When the model is "confident" (a few neurons dominate), the threshold is high. When activation is more diffuse, the threshold is lower.

---

## Step 4 — Count Neurons Above the Threshold

With the threshold determined, iterate back through every layer and count how many neurons satisfy:

```
contrib_i > threshold
```

Sum across all layers to produce the **total activated neuron count** for that token. Note that this count can exceed `TOP_K_NEURONS` (500) because the threshold was derived from a filtered candidate set (top-k per layer), while the final count checks *all* neurons against that threshold.

---

## Step 5 — Aggregate Per Chunk

Over a chunk of `CHUNK_SIZE` tokens (default: 32), the code tracks the **union** of activated `(layer, neuron_index)` pairs across all tokens in the chunk. When the chunk boundary is reached (or EOS is generated), the recorded neuron count is:

```
neuron_count = |activated_set|
```

where `activated_set` is the set of unique `(layer, neuron_index)` pairs that were activated in **at least one token** within the chunk. This union-based counting captures the total breadth of neural pathways the model engages over a window of reasoning.

---

## Configuration Summary

| Parameter          | Value | Description                                      |
|--------------------|-------|--------------------------------------------------|
| `CHUNK_SIZE`       | 32    | Number of tokens per chunk (paper: `B = 32`)     |
| `TOP_K_NEURONS`    | 500   | Global top-k for threshold (paper Appendix B)    |
| `TOP_K_PER_LAYER`  | 32    | Per-layer top-k candidates (paper: 64)           |
| `MAX_NEW_TOKENS`   | 256   | Maximum generated tokens per problem             |

---

## Core Intuition

A **high activated neuron count** means the model is spreading its computation across many neurons rather than relying on a small set of dominant ones. The paper's hypothesis — validated by this script — is that this diffuse activation pattern **correlates positively with output entropy**: when the model is uncertain about what to predict next, more neurons participate in the computation.

The paper reports correlations of approximately **Pearson r ≈ 0.633** and **Spearman ρ ≈ 0.660** for this relationship.
