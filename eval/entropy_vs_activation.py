"""
Activated Neuron Count vs. Entropy
===================================
Replicates Fig.2 from "Do LLMs Signal When They're Right?"
(arXiv:2510.26277)

For each B-token chunk during generation, measure:
  - Mean entropy of the output distribution
  - Count of activated neurons (SwiGLU contribution + threshold, Eq.1-2,5-6)
Then correlate count with entropy across all chunks.

"""

import os, json, gc
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.utils import get_sparse_model, get_tokenizer

# ── CONFIG ────────────────────────────────────────────────────────────
MODEL_PATH     = "Qwen/Qwen2.5-Math-7B-Instruct"
AIME_PATH      = "data/aime24.jsonl"
OUT            = f"{str(Path(__file__).resolve().parent.parent)}/allocation_results"
MAX_NEW_TOKENS = 8192
N_PROBLEMS     = 30
CHUNK_SIZE     = 64       # B=32 tokens per chunk (paper Eq.3 / Fig.2)
TOP_K_NEURONS  = 500      # k=500 for threshold (paper Appendix B)
TOP_K_PER_LAYER = 32      # 64 per layer in Eq.5
SAVE_DIR       = "activation_entropy"
DEVICE         = "auto"

os.makedirs(SAVE_DIR, exist_ok=True)

# ── LOAD MODEL + DATA ────────────────────────────────────────────────
tokenizer = get_tokenizer(MODEL_PATH)
model = get_sparse_model(
    MODEL_PATH, device=DEVICE,
    histogram_path=os.path.join(OUT, "histograms"),
    sparse_mode="wina", mask_by="topk", transform=False,
    torch_dtype=torch.float16,
)
model.load_greedy_sparsities(os.path.join(OUT, "lookup"), 0.0)
model.eval()

n_layers = len(model.model.layers)
eos_id = tokenizer.eos_token_id

with open(AIME_PATH) as f:
    aime = [json.loads(l) for l in f if l.strip()]
problems = [ex["problem"] for ex in aime][:N_PROBLEMS]
print(f"[INFO] {n_layers} layers, {len(problems)} problems, "
      f"chunk_size={CHUNK_SIZE}")

# ── HOOKS: capture gate & up projections for SwiGLU contribution ─────
# Paper Eq.1: contribution of neuron i in layer l ∝ |SiLU(x @ W_gate)_i * (x @ W_up)_i|
_gate_out = {}   # layer_idx -> tensor (intermediate_size,)
_up_out   = {}

def _gate_hook(li):
    def fn(module, inp, out):
        _gate_out[li] = out[:, -1, :].detach().float()
    return fn

def _up_hook(li):
    def fn(module, inp, out):
        _up_out[li] = out[:, -1, :].detach().float()
    return fn

hooks = []
for li, layer in enumerate(model.model.layers):
    hooks.append(layer.mlp.gate_proj.register_forward_hook(_gate_hook(li)))
    hooks.append(layer.mlp.up_proj.register_forward_hook(_up_hook(li)))


def count_activated_neurons():
    """
    Count activated neurons for the current token using the paper's
    threshold (Appendix B, Eq.5-6):
      1. Per layer: compute |SiLU(gate_i) * up_i| for each neuron i
      2. Keep top-64 values per layer
      3. Concatenate across layers, threshold = min of global top-500
      4. Count neurons exceeding threshold
    
    Returns: (n_activated, per_layer_counts dict)
    """
    # Step 1-2: per-layer contribution scores, keep top-64
    all_topk_vals = []
    layer_contribs = {}
    for li in range(n_layers):
        gate = _gate_out[li].squeeze(0)
        up   = _up_out[li].squeeze(0)
        contrib = (F.silu(gate) * up).abs()           # (intermediate_size,)
        layer_contribs[li] = contrib
        k = min(TOP_K_PER_LAYER, contrib.shape[0])
        topk_vals, _ = torch.topk(contrib, k)
        all_topk_vals.append(topk_vals)

    # Step 3: global threshold
    concatenated = torch.cat(all_topk_vals)
    k_global = min(TOP_K_NEURONS, concatenated.shape[0])
    threshold = torch.topk(concatenated, k_global).values[-1].item()

    # Step 4: count neurons exceeding threshold
    total = 0
    per_layer = {}
    for li in range(n_layers):
        n = (layer_contribs[li] > threshold).sum().item()
        per_layer[li] = n
        total += n

    return total, per_layer


# ── DECODE + COLLECT (per-chunk) ─────────────────────────────────────
# Each record = one chunk of CHUNK_SIZE tokens
chunk_records = []

for qi, question in enumerate(problems):
    msgs = [{"role": "user", "content": question}]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
    input_ids = tokenizer(txt, return_tensors="pt").input_ids.to(model.device)
    past = None

    # Accumulators for current chunk
    chunk_entropies = []
    chunk_activated_set = set()      # union of activated neurons in chunk
    chunk_per_layer_counts = [0] * n_layers

    for step in range(MAX_NEW_TOKENS):
        _gate_out.clear()
        _up_out.clear()

        with torch.no_grad():
            if past is None:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=input_ids[:, -1:],
                            past_key_values=past, use_cache=True)
            past = out.past_key_values

        logits = out.logits[:, -1, :].float()

        # ── Entropy ──
        probs  = F.softmax(logits, dim=-1).squeeze(0)
        entropy = -(probs * torch.log2(probs + 1e-12)).sum().item()
        chunk_entropies.append(entropy)

        # ── Activated neuron count (paper definition) ──
        n_act, per_layer = count_activated_neurons()

        # Union: add (layer, neuron_idx) pairs for this token
        for li in range(n_layers):
            contrib = (F.silu(_gate_out[li].squeeze(0)) *
                       _up_out[li].squeeze(0)).abs()
            # Use the same threshold logic to get indices
            # (reuse the per_layer count; only need indices for union)
            all_topk = []
            for lj in range(n_layers):
                g = _gate_out[lj].squeeze(0)
                u = _up_out[lj].squeeze(0)
                c = (F.silu(g) * u).abs()
                k = min(TOP_K_PER_LAYER, c.shape[0])
                all_topk.append(torch.topk(c, k).values)
            threshold = torch.topk(
                torch.cat(all_topk),
                min(TOP_K_NEURONS, torch.cat(all_topk).shape[0])
            ).values[-1].item()
            break  # only need threshold once

        for li in range(n_layers):
            g = _gate_out[li].squeeze(0)
            u = _up_out[li].squeeze(0)
            c = (F.silu(g) * u).abs()
            idxs = (c > threshold).nonzero(as_tuple=True)[0].tolist()
            for idx in idxs:
                chunk_activated_set.add((li, idx))

        # ── Greedy next token ──
        tok_id = logits.argmax(dim=-1).item()
        input_ids = torch.cat(
            [input_ids, torch.tensor([[tok_id]], device=model.device)], dim=-1)

        # ── End of chunk → save record ──
        if len(chunk_entropies) >= CHUNK_SIZE or tok_id == eos_id:
            chunk_records.append({
                "problem":       qi,
                "chunk_idx":     len([c for c in chunk_records
                                      if c["problem"] == qi]),
                "mean_entropy":  np.mean(chunk_entropies),
                "neuron_count":  len(chunk_activated_set),
                "n_tokens":      len(chunk_entropies),
            })
            chunk_entropies = []
            chunk_activated_set = set()

        if tok_id == eos_id:
            break

    del past; past = None
    del input_ids; input_ids = None
    del out                              # last forward-pass output
    _gate_out.clear()
    _up_out.clear()
    gc.collect()
    torch.cuda.empty_cache()

    if (qi + 1) % 5 == 0 or qi == 0:
        print(f"  [{qi+1}/{len(problems)}]  chunks so far: {len(chunk_records):,}")

for h in hooks:
    h.remove()
print(f"\n[INFO] {len(chunk_records):,} chunks collected.")

# ── ANALYSIS ─────────────────────────────────────────────────────────
counts   = np.array([r["neuron_count"]  for r in chunk_records])
ents     = np.array([r["mean_entropy"]  for r in chunk_records])

r_pearson, p_pearson   = sp_stats.pearsonr(counts, ents)
r_spearman, p_spearman = sp_stats.spearmanr(counts, ents)

print(f"\n{'='*60}")
print(f"Activated Neuron Count  vs  Entropy  (Paper Fig.2)")
print(f"{'='*60}")
print(f"  Chunks:     {len(chunk_records)}")
print(f"  Pearson  r  = {r_pearson:.3f}   (p = {p_pearson:.2e})")
print(f"  Spearman ρ  = {r_spearman:.3f}   (p = {p_spearman:.2e})")
print(f"  Expected: positive (more activated neurons ↔ higher entropy)")
print(f"  Paper reports: Pearson ~0.633, Spearman ~0.660")

# ── PLOTS ────────────────────────────────────────────────────────────

# 1. Scatter: neuron count vs entropy (replicating Paper Fig.2 right panel)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(counts, ents, s=4, alpha=0.15, color="#2166ac", edgecolors="none")
z = np.polyfit(counts, ents, 1)
xs = np.linspace(counts.min(), counts.max(), 200)
ax.plot(xs, np.polyval(z, xs), "k--", lw=1.5,
        label=f"Pearson r = {r_pearson:.3f}\nSpearman ρ = {r_spearman:.3f}")
ax.set_xlabel("Activated Neuron Count")
ax.set_ylabel("Average Entropy")
ax.set_title("Activated Neuron Count vs Entropy\n"
             f"({len(chunk_records)} chunks, {len(problems)} problems)")
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "neuron_count_vs_entropy.png"), dpi=150)
print(f"\n[SAVED] neuron_count_vs_entropy.png")

lo_q, hi_q = np.percentile(ents, [25, 75])

# 3. Per-layer: average neuron count in low-entropy vs high-entropy chunks
lo_mask = ents <= lo_q
hi_mask = ents >= hi_q

# Recompute per-layer counts (need a second pass or store them)
# For simplicity, report the aggregate finding
print(f"\n{'='*60}")
print(f"Low-Entropy vs High-Entropy Chunks")
print(f"{'='*60}")
print(f"  Low-entropy  (≤P25={lo_q:.2f}):  mean neuron count = {counts[lo_mask].mean():.1f}")
print(f"  High-entropy (≥P75={hi_q:.2f}):  mean neuron count = {counts[hi_mask].mean():.1f}")

# ── SAVE ─────────────────────────────────────────────────────────────
summary = {
    "n_problems":  len(problems),
    "n_chunks":    len(chunk_records),
    "chunk_size":  CHUNK_SIZE,
    "top_k_neurons": TOP_K_NEURONS,
    "correlation": {
        "pearson_r":    float(r_pearson),  # type: ignore
        "pearson_p":    float(p_pearson),  # type: ignore
        "spearman_rho": float(r_spearman), # type: ignore
        "spearman_p":   float(p_spearman), # type: ignore
    },
    "quartile_comparison": {
        "low_entropy_mean_count":  float(counts[lo_mask].mean()),
        "high_entropy_mean_count": float(counts[hi_mask].mean()),
    },
}
# with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
#     json.dump(summary, f, indent=2)
# with open(os.path.join(SAVE_DIR, "chunk_records.json"), "w") as f:
#     json.dump(chunk_records, f, indent=2)
print(f"[SAVED] summary.json, chunk_records.json\nDone.")