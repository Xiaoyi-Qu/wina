"""
Token-Entropy Visualizer
Usage: python visualize_token_entropy.py data1.json [data2.json ...]
 
Each JSON file should have:
  - "tokens": list of token strings
  - "token_entropies": list of float entropy values
  - "mean_entropy": float (optional)
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json, sys, os

matplotlib.use("Agg")
 
def clean_token(t):
    return t.replace("\u0120", " ").replace("\u010a", "\\n")
 
def plot_token_entropy(data, title, outfile):
    tokens = [clean_token(t) for t in data["tokens"]]
    token_entropies = data["token_entropies"]
    n = min(len(tokens), len(token_entropies))
    tokens, ents = tokens[:n], token_entropies[:n]
    mean_e = data.get("mean_entropy", np.mean(ents))
    cmap, norm = plt.cm.YlOrRd, mcolors.Normalize(0, max(ents)) # type: ignore
 
    fig, axes = plt.subplots(3, 1, figsize=(20, 12),
        gridspec_kw={"height_ratios": [3, 3, 1.2]})
 
    # Row 0-1: bar charts with token labels (first 100 tokens)
    per_row = n // 3
    for row in range(3):
        s, e = row * per_row, min((row + 1) * per_row, n)
        if s >= n:
            axes[row].axis("off"); continue
        x = range(e - s)
        colors = [cmap(norm(v)) for v in ents[s:e]]
        axes[row].bar(x, ents[s:e], color=colors, width=0.9)
        axes[row].set_xticks(list(x))
        axes[row].set_xticklabels(tokens[s:e], rotation=90,
            fontsize=5.5, fontfamily="monospace")
        axes[row].set_ylabel("Entropy")
        axes[row].set_title(f"Tokens {s}–{e-1}", fontsize=9)
 
    # Row 2: full-sequence heatmap
    axes[2].imshow(np.array(ents).reshape(1, -1),
        aspect="auto", cmap="YlOrRd", interpolation="nearest")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Token index")
    axes[2].set_title("Full-sequence entropy heatmap", fontsize=9)
 
    fig.suptitle(f"{title}  |  mean={mean_e:.4f}  |  {n} tokens", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {outfile}")
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for i, path in enumerate(sys.argv[1:], 1):
        with open(path) as f:
            data = json.load(f)
        name = os.path.splitext(os.path.basename(path))[0]
        breakpoint()
        plot_token_entropy(data, f"Document {i}: {name}", f"{name}_entropy.png")