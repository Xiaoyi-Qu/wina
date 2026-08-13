#!/usr/bin/env python3
"""
Postprocess token-entropy JSON to identify high-entropy minority tokens.

Based on: "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive
Effective Reinforcement Learning for LLM Reasoning"
(https://arxiv.org/pdf/2506.01939)

Reads a JSON file containing:
  { output, task_id, token_entropies, tokens, mean_entropy, token_count }

Produces a text report with:
  - Per-token entropy listing (with high-entropy flags)
  - Entropy distribution statistics
  - High-entropy minority token identification (configurable threshold)
  - Token-level breakdown showing "decision points"

Usage:
    python analyze_token_entropies.py input.json -o report.txt
    python analyze_token_entropies.py input.json --threshold 0.5 --top 30
    python analyze_token_entropies.py input.json --format json -o analysis.json
"""

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


# ── display helpers ──────────────────────────────────────────────────────────

def decode_bpe_token(tok: str) -> str:
    """Convert raw BPE token to readable form (Ġ→space, Ċ→newline)."""
    return tok.replace("\u0120", " ").replace("\u010a", "\n")


def readable_token(tok: str, max_len: int = 20) -> str:
    """Return a printable, truncated version of a token."""
    r = decode_bpe_token(tok)
    r = r.replace("\n", "\\n").replace("\t", "\\t")
    if len(r) > max_len:
        r = r[: max_len - 1] + "…"
    return r


# ── core analysis ────────────────────────────────────────────────────────────

def compute_percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0–100) of a sorted list."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def analyze(record: dict, threshold: float | None, top_n: int) -> dict:
    """
    Analyze a single record's token entropies.

    Parameters
    ----------
    record : dict with tokens, token_entropies, etc.
    threshold : explicit entropy threshold for "high-entropy".
                If None, uses mean + 1 std.
    top_n : how many top-entropy tokens to list.

    Returns a dict with all computed stats.
    """
    tokens = record["tokens"]
    entropies = record["token_entropies"]
    n = len(entropies)

    mean_e = statistics.mean(entropies)
    median_e = statistics.median(entropies)
    std_e = statistics.stdev(entropies) if n > 1 else 0.0
    min_e = min(entropies)
    max_e = max(entropies)

    p25 = compute_percentile(entropies, 25)
    p75 = compute_percentile(entropies, 75)
    p90 = compute_percentile(entropies, 90)
    p95 = compute_percentile(entropies, 95)
    p99 = compute_percentile(entropies, 99)

    # Determine threshold
    if threshold is None:
        thresh = mean_e + std_e
    else:
        thresh = threshold

    # Identify high-entropy minority tokens
    indexed = list(enumerate(zip(tokens, entropies)))
    high_entropy = [(i, tok, ent) for i, (tok, ent) in indexed if ent >= thresh]
    low_entropy = [(i, tok, ent) for i, (tok, ent) in indexed if ent < thresh]

    # Top-N by entropy
    top_tokens = sorted(indexed, key=lambda x: x[1][1], reverse=True)[:top_n]

    # Entropy contribution: what fraction of total entropy comes from the
    # high-entropy minority?
    total_entropy = sum(entropies)
    high_entropy_sum = sum(ent for _, _, ent in high_entropy)
    high_frac_count = len(high_entropy) / n if n else 0
    high_frac_entropy = high_entropy_sum / total_entropy if total_entropy else 0

    return {
        "task_id": record.get("task_id", 0),
        "token_count": n,
        "mean": mean_e,
        "median": median_e,
        "std": std_e,
        "min": min_e,
        "max": max_e,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "threshold": thresh,
        "high_count": len(high_entropy),
        "low_count": len(low_entropy),
        "high_frac_count": high_frac_count,
        "high_frac_entropy": high_frac_entropy,
        "total_entropy": total_entropy,
        "high_entropy_tokens": high_entropy,
        "top_tokens": top_tokens,
        "tokens": tokens,
        "entropies": entropies,
    }


# ── text report ──────────────────────────────────────────────────────────────

def format_report(a: dict, show_all_tokens: bool = False) -> str:
    lines = []
    w = lines.append

    w("=" * 78)
    w("  HIGH-ENTROPY MINORITY TOKEN ANALYSIS")
    w("  Based on: arxiv.org/abs/2506.01939")
    w("=" * 78)
    w("")
    w(f"  Task ID           : {a['task_id']}")
    w(f"  Total tokens      : {a['token_count']}")
    w(f"  Total entropy     : {a['total_entropy']:.4f}")
    w("")

    # ── Distribution stats ───────────────────────────────────────────────
    w("─── Entropy Distribution ───────────────────────────────────────────")
    w(f"  Mean              : {a['mean']:.6f}")
    w(f"  Median            : {a['median']:.6f}")
    w(f"  Std Dev           : {a['std']:.6f}")
    w(f"  Min               : {a['min']:.6f}")
    w(f"  Max               : {a['max']:.6f}")
    w(f"  P25               : {a['p25']:.6f}")
    w(f"  P75               : {a['p75']:.6f}")
    w(f"  P90               : {a['p90']:.6f}")
    w(f"  P95               : {a['p95']:.6f}")
    w(f"  P99               : {a['p99']:.6f}")
    w("")

    # ── Minority split ───────────────────────────────────────────────────
    w("─── High-Entropy Minority Tokens ────────────────────────────────────")
    w(f"  Threshold         : {a['threshold']:.6f}")
    w(f"  High-entropy      : {a['high_count']} / {a['token_count']}  "
      f"({a['high_frac_count']*100:.1f}% of tokens)")
    w(f"  Entropy share     : {a['high_frac_entropy']*100:.1f}% of total entropy")
    w(f"  Low-entropy       : {a['low_count']} / {a['token_count']}  "
      f"({(1-a['high_frac_count'])*100:.1f}% of tokens)")

    ratio_str = (
        f"{a['high_frac_count']*100:.0f}/{(1-a['high_frac_count'])*100:.0f}"
    )
    w(f"  Token/Entropy     : {ratio_str} token split carries "
      f"{a['high_frac_entropy']*100:.0f}% of entropy")
    w("")

    # ── Top tokens ───────────────────────────────────────────────────────
    w(f"─── Top {len(a['top_tokens'])} Tokens by Entropy (Decision Points) "
      "────────────────────")
    w(f"  {'Rank':<6}{'Pos':<8}{'Entropy':<14}{'Token':<25}{'Context'}")
    w(f"  {'─'*6}{'─'*8}{'─'*14}{'─'*25}{'─'*20}")
    for rank, (i, (tok, ent)) in enumerate(a["top_tokens"], 1):
        # Build a small context window
        toks = a["tokens"]
        start = max(0, i - 2)
        end = min(len(toks), i + 3)
        ctx_parts = []
        for j in range(start, end):
            t = readable_token(toks[j], 12)
            if j == i:
                t = f"[{t}]"
            ctx_parts.append(t)
        ctx = "".join(ctx_parts)
        w(f"  {rank:<6}{i:<8}{ent:<14.6f}{readable_token(tok):<25}{ctx}")
    w("")

    # ── Entropy histogram (text-based) ───────────────────────────────────
    w("─── Entropy Histogram ──────────────────────────────────────────────")
    entropies = a["entropies"]
    # Create buckets
    n_buckets = 20
    bucket_max = min(a["max"], a["p99"] * 1.5)  # cap outliers
    bucket_size = bucket_max / n_buckets if bucket_max > 0 else 1
    buckets = [0] * (n_buckets + 1)  # last catches overflow
    for e in entropies:
        idx = int(e / bucket_size) if bucket_size > 0 else 0
        idx = min(idx, n_buckets)
        buckets[idx] += 1
    max_count = max(buckets) if buckets else 1
    bar_width = 40

    for i_b in range(n_buckets + 1):
        lo = i_b * bucket_size
        if i_b < n_buckets:
            hi = (i_b + 1) * bucket_size
            label = f"  [{lo:6.3f}, {hi:6.3f})"
        else:
            label = f"  [{lo:6.3f},    ∞   )"
        cnt = buckets[i_b]
        bar_len = int(cnt / max_count * bar_width) if max_count else 0
        bar = "█" * bar_len
        marker = " ◄ threshold" if lo <= a["threshold"] < lo + bucket_size else ""
        if cnt > 0 or i_b <= n_buckets:
            w(f"{label} | {bar:<{bar_width}} {cnt:>5}{marker}")
    w("")

    # ── Full token listing (optional) ────────────────────────────────────
    if show_all_tokens:
        w("─── Full Token Listing ─────────────────────────────────────────────")
        w(f"  {'Pos':<7}{'Entropy':<14}{'HE?':<5}{'Token'}")
        w(f"  {'─'*7}{'─'*14}{'─'*5}{'─'*30}")
        thresh = a["threshold"]
        for i, (tok, ent) in enumerate(zip(a["tokens"], a["entropies"])):
            flag = " ★" if ent >= thresh else "  "
            w(f"  {i:<7}{ent:<14.6f}{flag:<5}{readable_token(tok, 50)}")
        w("")

    w("=" * 78)
    return "\n".join(lines)


# ── JSON output ──────────────────────────────────────────────────────────────

def format_json_output(a: dict) -> dict:
    """Return a JSON-serializable summary (no large arrays unless requested)."""
    high_tokens_detail = [
        {
            "position": i,
            "token": decode_bpe_token(tok),
            "raw_token": tok,
            "entropy": ent,
        }
        for i, tok, ent in a["high_entropy_tokens"]
    ]
    top_detail = [
        {
            "rank": rank,
            "position": i,
            "token": decode_bpe_token(a["tokens"][i]),
            "raw_token": a["tokens"][i],
            "entropy": a["entropies"][i],
        }
        for rank, (i, (tok, ent)) in enumerate(a["top_tokens"], 1)
    ]
    return {
        "task_id": a["task_id"],
        "token_count": a["token_count"],
        "total_entropy": a["total_entropy"],
        "distribution": {
            "mean": a["mean"],
            "median": a["median"],
            "std": a["std"],
            "min": a["min"],
            "max": a["max"],
            "p25": a["p25"],
            "p75": a["p75"],
            "p90": a["p90"],
            "p95": a["p95"],
            "p99": a["p99"],
        },
        "threshold": a["threshold"],
        "high_entropy_count": a["high_count"],
        "high_entropy_fraction": a["high_frac_count"],
        "high_entropy_share": a["high_frac_entropy"],
        "top_tokens": top_detail,
        "high_entropy_tokens": high_tokens_detail,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze token entropies to identify high-entropy minority tokens."
    )
    parser.add_argument("input", help="JSON file with token entropy data")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Entropy threshold for 'high-entropy' (default: mean + 1 std)",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top-entropy tokens to list (default: 20)",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Include full per-token listing in the report",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────
    with open(args.input, "r") as f:
        data = json.load(f)

    # Handle single record or list
    if isinstance(data, dict):
        records = [data]
    else:
        records = data

    # ── Process ──────────────────────────────────────────────────────────
    outputs = []
    for rec in records:
        a = analyze(rec, args.threshold, args.top)
        if args.format == "text":
            outputs.append(format_report(a, show_all_tokens=args.show_all))
        else:
            outputs.append(format_json_output(a))

    # ── Write ────────────────────────────────────────────────────────────
    if args.format == "text":
        result = "\n\n".join(outputs)
    else:
        result = json.dumps(
            outputs if len(outputs) > 1 else outputs[0],
            indent=2, ensure_ascii=False,
        )

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
            f.write("\n")
        print(f"Written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()