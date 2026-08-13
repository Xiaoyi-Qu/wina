#!/usr/bin/env python3
"""
Render token-entropy JSON as a color-coded HTML visualization.

Reads a JSON file with { tokens, token_entropies, ... } and produces
an HTML page where every token is colored on a gradient from cool (low
entropy / confident) to hot (high entropy / uncertain).

Usage:
    python visualize_token_entropies.py input.json -o visualization.html
    python visualize_token_entropies.py input.json --cmap blue-red --log
    python visualize_token_entropies.py input.json --percentile-cap 95
"""

import argparse
import colorsys
import html
import json
import math
import statistics
import sys
from pathlib import Path


# ── Color maps ───────────────────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    return tuple(lerp(a, b, t) for a, b in zip(c1, c2))


def multi_stop_lerp(stops: list[tuple[float, tuple]], t: float) -> tuple:
    """Interpolate through multiple color stops. Each stop is (position, (r,g,b))."""
    if t <= stops[0][0]:
        return stops[0][1]
    if t >= stops[-1][0]:
        return stops[-1][1]
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        if p0 <= t <= p1:
            local_t = (t - p0) / (p1 - p0) if p1 != p0 else 0
            return lerp_color(c0, c1, local_t)
    return stops[-1][1]


COLORMAPS = {
    "blue-red": [
        (0.0, (59, 130, 246)),    # blue-500
        (0.35, (147, 197, 253)),   # light blue
        (0.5, (250, 250, 250)),    # near white
        (0.65, (252, 165, 165)),   # light red
        (1.0, (220, 38, 38)),     # red-600
    ],
    "viridis": [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ],
    "cool-warm": [
        (0.0, (44, 123, 182)),
        (0.25, (171, 217, 233)),
        (0.5, (255, 255, 191)),
        (0.75, (253, 174, 97)),
        (1.0, (215, 48, 39)),
    ],
    "green-red": [
        (0.0, (34, 139, 34)),     # forest green
        (0.3, (144, 238, 144)),   # light green
        (0.5, (255, 255, 224)),   # light yellow
        (0.7, (255, 160, 122)),   # light salmon
        (1.0, (178, 34, 34)),     # firebrick
    ],
}


def entropy_to_color(entropy: float, e_min: float, e_max: float,
                     cmap_name: str, use_log: bool) -> str:
    """Map an entropy value to an RGB hex string."""
    if e_max <= e_min:
        t = 0.0
    else:
        if use_log:
            # Shift so minimum maps to log(1)=0
            val = math.log1p(entropy - e_min)
            top = math.log1p(e_max - e_min)
            t = val / top if top > 0 else 0.0
        else:
            t = (entropy - e_min) / (e_max - e_min)
    t = max(0.0, min(1.0, t))
    stops = COLORMAPS.get(cmap_name, COLORMAPS["blue-red"])
    r, g, b = multi_stop_lerp(stops, t)
    return f"rgb({int(r)},{int(g)},{int(b)})"


def text_color_for_bg(bg_rgb_str: str) -> str:
    """Return black or white text depending on background luminance."""
    # Parse "rgb(r,g,b)"
    inner = bg_rgb_str[4:-1]
    r, g, b = (int(x.strip()) for x in inner.split(","))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#1a1a1a" if luminance > 0.55 else "#f5f5f5"


# ── Token display ────────────────────────────────────────────────────────────

def decode_token(tok: str) -> str:
    """Convert BPE token to display form."""
    return tok.replace("\u0120", " ").replace("\u010a", "\n")


def compute_percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


# ── HTML generation ──────────────────────────────────────────────────────────

def build_html(records: list[dict], cmap: str, use_log: bool,
               percentile_cap: float) -> str:
    """Build a complete HTML page for one or more records."""

    html_parts = []

    # ── page header ──────────────────────────────────────────────────────
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Token Entropy Visualization</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');

  :root {{
    --bg:        #0e0e11;
    --surface:   #18181b;
    --surface-2: #232328;
    --border:    #2e2e35;
    --text:      #e4e4e7;
    --text-dim:  #a1a1aa;
    --accent:    #818cf8;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }}

  .page-header {{
    max-width: 960px;
    margin: 0 auto 2.5rem;
  }}
  .page-header h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
  }}
  .page-header p {{
    color: var(--text-dim);
    font-size: 0.85rem;
  }}

  .record {{
    max-width: 960px;
    margin: 0 auto 3rem;
  }}

  .record-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
  }}
  .record-header h2 {{
    font-size: 1.1rem;
    font-weight: 600;
  }}
  .record-header .stats {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
  }}

  /* ── token flow ──────────────────────────────────────── */
  .token-flow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    line-height: 2.2;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    white-space: normal;
    word-wrap: break-word;
  }}

  .tok {{
    padding: 2px 1px;
    border-radius: 3px;
    cursor: default;
    position: relative;
    transition: outline 0.15s;
  }}
  .tok:hover {{
    outline: 2px solid var(--accent);
    outline-offset: 1px;
    z-index: 2;
  }}

  /* tooltip */
  .tok .tip {{
    display: none;
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: #000;
    color: #e4e4e7;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 4px 8px;
    border-radius: 4px;
    white-space: nowrap;
    z-index: 10;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }}
  .tok:hover .tip {{ display: block; }}

  /* ── color bar legend ────────────────────────────────── */
  .legend {{
    margin-top: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.75rem;
    color: var(--text-dim);
  }}
  .legend-bar {{
    height: 14px;
    width: 220px;
    border-radius: 3px;
    border: 1px solid var(--border);
  }}

  /* ── summary stats cards ─────────────────────────────── */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.25rem;
  }}
  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }}
  .stat-card .label {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-bottom: 0.15rem;
  }}
  .stat-card .value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
  }}
  .stat-card .sub {{
    font-size: 0.7rem;
    color: var(--text-dim);
    margin-top: 0.1rem;
  }}

  /* ── high-entropy token list ─────────────────────────── */
  .he-section {{
    margin-top: 1.5rem;
  }}
  .he-section h3 {{
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--text);
  }}
  .he-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
  }}
  .he-table th {{
    text-align: left;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }}
  .he-table td {{
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
  }}
  .he-table tr:hover td {{
    background: var(--surface-2);
  }}
  .he-table .ctx {{
    color: var(--text-dim);
    font-size: 0.72rem;
  }}
  .he-table .ctx em {{
    font-style: normal;
    color: var(--text);
    font-weight: 500;
  }}
</style>
</head>
<body>

<div class="page-header">
  <h1>Token Entropy Visualization</h1>
  <p>Each token is colored by model uncertainty — cool tones are confident, warm tones are decision points.</p>
</div>
""")

    # ── per-record rendering ─────────────────────────────────────────────
    for rec_idx, rec in enumerate(records):
        tokens = rec["tokens"]
        entropies = rec["token_entropies"]
        n = len(entropies)
        task_id = rec.get("task_id", rec_idx)

        mean_e = statistics.mean(entropies)
        std_e = statistics.stdev(entropies) if n > 1 else 0.0
        median_e = statistics.median(entropies)

        e_min = min(entropies)
        e_max_raw = max(entropies)
        e_cap = compute_percentile(entropies, percentile_cap)
        e_max = e_cap  # cap outliers for color mapping

        thresh = mean_e + std_e
        high_count = sum(1 for e in entropies if e >= thresh)
        high_pct = high_count / n * 100 if n else 0
        total_ent = sum(entropies)
        high_ent_share = sum(e for e in entropies if e >= thresh) / total_ent * 100 if total_ent else 0

        html_parts.append(f'<div class="record">')
        html_parts.append(f'<div class="record-header">')
        html_parts.append(f'  <h2>Task {task_id}</h2>')
        html_parts.append(f'  <span class="stats">{n} tokens · threshold {thresh:.3f}</span>')
        html_parts.append(f'</div>')

        # stat cards
        html_parts.append('<div class="stats-grid">')
        cards = [
            ("Mean Entropy", f"{mean_e:.4f}", None),
            ("Median", f"{median_e:.4f}", None),
            ("Std Dev", f"{std_e:.4f}", None),
            ("Max", f"{e_max_raw:.4f}", None),
            ("High-Entropy", f"{high_count}", f"{high_pct:.1f}% of tokens"),
            ("Entropy Share", f"{high_ent_share:.1f}%", f"from {high_pct:.0f}% of tokens"),
        ]
        for label, value, sub in cards:
            sub_html = f'<div class="sub">{sub}</div>' if sub else ""
            html_parts.append(
                f'<div class="stat-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                f'{sub_html}</div>'
            )
        html_parts.append('</div>')

        # ── color-coded token flow ───────────────────────────────────────
        # Build all token spans as ONE string so no stray newlines appear
        # between them (the container uses white-space: pre-wrap).
        token_spans = []
        for i, (tok, ent) in enumerate(zip(tokens, entropies)):
            bg = entropy_to_color(ent, e_min, e_max, cmap, use_log)
            fg = text_color_for_bg(bg)
            display = decode_token(tok)
            # Replace real newlines with <br> so they render as line breaks
            # rather than being swallowed or creating layout issues.
            safe = html.escape(display, quote=True).replace("\n", "<br>")
            raw_safe = html.escape(tok, quote=True)
            tip_text = f"pos {i} · entropy {ent:.4f} · «{raw_safe}»"
            token_spans.append(
                f'<span class="tok" style="background:{bg};color:{fg}"'
                f' title="{tip_text}">'
                f'<span class="tip">{tip_text}</span>'
                f'{safe}</span>'
            )
        html_parts.append('<div class="token-flow">' + "".join(token_spans) + '</div>')

        # ── legend ───────────────────────────────────────────────────────
        stops = COLORMAPS.get(cmap, COLORMAPS["blue-red"])
        grad_parts = ", ".join(
            f"rgb({int(r)},{int(g)},{int(b)}) {int(p*100)}%"
            for p, (r, g, b) in stops
        )
        html_parts.append(
            f'<div class="legend">'
            f'<span>Low entropy ({e_min:.3f})</span>'
            f'<div class="legend-bar" style="background:linear-gradient(to right, {grad_parts})"></div>'
            f'<span>High entropy ({e_max:.3f})</span>'
            f'</div>'
        )

        # ── top high-entropy tokens table ────────────────────────────────
        indexed = sorted(enumerate(zip(tokens, entropies)),
                         key=lambda x: x[1][1], reverse=True)[:15]

        html_parts.append('<div class="he-section">')
        html_parts.append('<h3>Top Decision-Point Tokens</h3>')
        html_parts.append('<table class="he-table"><thead><tr>')
        html_parts.append('<th>Rank</th><th>Pos</th><th>Entropy</th><th>Token</th><th>Context</th>')
        html_parts.append('</tr></thead><tbody>')

        for rank, (idx, (tok, ent)) in enumerate(indexed, 1):
            bg = entropy_to_color(ent, e_min, e_max, cmap, use_log)
            fg = text_color_for_bg(bg)
            tok_display = html.escape(decode_token(tok))

            # context window
            start = max(0, idx - 3)
            end = min(n, idx + 4)
            ctx_parts = []
            for j in range(start, end):
                t = html.escape(decode_token(tokens[j]))
                if j == idx:
                    ctx_parts.append(f"<em>{t}</em>")
                else:
                    ctx_parts.append(t)
            ctx = "".join(ctx_parts)

            html_parts.append(
                f'<tr>'
                f'<td>{rank}</td>'
                f'<td>{idx}</td>'
                f'<td>{ent:.4f}</td>'
                f'<td><span style="background:{bg};color:{fg};padding:2px 6px;'
                f'border-radius:3px">{tok_display}</span></td>'
                f'<td class="ctx">{ctx}</td>'
                f'</tr>'
            )
        html_parts.append('</tbody></table>')
        html_parts.append('</div>')  # he-section
        html_parts.append('</div>')  # record

    html_parts.append('</body></html>')
    return "\n".join(html_parts)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize token entropies as a color-coded HTML page."
    )
    parser.add_argument("-i", "--input", help="JSON file with token entropy data")
    parser.add_argument("-o", "--output", default="entropy_vis",
                        help="Output path stem (default: entropy_vis). "
                             "For a single record produces <stem>.html; "
                             "for N records produces <stem>_0.html … <stem>_N-1.html")
    parser.add_argument("--cmap", choices=list(COLORMAPS.keys()),
                        default="blue-red",
                        help="Color map (default: blue-red)")
    parser.add_argument("--log", action="store_true",
                        help="Use log scale for color mapping")
    parser.add_argument("--percentile-cap", type=float, default=99,
                        help="Cap color scale at this percentile (default: 99)")
    args = parser.parse_args()

    with open(args.input) as f:
        raw = f.read().strip()

    # Detect format: JSON array, single JSON object, or JSONL (one object per line)
    try:
        data = json.loads(raw)
        records = [data] if isinstance(data, dict) else data
    except json.JSONDecodeError:
        # Likely JSONL — parse each non-empty line as a separate JSON object
        records = []
        for line_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {line_no}: {e}", file=sys.stderr)
        if not records:
            print("Error: no valid JSON records found.", file=sys.stderr)
            sys.exit(1)

    # Strip .html suffix from stem if the user passed one
    stem = args.output
    if stem.endswith(".html"):
        stem = stem[:-5]

    # Create visualization output directory
    out_dir = Path("visualization")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_name = Path(stem).name  # use just the filename part inside the folder

    if len(records) == 1:
        # Single record → single file
        page = build_html(records, args.cmap, args.log, args.percentile_cap)
        out_path = out_dir / f"{stem_name}.html"
        with open(out_path, "w") as f:
            f.write(page)
        print(f"Written to {out_path}")
    else:
        # Multiple records → one file per record
        for i, rec in enumerate(records):
            page = build_html([rec], args.cmap, args.log, args.percentile_cap)
            # breakpoint()
            out_path = out_dir/f"{args.input.split("/")[-2]}"/f"{args.input.split("/")[-1]}"/f"{stem_name}_{i}.html"
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(page)
            print(f"[{i+1}/{len(records)}] Written to {out_path}")
        print(f"Done — {len(records)} files in {out_dir}/")


if __name__ == "__main__":
    main()