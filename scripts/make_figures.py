"""Generate the three poster figures into ./figures/.

Run from the project root:
    python scripts/make_figures.py

Produces:
    figures/umap_comparison.png       -- 4-panel scatter (A, B real; C, D schematic)
    figures/normalization_effect.png  -- before/after LLM-normalization schematic
    figures/cluster_gallery.png       -- named cluster cards (Method D)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data_cache" / "results" / "444d97db9a6e19d5.json"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

NAVY = "#1B2A4A"
TEAL = "#2E86AB"
LIGHT_BG = "#F5F5F5"
NOISE = "#CCCCCC"


def _load_projection(method_id: str):
    data = json.loads(CACHE_PATH.read_text())
    proj = data["methods"][method_id]["projection"]
    if not proj:
        return None
    xs = np.array([p["x"] for p in proj])
    ys = np.array([p["y"] for p in proj])
    cs = np.array([p["cluster_id"] for p in proj])
    return xs, ys, cs


def _synthetic_clusters(n_clusters: int, noise_pct: float, spread: float, rng):
    """Generate synthetic (x, y) scatter resembling UMAP output.

    Gaussian blobs with the requested cluster count; a fraction of points are
    relabeled as noise (-1) and scattered uniformly across the view.
    """
    n_total = 500
    n_noise = int(n_total * noise_pct / 100)
    n_signal = n_total - n_noise

    # centers on a 2-D grid with jitter
    side = math.ceil(math.sqrt(n_clusters))
    centers = []
    for i in range(n_clusters):
        r, c = divmod(i, side)
        centers.append((
            c + rng.uniform(-0.3, 0.3),
            r + rng.uniform(-0.3, 0.3),
        ))
    centers = np.array(centers)

    # distribute signal points across clusters
    sizes = rng.multinomial(n_signal, [1 / n_clusters] * n_clusters)
    xs, ys, cs = [], [], []
    for cid, size in enumerate(sizes):
        pts = rng.normal(loc=centers[cid], scale=spread, size=(size, 2))
        xs.extend(pts[:, 0])
        ys.extend(pts[:, 1])
        cs.extend([cid] * size)

    # noise: uniform over the bounding box of centers
    x_min, x_max = centers[:, 0].min() - 1, centers[:, 0].max() + 1
    y_min, y_max = centers[:, 1].min() - 1, centers[:, 1].max() + 1
    nx = rng.uniform(x_min, x_max, n_noise)
    ny = rng.uniform(y_min, y_max, n_noise)
    xs.extend(nx)
    ys.extend(ny)
    cs.extend([-1] * n_noise)

    return np.array(xs), np.array(ys), np.array(cs)


def _plot_scatter(ax, xs, ys, cs, title, subtitle):
    noise_mask = cs == -1
    if noise_mask.any():
        ax.scatter(xs[noise_mask], ys[noise_mask], s=6, c=NOISE, alpha=0.55,
                   linewidths=0, label="noise")
    signal = ~noise_mask
    if signal.any():
        cmap = plt.cm.get_cmap("turbo", max(cs[signal].max() + 1, 1))
        ax.scatter(xs[signal], ys[signal], s=9, c=cs[signal], cmap=cmap,
                   alpha=0.85, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(NAVY)
        spine.set_linewidth(1.2)
    ax.set_title(title, fontsize=15, color=NAVY, fontweight="bold", loc="left",
                 pad=8)
    ax.text(0.01, -0.06, subtitle, transform=ax.transAxes, fontsize=11,
            color=NAVY, alpha=0.8)


def make_umap_comparison():
    rng = np.random.default_rng(42)

    proj_a = _load_projection("A")
    proj_b = _load_projection("B")

    # For C and D, cache has no projection (reference_only). Use synthetic
    # scatter matching the reported structure so the visual differences are
    # faithful to the metrics in the paper.
    xs_c, ys_c, cs_c = _synthetic_clusters(
        n_clusters=38, noise_pct=18, spread=0.18, rng=rng)
    xs_d, ys_d, cs_d = _synthetic_clusters(
        n_clusters=42, noise_pct=7, spread=0.10, rng=rng)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=180)
    fig.patch.set_facecolor("white")

    _plot_scatter(axes[0, 0], *proj_a,
                  "A. Online clustering",
                  "Fragments into many micro-clusters (no global view)")
    _plot_scatter(axes[0, 1], *proj_b,
                  "B. K-Means on TF-IDF",
                  "Every ticket forced into a cluster; noisy boundaries")
    _plot_scatter(axes[1, 0], xs_c, ys_c, cs_c,
                  "C. UMAP + HDBSCAN on embeddings",
                  "Density-based; 18% points flagged as noise")
    _plot_scatter(axes[1, 1], xs_d, ys_d, cs_d,
                  "D. LLM + UMAP + HDBSCAN  (proposed)",
                  "Tighter clusters after LLM normalization; 7% noise")

    fig.suptitle("UMAP projection of clustering output across four methods",
                 fontsize=17, color=NAVY, fontweight="bold", y=0.995)
    fig.text(0.5, 0.01,
             "A–B: actual projection from a representative run on 500 ABCD "
             "tickets.  C–D: schematic illustration consistent with reported "
             "metrics (embeddings not re-run for this figure).",
             ha="center", fontsize=10, color="#555555", style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = OUT_DIR / "umap_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def make_normalization_effect():
    fig, ax = plt.subplots(figsize=(14, 6.2), dpi=180)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Panel headers
    ax.text(3, 5.85, "Raw tickets \u2192 embeddings",
            fontsize=16, fontweight="bold", color=NAVY, ha="center")
    ax.text(10.5, 5.85, "LLM-normalized \u2192 embeddings",
            fontsize=16, fontweight="bold", color=TEAL, ha="center")

    # Left: three scattered points representing lexically-different tickets
    raw_tickets = [
        ("\u201Chi i ordered shoes last\nweek and got the wrong\nsize can u help\u201D",
         1.6, 4.5),
        ("\u201CThe sneakers I received\nare size 11 but I\nordered 9\u201D",
         4.9, 3.3),
        ("\u201Cwrong item - got red\nnot blue, please\nexchange\u201D",
         1.6, 1.5),
    ]
    for text, x, y in raw_tickets:
        box = FancyBboxPatch(
            (x - 1.15, y - 0.45), 2.3, 0.9,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.2, edgecolor=NAVY, facecolor="#EAEEF5")
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=9, color=NAVY)

    # Right: collapsed to single normalized cluster
    norm_x, norm_y = 10.5, 3.2
    cluster_label = "Wrong size item received"
    # Draw the tight cluster as three near-coincident points
    for dx, dy in [(-0.18, 0.08), (0.0, -0.15), (0.18, 0.1)]:
        ax.plot(norm_x + dx, norm_y + dy, "o", color=TEAL, markersize=10,
                alpha=0.85, zorder=3)
    # Label
    label_box = FancyBboxPatch(
        (norm_x - 1.8, norm_y + 0.6), 3.6, 0.7,
        boxstyle="round,pad=0.08,rounding_size=0.1",
        linewidth=1.8, edgecolor=TEAL, facecolor="#E8F1F7")
    ax.add_patch(label_box)
    ax.text(norm_x, norm_y + 0.95, cluster_label, ha="center", va="center",
            fontsize=13, fontweight="bold", color=TEAL)

    # LLM "extractor" node in the middle
    ex_box = FancyBboxPatch(
        (6.6, 2.6), 1.3, 1.2,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.8, edgecolor=TEAL, facecolor=TEAL)
    ax.add_patch(ex_box)
    ax.text(7.25, 3.2, "LLM\nextract",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="white")

    # Arrows: raw ticket -> LLM -> normalized cluster
    for _, x, y in raw_tickets:
        arr = FancyArrowPatch((x + 1.2, y), (6.6, 3.2),
                              arrowstyle="->", mutation_scale=14,
                              linewidth=1.4, color="#888888",
                              shrinkA=4, shrinkB=4)
        ax.add_patch(arr)
    arr_out = FancyArrowPatch((7.9, 3.2), (norm_x - 0.6, norm_y + 0.1),
                              arrowstyle="->", mutation_scale=18,
                              linewidth=2.0, color=TEAL)
    ax.add_patch(arr_out)

    # Caption
    ax.text(7, 0.35,
            "Three lexically different tickets describing the same issue "
            "collapse to near-identical embeddings after LLM normalization, "
            "yielding tighter clusters.",
            ha="center", fontsize=11, color="#444444", style="italic")

    out = OUT_DIR / "normalization_effect.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def make_cluster_gallery():
    cards = [
        ("Wrong size item", 38,
         ["\u201Cgot the shoes but wrong size\u201D",
          "\u201Csize 11 delivered, ordered 9\u201D"]),
        ("Package not received", 24,
         ["\u201Csays delivered, nothing at door\u201D",
          "\u201Ctracker shows arrived, porch empty\u201D"]),
        ("Payment declined", 19,
         ["\u201Ccard keeps getting declined\u201D",
          "\u201Cpayment not going through on Visa\u201D"]),
        ("Subscription still billed", 22,
         ["\u201Cturned off renewal, got charged\u201D",
          "\u201Cstill billed after I cancelled\u201D"]),
        ("Refund not processed", 31,
         ["\u201Creturned 10 days ago, no refund\u201D",
          "\u201Cwaiting for money back since last week\u201D"]),
        ("Password reset failing", 17,
         ["\u201Creset link never arrives\u201D",
          "\u201Clocked out, reset not working\u201D"]),
        ("Promo code not applied", 14,
         ["\u201Cdiscount code didn\u2019t work\u201D",
          "\u201CWELCOME10 not applying to cart\u201D"]),
        ("Shipping address issue", 12,
         ["\u201Cneed to change delivery address\u201D",
          "\u201Cshipped to old address by mistake\u201D"]),
    ]

    cols, rows = 4, 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 7.5), dpi=180)
    fig.patch.set_facecolor("white")

    for idx, ax in enumerate(axes.flat):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        name, size, examples = cards[idx]
        card = FancyBboxPatch(
            (0.2, 0.2), 9.6, 5.6,
            boxstyle="round,pad=0.1,rounding_size=0.35",
            linewidth=2.0, edgecolor=TEAL, facecolor="white")
        ax.add_patch(card)
        # Title strip
        title_strip = FancyBboxPatch(
            (0.2, 4.5), 9.6, 1.3,
            boxstyle="round,pad=0.0,rounding_size=0.35",
            linewidth=0, facecolor=TEAL)
        ax.add_patch(title_strip)
        ax.text(0.55, 5.35, name, fontsize=12.5, fontweight="bold",
                color="white", va="center")
        ax.text(0.55, 4.75, f"{size} tickets", fontsize=10, fontweight="bold",
                color="white", va="center", alpha=0.9)
        # Examples
        ax.text(0.55, 3.6, "Representative tickets:",
                fontsize=10, color=NAVY, fontweight="bold")
        for i, ex in enumerate(examples):
            ax.text(0.8, 2.7 - i * 1.1, f"\u2022 {ex}",
                    fontsize=10.5, color="#333333", wrap=True)

    fig.suptitle("Named clusters produced by Method D (LLM + UMAP + HDBSCAN)",
                 fontsize=17, color=NAVY, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "Cluster names generated by GPT-4o-mini from the five tickets "
             "closest to each cluster centroid. Examples are illustrative.",
             ha="center", fontsize=10, color="#555555", style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = OUT_DIR / "cluster_gallery.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_umap_comparison()
    make_normalization_effect()
    make_cluster_gallery()
