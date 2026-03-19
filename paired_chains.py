"""
paired_chains.py
================
Paired alpha/beta (TCR) or heavy/light (BCR) chain analysis.

When both chains are sequenced from the same cell (e.g. 10x Genomics,
paired-end bulk with barcode demultiplexing), this module:

1. Matches alpha/beta clonotypes sharing the same cell barcode or clone ID.
2. Computes per-chain and paired diversity metrics.
3. Detects dual-alpha (two productive TRA rearrangements) or dual-light cells.
4. Analyses V-gene pairing preferences as a co-occurrence heatmap.
5. Measures CDR3 length correlation between paired chains.
6. Reports sharing statistics: public clones present in multiple samples.

Input format
------------
A single DataFrame (or two separate DataFrames for chain A / chain B)
with at minimum:

    cell_id    : barcode / unique cell identifier  (REQUIRED for pairing)
    chain      : 'TRA', 'TRB', 'IGH', 'IGL', 'IGK'
    frequency  : clone count
    v_gene     : V-gene call      (optional but needed for pairing heatmap)
    cdr3_aa    : CDR3 amino-acid  (optional but needed for CDR3 correlations)
    sample_id  : sample label

Column aliases are resolved through data_loader._resolve_columns().
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from diversity_metrics import compute_all
from plots import METRIC_LABELS, PALETTE, _style
_style()

# Canonical chain pair definitions
CHAIN_PAIRS = {
    "TCR": ("TRA", "TRB"),
    "BCR_heavy_lambda": ("IGH", "IGL"),
    "BCR_heavy_kappa":  ("IGH", "IGK"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Pairing helpers
# ──────────────────────────────────────────────────────────────────────────────

def detect_receptor_type(df: pd.DataFrame) -> str:
    """Infer receptor type from chain column values."""
    if "chain" not in df.columns and "receptor" not in df.columns:
        return "unknown"
    col = "chain" if "chain" in df.columns else "receptor"
    chains = set(df[col].dropna().str.upper().unique())
    if "TRA" in chains or "TRB" in chains:
        return "TCR"
    if "IGH" in chains:
        return "BCR"
    return "unknown"


def pair_chains(
    df: pd.DataFrame,
    chain_a: str,
    chain_b: str,
    cell_col: str = "cell_id",
) -> pd.DataFrame:
    """
    Inner-join two chains on cell_id to create paired clonotype records.

    Returns a DataFrame with columns:
        cell_id,
        v_gene_a, cdr3_aa_a, freq_a,
        v_gene_b, cdr3_aa_b, freq_b,
        sample_id
    """
    col = cell_col if cell_col in df.columns else "clone_id"

    sub_a = df[df["chain"].str.upper() == chain_a].copy()
    sub_b = df[df["chain"].str.upper() == chain_b].copy()

    if sub_a.empty or sub_b.empty:
        warnings.warn(
            f"No data found for chain {chain_a} or {chain_b}. "
            "Check the 'chain' column values."
        )
        return pd.DataFrame()

    # Keep only cells with exactly one productive rearrangement per chain
    # (flag dual-chain cells separately)
    dup_a = sub_a[col].value_counts()
    dup_b = sub_b[col].value_counts()
    dual_a = dup_a[dup_a > 1].index.tolist()
    dual_b = dup_b[dup_b > 1].index.tolist()

    if dual_a:
        warnings.warn(
            f"{len(dual_a)} cell(s) have dual {chain_a} rearrangements — "
            "keeping first occurrence only."
        )
    sub_a = sub_a.drop_duplicates(subset=[col])
    sub_b = sub_b.drop_duplicates(subset=[col])

    # Rename columns for clarity
    rename_a = {"frequency": "freq_a", "v_gene": "v_gene_a",
                 "j_gene": "j_gene_a", "cdr3_aa": "cdr3_aa_a"}
    rename_b = {"frequency": "freq_b", "v_gene": "v_gene_b",
                 "j_gene": "j_gene_b", "cdr3_aa": "cdr3_aa_b"}
    sub_a = sub_a.rename(columns={k: v for k, v in rename_a.items() if k in sub_a.columns})
    sub_b = sub_b.rename(columns={k: v for k, v in rename_b.items() if k in sub_b.columns})

    keep_a = [col, "sample_id"] + [v for v in rename_a.values() if v in sub_a.columns]
    keep_b = [col] + [v for v in rename_b.values() if v in sub_b.columns]

    paired = sub_a[keep_a].merge(sub_b[[c for c in keep_b if c in sub_b.columns]],
                                 on=col, how="inner")
    return paired


# ──────────────────────────────────────────────────────────────────────────────
# Public clones (clones shared across samples)
# ──────────────────────────────────────────────────────────────────────────────

def find_public_clones(
    df: pd.DataFrame,
    sequence_col: str = "cdr3_aa",
    min_samples: int = 2,
) -> pd.DataFrame:
    """
    Identify clonotypes (by CDR3 sequence) present in ≥ min_samples samples.

    Returns a DataFrame of public clones with their per-sample frequencies.
    """
    if sequence_col not in df.columns:
        warnings.warn(f"Column '{sequence_col}' not found — cannot identify public clones.")
        return pd.DataFrame()

    pivot = (
        df.dropna(subset=[sequence_col])
        .groupby(["sample_id", sequence_col])["frequency"]
        .sum()
        .unstack("sample_id", fill_value=0)
    )
    sample_count = (pivot > 0).sum(axis=1)
    public = pivot[sample_count >= min_samples].copy()
    public.insert(0, "n_samples", sample_count[public.index])
    public = public.sort_values("n_samples", ascending=False)
    return public


# ──────────────────────────────────────────────────────────────────────────────
# Per-chain diversity
# ──────────────────────────────────────────────────────────────────────────────

def per_chain_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute diversity metrics separately for each chain × sample combination.

    Returns one row per (sample_id, chain).
    """
    if "chain" not in df.columns and "receptor" not in df.columns:
        warnings.warn("No 'chain' column found — treating all rows as one chain.")
        df = df.copy()
        df["chain"] = "unknown"

    col = "chain" if "chain" in df.columns else "receptor"
    rows = []
    for (sample, chain), sub in df.groupby(["sample_id", col]):
        m = compute_all(sub["frequency"].values, sample_id=f"{sample}|{chain}")
        m["sample_id"] = sample
        m["chain"] = chain
        rows.append(m)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_vgene_pairing_heatmap(
    paired_df: pd.DataFrame,
    top_n: int = 12,
    output_dir: str | Path = "output",
    sample_id: Optional[str] = None,
) -> Optional[Path]:
    """
    V-gene co-usage heatmap for paired alpha/beta or heavy/light chains.
    Colour = log10(number of paired cells).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "v_gene_a" not in paired_df.columns or "v_gene_b" not in paired_df.columns:
        warnings.warn("V-gene columns missing from paired DataFrame — skipping heatmap.")
        return None

    sub = paired_df.copy()
    if sample_id:
        sub = sub[sub["sample_id"] == sample_id]

    # Top N V-genes per chain by total cell count
    top_a = sub["v_gene_a"].value_counts().head(top_n).index.tolist()
    top_b = sub["v_gene_b"].value_counts().head(top_n).index.tolist()
    sub = sub[sub["v_gene_a"].isin(top_a) & sub["v_gene_b"].isin(top_b)]

    pivot = (
        sub.groupby(["v_gene_a", "v_gene_b"])
        .size()
        .unstack("v_gene_b", fill_value=0)
        .reindex(index=top_a, columns=top_b, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(max(7, top_n * 0.7), max(6, top_n * 0.65)))
    im = ax.imshow(
        pivot.values + 0.5,  # +0.5 avoids log(0)
        norm=LogNorm(vmin=0.5),
        cmap="YlOrRd", aspect="auto"
    )
    plt.colorbar(im, ax=ax, label="Paired cells (log scale)", fraction=0.03)

    ax.set_xticks(range(len(top_b)))
    ax.set_xticklabels(top_b, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_a)))
    ax.set_yticklabels(top_a, fontsize=8)
    ax.set_xlabel("Chain B / Heavy V-gene")
    ax.set_ylabel("Chain A / Light V-gene")
    title = f"V-gene pairing heatmap{' — ' + sample_id if sample_id else ''}"
    ax.set_title(title)
    fig.tight_layout()

    fname = f"vgene_pairing{'_' + sample_id if sample_id else ''}.png"
    path = out / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cdr3_length_correlation(
    paired_df: pd.DataFrame,
    output_dir: str | Path = "output",
) -> Optional[Path]:
    """
    Scatter plot of CDR3 length for chain A vs chain B in each paired cell.
    Includes Spearman correlation annotation.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "cdr3_aa_a" not in paired_df.columns or "cdr3_aa_b" not in paired_df.columns:
        warnings.warn("CDR3 columns missing — skipping length correlation plot.")
        return None

    sub = paired_df.dropna(subset=["cdr3_aa_a", "cdr3_aa_b"]).copy()
    sub["len_a"] = sub["cdr3_aa_a"].str.len()
    sub["len_b"] = sub["cdr3_aa_b"].str.len()

    samples = sub["sample_id"].unique()
    cmap = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sorted(samples))}

    fig, ax = plt.subplots(figsize=(7, 6))
    for s in samples:
        ss = sub[sub["sample_id"] == s]
        ax.scatter(ss["len_a"], ss["len_b"],
                   c=cmap[s], alpha=0.5, s=18, label=s, linewidths=0)

    rho, pval = stats_spearman(sub["len_a"].values, sub["len_b"].values)
    ax.set_xlabel("Chain A CDR3 length (aa)")
    ax.set_ylabel("Chain B CDR3 length (aa)")
    ax.set_title("CDR3 length correlation (paired chains)")
    ax.text(0.05, 0.95,
            f"Spearman ρ = {rho:.3f}  (p = {pval:.2e})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    if len(samples) > 1:
        ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout()

    path = out / "cdr3_length_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_chain_diversity(
    chain_div_df: pd.DataFrame,
    output_dir: str | Path = "output",
    metrics: Optional[List[str]] = None,
) -> Path:
    """
    Grouped bar chart comparing diversity metrics side-by-side per chain.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = [m for m in ["low_q", "high_q", "shannon_H", "simpson_1minusD"]
                   if m in chain_div_df.columns]

    chains = chain_div_df["chain"].unique()
    cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(chains)}

    n = len(metrics)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        pivot = chain_div_df.pivot_table(
            index="sample_id", columns="chain", values=metric
        )
        pivot.plot(kind="bar", ax=ax,
                   color=[cmap.get(c, "#888") for c in pivot.columns],
                   edgecolor="white", linewidth=0.4)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.legend(title="Chain", fontsize=7)

    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Per-chain diversity comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    path = out / "per_chain_diversity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── shim so we don't need scipy import at top ────────────────────────────────
def stats_spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return r.statistic, r.pvalue
