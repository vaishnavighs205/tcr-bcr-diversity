"""
Plotting helpers for repertoire diversity analysis.
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diversity_metrics import hill_number, rarefaction_curve

PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]

METRIC_LABELS = {
    "low_q": "Low-Q richness",
    "high_q": "High-Q / inverse Simpson",
    "ip_slope": "IP slope",
    "ipq": "IPQ",
    "shannon_H": "Shannon H′",
    "shannon_exp_H": "exp(Shannon H′)",
    "simpson_1minusD": "Simpson 1-D",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": True,
    })


_style()


def _sample_groups(df: pd.DataFrame):
    if "sample_id" not in df.columns:
        raise ValueError("DataFrame must contain sample_id column.")
    return list(df.groupby("sample_id", sort=True))


def plot_rarefaction(df: pd.DataFrame, steps: int = 40):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, (sample, grp) in enumerate(_sample_groups(df)):
        xs, ys = rarefaction_curve(grp["frequency"].values, steps=steps, seed=i)
        ax.plot(xs, ys, lw=2, label=sample, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Reads sampled")
    ax.set_ylabel("Expected observed clones")
    ax.set_title("Rarefaction curves")
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    return fig, ax


def plot_hill_profile(df: pd.DataFrame, q_values: Optional[np.ndarray] = None):
    q_values = np.asarray(q_values if q_values is not None else np.linspace(0, 3, 31), dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, (sample, grp) in enumerate(_sample_groups(df)):
        values = [hill_number(grp["frequency"].values, float(q if q != 1 else 1)) for q in q_values]
        ax.plot(q_values, values, lw=2, label=sample, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Hill order q")
    ax.set_ylabel("Effective diversity")
    ax.set_title("Hill diversity profile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_vgene_usage(df: pd.DataFrame, top_n: int = 12):
    if "v_gene" not in df.columns:
        warnings.warn("No v_gene column found — skipping V-gene usage plot.")
        return None, None
    data = df.dropna(subset=["v_gene"]).copy()
    if data.empty:
        return None, None
    order = data.groupby("v_gene")["frequency"].sum().sort_values(ascending=False).head(top_n).index
    pivot = (
        data[data["v_gene"].isin(order)]
        .pivot_table(index="v_gene", columns="sample_id", values="frequency", aggfunc="sum", fill_value=0)
        .reindex(order)
    )
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(order) * 0.35)))
    pivot.plot(kind="barh", stacked=True, ax=ax, color=[PALETTE[i % len(PALETTE)] for i in range(pivot.shape[1])])
    ax.set_xlabel("Frequency")
    ax.set_ylabel("V-gene")
    ax.set_title(f"Top {len(order)} V-gene usage")
    ax.legend(title="Sample", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_cdr3_length(df: pd.DataFrame):
    if "cdr3_aa" not in df.columns:
        warnings.warn("No cdr3_aa column found — skipping CDR3 length plot.")
        return None, None
    data = df.dropna(subset=["cdr3_aa"]).copy()
    if data.empty:
        return None, None
    data["cdr3_len"] = data["cdr3_aa"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for i, (sample, grp) in enumerate(data.groupby("sample_id", sort=True)):
        vals = grp["cdr3_len"].values
        bins = np.arange(max(1, vals.min()) - 0.5, vals.max() + 1.5)
        ax.hist(vals, bins=bins, alpha=0.4, label=sample, color=PALETTE[i % len(PALETTE)], density=True)
    ax.set_xlabel("CDR3 length (aa)")
    ax.set_ylabel("Density")
    ax.set_title("CDR3 length distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax
