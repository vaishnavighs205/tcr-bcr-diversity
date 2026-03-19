"""
stats_compare.py
================
Statistical comparison of diversity metrics across sample groups.

Methods
-------
- Permutation test (non-parametric, no distribution assumptions)
- Mann-Whitney U test (two groups)
- Kruskal-Wallis H test (3+ groups)
- Pairwise comparisons with Benjamini-Hochberg FDR correction
- Effect size: rank-biserial correlation (r) for Mann-Whitney

Usage
-----
    from stats_compare import GroupComparison

    groups = {
        "Healthy": ["Healthy_1", "Healthy_2", "Healthy_3"],
        "Patient": ["Patient_1", "Patient_2", "Patient_3"],
    }
    cmp = GroupComparison(summary_df, groups)
    results = cmp.run_all()
    cmp.plot_comparison(output_dir="output/")
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plots import METRIC_LABELS, PALETTE, _style
_style()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    cummin = np.inf
    for i in range(n - 1, -1, -1):
        idx = order[i]
        adjusted[idx] = cummin = min(cummin, pvals[idx] * n / (i + 1))
    return np.clip(adjusted, 0, 1)


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size r = 1 - 2U / (n1*n2) for Mann-Whitney U."""
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1 - 2 * u / (len(x) * len(y)))


def _permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 9999,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Two-sample permutation test on the difference of means.
    Returns two-sided p-value.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs = abs(np.mean(x) - np.mean(y))
    combined = np.concatenate([x, y])
    n = len(x)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        count += abs(np.mean(perm[:n]) - np.mean(perm[n:])) >= obs
    return (count + 1) / (n_perm + 1)


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_METRICS = [
    "low_q", "high_q", "ip_slope", "ipq",
    "shannon_H", "shannon_exp_H", "simpson_1minusD",
]


class GroupComparison:
    """
    Compare diversity metrics across labelled sample groups.

    Parameters
    ----------
    summary_df : DataFrame output of analyse.run_analysis()
    groups     : dict mapping group_name → list of sample_id strings
    metrics    : list of metric columns to test (default: all 7 core metrics)
    n_perm     : permutation iterations (default 9999)
    """

    def __init__(
        self,
        summary_df: pd.DataFrame,
        groups: Dict[str, List[str]],
        metrics: Optional[List[str]] = None,
        n_perm: int = 9999,
    ):
        self.df = summary_df.copy()
        self.groups = groups
        self.metrics = metrics or [m for m in DEFAULT_METRICS if m in summary_df.columns]
        self.n_perm = n_perm
        self._rng = np.random.default_rng(42)

        # Tag each row with its group label
        sid_to_group: Dict[str, str] = {}
        requested: set[str] = set()
        for g, sids in groups.items():
            for s in sids:
                sid_to_group[s] = g
                requested.add(s)
        self.df["group"] = self.df["sample"].map(sid_to_group)

        unmatched = int(self.df["group"].isna().sum())
        missing_requested = requested.difference(set(self.df["sample"].astype(str)))
        if unmatched or missing_requested:
            n_missing = unmatched + len(missing_requested)
            warnings.warn(
                f"{n_missing} sample(s) not assigned to any group or not found — they will be excluded.",
                UserWarning,
            )
        self.df = self.df.dropna(subset=["group"])
        self.group_names = [g for g in groups.keys() if (self.df["group"] == g).any()]

    # ── Pairwise stats ────────────────────────────────────────────────────────

    def _test_pair(self, g1: str, g2: str, metric: str) -> dict:
        x = self.df.loc[self.df["group"] == g1, metric].values.astype(float)
        y = self.df.loc[self.df["group"] == g2, metric].values.astype(float)

        result = {
            "group1": g1, "group2": g2, "metric": metric,
            "n1": len(x), "n2": len(y),
            "mean1": float(np.mean(x)) if len(x) else np.nan,
            "mean2": float(np.mean(y)) if len(y) else np.nan,
            "median1": float(np.median(x)) if len(x) else np.nan,
            "median2": float(np.median(y)) if len(y) else np.nan,
        }

        if len(x) < 2 or len(y) < 2:
            result.update(p_perm=np.nan, p_mwu=np.nan, effect_r=np.nan)
            return result

        result["p_perm"] = _permutation_test(x, y, n_perm=self.n_perm, rng=self._rng)

        if len(x) >= 3 and len(y) >= 3:
            mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
            result["p_mwu"] = float(mwu.pvalue)
            result["effect_r"] = _rank_biserial(x, y)
        else:
            result["p_mwu"] = np.nan
            result["effect_r"] = np.nan

        return result

    def run_all(self) -> pd.DataFrame:
        """
        Run pairwise tests for all group pairs × all metrics.
        Applies BH FDR correction across all tests per metric.

        Returns
        -------
        DataFrame with one row per (group1, group2, metric) comparison.
        """
        rows = []
        pairs = list(combinations(self.group_names, 2))

        for metric in self.metrics:
            pair_rows = [self._test_pair(g1, g2, metric) for g1, g2 in pairs]

            # BH correction on permutation p-values for this metric
            raw_p = np.array([r.get("p_perm", np.nan) for r in pair_rows], dtype=float)
            valid = ~np.isnan(raw_p)
            adj = np.full(len(raw_p), np.nan)
            if valid.sum() > 0:
                adj[valid] = _bh_correct(raw_p[valid])
            for r, a in zip(pair_rows, adj):
                r["p_perm_adj"] = float(a)
                r["sig"] = sig_stars(float(a)) if not np.isnan(a) else ""

            rows.extend(pair_rows)

        self.results_ = pd.DataFrame(rows)
        return self.results_

    # ── Kruskal-Wallis (3+ groups) ────────────────────────────────────────────

    def kruskal_wallis(self) -> pd.DataFrame:
        """
        Kruskal-Wallis H test across all groups simultaneously.
        Useful as an omnibus test before pairwise comparisons.
        """
        rows = []
        for metric in self.metrics:
            group_data = [
                self.df.loc[self.df["group"] == g, metric].values.astype(float)
                for g in self.group_names
            ]
            group_data = [d for d in group_data if len(d) >= 2]
            if len(group_data) < 2:
                continue
            try:
                h, p = stats.kruskal(*group_data)
                rows.append({
                    "metric": metric, "label": METRIC_LABELS.get(metric, metric),
                    "H_stat": round(h, 4), "p_value": round(p, 6),
                    "sig": sig_stars(p),
                })
            except Exception:
                pass
        return pd.DataFrame(rows)

    # ── Plots ─────────────────────────────────────────────────────────────────

    def plot_comparison(
        self,
        output_dir: str | Path = "output",
        metrics: Optional[List[str]] = None,
    ) -> Path:
        """
        Grouped strip + box plots for each metric, annotated with
        significance stars from the permutation test.

        Returns path to saved PNG.
        """
        if not hasattr(self, "results_"):
            self.run_all()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metrics = metrics or self.metrics
        n = len(metrics)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        cmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(self.group_names)}

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 4))
        axes = np.array(axes).flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            positions = np.arange(len(self.group_names))

            for j, g in enumerate(self.group_names):
                vals = self.df.loc[self.df["group"] == g, metric].values.astype(float)
                # Box
                bp = ax.boxplot(
                    vals, positions=[j], widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor=cmap[g], alpha=0.35, linewidth=0.8),
                    medianprops=dict(color=cmap[g], linewidth=2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker=".", markersize=4, color=cmap[g]),
                )
                # Strip (jitter)
                jitter = np.random.default_rng(i * 100 + j).uniform(-0.12, 0.12, len(vals))
                ax.scatter(np.full(len(vals), j) + jitter, vals,
                           color=cmap[g], s=28, zorder=3, alpha=0.85, linewidths=0)

            ax.set_xticks(positions)
            ax.set_xticklabels(self.group_names, fontsize=8, rotation=20, ha="right")
            ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10, pad=4)
            ax.set_ylabel(metric, fontsize=8)

            # Significance annotations
            pairs = list(combinations(range(len(self.group_names)), 2))
            y_max = ax.get_ylim()[1]
            y_step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.12

            for k, (pi, pj) in enumerate(pairs):
                g1, g2 = self.group_names[pi], self.group_names[pj]
                row = self.results_[
                    (self.results_["metric"] == metric) &
                    (self.results_["group1"] == g1) &
                    (self.results_["group2"] == g2)
                ]
                if row.empty:
                    continue
                star = row.iloc[0]["sig"]
                if star == "ns":
                    continue
                y_bar = y_max + y_step * (k + 0.8)
                ax.plot([pi, pi, pj, pj],
                        [y_bar - y_step * 0.3, y_bar, y_bar, y_bar - y_step * 0.3],
                        lw=0.8, color="gray")
                ax.text((pi + pj) / 2, y_bar + y_step * 0.05,
                        star, ha="center", va="bottom", fontsize=9, color="gray")

        legend_patches = [
            mpatches.Patch(color=cmap[g], label=g, alpha=0.7)
            for g in self.group_names
        ]
        fig.legend(
            handles=legend_patches, loc="lower center",
            ncol=len(self.group_names), framealpha=0.8,
            bbox_to_anchor=(0.5, -0.02), fontsize=9,
        )

        for j in range(len(metrics), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Group comparison — diversity metrics", fontsize=13, y=1.01)
        fig.tight_layout()

        path = out / "group_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def summary_table(self) -> pd.DataFrame:
        """
        Compact mean ± std table per group and metric,
        plus significance from pairwise tests.
        """
        rows = []
        for metric in self.metrics:
            row = {"metric": METRIC_LABELS.get(metric, metric)}
            for g in self.group_names:
                vals = self.df.loc[self.df["group"] == g, metric].values.astype(float)
                row[g] = f"{np.mean(vals):.3g} ± {np.std(vals):.3g}" if len(vals) else "—"
            rows.append(row)
        return pd.DataFrame(rows).set_index("metric")
