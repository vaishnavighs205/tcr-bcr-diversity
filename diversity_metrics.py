"""Core diversity metric calculations for TCR/BCR repertoires."""

from __future__ import annotations

from typing import Iterable
import math

import numpy as np
from scipy.special import gammaln


ArrayLike = Iterable[float] | np.ndarray


def _to_freqs(counts: ArrayLike) -> np.ndarray:
    """Return finite, strictly positive frequencies as a 1D float array."""
    arr = np.asarray(counts, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    return arr.astype(float, copy=False)


def _proportions(counts: ArrayLike) -> np.ndarray:
    """Convert counts or proportions to a normalised probability vector."""
    freqs = _to_freqs(counts)
    if freqs.size == 0:
        return np.array([], dtype=float)
    total = float(freqs.sum())
    return freqs / total if total > 0 else np.array([], dtype=float)


def low_q(counts: ArrayLike) -> float:
    """Hill number of order q=0 (richness)."""
    return float(len(_to_freqs(counts)))


def shannon(counts: ArrayLike) -> dict[str, float]:
    """Shannon entropy and its exponential effective number of clones."""
    p = _proportions(counts)
    if p.size == 0:
        return {"H": 0.0, "exp_H": 0.0}
    h = float(-np.sum(p * np.log(p)))
    return {"H": h, "exp_H": float(math.exp(h))}


def simpson(counts: ArrayLike) -> dict[str, float]:
    """Simpson concentration/diversity measures."""
    p = _proportions(counts)
    if p.size == 0:
        return {"D": 0.0, "one_minus_D": 0.0, "inv_D": 0.0}
    d = float(np.sum(p**2))
    return {
        "D": d,
        "one_minus_D": float(1.0 - d),
        "inv_D": float(1.0 / d) if d > 0 else 0.0,
    }


def high_q(counts: ArrayLike) -> float:
    """Hill number of order q=2 (inverse Simpson)."""
    return float(simpson(counts)["inv_D"])


def hill_number(counts: ArrayLike, q: float) -> float:
    """General Hill number."""
    p = _proportions(counts)
    if p.size == 0:
        return 0.0
    if q == 0:
        return low_q(p)
    if q == 1:
        return float(shannon(p)["exp_H"])
    if q == 2:
        return high_q(p)
    return float(np.sum(p**q) ** (1.0 / (1.0 - q)))


def _log_choose(n: np.ndarray | float, k: np.ndarray | float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rarefaction_curve(counts: ArrayLike, steps: int = 50, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Expected rarefaction curve using the exact Hurlbert hypergeometric formula."""
    del seed  # kept for backward compatibility with older plotting calls
    freqs = np.rint(_to_freqs(counts)).astype(int)
    if freqs.size == 0:
        return np.array([0.0]), np.array([0.0])

    n_reads = int(freqs.sum())
    if n_reads <= 0:
        return np.array([0.0]), np.array([0.0])

    n_steps = max(2, min(int(steps), n_reads))
    sample_sizes = np.unique(np.linspace(1, n_reads, num=n_steps, dtype=int))
    denom = _log_choose(float(n_reads), sample_sizes)
    ys: list[float] = []
    for m, log_denom in zip(sample_sizes, denom):
        absent_prob = np.zeros(len(freqs), dtype=float)
        mask = (n_reads - freqs) >= m
        absent_prob[mask] = np.exp(_log_choose((n_reads - freqs[mask]).astype(float), float(m)) - log_denom)
        ys.append(float(np.sum(1.0 - absent_prob)))
    return sample_sizes.astype(float), np.asarray(ys, dtype=float)


def ip_slope(counts: ArrayLike, steps: int = 50) -> float:
    """Initial slope of the normalised rarefaction curve, clipped to [0, 1]."""
    xs, ys = rarefaction_curve(counts, steps=steps)
    richness = max(low_q(counts), 1.0)
    if len(xs) < 2 or xs[-1] <= 0:
        return 0.0
    xn = xs / xs[-1]
    yn = ys / richness
    slopes = np.diff(yn) / np.diff(xn)
    if slopes.size == 0 or not np.isfinite(slopes[0]):
        return 0.0
    return float(np.clip(slopes[0], 0.0, 1.0))


def ipq(counts: ArrayLike, steps: int = 50) -> float:
    """Area under the normalised rarefaction curve, scaled to [0, 1]."""
    xs, ys = rarefaction_curve(counts, steps=steps)
    richness = max(low_q(counts), 1.0)
    if len(xs) < 2 or xs[-1] <= 0:
        return 0.0
    xn = xs / xs[-1]
    yn = ys / richness
    area = float(np.trapezoid(yn, xn))
    return float(np.clip(area, 0.0, 1.0))


def compute_all(counts: ArrayLike, sample_id: str | None = None, steps: int = 50) -> dict[str, float | str | int]:
    """Compute the full diversity summary for one sample."""
    freqs = _to_freqs(counts)
    sh = shannon(freqs)
    si = simpson(freqs)
    return {
        "sample": sample_id if sample_id is not None else "sample",
        "n_clones": int(low_q(freqs)),
        "n_reads": int(freqs.sum()) if freqs.size else 0,
        "low_q": float(low_q(freqs)),
        "high_q": float(high_q(freqs)),
        "ip_slope": float(ip_slope(freqs, steps=steps)),
        "ipq": float(ipq(freqs, steps=steps)),
        "shannon_H": float(sh["H"]),
        "shannon_exp_H": float(sh["exp_H"]),
        "simpson_D": float(si["D"]),
        "simpson_1minusD": float(si["one_minus_D"]),
        "simpson_inv_D": float(si["inv_D"]),
    }
