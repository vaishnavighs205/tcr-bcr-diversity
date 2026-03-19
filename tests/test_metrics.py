"""
tests/test_metrics.py
Pytest unit tests for diversity_metrics.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from diversity_metrics import (
    low_q, high_q, ip_slope, ipq,
    shannon, simpson, compute_all,
    _proportions,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def uniform():
    """10 clones each with the same count — maximally even."""
    return np.ones(10) * 100

@pytest.fixture
def monopoly():
    """One dominant clone with 999, one rare with 1."""
    return np.array([999.0, 1.0])

@pytest.fixture
def realistic():
    """Power-law-like counts, 50 clones."""
    rng = np.random.default_rng(0)
    return rng.dirichlet(np.ones(50) * 0.3) * 10_000


# ── _proportions ─────────────────────────────────────────────────────────────

def test_proportions_sum_to_one(realistic):
    p = _proportions(realistic)
    assert abs(p.sum() - 1.0) < 1e-9

def test_proportions_drops_zeros():
    counts = np.array([10.0, 0.0, 5.0, 0.0, 1.0])
    p = _proportions(counts)
    assert len(p) == 3

def test_proportions_single_clone():
    p = _proportions(np.array([42.0]))
    assert p[0] == pytest.approx(1.0)


# ── low_q ─────────────────────────────────────────────────────────────────────

def test_low_q_uniform(uniform):
    assert low_q(uniform) == 10.0

def test_low_q_monopoly(monopoly):
    # Both clones present — richness is 2
    assert low_q(monopoly) == 2.0

def test_low_q_scales_with_richness():
    a = low_q(np.ones(5))
    b = low_q(np.ones(50))
    assert b > a


# ── high_q ────────────────────────────────────────────────────────────────────

def test_high_q_uniform_equals_richness(uniform):
    # For uniform distribution, inv-Simpson == n
    assert high_q(uniform) == pytest.approx(10.0, rel=1e-6)

def test_high_q_monopoly_near_one(monopoly):
    # Almost all reads in one clone → effective diversity ≈ 1
    assert high_q(monopoly) < 1.01

def test_high_q_less_than_low_q_for_skewed(realistic):
    # Skewed distribution: high_q < low_q (dominance reduces effective diversity)
    assert high_q(realistic) < low_q(realistic)


# ── shannon ───────────────────────────────────────────────────────────────────

def test_shannon_uniform_maximum(uniform):
    result = shannon(uniform)
    expected_H = np.log(10)
    assert result["H"] == pytest.approx(expected_H, rel=1e-6)
    assert result["exp_H"] == pytest.approx(10.0, rel=1e-6)

def test_shannon_monopoly_near_zero(monopoly):
    result = shannon(monopoly)
    assert result["H"] < 0.02   # near-zero entropy

def test_shannon_exp_H_between_1_and_richness(realistic):
    s = shannon(realistic)
    assert 1.0 < s["exp_H"] < low_q(realistic)


# ── simpson ───────────────────────────────────────────────────────────────────

def test_simpson_uniform(uniform):
    s = simpson(uniform)
    # For n equal clones: D = 1/n
    assert s["D"] == pytest.approx(1.0 / 10, rel=1e-6)
    assert s["one_minus_D"] == pytest.approx(9.0 / 10, rel=1e-6)
    assert s["inv_D"] == pytest.approx(10.0, rel=1e-6)

def test_simpson_monopoly(monopoly):
    s = simpson(monopoly)
    assert s["D"] > 0.99
    assert s["one_minus_D"] < 0.01

def test_simpson_inv_D_equals_high_q(realistic):
    assert simpson(realistic)["inv_D"] == pytest.approx(high_q(realistic), rel=1e-6)


# ── ip_slope ──────────────────────────────────────────────────────────────────

def test_ip_slope_range(realistic):
    s = ip_slope(realistic, steps=20)
    # Slope must be between 0 and 1 for a normalised curve
    assert 0.0 < s <= 1.0

def test_ip_slope_higher_for_rich(uniform, monopoly):
    # Uniform (10 clones) should have steeper slope than monopoly at same read depth
    # (monopoly saturates almost immediately)
    s_uniform = ip_slope(np.ones(10) * 100, steps=20)
    s_monopoly = ip_slope(np.array([999.0, 1.0]), steps=20)
    assert s_monopoly < s_uniform


# ── ipq ───────────────────────────────────────────────────────────────────────

def test_ipq_range(realistic):
    val = ipq(realistic, steps=20)
    assert 0.0 < val <= 1.0

def test_ipq_monopoly_moderate(monopoly):
    # Monopoly: rarefaction curve saturates almost instantly → AUC near 1
    val = ipq(monopoly, steps=20)
    assert 0.5 < val < 0.95

def test_ipq_diverse_higher_than_monopoly(realistic, monopoly):
    val_div = ipq(realistic, steps=20)
    val_mono = ipq(monopoly, steps=20)
    assert val_div > val_mono


# ── compute_all ───────────────────────────────────────────────────────────────

def test_compute_all_keys(realistic):
    result = compute_all(realistic, sample_id="test", steps=10)
    required = {
        "sample", "n_clones", "n_reads",
        "low_q", "high_q", "ip_slope", "ipq",
        "shannon_H", "shannon_exp_H",
        "simpson_D", "simpson_1minusD", "simpson_inv_D",
    }
    assert required.issubset(set(result.keys()))

def test_compute_all_sample_label(realistic):
    result = compute_all(realistic, sample_id="MySample", steps=10)
    assert result["sample"] == "MySample"

def test_compute_all_n_reads(realistic):
    result = compute_all(realistic, sample_id="x", steps=10)
    assert result["n_reads"] == pytest.approx(int(realistic.sum()), rel=1e-3)

def test_compute_all_no_zeros():
    counts = np.array([10.0, 0.0, 5.0, 0.0, 3.0])
    result = compute_all(counts, sample_id="x", steps=5)
    assert result["n_clones"] == 3   # zeros dropped

def test_compute_all_proportions_input():
    """Should work whether input is raw counts or proportions."""
    counts = np.array([500.0, 300.0, 150.0, 50.0])
    props  = counts / counts.sum()
    r1 = compute_all(counts, sample_id="x", steps=5)
    r2 = compute_all(props,  sample_id="x", steps=5)
    # Shannon and Simpson should be identical; low_q richness also
    assert r1["shannon_H"]       == pytest.approx(r2["shannon_H"],       rel=1e-5)
    assert r1["simpson_1minusD"] == pytest.approx(r2["simpson_1minusD"], rel=1e-5)
    assert r1["low_q"] == r2["low_q"]
