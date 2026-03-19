"""
tests/test_loader_stats.py
Tests for data_loader.py and stats_compare.py
"""
import sys, os, io, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from data_loader import load_file, load_multiple, _resolve_columns
from diversity_metrics import compute_all
from stats_compare import GroupComparison, sig_stars, _bh_correct


# ── data_loader ───────────────────────────────────────────────────────────────

def _write_csv(content: str) -> str:
    """Write content to a temp CSV file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_load_basic_frequency_column():
    path = _write_csv("frequency\n100\n50\n20\n")
    df = load_file(path, sample_id="S1")
    assert len(df) == 3
    assert "frequency" in df.columns
    assert df["sample_id"].iloc[0] == "S1"


def test_load_alias_count():
    path = _write_csv("count\n100\n50\n")
    df = load_file(path, sample_id="S1")
    assert "frequency" in df.columns
    assert df["frequency"].sum() == 150


def test_load_alias_cloneCount():
    path = _write_csv("cloneCount,v_gene\n200,TRBV1\n100,TRBV2\n")
    df = load_file(path, sample_id="S1")
    assert "frequency" in df.columns
    assert "v_gene" in df.columns


def test_load_drops_zero_freq():
    path = _write_csv("frequency\n100\n0\n50\n0\n")
    df = load_file(path, sample_id="S1")
    assert len(df) == 2


def test_load_raises_on_missing_freq_col():
    path = _write_csv("v_gene,cdr3\nTRBV1,CASSL\n")
    with pytest.raises(ValueError, match="frequency"):
        load_file(path)


def test_load_tsv():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
    f.write("frequency\tv_gene\n100\tTRBV1\n50\tTRBV2\n")
    f.close()
    df = load_file(f.name, sample_id="S1")
    assert len(df) == 2
    assert "v_gene" in df.columns


def test_load_multiple():
    p1 = _write_csv("frequency\n100\n50\n")
    p2 = _write_csv("frequency\n200\n30\n")
    df = load_multiple([p1, p2], sample_ids=["A", "B"])
    assert set(df["sample_id"].unique()) == {"A", "B"}
    assert len(df) == 4


def test_load_multiple_mismatched_ids_raises():
    p1 = _write_csv("frequency\n100\n")
    p2 = _write_csv("frequency\n200\n")
    with pytest.raises(ValueError):
        load_multiple([p1, p2], sample_ids=["A", "B", "C"])


def test_resolve_columns_case_insensitive():
    df = pd.DataFrame({"Count": [10, 5], "V_Gene": ["TRBV1", "TRBV2"]})
    resolved = _resolve_columns(df)
    assert "frequency" in resolved.columns
    assert "v_gene" in resolved.columns


# ── stats_compare ─────────────────────────────────────────────────────────────

@pytest.fixture
def two_group_summary():
    rng = np.random.default_rng(1)
    rows = []
    # Healthy: high diversity
    for i in range(4):
        freqs = rng.dirichlet(np.ones(300) * 0.3) * 5000
        rows.append(compute_all(freqs, sample_id=f"H{i}", steps=10))
    # Patient: low diversity (clonally expanded)
    for i in range(4):
        freqs = rng.dirichlet(np.ones(30) * 3.0) * 5000
        rows.append(compute_all(freqs, sample_id=f"P{i}", steps=10))
    return pd.DataFrame(rows)


def test_sig_stars():
    assert sig_stars(0.0001) == "***"
    assert sig_stars(0.005)  == "**"
    assert sig_stars(0.03)   == "*"
    assert sig_stars(0.1)    == "ns"


def test_bh_correct_length():
    pvals = np.array([0.01, 0.04, 0.2, 0.5])
    adj = _bh_correct(pvals)
    assert len(adj) == 4


def test_bh_correct_monotone():
    pvals = np.array([0.001, 0.01, 0.05, 0.1])
    adj = _bh_correct(pvals)
    assert np.all(adj >= pvals - 1e-12)   # adjusted >= raw (BH is conservative)


def test_group_comparison_results_shape(two_group_summary):
    groups = {
        "Healthy": [f"H{i}" for i in range(4)],
        "Patient": [f"P{i}" for i in range(4)],
    }
    cmp = GroupComparison(two_group_summary, groups, n_perm=99)
    results = cmp.run_all()
    # 7 default metrics × 1 pair = 7 rows
    assert len(results) == 7
    assert "p_perm_adj" in results.columns
    assert "sig" in results.columns


def test_group_comparison_detects_difference(two_group_summary):
    """Healthy vs Patient should show significant difference in at least low_q."""
    groups = {
        "Healthy": [f"H{i}" for i in range(4)],
        "Patient": [f"P{i}" for i in range(4)],
    }
    cmp = GroupComparison(two_group_summary, groups, n_perm=499)
    results = cmp.run_all()
    low_q_row = results[results["metric"] == "low_q"].iloc[0]
    assert low_q_row["p_perm"] < 0.05   # should be clearly significant


def test_kruskal_wallis_returns_df(two_group_summary):
    groups = {
        "Healthy": [f"H{i}" for i in range(4)],
        "Patient": [f"P{i}" for i in range(4)],
    }
    cmp = GroupComparison(two_group_summary, groups, n_perm=49)
    kw = cmp.kruskal_wallis()
    assert isinstance(kw, pd.DataFrame)
    assert "H_stat" in kw.columns
    assert "p_value" in kw.columns


def test_summary_table(two_group_summary):
    groups = {
        "Healthy": [f"H{i}" for i in range(4)],
        "Patient": [f"P{i}" for i in range(4)],
    }
    cmp = GroupComparison(two_group_summary, groups, n_perm=49)
    cmp.run_all()
    tbl = cmp.summary_table()
    assert "Healthy" in tbl.columns
    assert "Patient" in tbl.columns


def test_unmatched_samples_warning(two_group_summary):
    groups = {
        "Healthy": [f"H{i}" for i in range(4)],
        "Patient": [f"P{i}" for i in range(4)],
        "Ghost":   ["nonexistent_sample"],
    }
    with pytest.warns(UserWarning, match="not assigned"):
        cmp = GroupComparison(two_group_summary, groups, n_perm=9)
