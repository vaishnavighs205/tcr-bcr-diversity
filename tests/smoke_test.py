from pathlib import Path

from analyse import run_analysis
from data_loader import load_multiple
from generate_demo import generate_demo
from paired_chains import find_public_clones, per_chain_diversity
from stats_compare import GroupComparison


def test_smoke(tmp_path: Path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "output"
    paths = generate_demo(data_dir)
    df = load_multiple(paths)
    summary = run_analysis(df, output_dir=out_dir)
    assert not summary.empty
    assert (out_dir / "diversity_report.html").exists()

    groups = {
        "Healthy": [s for s in summary["sample"] if "Healthy" in s],
        "Patient": [s for s in summary["sample"] if "Patient" in s],
    }
    cmp = GroupComparison(summary, groups, n_perm=99)
    results = cmp.run_all()
    assert not results.empty

    chain_div = per_chain_diversity(df)
    assert not chain_div.empty
    public = find_public_clones(df, min_samples=1)
    assert not public.empty
