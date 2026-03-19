"""
make_notebook.py — generates the Jupyter notebook as a .ipynb JSON file
without needing Jupyter installed in the build environment.
Run: python make_notebook.py
"""
import json, textwrap

def md(source): return {"cell_type":"markdown","metadata":{},"source":source}
def code(source, outputs=None):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":outputs or [],"source":source}

cells = [

# ── Title ──────────────────────────────────────────────────────────────────
md(textwrap.dedent("""\
# TCR / BCR Receptor Diversity Analysis
### Quantifying T-cell and B-cell repertoire diversity with Hill numbers, rarefaction, and entropy metrics

This notebook walks through a complete analysis pipeline:

1. **Load** repertoire data from CSV/TSV files  
2. **Compute** six diversity metrics per sample  
3. **Visualise** rank-abundance, rarefaction, Hill profile, V-gene usage, CDR3 length  
4. **Compare** groups statistically (permutation tests + FDR correction)  
5. **Paired-chain** analysis for dual-sequenced alpha/beta or heavy/light data  
6. **Export** a self-contained HTML report  

---
""")),

# ── Setup ──────────────────────────────────────────────────────────────────
md("## 0. Setup"),
code("""\
# Install dependencies if needed
# !pip install numpy pandas matplotlib scipy

import sys, warnings
sys.path.insert(0, "..")   # adjust if running from notebooks/
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 110

print("Environment ready.")
"""),

# ── Section 1 ──────────────────────────────────────────────────────────────
md(textwrap.dedent("""\
## 1. Generate synthetic data (or load your own)

The `generate_demo.py` script creates five synthetic samples:
- **Healthy_T / Patient_T / Recovery_T** — TRB chain samples with varying degrees of clonal expansion  
- **Healthy_B / Patient_B** — IGH chain samples  

If you have real data, replace the `paths` list below with your own CSV/TSV files.

### Expected CSV format
```
clone_id, frequency, v_gene, j_gene, cdr3_aa, sample_id
clone_001, 1240, TRBV12-3, TRBJ1-2, CASSLAPGATNEKLFF, Patient_T
...
```
Column names are flexible — see `data_loader.py` for the full alias list.
""")),

code("""\
import subprocess, os
os.chdir("..")  # run from project root

result = subprocess.run(
    ["python", "generate_demo.py"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
"""),

# ── Section 2 ──────────────────────────────────────────────────────────────
md("## 2. Load data"),
code("""\
from data_loader import load_multiple, validate
from pathlib import Path

# Load all demo CSV files
paths = sorted(Path("data").glob("*.csv"))
print(f"Found {len(paths)} files: {[p.name for p in paths]}")

df = load_multiple(paths)
validate(df)
df.head(6)
"""),

code("""\
# Quick overview of clone-size distribution per sample
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, log in zip(axes, [False, True]):
    for s, grp in df.groupby("sample_id"):
        freqs = np.sort(grp["frequency"].values)[::-1][:200]
        ax.plot(np.arange(1, len(freqs)+1), freqs / freqs.sum(), lw=1.5, label=s)
    ax.set_xlabel("Clone rank")
    ax.set_ylabel("Relative frequency")
    ax.set_title(f"Rank-abundance {'(log-log)' if log else '(linear top-200)'}")
    if log:
        ax.set_xscale("log"); ax.set_yscale("log")

axes[0].legend(fontsize=7)
plt.tight_layout()
plt.show()
"""),

# ── Section 3 ──────────────────────────────────────────────────────────────
md(textwrap.dedent("""\
## 3. Compute diversity metrics

All six metrics are computed by `diversity_metrics.compute_all()`:

| Metric | Formula | What it captures |
|---|---|---|
| **Low-Q** | Hill q=0 | Clonal richness (count of distinct clones) |
| **High-Q** | 1/Σpᵢ² | Inverse Simpson — dominant-clone sensitivity |
| **IP Slope** | slope(rarefaction) | Unseen diversity at current depth |
| **IPQ** | AUC(rarefaction) | Sampling saturation (0=unsaturated, 1=saturated) |
| **Shannon H′** | −Σpᵢ ln pᵢ | Balanced richness + evenness |
| **Simpson 1−D** | 1−Σpᵢ² | Probability two reads come from different clones |
""")),

code("""\
from diversity_metrics import compute_all

rows = []
for s, grp in df.groupby("sample_id"):
    rows.append(compute_all(grp["frequency"].values, sample_id=s, steps=50))

summary = pd.DataFrame(rows)

# Display rounded for readability
display_cols = ["sample","n_clones","n_reads","low_q","high_q",
                "ip_slope","ipq","shannon_H","simpson_1minusD"]
summary[display_cols].round(3)
"""),

code("""\
# Visual: metrics as a styled heatmap (pandas styler)
metric_cols = ["low_q","high_q","ip_slope","ipq","shannon_H","simpson_1minusD"]
normed = summary.set_index("sample")[metric_cols]
normed = (normed - normed.mean()) / (normed.std() + 1e-9)

normed.style.background_gradient(cmap="RdYlBu_r", axis=0).format("{:.2f}")
"""),

# ── Section 4 ──────────────────────────────────────────────────────────────
md("## 4. Rarefaction curves and Hill profile"),
code("""\
from plots import plot_rarefaction, plot_hill_profile

fig, ax = plot_rarefaction(df, steps=40)
ax.set_title("Rarefaction curves — all samples")
plt.tight_layout(); plt.show()
"""),

code("""\
fig, ax = plot_hill_profile(df)
plt.tight_layout(); plt.show()
print("""
q=0 -> Low-Q  (every clone counts equally)
q=1 -> Shannon effective diversity
q=2 -> High-Q (inverse Simpson, dominant clones weighted)
""")
"""),

# ── Section 5 ──────────────────────────────────────────────────────────────
md(textwrap.dedent("""\
## 5. V-gene usage and CDR3 length

If your data includes `v_gene` and `cdr3_aa` columns, these plots are
automatically generated. They can reveal:
- Biased V-gene selection during immune response  
- CDR3 length contraction (often seen in antigen-driven expansion)
""")),

code("""\
from plots import plot_vgene_usage, plot_cdr3_length

fig, ax = plot_vgene_usage(df, top_n=12)
if fig: plt.tight_layout(); plt.show()

fig, ax = plot_cdr3_length(df)
if fig: plt.tight_layout(); plt.show()
"""),

# ── Section 6 ──────────────────────────────────────────────────────────────
md(textwrap.dedent("""\
## 6. Statistical comparison across groups

`GroupComparison` runs pairwise tests for each metric:
- **Permutation test** — non-parametric, no distribution assumptions  
- **Mann-Whitney U** — when n ≥ 3 per group  
- **BH FDR correction** — corrects for multiple comparisons across metrics  
- **Effect size r** — rank-biserial correlation (|r| > 0.5 = large effect)

Significance stars: `*` p<0.05, `**` p<0.01, `***` p<0.001
""")),

code("""\
from stats_compare import GroupComparison

groups = {
    "Healthy": [s for s in summary["sample"] if "Healthy" in s],
    "Patient": [s for s in summary["sample"] if "Patient" in s],
}
print("Groups:", {k: v for k, v in groups.items()})

cmp = GroupComparison(summary, groups, n_perm=999)
results = cmp.run_all()

# Show pairwise results sorted by adjusted p-value
results[["group1","group2","metric","mean1","mean2","p_perm","p_perm_adj","sig","effect_r"]]\\
    .sort_values("p_perm_adj").round(4)
"""),

code("""\
# Kruskal-Wallis omnibus test (if you have 3+ groups)
all_groups = {
    "Healthy": [s for s in summary["sample"] if "Healthy" in s],
    "Patient": [s for s in summary["sample"] if "Patient" in s],
    "Recovery":[s for s in summary["sample"] if "Recovery" in s],
}
cmp3 = GroupComparison(summary, all_groups, n_perm=999)
cmp3.run_all()
kw = cmp3.kruskal_wallis()
kw
"""),

code("""\
# Plot: grouped strip + box charts with significance bars
path = cmp.plot_comparison(output_dir="output/")
from IPython.display import Image
Image(str(path))
"""),

code("""\
# Summary table: mean ± std per group
cmp.summary_table()
"""),

# ── Section 7 ──────────────────────────────────────────────────────────────
md(textwrap.dedent("""\
## 7. Paired alpha/beta chain analysis

If you have paired single-cell or barcode-resolved bulk data, use
`paired_chains.pair_chains()` to match productive TRA+TRB (or IGH+IGL/K)
rearrangements from the same cell.

### Generate paired demo data
""")),

code("""\
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)
vg_a = [f"TRAV{i}" for i in range(1, 22)]
vg_b = [f"TRBV{i}" for i in range(1, 16)]
aa = list("ACDEFGHIKLMNPQRSTVWY")

rows = []
for sample in ["Healthy_T", "Patient_T"]:
    # Create pairing bias for Patient: TRAV14 + TRBV12 over-represented
    for cell_i in range(300):
        cid = f"{sample}_cell{cell_i:04d}"
        if sample == "Patient_T" and cell_i < 80:
            va, vb = "TRAV14", "TRBV12"
        else:
            va = rng.choice(vg_a)
            vb = rng.choice(vg_b)
        for chain, vg, freq_max in [("TRA", va, 50), ("TRB", vb, 80)]:
            rows.append({
                "cell_id": cid,
                "chain": chain,
                "v_gene": vg,
                "frequency": int(rng.integers(1, freq_max)),
                "cdr3_aa": "".join(rng.choice(aa, size=rng.integers(10,18))),
                "sample_id": sample,
            })

paired_raw = pd.DataFrame(rows)
print(paired_raw.groupby(["sample_id","chain"]).size().to_string())
paired_raw.head()
"""),

code("""\
from paired_chains import (
    pair_chains, per_chain_diversity,
    plot_vgene_pairing_heatmap, plot_cdr3_length_correlation,
    plot_per_chain_diversity, find_public_clones
)

# Pair TRA + TRB within each cell
paired = pair_chains(paired_raw, chain_a="TRA", chain_b="TRB", cell_col="cell_id")
print(f"Paired cells: {len(paired):,}")
paired.head()
"""),

code("""\
# V-gene pairing heatmap — reveals co-selection bias
for s in paired["sample_id"].unique():
    path = plot_vgene_pairing_heatmap(paired, top_n=10,
                                       output_dir="output/", sample_id=s)
    if path:
        from IPython.display import Image, display
        display(Image(str(path)))
"""),

code("""\
# CDR3 length correlation between paired chains
path = plot_cdr3_length_correlation(paired, output_dir="output/")
if path:
    from IPython.display import Image
    Image(str(path))
"""),

code("""\
# Per-chain diversity (TRA vs TRB side by side)
chain_div = per_chain_diversity(paired_raw)
chain_div[["sample_id","chain","n_clones","low_q","high_q","shannon_H"]].round(3)
"""),

code("""\
path = plot_per_chain_diversity(chain_div, output_dir="output/")
from IPython.display import Image
Image(str(path))
"""),

code("""\
# Public clones — CDR3 sequences shared between Healthy_T and Patient_T
public = find_public_clones(paired_raw, sequence_col="cdr3_aa", min_samples=2)
print(f"Public clones (shared across ≥2 samples): {len(public)}")
public.head(10)
"""),

# ── Section 8 ──────────────────────────────────────────────────────────────
md("## 8. Export full HTML report"),
code("""\
from analyse import run_analysis

# Run the full pipeline including report generation
summary_full = run_analysis(df, output_dir="output/", verbose=True)
print("\\nReport saved to output/diversity_report.html")
"""),

code("""\
# Open report inline (works in JupyterLab)
from IPython.display import IFrame
IFrame("output/diversity_report.html", width="100%", height=700)
"""),

# ── Section 9 ──────────────────────────────────────────────────────────────
md(textwrap.dedent("""\
## 9. Using your own data — quick guide

### Minimal CSV (counts only)
```csv
frequency
1240
890
340
...
```

### Full CSV (recommended)
```csv
clone_id,   frequency, v_gene,    j_gene,   cdr3_aa,              sample_id
clone_001,  1240,      TRBV12-3,  TRBJ1-2,  CASSLAPGATNEKLFF,     Patient_01
clone_002,  890,       TRBV20-1,  TRBJ2-7,  CSARDLGQPQHF,         Patient_01
...
```

### MiXCR export
```bash
mixcr exportClones -f cloneId,cloneCount,allVHitsWithScore,allJHitsWithScore,aaSeqCDR3 clones.clns > mixcr_output.tsv
python analyse.py --input mixcr_output.tsv
```

### VDJtools export
VDJtools output uses `freq`, `count`, `v`, `j`, `CDR3aa` — all resolved automatically.

### Multi-sample
Put each sample in its own file; filenames become sample labels:
```bash
python analyse.py --input data/Pt1.csv data/Pt2.csv data/HC1.csv data/HC2.csv
```

Or include a `sample_id` column in a combined file:
```bash
python analyse.py --input data/all_samples.csv
```
""")),

]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out_path = "notebooks/TCR_BCR_Diversity_Analysis.ipynb"
import os; os.makedirs("notebooks", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Notebook written -> {out_path}")
