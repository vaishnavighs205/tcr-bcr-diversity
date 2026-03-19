"""Generate synthetic demo data and run the full analysis pipeline."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from analyse import run_analysis


AMINO = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def _rand_cdr3(rng: np.random.Generator, length: int) -> str:
    core = "".join(rng.choice(AMINO, size=max(3, length - 2)))
    return f"C{core}F"


def _sample_table(sample_id: str, chain: str, n_clones: int, total_reads: int, dominant: float, rng: np.random.Generator) -> pd.DataFrame:
    alpha = np.full(n_clones, 1.0)
    alpha[0] = dominant
    probs = rng.dirichlet(alpha)
    counts = np.maximum(1, rng.multinomial(total_reads, probs))
    counts[0] += total_reads - counts.sum()

    prefixes = {"TRA": "TRAV", "TRB": "TRBV", "IGH": "IGHV", "IGL": "IGLV", "IGK": "IGKV"}
    jprefix = {"TRA": "TRAJ", "TRB": "TRBJ", "IGH": "IGHJ", "IGL": "IGLJ", "IGK": "IGKJ"}

    rows = []
    for i, count in enumerate(counts, start=1):
        rows.append(
            {
                "clone_id": f"{sample_id}_clone_{i:04d}",
                "frequency": int(count),
                "v_gene": f"{prefixes.get(chain, 'V')}{rng.integers(1, 31)}-{rng.integers(1, 4)}",
                "j_gene": f"{jprefix.get(chain, 'J')}{rng.integers(1, 7)}",
                "cdr3_aa": _rand_cdr3(rng, int(rng.integers(10, 22))),
                "sample_id": sample_id,
                "chain": chain,
            }
        )
    return pd.DataFrame(rows)


def generate_demo(data_dir: str | Path = "data", output_dir: str | Path = "output", seed: int = 42) -> list[Path]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    specs = [
        ("Healthy_T", "TRB", 220, 26000, 0.8),
        ("Patient_T", "TRB", 120, 26000, 6.0),
        ("Recovery_T", "TRB", 170, 26000, 2.4),
        ("Healthy_B", "IGH", 210, 24000, 1.0),
        ("Patient_B", "IGH", 95, 24000, 5.5),
    ]

    paths: list[Path] = []
    for sample_id, chain, n_clones, total_reads, dominant in specs:
        table = _sample_table(sample_id, chain, n_clones, total_reads, dominant, rng)
        path = data_dir / f"{sample_id}.csv"
        table.to_csv(path, index=False)
        paths.append(path)

    run_analysis(paths, output_dir=output_dir, verbose=False)
    return paths


if __name__ == "__main__":
    created = generate_demo()
    print("Generated demo files:")
    for path in created:
        print(f" - {path}")
    print("Generated output/diversity_report.html")
