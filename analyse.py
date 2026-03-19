"""End-to-end analysis runner and CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_loader import load_multiple, validate
from diversity_metrics import compute_all
from plots import plot_cdr3_length, plot_hill_profile, plot_rarefaction, plot_vgene_usage
from report import write_html_report


def _save_plot(result: tuple[object, object], path: Path) -> bool:
    fig, _ = result
    if fig is None:
        return False
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.clf()
    return True


def _summary_from_df(df: pd.DataFrame, steps: int = 50) -> pd.DataFrame:
    rows = []
    for sample, grp in df.groupby("sample_id", sort=True):
        rows.append(compute_all(grp["frequency"].values, sample_id=sample, steps=steps))
    return pd.DataFrame(rows)


def run_analysis(
    data: pd.DataFrame | Iterable[str | Path],
    output_dir: str | Path = "output",
    sample_ids: list[str] | None = None,
    steps: int = 50,
    top_vgenes: int = 12,
    verbose: bool = True,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        validate(df)
    else:
        df = load_multiple(data, sample_ids=sample_ids)

    summary = _summary_from_df(df, steps=steps)
    summary_path = output_dir / "diversity_summary.csv"
    legacy_path = output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(legacy_path, index=False)

    images: list[tuple[str, Path]] = []

    rarefaction_path = output_dir / "rarefaction.png"
    if _save_plot(plot_rarefaction(df, steps=max(10, min(steps, 80))), rarefaction_path):
        images.append(("Rarefaction curves", rarefaction_path))

    hill_path = output_dir / "hill_profile.png"
    if _save_plot(plot_hill_profile(df), hill_path):
        images.append(("Hill diversity profile", hill_path))

    vgene_path = output_dir / "vgene_usage.png"
    if _save_plot(plot_vgene_usage(df, top_n=top_vgenes), vgene_path):
        images.append(("V-gene usage", vgene_path))

    cdr3_path = output_dir / "cdr3_length.png"
    if _save_plot(plot_cdr3_length(df), cdr3_path):
        images.append(("CDR3 length distribution", cdr3_path))

    report_path = write_html_report(summary, output_dir, images)

    if verbose:
        print(f"Saved summary: {summary_path}")
        print(f"Saved report: {report_path}")
        for _, image_path in images:
            print(f"Saved plot: {image_path}")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TCR/BCR diversity analysis on CSV/TSV repertoire files.")
    parser.add_argument("--input", nargs="+", required=True, help="Input CSV/TSV files.")
    parser.add_argument("--output", default="output", help="Output directory.")
    parser.add_argument("--sample-ids", nargs="+", help="Optional sample IDs corresponding to input files.")
    parser.add_argument("--steps", type=int, default=50, help="Rarefaction resolution.")
    parser.add_argument("--top-vgenes", type=int, default=12, help="Number of V-genes to show in the bar plot.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_analysis(
        args.input,
        output_dir=args.output,
        sample_ids=args.sample_ids,
        steps=args.steps,
        top_vgenes=args.top_vgenes,
        verbose=not args.quiet,
    )
