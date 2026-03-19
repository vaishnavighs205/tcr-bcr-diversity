"""Flexible CSV/TSV loading and column alias resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


COLUMN_ALIASES = {
    "frequency": ["frequency", "count", "counts", "reads", "templates", "clonecount", "clone_count"],
    "v_gene": ["v_gene", "v_call", "vgene", "bestvgene", "bestvhit", "v"],
    "j_gene": ["j_gene", "j_call", "jgene", "bestjhit", "j"],
    "cdr3_aa": ["cdr3_aa", "cdr3", "aminoacid", "junction_aa", "aaseqcdr3", "cdr3aminoacid", "cdr3 amino acid"],
    "chain": ["chain", "receptor", "locus", "chaintype"],
    "sample_id": ["sample_id", "sample", "sampleid", "sample_name"],
    "clone_id": ["clone_id", "clone", "clonotype_id", "sequence_id", "id"],
    "cell_id": ["cell_id", "barcode", "cell_barcode", "cell", "cb"],
}


def _normalise_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to canonical names, case-insensitively."""
    rename_map: dict[str, str] = {}
    normalised = {_normalise_name(col): col for col in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalise_name(alias)
            if key in normalised:
                rename_map[normalised[key]] = canonical
                break
    return df.rename(columns=rename_map).copy()


def validate(df: pd.DataFrame) -> None:
    if "frequency" not in df.columns:
        raise ValueError("Input table must contain a frequency/count column.")
    if "sample_id" not in df.columns:
        raise ValueError("Input table must contain sample identifiers.")
    freq = pd.to_numeric(df["frequency"], errors="coerce")
    if freq.isna().any():
        raise ValueError("Frequency column contains non-numeric or missing values.")
    if (freq < 0).any():
        raise ValueError("Frequency values must be non-negative.")


def load_file(path: str | Path, sample_id: str | None = None) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    df = _resolve_columns(df)

    if "frequency" not in df.columns:
        raise ValueError("Input table must contain a frequency/count column.")

    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce")
    if df["frequency"].isna().any():
        raise ValueError("Frequency column contains non-numeric or missing values.")

    df = df[df["frequency"] > 0].copy()
    df["frequency"] = df["frequency"].astype(float)

    if "sample_id" not in df.columns:
        df["sample_id"] = sample_id if sample_id is not None else path.stem
    elif sample_id is not None:
        df["sample_id"] = sample_id

    if "clone_id" not in df.columns:
        df["clone_id"] = [f"clone_{i + 1:05d}" for i in range(len(df))]

    validate(df)
    return df.reset_index(drop=True)


def load_multiple(paths: Iterable[str | Path], sample_ids: list[str] | None = None) -> pd.DataFrame:
    paths = list(paths)
    if not paths:
        raise ValueError("No input files were provided.")
    if sample_ids is not None and len(sample_ids) != len(paths):
        raise ValueError("sample_ids must match the number of input files.")

    parts = []
    for i, path in enumerate(paths):
        sid = None if sample_ids is None else sample_ids[i]
        parts.append(load_file(path, sample_id=sid))
    out = pd.concat(parts, ignore_index=True, sort=False)
    validate(out)
    return out
