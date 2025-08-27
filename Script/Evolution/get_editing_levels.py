#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI tool to process RNA editing tables:
- Filters A->G sites
- Aggregates base counts across replicates
- Computes weighted editing level per site
- Exports a tidy CSV and a BED6 file

Assumptions / Notes
-------------------
1) The input table contains:
   - '#1.Chromosome_ID'   (chromosome)
   - '2.Coordinate'       (1-based genomic coordinate)
   - '3.Strand'           ('+' or '-')
   - '4.Editing_Type'     (e.g., 'A->G')
   - Replicate columns like:
     '6.rep1.RNA_BaseCount[A,C,G,T];Editing_Level',
     '8.rep2.RNA_BaseCount[A,C,G,T];Editing_Level', etc.

2) Replicate cell format can be either:
   'A,C,G,T;editing_level'  OR  'editing_level;A,C,G,T'.
   The parser auto-detects which side contains the 4 integer counts.

3) Editing level definition:
   Strand '+': G / (A + G)
   Strand '-': C / (T + C)

4) BED6 output is 0-based half-open:
   chr, start=position-1, end=position, name=0, score=0, strand
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


def detect_replicate_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect replicate columns that contain base counts and editing level.
    We look for column names containing 'RNA_BaseCount' and ';Editing_Level'.
    """
    candidates = [
        c for c in df.columns
        if "RNA_BaseCount" in c and ";Editing_Level" in c
    ]
    # Keep original order as in the dataframe
    return candidates


_counts_pattern = re.compile(r"^\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*$")


def _parse_counts_cell(value: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse a cell that should contain base counts and an editing level separated by ';'.
    The counts may appear either before or after the semicolon.

    Returns:
        (A, C, G, T) if successful, otherwise None.
    """
    if pd.isna(value):
        return None

    try:
        text = str(value)
        parts = text.split(";")

        # Prefer the part that looks like "A,C,G,T"
        for part in parts:
            if _counts_pattern.match(part.strip()):
                A, C, G, T = [int(x) for x in part.strip().split(",")]
                return A, C, G, T

        # As a last resort, if the entire value is just the counts
        if _counts_pattern.match(text.strip()):
            A, C, G, T = [int(x) for x in text.strip().split(",")]
            return A, C, G, T

    except Exception:
        return None

    return None


def calculate_row_aggregates(
    row: pd.Series,
    replicate_cols: Iterable[str],
    strand_col: str = "3.Strand",
) -> Tuple[int, int, int, int, float, int]:
    """
    Sum base counts across provided replicate columns and compute per-row metrics.

    Returns:
        total_A, total_T, total_C, total_G, editing_level, total_coverage
    """
    total_A = total_C = total_G = total_T = 0

    for col in replicate_cols:
        if col not in row.index:
            continue
        counts = _parse_counts_cell(row[col])
        if counts is None:
            continue
        A, C, G, T = counts
        total_A += A
        total_C += C
        total_G += G
        total_T += T

    # Compute editing level by strand
    strand = str(row.get(strand_col, "+")).strip()
    if strand == "+":
        denom = total_A + total_G
        editing_level = (total_G / denom) if denom > 0 else 0.0
    else:  # '-' strand
        denom = total_T + total_C
        editing_level = (total_C / denom) if denom > 0 else 0.0

    total_coverage = total_A + total_C + total_G + total_T
    return total_A, total_T, total_C, total_G, float(editing_level), total_coverage


def process_rna_editing_table(
    input_file: Path,
    out_dir: Path,
    sep: str = "\t",
    editing_type: str = "A->G",
    replicate_cols: Optional[List[str]] = None,
    min_coverage: int = 0,
    basename: str = "A2IEditingSite",
) -> None:
    """
    End-to-end processing:
      - Read input table
      - Filter by editing type (default 'A->G')
      - Aggregate counts across replicates
      - Compute editing level and total coverage
      - Save CSV and BED6 outputs
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / f"{basename}.csv"
    output_bed = out_dir / f"{basename}.bed"

    logging.info("Reading input: %s", input_file)
    df = pd.read_csv(input_file, sep=sep)

    required_cols = ["#1.Chromosome_ID", "2.Coordinate", "3.Strand", "4.Editing_Type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    logging.info("Filtering rows by editing type == %r", editing_type)
    df = df[df["4.Editing_Type"] == editing_type].copy()

    # Determine replicate columns
    if replicate_cols:
        cols_to_use = [c for c in replicate_cols if c in df.columns]
        missing_rep = [c for c in replicate_cols if c not in df.columns]
        if missing_rep:
            logging.warning("Some replicate columns were not found and will be ignored: %s", missing_rep)
    else:
        cols_to_use = detect_replicate_columns(df)

    if not cols_to_use:
        raise ValueError(
            "No replicate columns were found. "
            "Specify them with --rep-col or ensure your input has columns like "
            "'*.RNA_BaseCount[A,C,G,T];Editing_Level'."
        )
    logging.info("Using replicate columns: %s", cols_to_use)

    # Compute aggregates
    agg_cols = ["Total_A", "Total_T", "Total_C", "Total_G", "EditingLevel", "Total_Coverage"]
    df[agg_cols] = df.apply(
        lambda r: calculate_row_aggregates(r, cols_to_use, strand_col="3.Strand"),
        axis=1, result_type="expand"
    )

    # Optional coverage filter
    if min_coverage > 0:
        before = len(df)
        df = df[df["Total_Coverage"] >= min_coverage].copy()
        logging.info("Coverage filter kept %d/%d rows (min_coverage=%d).", len(df), before, min_coverage)

    # Select and rename columns
    output_columns = [
        "#1.Chromosome_ID", "2.Coordinate", "3.Strand", "4.Editing_Type",
        "Total_A", "Total_T", "Total_C", "Total_G", "Total_Coverage", "EditingLevel"
    ]
    df_out = df[output_columns].rename(columns={
        "#1.Chromosome_ID": "Chr",
        "2.Coordinate": "Position",
        "3.Strand": "Strand",
        "4.Editing_Type": "Editing_Type",
        "Total_A": "A_Count",
        "Total_T": "T_Count",
        "Total_C": "C_Count",
        "Total_G": "G_Count",
        "Total_Coverage": "Total_Coverage",
        "EditingLevel": "EditingLevel",
    })

    # Ensure Position is integer
    try:
        df_out["Position"] = df_out["Position"].astype(int)
    except Exception:
        logging.warning("Could not cast 'Position' to int; keeping original dtype.")

    # Save CSV
    logging.info("Writing CSV: %s", output_csv)
    df_out.to_csv(output_csv, index=False)

    # Build BED6
    bed_df = df_out[["Chr", "Position", "Strand"]].copy()
    bed_df["Start"] = bed_df["Position"].astype(int) - 1
    bed_df["End"] = bed_df["Position"].astype(int)
    bed_df["Name"] = 0
    bed_df["Score"] = 0
    bed_bed6 = bed_df[["Chr", "Start", "End", "Name", "Score", "Strand"]]

    # Save BED6 (tab-separated, no header)
    logging.info("Writing BED6: %s", output_bed)
    bed_bed6.to_csv(output_bed, sep="\t", header=False, index=False)

    logging.info("Done.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process RNA editing result table into tidy CSV and BED6."
    )
    p.add_argument("-i", "--input", required=True, type=Path, help="Path to input table (TSV by default).")
    p.add_argument("-d", "--out-dir", required=True, type=Path, help="Directory to write outputs into.")
    p.add_argument("--sep", default="\t", help=r"Input separator (default: '\t').")
    p.add_argument("--editing-type", default="A->G", help="Editing type to keep (default: 'A->G').")
    p.add_argument(
        "--rep-col", action="append", default=None,
        help=("Replicate column to use; may be repeated. "
              "If omitted, columns containing 'RNA_BaseCount' and ';Editing_Level' are auto-detected.")
    )
    p.add_argument("--min-coverage", type=int, default=0, help="Minimum total coverage to keep a site (default: 0).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        process_rna_editing_table(
        input_file=args.input,
        out_dir=args.out_dir,
        sep=args.sep,
        editing_type=args.editing_type,
        replicate_cols=args.rep_col,
        min_coverage=args.min_coverage,
        basename="A2IEditingSite",
    )
        return 0
    except Exception as e:
        logging.error("Failed: %s", e, exc_info=(args.log_level == "DEBUG"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
