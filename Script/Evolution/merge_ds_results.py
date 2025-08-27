#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge dsRNA structure results with per-site editing levels.

- Reads:
    1) editing level table (A2IEditingSite.csv)
    2) all_data_results.csv (from get_ds_with_majority_ES.py)
- Expands rows by:
    a) relative_positions: (local_position, genomic_position)
    b) A_20d: (genomic_position, local_position, editing_site)

For each expanded entry, it joins the matching row from the editing table using
(Chr, Position, Strand) and writes a combined CSV.

Notes:
- Assumes editing table has columns: Chr, Position, Strand, A_Count, C_Count, G_Count, T_Count, EditingLevel, Total_Coverage, ...
- Assumes all_data_results.csv has columns: chr, strand, relative_positions, A_20d, ...
- All comments are in English as requested.
"""

import os
import ast
import argparse
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Global editing table index for workers
_EDIT_IDX: pd.DataFrame | None = None


def _prepare_edit_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure types and create a MultiIndex on (Chr, Position, Strand)."""
    needed = ["Chr", "Position", "Strand"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Editing table missing required columns: {missing}")

    df = df.copy()
    # Normalize dtypes
    df["Chr"] = df["Chr"].astype(str)
    # Robust int casting for Position
    df = df[pd.notnull(df["Position"])]
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df[pd.notnull(df["Position"])]
    df["Position"] = df["Position"].astype(int)
    df["Strand"] = df["Strand"].astype(str)

    # Set MultiIndex for fast lookup
    df = df.set_index(["Chr", "Position", "Strand"], drop=False)
    return df


def _init_worker(edit_idx: pd.DataFrame):
    """Initializer for each worker process—sets the global editing index."""
    global _EDIT_IDX
    _EDIT_IDX = edit_idx


def _safe_eval_list(val: Any) -> list:
    """Safely parse a list literal from CSV cell; return [] on NA/empty/invalid."""
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s or s == "nan":
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _join_edit_row(chr_val: str, pos_val: int, strand_val: str) -> Dict[str, Any] | None:
    """Lookup a row in the global editing index; return dict or None."""
    global _EDIT_IDX
    if _EDIT_IDX is None:
        raise RuntimeError("Editing index is not initialized in worker.")
    key = (str(chr_val), int(pos_val), str(strand_val))
    try:
        row = _EDIT_IDX.loc[key]
    except KeyError:
        return None
    # loc with MultiIndex returns a Series (single row)
    if isinstance(row, pd.Series):
        return row.to_dict()
    # In very rare cases there might be duplicated keys; take the first
    return row.iloc[0].to_dict()


def _process_single_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a single input row using 'relative_positions' and 'A_20d' and join editing info."""
    expanded_rows: List[Dict[str, Any]] = []

    # Normalize minimal fields
    row_chr = str(row.get("chr"))
    row_strand = str(row.get("strand"))

    # --- Handle relative_positions: [(local_position, genomic_position), ...]
    rel_positions = _safe_eval_list(row.get("relative_positions"))
    for tup in rel_positions:
        try:
            local_position, genomic_position = tup
            genomic_position = int(genomic_position)
        except Exception:
            continue

        matched = _join_edit_row(row_chr, genomic_position, row_strand)
        if matched is None:
            continue

        combined = dict(row)  # include all original columns
        combined.update(matched)  # add all editing table columns
        combined["Local_Position"] = local_position  # keep reported local index
        expanded_rows.append(combined)

    # --- Handle A_20d: [(genomic_position, local_position, editing_site), ...]
    a20_list = _safe_eval_list(row.get("A_20d"))
    for tup in a20_list:
        try:
            genomic_position, local_position_ad, editing_site = tup
            genomic_position = int(genomic_position)
            editing_site = int(editing_site)
        except Exception:
            continue

        matched = _join_edit_row(row_chr, editing_site, row_strand)
        if matched is None:
            continue

        combined = dict(row)
        combined.update(matched)
        combined["Position"] = genomic_position              # set to the adenosine position
        combined["Local_Position"] = local_position_ad       # local relative index
        combined["Editing_Type"] = str(editing_site)         # store the linked editing site
        # Override counts/level to neutral defaults for A candidates
        combined["A_Count"] = None
        combined["G_Count"] = None
        combined["C_Count"] = None
        combined["T_Count"] = None
        combined["EditingLevel"] = 0
        expanded_rows.append(combined)

    return expanded_rows


def merge_tables_with_relative_positions_parallel(
    editing_level_csv: Path,
    all_data_results_csv: Path,
    output_csv: Path,
    workers: int = 0,
    chunksize: int | None = None,
) -> None:
    """Driver function: load inputs, run multiprocessing expansion, write output."""
    logging.info("Reading editing level table: %s", editing_level_csv)
    edit_df = pd.read_csv(editing_level_csv)
    edit_idx = _prepare_edit_index(edit_df)

    logging.info("Reading dsRNA results table: %s", all_data_results_csv)
    table2 = pd.read_csv(all_data_results_csv)

    # Dict records are cheap to pickle between processes
    records: List[Dict[str, Any]] = table2.to_dict(orient="records")
    n = len(records)
    if n == 0:
        logging.warning("No rows in dsRNA results table. Writing empty output.")
        pd.DataFrame([]).to_csv(output_csv, index=False)
        return

    # Determine pool size and chunking
    if workers <= 0:
        workers = max(1, min(15, (cpu_count() or 1)))
    if chunksize is None:
        chunksize = max(1, n // (workers * 4) or 1)

    logging.info("Starting pool: workers=%d, chunksize=%d, rows=%d", workers, chunksize, n)
    with Pool(processes=workers, initializer=_init_worker, initargs=(edit_idx,)) as pool:
        # Use imap for streaming
        results_iter = pool.imap(_process_single_row, records, chunksize=chunksize)
        expanded_rows: List[Dict[str, Any]] = []
        for lst in results_iter:
            if lst:
                expanded_rows.extend(lst)

    logging.info("Expanded rows: %d", len(expanded_rows))
    out_df = pd.DataFrame(expanded_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    logging.info("Wrote output: %s", output_csv)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge dsRNA structure results with per-site editing levels (parallel)."
    )
    p.add_argument(
        "-e", "--editing-level",
        required=True,
        type=Path,
        help="Path to A2IEditingSite.csv (editing level table)."
    )
    p.add_argument(
        "-a", "--all-data-results",
        required=True,
        type=Path,
        help="Path to all_data_results.csv (from get_ds_with_majority_ES.py)."
    )
    p.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        help="Output CSV path."
    )
    p.add_argument(
        "-w", "--workers",
        type=int,
        default=0,
        help="Number of worker processes (default: auto, up to 15)."
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Chunk size per worker task (default: auto)."
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity."
    )
    return p


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        merge_tables_with_relative_positions_parallel(
            editing_level_csv=args.editing_level,
            all_data_results_csv=args.all_data_results,
            output_csv=args.output,
            workers=args.workers,
            chunksize=args.chunksize,
        )
        return 0
    except Exception as e:
        logging.error("Failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
