#!/usr/bin/env python3
"""Select the exact published human benchmark records from the raw GTEx tables.

The binary labels are derived from EditingIndex (yes >= 15; no < 1), and each
tissue is downsampled before splitting. Random sampling can differ across R
versions and input order, so the exact selected record multiset is stored in
``published_site_selection.tsv.gz``. This script
matches that immutable manifest against the five downloaded raw tables and
writes one exact, unsplit pool per context.

No released data are overwritten.  The output directory must not exist.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


TISSUE_FILES = {
    "Artery": "Artery_Site_in_PairAlu_cov100.csv",
    "Brain": "Brain_Site_in_PairAlu_cov100.csv",
    "Liver": "Liver_Site_in_PairAlu_cov100.csv",
    "MuscleSkeletal": "MuscleSkeletal_Site_in_PairAlu_cov100.csv",
    "Combined": "Combined_Site_in_PairAlu_cov100.csv",
}


def record_fingerprint(
    structure: str, left: str, right: str, label: str
) -> str:
    payload = json.dumps(
        [structure, left, right, label],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> dict[str, Counter[str]]:
    selected = {tissue: Counter() for tissue in TISSUE_FILES}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"tissue", "record_sha256", "count"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            tissue = row["tissue"]
            if tissue not in selected:
                raise RuntimeError(
                    f"{path}:{row_number}: unknown tissue {tissue!r}"
                )
            fingerprint = row["record_sha256"].lower()
            if len(fingerprint) != 64:
                raise RuntimeError(
                    f"{path}:{row_number}: invalid SHA256 {fingerprint!r}"
                )
            count = int(row["count"])
            if count < 1:
                raise RuntimeError(
                    f"{path}:{row_number}: count must be positive"
                )
            selected[tissue][fingerprint] += count
    return selected


def label(editing_index: str, yes_cutoff: float, no_cutoff: float):
    try:
        value = float(editing_index)
    except (TypeError, ValueError):
        return None
    if value >= yes_cutoff:
        return "yes"
    if value < no_cutoff:
        return "no"
    return None


def select_tissue(
    tissue: str,
    raw_path: Path,
    expected: Counter[str],
    output_path: Path,
    yes_cutoff: float,
    no_cutoff: float,
) -> dict:
    remaining = expected.copy()
    selected_rows = []
    raw_rows = 0
    eligible_rows = 0

    with raw_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"structure", "L", "R", "EditingIndex"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{raw_path}: missing columns {sorted(missing)}")

        for source_row, row in enumerate(reader, start=2):
            raw_rows += 1
            yes_no = label(row.get("EditingIndex"), yes_cutoff, no_cutoff)
            if yes_no is None:
                continue
            eligible_rows += 1
            structure = (row.get("structure") or "").strip()
            left = (row.get("L") or "").strip().upper()
            right = (row.get("R") or "").strip().upper()
            # CSV serialization represents an empty arm at a terminal adenosine
            # as the literal string "NA". Preserve
            # that representation so the pre-filter selection manifest can be
            # verified; build_human_global_split.py then reports and removes
            # these length-mismatched records.
            if not left:
                left = "NA"
            if not right:
                right = "NA"
            fingerprint = record_fingerprint(
                structure, left, right, yes_no
            )
            if remaining[fingerprint] < 1:
                continue
            selected_rows.append(
                {
                    "structure": structure,
                    "L": left,
                    "R": right,
                    "yes_no": yes_no,
                    "EditingIndex": row.get("EditingIndex"),
                    "source_row": source_row,
                }
            )
            remaining[fingerprint] -= 1
            if remaining[fingerprint] == 0:
                del remaining[fingerprint]

    if remaining:
        missing_count = sum(remaining.values())
        examples = ", ".join(sorted(remaining)[:5])
        raise RuntimeError(
            f"{tissue}: {missing_count} published records were not found in "
            f"{raw_path}; first missing fingerprints: {examples}"
        )

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "structure",
                "L",
                "R",
                "yes_no",
                "EditingIndex",
                "source_row",
            ],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    classes = Counter(row["yes_no"] for row in selected_rows)
    return {
        "raw_rows": raw_rows,
        "eligible_rows": eligible_rows,
        "manifest_records": sum(expected.values()),
        "selected_rows": len(selected_rows),
        "selected_yes": classes["yes"],
        "selected_no": classes["no"],
        "missing_manifest_records": 0,
        "output": str(output_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory containing the five downloaded GTEx CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for <Tissue>.balanced.csv and selection_report.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=here / "published_site_selection.tsv.gz",
    )
    parser.add_argument("--yes-cutoff", type=float, default=15.0)
    parser.add_argument("--no-cutoff", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.yes_cutoff <= args.no_cutoff:
        raise RuntimeError("--yes-cutoff must be greater than --no-cutoff")
    if args.output_dir.exists():
        raise RuntimeError(
            f"refusing to overwrite existing output path: {args.output_dir}"
        )

    manifest = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True)
    report = {}
    try:
        for tissue, filename in TISSUE_FILES.items():
            raw_path = args.data_dir / filename
            if not raw_path.exists():
                raise RuntimeError(f"missing required raw table: {raw_path}")
            output_path = args.output_dir / f"{tissue}.balanced.csv"
            report[tissue] = select_tissue(
                tissue,
                raw_path,
                manifest[tissue],
                output_path,
                args.yes_cutoff,
                args.no_cutoff,
            )
            print(
                f"[{tissue}] selected {report[tissue]['selected_rows']} "
                f"published rows"
            )

        payload = {
            "status": "PASS",
            "label_definition": {
                "yes": f"EditingIndex >= {args.yes_cutoff:g}",
                "no": f"EditingIndex < {args.no_cutoff:g}",
                "excluded": (
                    f"{args.no_cutoff:g} <= EditingIndex < "
                    f"{args.yes_cutoff:g}"
                ),
            },
            "manifest": str(args.manifest.resolve()),
            "tissues": report,
        }
        (args.output_dir / "selection_report.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    except Exception:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise

    print(f"PASS: exact published pools written to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
