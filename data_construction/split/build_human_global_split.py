#!/usr/bin/env python3
"""Build the human benchmark from balanced site pools and the canonical map.

Exact input files are produced by
``human_alu/select_published_site_pool.py``:

    <balanced-dir>/{Artery,Brain,Liver,MuscleSkeletal,Combined}.balanced.csv

Each valid full substrate (``L + "A" + R``) receives a stable integer pair_id
by sorting the substrate sequences. The shipped ``global_pair_split.json`` is
then applied to all five contexts. Consequently, a substrate is assigned to the
same train/validation/test partition everywhere it occurs.

``human_alu/build_cross_splits.R`` documents the label-and-balance procedure.
The selector plus its immutable manifest make the sampled record set invariant
to R version and input ordering.

The script never overwrites a non-empty output directory. It reports malformed
rows rather than silently placing them in a split. With ``--canonical-reference``
it also verifies that the reconstructed split has exactly the same records as
the data shipped with the paper (comparison is order independent).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


TISSUES = ("Artery", "Brain", "Liver", "MuscleSkeletal", "Combined")
SPLITS = ("train", "valid", "test")
SYSTEM_MESSAGE = (
    "Predict if the central adenosine (A) in the given RNA sequence context "
    "within an Alu element will be edited to inosine (I) by ADAR enzymes."
)


def record_json(structure: str, left: str, right: str, label: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": (
                    f"L:{left}, A:A, R:{right}, "
                    f"Alu Vienna Structure:{structure}"
                ),
            },
            {"role": "assistant", "content": label},
        ]
    }


def canonical_line(record: dict) -> str:
    return json.dumps(record)


def parse_reference_line(line: str) -> tuple[str, str, str, str]:
    record = json.loads(line)
    user = next(x["content"] for x in record["messages"] if x["role"] == "user")
    label = next(
        x["content"].strip().lower()
        for x in record["messages"]
        if x["role"] == "assistant"
    )
    left = user.split("L:", 1)[1].split(", A:", 1)[0]
    right = user.split(", A:A, R:", 1)[1].split(
        ", Alu Vienna Structure:", 1
    )[0]
    structure = user.split(", Alu Vienna Structure:", 1)[1]
    return structure, left, right, label


def row_fingerprint(structure: str, left: str, right: str, label: str) -> str:
    payload = json.dumps(
        [structure, left, right, label],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def read_balanced_pool(path: Path, invalid_policy: str):
    valid = []
    invalid = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"structure", "L", "R", "yes_no"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            structure = (row.get("structure") or "").strip()
            left = (row.get("L") or "").strip().upper()
            right = (row.get("R") or "").strip().upper()
            label = (row.get("yes_no") or "").strip().lower()
            reasons = []
            if label not in {"yes", "no"}:
                reasons.append(f"invalid label {label!r}")
            sequence = left + "A" + right
            if len(sequence) != len(structure):
                reasons.append(
                    f"sequence length {len(sequence)} != structure length {len(structure)}"
                )
            if not sequence:
                reasons.append("empty sequence")
            if reasons:
                invalid.append(
                    {
                        "file": str(path),
                        "row": row_number,
                        "reasons": "; ".join(reasons),
                        "sequence_length": len(sequence),
                        "structure_length": len(structure),
                    }
                )
                continue
            valid.append(
                {
                    "structure": structure,
                    "L": left,
                    "R": right,
                    "yes_no": label,
                    "substrate": sequence,
                }
            )
    if invalid and invalid_policy == "error":
        first = invalid[0]
        raise RuntimeError(
            f"{path}: {len(invalid)} invalid rows; first is row {first['row']}: "
            f"{first['reasons']}"
        )
    return valid, invalid


def load_split_map(path: Path, expected_ids: set[str]) -> dict[str, str]:
    mapping = json.loads(path.read_text())
    bad_values = {value for value in mapping.values() if value not in SPLITS}
    if bad_values:
        raise RuntimeError(f"{path}: invalid split values {sorted(bad_values)}")
    keys = set(mapping)
    if keys != expected_ids:
        missing = sorted(expected_ids - keys, key=int)
        extra = sorted(keys - expected_ids, key=int)
        raise RuntimeError(
            f"{path}: map/substrate mismatch; missing IDs={missing[:10]}, "
            f"extra IDs={extra[:10]}, map={len(keys)}, substrates={len(expected_ids)}"
        )
    return mapping


def multiset_for_jsonl(path: Path) -> Counter[str]:
    values: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            structure, left, right, label = parse_reference_line(line)
            values[row_fingerprint(structure, left, right, label)] += 1
    return values


def compare_to_reference(output_dir: Path, reference: Path) -> dict:
    report = {}
    failures = []
    for tissue in TISSUES:
        report[tissue] = {}
        for split in SPLITS:
            observed_path = output_dir / tissue / f"{split}.jsonl"
            expected_path = reference / tissue / f"{split}.jsonl"
            if not expected_path.exists():
                raise RuntimeError(f"canonical reference is missing {expected_path}")
            observed = multiset_for_jsonl(observed_path)
            expected = multiset_for_jsonl(expected_path)
            missing = sum((expected - observed).values())
            extra = sum((observed - expected).values())
            status = "PASS" if not missing and not extra else "FAIL"
            report[tissue][split] = {
                "observed_rows": sum(observed.values()),
                "reference_rows": sum(expected.values()),
                "missing_rows": missing,
                "extra_rows": extra,
                "status": status,
            }
            if status == "FAIL":
                failures.append(f"{tissue}/{split}: missing={missing}, extra={extra}")
    if failures:
        raise RuntimeError(
            "reconstructed data do not match the canonical shipped benchmark: "
            + "; ".join(failures)
        )
    return report


def write_outputs(
    pools: dict[str, list[dict]],
    pair_ids: dict[str, int],
    split_map: dict[str, str],
    output_dir: Path,
) -> dict:
    summary = {}
    for tissue in TISSUES:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in pools[tissue]:
            pair_id = pair_ids[row["substrate"]]
            split = split_map[str(pair_id)]
            buckets[split].append({**row, "pair_id": pair_id, "split": split})

        tissue_dir = output_dir / tissue
        tissue_dir.mkdir(parents=True)
        summary[tissue] = {}
        for split in SPLITS:
            jsonl_path = tissue_dir / f"{split}.jsonl"
            metadata_path = tissue_dir / f"{split}.metadata.csv"
            with jsonl_path.open("w") as jsonl, metadata_path.open(
                "w", newline=""
            ) as metadata:
                writer = csv.DictWriter(
                    metadata,
                    fieldnames=["pair_id", "pair_prefix", "yes_no", "split"],
                )
                writer.writeheader()
                for row in buckets[split]:
                    jsonl.write(
                        canonical_line(
                            record_json(
                                row["structure"],
                                row["L"],
                                row["R"],
                                row["yes_no"],
                            )
                        )
                        + "\n"
                    )
                    writer.writerow(
                        {
                            "pair_id": row["pair_id"],
                            "pair_prefix": row["substrate"][:24],
                            "yes_no": row["yes_no"],
                            "split": split,
                        }
                    )
            counts = Counter(row["yes_no"] for row in buckets[split])
            summary[tissue][split] = {
                "rows": len(buckets[split]),
                "yes": counts["yes"],
                "no": counts["no"],
                "pairs": len({row["pair_id"] for row in buckets[split]}),
            }
    return summary


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-map",
        type=Path,
        default=here / "global_pair_split.json",
        help="canonical pair_id-to-split JSON map",
    )
    parser.add_argument(
        "--invalid-policy",
        choices=("drop", "error"),
        default="drop",
        help="drop malformed graph inputs with a report, or fail immediately",
    )
    parser.add_argument(
        "--canonical-reference",
        type=Path,
        default=None,
        help=(
            "optional data/human directory; require exact order-independent "
            "record equality with the shipped benchmark"
        ),
    )
    parser.add_argument(
        "--expected-substrates",
        type=int,
        default=884,
        help="expected union of graph-buildable substrates [default: 884]",
    )
    parser.add_argument(
        "--overwrite-empty",
        action="store_true",
        help="allow removal of an existing empty output directory only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        entries = list(args.output_dir.iterdir()) if args.output_dir.is_dir() else [args.output_dir]
        if entries:
            raise RuntimeError(
                f"refusing to overwrite non-empty output path: {args.output_dir}"
            )
        if not args.overwrite_empty:
            raise RuntimeError(
                f"output directory already exists: {args.output_dir}; "
                "pass --overwrite-empty only if it is empty"
            )
        args.output_dir.rmdir()

    pools = {}
    invalid_rows = {}
    for tissue in TISSUES:
        path = args.balanced_dir / f"{tissue}.balanced.csv"
        if not path.exists():
            raise RuntimeError(f"missing balanced pool: {path}")
        pools[tissue], invalid_rows[tissue] = read_balanced_pool(
            path, args.invalid_policy
        )

    substrates = sorted(
        {row["substrate"] for tissue in TISSUES for row in pools[tissue]}
    )
    if len(substrates) != args.expected_substrates:
        raise RuntimeError(
            f"found {len(substrates)} graph-buildable substrates; "
            f"expected {args.expected_substrates}. Check the raw-table version, "
            "thresholds, balancing seed, and invalid-row report."
        )
    pair_ids = {sequence: index for index, sequence in enumerate(substrates, start=1)}
    split_map = load_split_map(
        args.split_map, {str(index) for index in range(1, len(substrates) + 1)}
    )

    args.output_dir.mkdir(parents=True)
    try:
        split_summary = write_outputs(
            pools, pair_ids, split_map, args.output_dir
        )
        reconstruction = None
        if args.canonical_reference is not None:
            reconstruction = compare_to_reference(
                args.output_dir, args.canonical_reference
            )
        report = {
            "status": "PASS",
            "balanced_dir": str(args.balanced_dir.resolve()),
            "split_map": str(args.split_map.resolve()),
            "unique_substrates": len(substrates),
            "invalid_policy": args.invalid_policy,
            "invalid_rows": {
                tissue: {
                    "count": len(rows),
                    "examples": rows[:10],
                }
                for tissue, rows in invalid_rows.items()
            },
            "splits": split_summary,
            "canonical_record_comparison": reconstruction,
        }
        (args.output_dir / "construction_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    except Exception:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        raise

    print(
        f"PASS: wrote the globally pair-disjoint human benchmark to "
        f"{args.output_dir}"
    )
    print(
        f"      substrates={len(substrates)}; invalid rows dropped="
        f"{sum(len(rows) for rows in invalid_rows.values())}"
    )
    if args.canonical_reference is not None:
        print("      canonical order-independent record comparison: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
