#!/usr/bin/env python3
"""Run the complete non-Alu species benchmark construction workflow.

The input manifest is a CSV with columns:

    species,editing_table,genome

Each editing table is the Zhang et al. per-site table consumed by
species/get_editing_levels.py. Each genome is the matching reference FASTA.
All output is written below --out-dir; the shipped repository data are never
modified by this program.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPECIES_SCRIPTS = HERE / "species"
SPLIT_SCRIPTS = HERE / "split"


def require_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required program not found in PATH: {name}")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"species", "editing_table", "genome"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing manifest columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty manifest")

    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        species = row["species"].strip()
        if not species or species in seen:
            raise ValueError(f"{path}:{line_number}: empty or duplicate species {species!r}")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", species) is None:
            raise ValueError(
                f"{path}:{line_number}: unsafe species name {species!r}; "
                "use only letters, digits, dot, underscore or hyphen"
            )
        seen.add(species)
        for field in ("editing_table", "genome"):
            candidate = Path(row[field]).expanduser().resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"{path}:{line_number}: {field} not found: {candidate}")
            row[field] = str(candidate)
        row["species"] = species
    return rows


def display(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def run(command: list[str], commands: list[list[str]], dry_run: bool) -> None:
    commands.append(command)
    print(f"[run] {display(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def ensure_empty(path: Path, dry_run: bool) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"--out-dir must be empty or absent: {path}")
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--merge-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--cluster-distance", type=int, default=1000)
    parser.add_argument("--cluster-min-count", type=int, default=5)
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--min-coverage", type=int, default=100)
    parser.add_argument("--positive-threshold", type=float, default=0.15)
    parser.add_argument("--negative-threshold", type=float, default=0.001)
    parser.add_argument(
        "--bprna", default=None,
        help="Path to bpRNA.pl (default: BPRNA_PL or bpRNA.pl in PATH)",
    )
    parser.add_argument(
        "--invalid-target-policy", choices=("error", "drop"), default="error"
    )
    parser.add_argument(
        "--no-equalize-across", action="store_true",
        help="Balance each species independently instead of using a shared size",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    require_program("Rscript")
    require_program("bedtools")
    bprna = args.bprna or os.environ.get("BPRNA_PL") or shutil.which("bpRNA.pl")
    if bprna is None:
        raise RuntimeError(
            "bpRNA.pl was not found. Provide --bprna /path/to/bpRNA.pl or add it to PATH."
        )
    bprna_path = Path(bprna).expanduser().resolve()
    if not bprna_path.is_file():
        raise FileNotFoundError(f"bpRNA.pl not found: {bprna_path}")
    rows = load_manifest(args.manifest)
    output = args.out_dir.expanduser().resolve()
    ensure_empty(output, args.dry_run)

    work_root = output / "work"
    balanced_root = output / "balanced"
    benchmark_root = output / "benchmark"
    commands: list[list[str]] = []
    filtered_inputs: list[str] = []

    for row in rows:
        species = row["species"]
        work = work_root / species
        edit_dir = work / "editing"
        structure_dir = work / "structures"
        if not args.dry_run:
            edit_dir.mkdir(parents=True, exist_ok=True)
            structure_dir.mkdir(parents=True, exist_ok=True)

        run(
            [
                sys.executable,
                str(SPECIES_SCRIPTS / "get_editing_levels.py"),
                "--input", row["editing_table"],
                "--out-dir", str(edit_dir),
            ],
            commands,
            args.dry_run,
        )
        editing_csv = edit_dir / "A2IEditingSite.csv"
        editing_bed = edit_dir / "A2IEditingSite.bed"
        clusters = work / "clusters.bed"
        run(
            [
                sys.executable,
                str(SPECIES_SCRIPTS / "cluster_editing_sites.py"),
                "--input-bed", str(editing_bed),
                "--distance", str(args.cluster_distance),
                "--min-count", str(args.cluster_min_count),
                "--output-mode", "bed6",
                "--out-file", str(clusters),
            ],
            commands,
            args.dry_run,
        )
        run(
            [
                sys.executable,
                str(SPECIES_SCRIPTS / "get_ds_with_majority_ES.py"),
                "--input_regions", str(clusters),
                "--output_dir", str(structure_dir),
                "--editing_site", str(editing_bed),
                "--genome", row["genome"],
                "--window", str(args.window),
                "--num_processes", str(args.num_processes),
                "--bprna", str(bprna_path),
            ],
            commands,
            args.dry_run,
        )
        merged = work / "merged_sites.csv"
        run(
            [
                sys.executable,
                str(SPECIES_SCRIPTS / "merge_ds_results.py"),
                "--editing-level", str(editing_csv),
                "--all-data-results", str(structure_dir / "all_data_results.csv"),
                "--output", str(merged),
                "--workers", str(args.merge_workers),
            ],
            commands,
            args.dry_run,
        )
        filtered = work / "filtered_sites.csv"
        run(
            [
                "Rscript",
                str(SPECIES_SCRIPTS / "filter_ds_groups.R"),
                "--input", str(merged),
                "--output", str(filtered),
                "--min-length", str(args.min_length),
                "--min-coverage", str(args.min_coverage),
            ],
            commands,
            args.dry_run,
        )
        filtered_inputs.append(f"{species}={filtered}")

    balance_command = [
        "Rscript",
        str(SPECIES_SCRIPTS / "prepare_balanced_ml_sets.R"),
        "--inputs", ",".join(filtered_inputs),
        "--out-dir", str(balanced_root),
        "--pos-threshold", str(args.positive_threshold),
        "--neg-threshold", str(args.negative_threshold),
        "--equalize-across", "FALSE" if args.no_equalize_across else "TRUE",
        "--seed", str(args.seed),
        "--invalid-target-policy", args.invalid_target_policy,
    ]
    run(balance_command, commands, args.dry_run)
    run(
        [
            sys.executable,
            str(SPLIT_SCRIPTS / "build_species_benchmark.py"),
            "--balanced-root", str(balanced_root),
            "--out-dir", str(benchmark_root),
            "--seed", str(args.seed),
            "--fractions", "0.64,0.16,0.20",
        ],
        commands,
        args.dry_run,
    )
    run(
        [
            sys.executable,
            str(HERE / "verify_species_split.py"),
            "--data-root", str(benchmark_root),
        ],
        commands,
        args.dry_run,
    )

    if not args.dry_run:
        provenance = {
            "manifest": str(args.manifest.resolve()),
            "parameters": vars(args) | {"manifest": str(args.manifest), "out_dir": str(args.out_dir)},
            "commands": commands,
            "final_data_root": str(benchmark_root),
        }
        (output / "pipeline_provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str) + "\n"
        )
        print(f"PASS: complete species benchmark available at {benchmark_root}")
    else:
        print("DRY RUN PASS: commands validated and displayed; nothing was written")


if __name__ == "__main__":
    main()
