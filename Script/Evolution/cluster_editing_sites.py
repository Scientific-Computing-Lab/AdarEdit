#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster editing sites from a BED6 file using bedtools:
1) bedtools sort
2) bedtools merge -d <distance> [-s] -c 3,3,6 -o count,collapse,distinct
3) Filter clusters by minimum count
4) Write BED output

Default output columns (BED6-valid):
    chrom  start  end  name=count  score=0  strand

Notes
-----
- Requires `bedtools` in PATH.
- Input must be BED6: chrom, start, end, name, score, strand.
- Merging is strand-specific by default (-s).
- File name pattern: cluster_d<distance>_up<mincount>editingsite.bed
- For compatibility with your original pipeline printing ($1,$2,$3,$5,$6,$4),
  use --output-mode compat (not BED6-valid).
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def ensure_bedtools_available() -> None:
    if shutil.which("bedtools") is None:
        raise RuntimeError("bedtools not found in PATH. Please install bedtools.")


def compute_outfile(out_dir: Path, distance: int, min_count: int, name_prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Match your naming pattern from the example:
    fname = f"{name_prefix}_d{distance}_up{min_count}editingsite.bed"
    return out_dir / fname


def cluster_sites(
    input_bed: Path,
    output_bed: Path,
    distance: int = 1000,
    min_count: int = 5,
    strand_specific: bool = True,
    output_mode: str = "bed6",  # "bed6" (default) or "compat"
    log_level: str = "INFO",
) -> None:
    """
    Run: bedtools sort | bedtools merge -d D [-s] -c 3,3,6 -o count,collapse,distinct
    Filter by min_count on the 'count' field, and write output.

    output_mode:
      - "bed6": write chrom, start, end, name=count, score=0, strand
      - "compat": replicate your awk '{print $1,$2,$3,$5,$6,$4}'
                  (NOT strictly BED6; name=collapsed list, score=strand, strand=count)
    """
    logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s | %(message)s")
    ensure_bedtools_available()

    # bedtools sort -i input
    sort_cmd = ["bedtools", "sort", "-i", str(input_bed)]
    logging.info("Running: %s", " ".join(sort_cmd))
    sort_proc = subprocess.Popen(sort_cmd, stdout=subprocess.PIPE, text=True)

    # bedtools merge -d D [-s] -c 3,3,6 -o count,collapse,distinct
    merge_cmd = ["bedtools", "merge", "-d", str(distance)]
    if strand_specific:
        merge_cmd.append("-s")  # place -s AFTER the distance
    merge_cmd += ["-c", "3,3,6", "-o", "count,collapse,distinct"]
    logging.info("Running: %s", " ".join(merge_cmd))
    merge_proc = subprocess.Popen(merge_cmd, stdin=sort_proc.stdout, stdout=subprocess.PIPE, text=True)

    # We will stream merged lines, filter, and write.
    kept = 0
    total = 0
    with output_bed.open("w", encoding="utf-8") as fout:
        assert merge_proc.stdout is not None
        for line in merge_proc.stdout:
            total += 1
            parts = line.rstrip("\n").split("\t")
            # Expected columns: chr(0), start(1), end(2), count(3), collapsed(4), strand(5)
            if len(parts) < 6:
                # Skip malformed rows
                continue

            chrom, start, end, count_str, collapsed, strand = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]

            try:
                cnt = int(count_str)
            except ValueError:
                # If count is not integer, skip row
                continue

            if cnt <= min_count:
                continue  # filter out

            # Write according to output mode
            if output_mode == "compat":
                # Replicate: print $1, $2, $3, $5, $6, $4
                # -> chrom, start, end, collapsed, strand, count
                fout.write("\t".join([chrom, start, end, collapsed, strand, str(cnt)]) + "\n")
            else:
                # BED6-valid: name=count, score=0, strand=strand
                fout.write("\t".join([chrom, start, end, str(cnt), collapsed, strand]) + "\n")
            kept += 1

    # Ensure processes finish
    sort_ret = sort_proc.wait()
    merge_ret = merge_proc.wait()
    if sort_ret != 0 or merge_ret != 0:
        raise RuntimeError(f"bedtools error (sort exit={sort_ret}, merge exit={merge_ret}).")

    logging.info("Clusters processed: %d, kept after threshold: %d", total, kept)
    logging.info("Wrote: %s", output_bed)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cluster BED6 editing sites using bedtools merge and filter by count.")
    p.add_argument("-i", "--input-bed", required=True, type=Path, help="Input A2IEditingSite.bed (BED6).")
    p.add_argument("-d", "--distance", type=int, default=1000, help="Merge distance for bedtools (-d). Default: 1000.")
    p.add_argument("-m", "--min-count", type=int, default=5, help="Minimum merged site count to keep. Default: 5.")
    p.add_argument("--no-strand", dest="strand_specific", action="store_false", help="Disable strand-specific merge.")
    p.add_argument("-O", "--output-mode", choices=["bed6", "compat"], default="bed6",
                   help="Output format: 'bed6' (default) or 'compat' (replicates awk print).")
    p.add_argument("-o", "--out-file", type=Path, help="Explicit output BED path. If omitted, use --out-dir pattern.")
    p.add_argument("-D", "--out-dir", type=Path, default=Path("."), help="Output directory (if --out-file not given).")
    p.add_argument("--name-prefix", default="cluster", help="Filename prefix when using --out-dir. Default: cluster")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[list] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    if args.out_file:
        out_path = args.out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = compute_outfile(args.out_dir, args.distance, args.min_count, args.name_prefix)

    try:
        cluster_sites(
            input_bed=args.input_bed,
            output_bed=out_path,
            distance=args.distance,
            min_count=args.min_count,
            strand_specific=args.strand_specific,
            output_mode=args.output_mode,
            log_level=args.log_level,
        )
        return 0
    except Exception as e:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s | %(message)s")
        logging.error("Failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
