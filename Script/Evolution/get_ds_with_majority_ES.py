#!/usr/bin/env python3

import os
import argparse
import logging
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from ViennaRNA import RNA

import sys, shutil

def ensure_rnastructure_datapath():
    if "DATAPATH" not in os.environ:
        candidate = os.path.join(sys.prefix, "share", "rnastructure", "data_tables")
        if os.path.isdir(candidate):
            os.environ["DATAPATH"] = candidate
        else:
            logging.warning("DATAPATH not set and default path not found: %s", candidate)

def run_dot2ct(dbn_file: str, ct_file: str):
    ensure_rnastructure_datapath()
    subprocess.run(["dot2ct", dbn_file, ct_file], check=True)

def run_draw(ct_file: str, svg_out: str, shape_file: str | None = None):
    ensure_rnastructure_datapath()
    cmd = ["draw", ct_file, svg_out, "--svg", "-n", "1"]
    if shape_file:
        cmd.extend(["-s", shape_file])
    subprocess.run(cmd, check=True)

def run_bprna(dbn_file: str, output_dir: str) -> str:
    print("open run_bprna")
    perl = shutil.which("perl") or "perl"
    bprna = shutil.which("bpRNA.pl")
    if not bprna:
        bprna = os.environ.get("BPRNA_PL")
        if not bprna:
            raise RuntimeError("bpRNA.pl not found. Add bpRNA repo to PATH or set $BPRNA_PL")

    base = os.path.splitext(dbn_file)[0] + ".st"
    out_st = os.path.join(output_dir, os.path.basename(base))
    subprocess.run([perl, bprna, dbn_file], cwd=output_dir, check=True)
    print("***********************************************************")
    print(out_st)
    return out_st


class genome_reader:
    """Loads genome once and provides FASTA queries via bedtools."""

    def __init__(self, bedtools: str, genome: str):
        """
        Args:
            bedtools: Path to the bedtools binary.
            genome: Path to the genome FASTA file.
        """
        self.bedtools = bedtools
        self.genome = genome
        self.chr_lengths = self._load_chromosome_lengths()

    def _load_chromosome_lengths(self):
        """Load chromosome lengths from the genome FASTA index."""
        from pyfaidx import Fasta
        genome = Fasta(self.genome)
        return {chrom: len(seq) for chrom, seq in genome.items()}

    def get_chromosome_lengths(self):
        """Return chromosome lengths dict."""
        return self.chr_lengths

    def get_fasta(self, chrom: str, start: int, end: int, strand: str = "+") -> str:
        """Fetch FASTA sequence with bedtools getfasta (-s with strand awareness).

        Args:
            chrom: Chromosome name.
            start: Start position (bedtools expects 0-based BED-like intervals).
            end: End position (half-open).
            strand: '+' or '-'.

        Returns:
            FASTA text (header + sequence).
        """
        try:
            line = f"{chrom}\t{start}\t{end}\t0\t0\t{strand}"
            result = subprocess.run(
                [self.bedtools, "getfasta", "-fi", self.genome, "-bed", "stdin", "-s"],
                input=line.encode("utf-8"),
                capture_output=True,
                check=True,
                text=False,
            )
            return result.stdout.decode("utf-8")
        except subprocess.CalledProcessError as e:
            logging.error("Error fetching FASTA sequence: %s", e)
            raise



def get_seq(
    chr1: str,
    start1: int,
    end1: int,
    start2: int,
    end2: int,
    strand: str,
    gr: genome_reader,
    upstream: int = 0,
    downstream: int = 0,
) -> Tuple[str, str]:
    """Build a combined sequence window covering both regions plus optional flanks."""
    chr_lengths = gr.get_chromosome_lengths()
    if strand == "+":
        start = min(start1, start2) - upstream
        end = max(end1, end2) + downstream
    else:
        start = min(start1, start2) - downstream
        end = max(end1, end2) + upstream

    # Clamp to genome bounds defensively
    start = max(0, start)
    end = min(chr_lengths[chr1], end)

    fasta = gr.get_fasta(chr1, start, end, strand)
    seq = fasta.split("\n")[1].upper().replace("T", "U")
    pos_id = "_".join([chr1, str(start), str(end)])
    return seq, pos_id


def create_SHAPE_file_for_editing_site(
    fold_region: Tuple[int, int],
    sites: List[int],
    shape_filename: str,
    strand: str,
):
    """Create a .shape file marking given editing sites within a fold region.

    The coordinate written is 1-based relative index.
    """
    fold_region_start, fold_region_end = fold_region
    out_lines = []
    for position in sites:
        if strand == "+":
            rel = position - fold_region_start
        else:  # '-'
            rel = fold_region_end - position + 1
        out_lines.append(f"{rel} 0.9")

    with open(shape_filename, "w") as f:
        f.write("\n".join(out_lines) + "\n")


def get_realtive_position_list(
    fold_region: Tuple[int, int],
    sites: List[int],
    strand: str,
) -> List[int]:
    """Return list of relative positions (1-based index) of sites within the fold region."""
    fold_region_start, fold_region_end = fold_region
    rel = []
    for position in sites:
        if strand == "+":
            rel.append(position - fold_region_start)
        else:
            rel.append(fold_region_end - position + 1)
    return rel


def get_genome_position(
    fold_region: Tuple[int, int],
    coords: Tuple[int, int, int, int],
    strand: str,
) -> Tuple[int, int, int, int]:
    """Map segment-relative coords back to genomic coords."""
    fold_region_start, fold_region_end = fold_region
    start_1ds, end_1ds, start_2ds, end_2ds = coords

    if strand == "+":
        s1g = fold_region_start + start_1ds
        e1g = fold_region_start + end_1ds
        s2g = fold_region_start + start_2ds
        e2g = fold_region_start + end_2ds
    else:
        s1g = fold_region_end - end_1ds
        e1g = fold_region_end - start_1ds
        s2g = fold_region_end - end_2ds
        e2g = fold_region_end - start_2ds
    return s1g, e1g, s2g, e2g


def parse_position_id(position_id: str) -> Tuple[int, int] | None:
    """Parse position_id like 'chr_start_end' and return (start, end)."""
    parts = [p for p in position_id.split("_") if p.isdigit()]
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    logging.warning("Invalid position_id format: %s", position_id)
    return None


def get_segment_majority_ES(file: str, sites: List[int]) -> Tuple[Tuple[int, int, int, int], Tuple[str, str]] | Tuple[int, int]:
    """From a .st file, pick the segment containing the majority of given editing sites."""
    with open(file, "r") as bpf:
        data = bpf.readlines()

    segment_regex = re.compile(r"(\s\d+\.\.\d+\s)")
    segment_coords = []
    segment_sequences = []

    for line in data:
        if "segment" in line:
            parts = segment_regex.split(line)
            if len(parts) >= 4:
                start1, end1 = map(int, parts[1].strip().split(".."))
                start2, end2 = map(int, parts[3].strip().split(".."))
                segment_coords.append((start1, end1, start2, end2))
                segment_sequences.append((parts[2].strip(), parts[-1].strip("\n")))

    if not segment_coords:
        logging.warning("No segments found in .st file: %s", file)
        return (0, 0)

    counts = [0] * len(segment_coords)
    for site in sites:
        for idx, (s1, e1, s2, e2) in enumerate(segment_coords):
            if s1 <= site <= e1 or s2 <= site <= e2:
                counts[idx] += 1
                break

    maj_idx = counts.index(max(counts))
    coords = segment_coords[maj_idx]
    seqs = segment_sequences[maj_idx]
    logging.info("Selected segment %s with %d sites.", coords, max(counts))
    return (coords, seqs)


def intersect_with_all_site(
    site_file: str,
    start_1ds_genome: int,
    end_1ds_genome: int,
    start_2ds_genome: int,
    end_2ds_genome: int,
    chrom: str,
    strand: str,
) -> Tuple[List[str], List[str]]:
    """Intersect two dsRNA subregions with a BED6 of all editing sites (strand-aware)."""
    bedtools_cmd = ["bedtools", "intersect", "-wa", "-a", site_file, "-b", "stdin"]

    # Build two BED lines for the subregions and run two intersects
    def run_intersect(s: int, e: int) -> List[str]:
        bed_line = f"{chrom}\t{s}\t{e-1}\t0\t0\t{strand}"
        p = subprocess.run(bedtools_cmd, input=bed_line.encode("utf-8"), capture_output=True, text=False)
        return [ln for ln in p.stdout.decode("utf-8").splitlines() if ln]

    inter1 = run_intersect(start_1ds_genome, end_1ds_genome)
    inter2 = run_intersect(start_2ds_genome, end_2ds_genome)
    return inter1, inter2


def extract_sites(intersected_regions: List[str], strand: str) -> List[int]:
    """Extract site positions from intersect output lines (BED6), matching strand."""
    sites = []
    for region in intersected_regions:
        fields = region.split("\t")
        if len(fields) < 6:
            continue
        if fields[5] != strand:
            continue
        # Using end column (Position in your A2IEditingSite.bed)
        sites.append(int(fields[2]))
    return sites


def get_relative_site(
    start_1ds_genome: int,
    end_1ds_genome: int,
    start_2ds_genome: int,
    end_2ds_genome: int,
    position: int,
    strand: str,
    link: str,
) -> int:
    """Compute 0-based relative index of a genomic position within concatenated ds1 + link + ds2."""
    ds_1 = start_1ds_genome <= position <= end_1ds_genome
    ds_2 = not ds_1

    if strand == "+":
        if ds_1:
            rel = position - start_1ds_genome
        else:
            rel = (end_1ds_genome - start_1ds_genome + 1) + len(link) + (position - start_2ds_genome)
    else:
        if ds_1:
            rel = end_1ds_genome - position + 1
        else:
            rel = (end_1ds_genome - start_1ds_genome + 1) + len(link) + (end_2ds_genome - position + 1)
    return rel


def create_shape_file_small_seq(
    start_1ds_genome: int,
    end_1ds_genome: int,
    start_2ds_genome: int,
    end_2ds_genome: int,
    strand: str,
    path_shape_file: str,
    more_editing_site: List[int],
    link: str,
) -> List[Tuple[int, int]]:
    """Create shape file for small ds sequence and return list of (relative_position, genomic_position)."""
    rows = []
    rel_positions = []
    for p in more_editing_site:
        rel = get_relative_site(
            start_1ds_genome, end_1ds_genome, start_2ds_genome, end_2ds_genome, p, strand, link
        ) + 1
        rows.append({"Position": str(rel), "Frequency": 0.5})
        rel_positions.append((rel, p))

    pd.DataFrame(rows).to_csv(path_shape_file, header=None, index=False, sep=" ")
    return rel_positions


def get_a_positions_in_window(
    sequence: str,
    start: int,
    strand: str,
    editing_sites: List[int],
    start_1ds_genome: int,
    end_1ds_genome: int,
    start_2ds_genome: int,
    end_2ds_genome: int,
    link: str,
    max_distance: int = 20,
) -> List[Tuple[int, int, int]]:
    """Find As within max_distance of any editing site (exclude the sites themselves)."""
    seq = sequence[::-1] if strand == "-" else sequence
    out = []
    for i, nt in enumerate(seq):
        genomic_pos = start + i if strand == "+" else start + i + 1
        if nt == "A" and genomic_pos not in editing_sites:
            for site in editing_sites:
                if abs(genomic_pos - site) <= max_distance:
                    rel = get_relative_site(
                        start_1ds_genome, end_1ds_genome, start_2ds_genome, end_2ds_genome, genomic_pos, strand, link
                    ) + 1
                    out.append((genomic_pos, rel, site))
                    break
    return out


def process_row(row, gr: genome_reader, all_chr_length, args, link: str):
    chr1, start1, end1, chr2, start2, end2, sites_column, strand = row
    position = end1
    seq_id = f"{start1}_{end1}"
    sites = [int(num) for num in str(sites_column).split(",") if str(num).strip().isdigit()]

    chr_length = all_chr_length[str(chr1)]
    if args.window != 0 and (start1 - args.window < 0 or end1 + args.window > chr_length):
        return None

    sequence, position_id = get_seq(
        chr1=str(chr1),
        start1=start1,
        end1=end1,
        start2=start2,
        end2=end2,
        strand=strand,
        gr=gr,
        upstream=args.upstream if args.window == 0 else args.window,
        downstream=args.downstream if args.window == 0 else args.window,
    )

    # Fold (big window)
    fc = RNA.fold_compound(sequence)
    mfe_struct, mfe = fc.mfe()
    logging.info("Computed MFE: %s for sequence ID: %s", mfe, seq_id)

    mfe_data = {
        "sequence_id": seq_id,
        "region_id": "_".join(
            {
                "_".join([chr1, str(start1), str(end1)]),
                "_".join([chr2, str(start2), str(end2)]),
            }
        ),
        "strand": strand,
        "fold_region": position_id,
        "mfe": mfe,
    }

    # Write .dbn
    dbn_filename = os.path.join(args.output_dir, f"{seq_id}_{position_id}.dbn")
    with open(dbn_filename, "w") as dbn_file:
        dbn_file.write(f">{position_id}\n{sequence}\n{mfe_struct}\n")

    # .ct
    ct_filename = os.path.join(args.output_dir, f"{seq_id}_{position_id}.ct")
    run_dot2ct(dbn_filename, ct_filename)

    # SHAPE for editing sites (big region)
    shape_filename = os.path.join(args.output_dir, f"{seq_id}_{position_id}.shape")
    fold_region = parse_position_id(position_id)
    if fold_region:
        create_SHAPE_file_for_editing_site(fold_region, sites, shape_filename, strand)
    else:
        logging.warning("Skipping SHAPE file creation for invalid position_id: %s", position_id)

    # Draw (big)
    output_image = os.path.join(args.output_dir, f"{seq_id}_{position_id}.svg")
    run_draw(ct_filename, output_image, shape_filename)
    logging.info("Structure plot saved to %s", output_image)

    # bpRNA and choose segment
    print("hii")
    st_file = run_bprna(dbn_filename, args.output_dir)
    print
    relative_site_list = get_realtive_position_list(fold_region, sites, strand)
    coords, near_seqs = get_segment_majority_ES(st_file, relative_site_list)
    if coords == 0:
        return None

    # Genomic coords for chosen segment
    start_1ds_genome, end_1ds_genome, start_2ds_genome, end_2ds_genome = get_genome_position(fold_region, coords, strand)

    # Intersect with all editing sites
    es_1, es_2 = intersect_with_all_site(
        args.editing_site, start_1ds_genome, end_1ds_genome, start_2ds_genome, end_2ds_genome, chr1, strand
    )
    edit_1 = extract_sites(es_1, strand)
    edit_2 = extract_sites(es_2, strand)
    edit_both = edit_1 + edit_2

    small_ds_seq = near_seqs[0] + link + near_seqs[1]

    a_pos_ds1 = get_a_positions_in_window(
        near_seqs[0], start_1ds_genome, strand, edit_1, start_1ds_genome, end_1ds_genome, start_2ds_genome, end_2ds_genome, link
    )
    a_pos_ds2 = get_a_positions_in_window(
        near_seqs[1], start_2ds_genome, strand, edit_2, start_1ds_genome, end_1ds_genome, start_2ds_genome, end_2ds_genome, link
    )
    all_a = a_pos_ds1 + a_pos_ds2

    # Fold (small)
    fc = RNA.fold_compound(small_ds_seq)
    mfe_struct_small, mfe_small = fc.mfe()
    logging.info("small_seq - Computed MFE: %s for sequence ID: %s", mfe_small, seq_id)

    # .dbn (small)
    dbn_small = os.path.join(args.output_dir, f"small_{seq_id}_{position_id}.dbn")
    with open(dbn_small, "w") as dbn_file:
        dbn_file.write(f">{start_1ds_genome}_{end_1ds_genome}_{start_2ds_genome}_{end_2ds_genome}\n")
        dbn_file.write(f"{small_ds_seq}\n{mfe_struct_small}\n")

    # .ct (small)
    ct_small = os.path.join(args.output_dir, f"small_{seq_id}_{position_id}.ct")
    run_dot2ct(dbn_small, ct_small)

    # SHAPE (small)
    shape_small = os.path.join(args.output_dir, f"small_{seq_id}_{position_id}.shape")
    relative_positions = create_shape_file_small_seq(
        start_1ds_genome, end_1ds_genome, start_2ds_genome, end_2ds_genome, strand, shape_small, edit_both, link
    )

    # Draw (small)
    out_svg_small = os.path.join(args.output_dir, f"small_{seq_id}_{position_id}.svg")  
    logging.info("Structure plot saved to %s", out_svg_small)
    run_draw(ct_small, out_svg_small, shape_small)

    all_data = {
        "sequence_id": seq_id,
        "chr": chr1,
        "position": position,
        "start_cluster": start2,
        "end_cluster": end2,
        "strand": strand,
        "fold_region_big": position_id,
        "small_ds_seq": small_ds_seq,
        "mfe_struct": mfe_struct,
        "mfe_small": mfe_small,
        "start_1ds_genome": start_1ds_genome,
        "end_1ds_genome": end_1ds_genome,
        "start_2ds_genome": start_2ds_genome,
        "end_2ds_genome": end_2ds_genome,
        "length_small_ds": len(small_ds_seq),
        "editing_site_intersected_1ds": edit_1,
        "num_editing_site_intersected_1ds": len(edit_1),
        "editing_site_intersected_2ds": edit_2,
        "num_editing_site_intersected_2ds": len(edit_2),
        "relative_positions": relative_positions,
        "A_20d": all_a,
    }

    return all_data, mfe_data


if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description="Create a dataset of secondary structures of editing sites and augmentations.",
    )
    parser.add_argument(
        "-i", "--input_regions", dest="regions",
        help=".",
        type=str,
    )
    parser.add_argument(
        "-o", "--output_dir", dest="output_dir",
        help="Directory for output files (SVG/CT/DBN/SHAPE and CSV summaries).",
        type=str,
    )
    parser.add_argument(
        "-e", "--editing_site", dest="editing_site",
        help="BED6 file with all editing sites (e.g., A2IEditingSite.bed).",
        type=str,
    )
    parser.add_argument(
        "-g", "--genome", dest="genome",
        help="Genome FASTA file.",
        type=str,
    )
    parser.add_argument("--bedtools", dest="bedtools_path", help="Path to bedtools binary.", type=str, default="bedtools")
    parser.add_argument("--window", dest="window", help="Symmetric window size.", type=int, default=1000)
    parser.add_argument("--upstream", dest="upstream", help="Upstream extension (used when --window=0).", type=int, default=0)
    parser.add_argument("--downstream", dest="downstream", help="Downstream extension (used when --window=0).", type=int, default=0)
    parser.add_argument("--num_processes", type=int, default=1, help="Number of parallel processes.")

    args = parser.parse_args()
    link = "N" * 10  # linker between ds segments

    # Mutually exclusive window vs. upstream/downstream
    if args.window != 0 and not (args.downstream == 0 and args.upstream == 0):
        parser.error("--upstream and --downstream cannot be used with --window != 0")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load regions table
    input_file = args.regions
    if input_file.endswith(".csv"):
        alu_region = pd.read_csv(input_file)
        alu_region.columns = ["chr1", "start1", "end1", "chr2", "start2", "end2", "id", "strand"]
    elif input_file.endswith(".bed"):
        alu_region = pd.read_csv(input_file, sep="\t", header=None)
        alu_region.columns = ["chr", "start", "end", "num_site", "sites", "strand"]
        alu_region = alu_region.drop(columns=["num_site"])
        # Map to paired region format
        alu_region["chr1"] = alu_region["chr"]
        alu_region["start1"] = alu_region["start"]
        alu_region["end1"] = alu_region["end"]
        alu_region["chr2"] = alu_region["chr"]
        alu_region["start2"] = alu_region["start"]
        alu_region["end2"] = alu_region["end"]
        alu_region = alu_region[["chr1", "start1", "end1", "chr2", "start2", "end2", "sites", "strand"]]
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or BED6 file.")

    gr = genome_reader(args.bedtools_path, args.genome)
    all_chr_length = gr.get_chromosome_lengths()

    # Parallel processing
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        process_func = partial(process_row, gr=gr, all_chr_length=all_chr_length, args=args, link=link)
        results = list(executor.map(process_func, alu_region.itertuples(index=False, name=None)))

    # Collect and save CSV outputs
    all_data_results = [res[0] for res in results if res is not None]
    mfe_data_results = [res[1] for res in results if res is not None]

    pd.DataFrame(all_data_results).to_csv(os.path.join(args.output_dir, "all_data_results.csv"), index=False)
    pd.DataFrame(mfe_data_results).to_csv(os.path.join(args.output_dir, "mfe_data_results.csv"), index=False)

    logging.info("Processing complete. Results saved.")
