#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(data.table)
})

# -------- CLI --------
option_list <- list(
  make_option(c("-i","--input"), type="character",
              help="Input CSV from merge step (dsRNA_structure_with_editing_sites_andA20.csv).", metavar="FILE"),
  make_option(c("-o","--output"), type="character",
              help="Output CSV path for filtered, deduplicated results.", metavar="FILE"),

  # Defaults aligned to your original script:
  make_option(c("--tolerance"), type="integer", default=20,
              help="Max allowed range (bp) within a group for each ds boundary [default: %default]."),
  make_option(c("--min-length"), type="integer", default=200,
              help="Minimum length_small_ds to keep [default: %default]."),
  make_option(c("--min-coverage"), type="integer", default=100,
              help="Minimum Total_Coverage to keep [default: %default]."),
  make_option(c("--editing-threshold"), type="double", default=0.1,
              help="Threshold for 'above' subset (not saved unless --save-above-threshold is provided) [default: %default]."),

  # optional extra output (off by default)
  make_option(c("--save-above-threshold"), type="character", default=NULL,
              help="Optional path to save rows with EditingLevel > editing-threshold.")
)

opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$input) || is.null(opt$output)) {
  stop("Missing required --input and/or --output", call. = FALSE)
}

# -------- I/O --------
message("Reading: ", opt$input)
all_data <- fread(opt$input)
setDT(all_data)

# Required columns (from your code)
required_cols <- c(
  "Chr","Position","Strand","Editing_Type","Total_Coverage","Local_Position",
  "EditingLevel","small_ds_seq","mfe_struct",
  "start_1ds_genome","end_1ds_genome","start_2ds_genome","end_2ds_genome",
  "length_small_ds","num_editing_site_intersected_1ds","num_editing_site_intersected_2ds",
  "start_cluster","end_cluster"
)
missing <- setdiff(required_cols, names(all_data))
if (length(missing)) {
  stop(sprintf("Input is missing required columns: %s", paste(missing, collapse=", ")), call. = FALSE)
}

data_filter_for_analysis <- all_data[, ..required_cols]

# -------- Functions --------
check_group_fast_dt <- function(dt, tol) {
  r1s <- max(dt$start_1ds_genome) - min(dt$start_1ds_genome)
  r1e <- max(dt$end_1ds_genome)   - min(dt$end_1ds_genome)
  r2s <- max(dt$start_2ds_genome) - min(dt$start_2ds_genome)
  r2e <- max(dt$end_2ds_genome)   - min(dt$end_2ds_genome)
  (r1s <= tol) & (r1e <= tol) & (r2s <= tol) & (r2e <= tol)
}

tol <- as.integer(opt$tolerance)

message("Computing valid groups (tolerance = ", tol, ") ...")
valid_groups <- data_filter_for_analysis[, .(valid = check_group_fast_dt(.SD, tol)), by = .(Chr, Position, Strand)]

data_with_flags <- merge(data_filter_for_analysis, valid_groups,
                         by = c("Chr","Position","Strand"), all.x = TRUE)

message("Selecting best row per (Chr,Position,Strand) group ...")
filtered_data <- data_with_flags[valid == TRUE,
  .SD[which.max(num_editing_site_intersected_1ds + num_editing_site_intersected_2ds)],
  by = .(Chr, Position, Strand)
]

min_len <- as.integer(opt$`min-length`)
min_cov <- as.integer(opt$`min-coverage`)
message("Applying final filters: length_small_ds >= ", min_len, ", Total_Coverage >= ", min_cov)
data_for_analysis <- filtered_data[length_small_ds >= min_len & Total_Coverage >= min_cov]

# Optional subset like your above_15p (default threshold 0.1); save only if path provided
thr <- as.numeric(opt$`editing-threshold`)
if (!is.null(opt$`save-above-threshold`)) {
  out_thr <- opt$`save-above-threshold`
  message("Saving rows with EditingLevel > ", thr, " to: ", out_thr)
  above_thr <- data_for_analysis[EditingLevel > thr]
  fwrite(above_thr, out_thr)
}

# Write main output
out_path <- opt$output
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
message("Writing: ", out_path)
fwrite(data_for_analysis, out_path)

message("Done.")
