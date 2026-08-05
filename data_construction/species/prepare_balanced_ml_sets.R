#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(stringr)
})

# This stage deliberately does not create train/validation/test partitions.
# It labels and balances the complete per-species site pools while retaining
# the genomic region identifiers needed by the following region-disjoint split.

option_list <- list(
  make_option(
    c("--inputs"), type = "character",
    help = paste0(
      "Comma-separated name=CSV inputs from filter_ds_groups.R, for example ",
      "'Octopus=/data/octopus_filtered.csv,Ptychodera=/data/ptychodera_filtered.csv'."
    )
  ),
  make_option(
    c("-o", "--out-dir"), type = "character", metavar = "DIR",
    help = "Output directory. One <species>/balanced_sites.csv file is written per species."
  ),
  make_option(
    c("--pos-threshold"), type = "double", default = 0.15,
    help = "Strict positive threshold: EditingLevel > threshold. [default: %default]"
  ),
  make_option(
    c("--neg-threshold"), type = "double", default = 0.001,
    help = "Strict negative threshold: EditingLevel < threshold. [default: %default]"
  ),
  make_option(
    c("--equalize-across"), type = "logical", default = TRUE,
    help = "Downsample every species to the same per-class count. [default: %default]"
  ),
  make_option(
    c("--seed"), type = "integer", default = 42,
    help = "Random seed used only for class downsampling. [default: %default]"
  ),
  make_option(
    c("--min-local-pos"), type = "integer", default = 1,
    help = "Minimum accepted one-based Local_Position. [default: %default]"
  ),
  make_option(
    c("--invalid-target-policy"), type = "character", default = "error",
    help = "Action when Local_Position is invalid or not an A: error or drop. [default: %default]"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$inputs) || is.null(opt$`out-dir`)) {
  stop("Missing required --inputs and/or --out-dir", call. = FALSE)
}
if (!(opt$`invalid-target-policy` %in% c("error", "drop"))) {
  stop("--invalid-target-policy must be 'error' or 'drop'", call. = FALSE)
}
if (opt$`pos-threshold` <= opt$`neg-threshold`) {
  stop("--pos-threshold must be greater than --neg-threshold", call. = FALSE)
}

split_inputs <- function(value) {
  items <- str_trim(str_split(value, ",", simplify = FALSE)[[1]])
  items[nchar(items) > 0]
}

parse_item <- function(item) {
  match <- regexpr("=", item, fixed = TRUE)
  if (match < 1) {
    stop(sprintf("Input must use name=/path/file.csv syntax: '%s'", item), call. = FALSE)
  }
  name <- str_trim(substr(item, 1, match - 1))
  path <- str_trim(substr(item, match + 1, nchar(item)))
  if (name == "" || path == "") {
    stop(sprintf("Invalid input item: '%s'", item), call. = FALSE)
  }
  if (!str_detect(name, "^[A-Za-z0-9_.-]+$")) {
    stop(sprintf(
      "Invalid species name '%s'; use only letters, digits, dot, underscore or hyphen",
      name
    ), call. = FALSE)
  }
  list(name = name, path = path)
}

required_columns <- c(
  "Chr", "Position", "Strand", "Local_Position", "EditingLevel",
  "small_ds_seq", "mfe_struct", "start_cluster", "end_cluster"
)

prepare_species <- function(species_name, input_path, min_local_pos,
                            pos_threshold, neg_threshold, invalid_policy) {
  if (!file.exists(input_path)) {
    stop(sprintf("[%s] Input file not found: %s", species_name, input_path), call. = FALSE)
  }

  message(sprintf("[%s] Reading %s", species_name, input_path))
  raw <- suppressMessages(read_csv(input_path, show_col_types = FALSE))
  missing <- setdiff(required_columns, names(raw))
  if (length(missing) > 0) {
    stop(sprintf(
      "[%s] Missing required columns: %s",
      species_name, paste(missing, collapse = ", ")
    ), call. = FALSE)
  }

  prepared <- raw %>%
    mutate(
      source_row_id = row_number(),
      Local_Position = suppressWarnings(as.integer(Local_Position)),
      EditingLevel = suppressWarnings(as.numeric(EditingLevel)),
      small_ds_seq = toupper(as.character(small_ds_seq)),
      mfe_struct = as.character(mfe_struct),
      Strand = as.character(Strand),
      sequence_length = nchar(small_ds_seq),
      structure_length = nchar(mfe_struct)
    )

  missing_region <- prepared %>% filter(
    is.na(Chr) | str_trim(as.character(Chr)) == "" |
      is.na(Strand) | !(Strand %in% c("+", "-")) |
      is.na(start_cluster) | is.na(end_cluster)
  )
  if (nrow(missing_region) > 0) {
    examples <- paste(head(missing_region$source_row_id, 10), collapse = ", ")
    stop(sprintf(
      "[%s] %d rows have missing/invalid region coordinates or strand (source rows: %s)",
      species_name, nrow(missing_region), examples
    ), call. = FALSE)
  }

  missing_sequence <- prepared %>% filter(
    is.na(small_ds_seq) | is.na(mfe_struct)
  )
  if (nrow(missing_sequence) > 0) {
    examples <- paste(head(missing_sequence$source_row_id, 10), collapse = ", ")
    stop(sprintf(
      "[%s] %d rows have missing sequence or structure (source rows: %s)",
      species_name, nrow(missing_sequence), examples
    ), call. = FALSE)
  }

  invalid_local <- prepared %>% filter(
    is.na(Local_Position) | Local_Position < min_local_pos |
      Local_Position > sequence_length
  )
  invalid_local_n <- nrow(invalid_local)
  if (invalid_local_n > 0) {
    examples <- invalid_local %>%
      transmute(example = paste0(
        "row=", source_row_id, ",local=", Local_Position,
        ",length=", sequence_length
      )) %>%
      pull(example) %>% head(10) %>% paste(collapse = "; ")
    message(sprintf(
      "[%s] Found %d invalid focal indices (%s)",
      species_name, invalid_local_n, examples
    ))
    if (invalid_policy == "error") {
      stop(sprintf(
        "[%s] Refusing invalid Local_Position values. Correct the upstream coordinate/strand mapping or rerun with --invalid-target-policy drop.",
        species_name
      ), call. = FALSE)
    }
  }

  prepared <- prepared %>%
    filter(
      !is.na(Local_Position), Local_Position >= min_local_pos,
      Local_Position <= sequence_length
    ) %>%
    mutate(
      focal_base = substr(small_ds_seq, Local_Position, Local_Position),
      L = substr(small_ds_seq, 1, pmax(Local_Position - 1, 0)),
      R = substr(small_ds_seq, Local_Position + 1, sequence_length),
      region_id = paste(Chr, Strand, start_cluster, end_cluster, sep = ":"),
      EditingClass = case_when(
        !is.na(EditingLevel) & EditingLevel > pos_threshold ~ "yes",
        !is.na(EditingLevel) & EditingLevel < neg_threshold ~ "no",
        TRUE ~ NA_character_
      )
    )

  length_bad <- prepared %>% filter(sequence_length != structure_length)
  if (nrow(length_bad) > 0) {
    examples <- paste(head(length_bad$source_row_id, 10), collapse = ", ")
    stop(sprintf(
      "[%s] %d rows have sequence/structure length mismatches (source rows: %s)",
      species_name, nrow(length_bad), examples
    ), call. = FALSE)
  }

  invalid_target <- prepared %>% filter(focal_base != "A")
  invalid_n <- nrow(invalid_target)
  if (invalid_n > 0) {
    examples <- invalid_target %>%
      transmute(example = paste0("row=", source_row_id, ",base=", focal_base,
                                 ",local=", Local_Position)) %>%
      pull(example) %>% head(10) %>% paste(collapse = "; ")
    message(sprintf(
      "[%s] Found %d non-A focal coordinates (%s)", species_name, invalid_n, examples
    ))
    if (invalid_policy == "error") {
      stop(sprintf(
        "[%s] Refusing to replace non-A focal coordinates with A. Correct the upstream coordinate/strand mapping or rerun with --invalid-target-policy drop.",
        species_name
      ), call. = FALSE)
    }
    prepared <- prepared %>% filter(focal_base == "A")
  }

  labelled <- prepared %>% filter(!is.na(EditingClass))
  counts <- table(factor(labelled$EditingClass, levels = c("no", "yes")))
  per_class <- min(as.integer(counts))
  if (per_class < 1) {
    stop(sprintf("[%s] One class is empty after thresholding", species_name), call. = FALSE)
  }

  list(
    data = labelled,
    raw_n = nrow(raw),
    eligible_n = nrow(prepared),
    invalid_local_position_rows = invalid_local_n,
    invalid_target_n = invalid_n,
    no_n = as.integer(counts[["no"]]),
    yes_n = as.integer(counts[["yes"]]),
    available_per_class = per_class
  )
}

sample_per_class <- function(data, per_class, seed) {
  set.seed(seed)
  no_rows <- data %>% filter(EditingClass == "no")
  yes_rows <- data %>% filter(EditingClass == "yes")
  selected_no <- no_rows[sample.int(nrow(no_rows), per_class), , drop = FALSE]
  selected_yes <- yes_rows[sample.int(nrow(yes_rows), per_class), , drop = FALSE]
  bind_rows(selected_no, selected_yes) %>%
    arrange(region_id, Position, source_row_id)
}

items <- lapply(split_inputs(opt$inputs), parse_item)
if (length(items) == 0) {
  stop("No inputs were supplied", call. = FALSE)
}
item_names <- vapply(items, function(item) item$name, character(1))
if (anyDuplicated(item_names)) {
  duplicate_names <- unique(item_names[duplicated(item_names)])
  stop(sprintf(
    "Duplicate species names: %s", paste(duplicate_names, collapse = ", ")
  ), call. = FALSE)
}

prepared <- list()
for (index in seq_along(items)) {
  item <- items[[index]]
  prepared[[item$name]] <- prepare_species(
    species_name = item$name,
    input_path = item$path,
    min_local_pos = opt$`min-local-pos`,
    pos_threshold = opt$`pos-threshold`,
    neg_threshold = opt$`neg-threshold`,
    invalid_policy = opt$`invalid-target-policy`
  )
}

available <- vapply(prepared, function(x) x$available_per_class, integer(1))
if (isTRUE(opt$`equalize-across`)) {
  target_per_class <- min(available)
} else {
  target_per_class <- NA_integer_
}

out_dir <- opt$`out-dir`
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
summary_rows <- list()

for (index in seq_along(prepared)) {
  species_name <- names(prepared)[[index]]
  info <- prepared[[species_name]]
  selected_per_class <- if (isTRUE(opt$`equalize-across`)) {
    target_per_class
  } else {
    info$available_per_class
  }
  selected <- sample_per_class(
    info$data, selected_per_class, as.integer(opt$seed)
  )

  species_dir <- file.path(out_dir, species_name)
  dir.create(species_dir, recursive = TRUE, showWarnings = FALSE)
  output_path <- file.path(species_dir, "balanced_sites.csv")
  suppressMessages(write_csv(selected, output_path))

  summary_rows[[species_name]] <- tibble(
    species = species_name,
    input_file = basename(items[[index]]$path),
    raw_rows = info$raw_n,
    eligible_rows = info$eligible_n,
    invalid_local_position_rows = info$invalid_local_position_rows,
    invalid_target_rows = info$invalid_target_n,
    labelled_no_before_balance = info$no_n,
    labelled_yes_before_balance = info$yes_n,
    selected_per_class = selected_per_class,
    selected_total = nrow(selected),
    positive_threshold_strict_gt = opt$`pos-threshold`,
    negative_threshold_strict_lt = opt$`neg-threshold`,
    seed = as.integer(opt$seed)
  )
  message(sprintf(
    "[%s] Wrote %d balanced sites (%d per class) to %s",
    species_name, nrow(selected), selected_per_class, output_path
  ))
}

summary <- bind_rows(summary_rows)
suppressMessages(write_csv(summary, file.path(out_dir, "balance_summary.csv")))
message(sprintf("PASS: balanced species pools written under %s", out_dir))
