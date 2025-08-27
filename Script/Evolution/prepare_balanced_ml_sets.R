#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(tools)
})

# ---------- CLI ----------
option_list <- list(
  make_option(c("--inputs"), type="character",
              help="Comma-separated list of inputs. Each item is either name=/path/file.csv or /path/file.csv (species inferred from parent folder). Example: --inputs 'Octopus=/a/b/oct.csv,/a/b/Strongy/ds.csv'"),
  make_option(c("-o","--out-dir"), type="character", metavar="DIR",
              help="Output directory (species subfolders will be created)."),
  make_option(c("--pos-threshold"), type="double", default=0.1,
              help="EditingLevel threshold for positive class (yes). [default: %default]"),
  make_option(c("--neg-threshold"), type="double", default=0.001,
              help="EditingLevel threshold for negative class (no). [default: %default]"),
  make_option(c("--train-frac"), type="double", default=0.8,
              help="Train split fraction. [default: %default]"),
  make_option(c("--equalize-across"), type="logical", default=TRUE,
              help="Downsample each species to the same size (per-class = min across species). [default: %default]"),
  make_option(c("--seed"), type="integer", default=42,
              help="Random seed. [default: %default]"),
  make_option(c("--min-local-pos"), type="integer", default=1,
              help="Filter rows with Local_Position < this value. [default: %default]")
)

opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$inputs) || is.null(opt$`out-dir`)) {
  stop("Missing required --inputs and/or --out-dir", call. = FALSE)
}

# ---------- Helpers ----------
trim_ws <- function(x) str_trim(x)

split_inputs <- function(s) {
  items <- str_split(s, ",")[[1]]
  items <- trim_ws(items)
  items[nchar(items) > 0]
}

parse_item <- function(item) {
  # allow either "name=/path" or "name:/path" or just "/path"
  if (str_detect(item, "=") || str_detect(item, ":")) {
    # split on first '=' or ':'
    m <- regexpr("=|:", item)
    name <- substr(item, 1, m - 1)
    path <- substr(item, m + 1, nchar(item))
  } else {
    path <- item
    # infer species name from parent folder; fallback to file stem
    name <- basename(dirname(normalizePath(path, mustWork = FALSE)))
    if (is.na(name) || name == "" || name == ".") {
      name <- file_path_sans_ext(basename(path))
    }
  }
  if (name == "" || path == "") stop(sprintf("Invalid input item: '%s'", item), call. = FALSE)
  list(name=name, path=path)
}

safe_read <- function(path) {
  if (!file.exists(path)) stop(sprintf("Input file not found: %s", path), call. = FALSE)
  suppressMessages(read_csv(path, show_col_types = FALSE))
}

split_sequence <- function(df, min_local_pos) {
  df %>%
    mutate(
      Local_Position = suppressWarnings(as.integer(Local_Position)),
      small_ds_seq   = as.character(small_ds_seq)
    ) %>%
    filter(!is.na(Local_Position), Local_Position >= min_local_pos,
           !is.na(small_ds_seq), nchar(small_ds_seq) >= Local_Position) %>%
    mutate(
      L = substr(small_ds_seq, 1, pmax(Local_Position - 1, 0)),
      R = substr(small_ds_seq, Local_Position + 1, nchar(small_ds_seq))
    )
}

balance_data <- function(df, pos_thr, neg_thr, seed) {
  df2 <- df %>%
    mutate(EditingLevel = suppressWarnings(as.numeric(EditingLevel))) %>%
    mutate(EditingClass = case_when(
      !is.na(EditingLevel) & EditingLevel >  pos_thr ~ "yes",
      !is.na(EditingLevel) & EditingLevel <  neg_thr ~ "no",
      TRUE ~ NA_character_
    )) %>%
    tidyr::drop_na(EditingClass)

  count_yes <- sum(df2$EditingClass == "yes")
  count_no  <- sum(df2$EditingClass == "no")
  min_count <- min(count_yes, count_no)

  if (min_count == 0) {
    warning("Skipping species: one class has zero samples after thresholds.")
    return(NULL)
  }

  set.seed(seed)
  balanced_yes <- df2 %>% filter(EditingClass == "yes") %>% dplyr::sample_n(min_count)
  balanced_no  <- df2 %>% filter(EditingClass == "no")  %>% dplyr::sample_n(min_count)
  bind_rows(balanced_yes, balanced_no)
}

final_downsample <- function(df_balanced, per_class, seed) {
  set.seed(seed)
  df_balanced %>%
    group_by(EditingClass) %>%
    dplyr::sample_n(size = per_class) %>%
    ungroup()
}

write_species_outputs <- function(sp_name, df_final, out_dir, train_frac, seed) {
  df_ml <- df_final %>%
    mutate(structure = mfe_struct, yes_no = EditingClass) %>%
    select(structure, L, R, yes_no)

  set.seed(seed)
  df_ml <- df_ml %>% dplyr::sample_frac(1.0)

  n <- nrow(df_ml)
  n_train <- max(1, floor(train_frac * n))
  idx <- sample.int(n, size = n_train)
  train <- df_ml[idx, , drop = FALSE]
  valid <- df_ml[-idx, , drop = FALSE]

  species_dir <- file.path(out_dir, sp_name)
  dir.create(species_dir, recursive = TRUE, showWarnings = FALSE)

  data_for_prepare_path <- file.path(species_dir, sprintf("data_for_prepare_%s.csv", sp_name))
  suppressMessages(write_csv(df_final %>% select(mfe_struct, L, R, EditingClass), data_for_prepare_path))

  train_path <- file.path(species_dir, sprintf("final_%s_train.csv", sp_name))
  valid_path <- file.path(species_dir, sprintf("final_%s_valid.csv", sp_name))
  suppressMessages(write_csv(train, train_path))
  suppressMessages(write_csv(valid, valid_path))

  message(sprintf("[%s] saved: %s ; %s ; %s",
                  sp_name, data_for_prepare_path, train_path, valid_path))
}

# ---------- Main ----------
items <- split_inputs(opt$inputs)
if (length(items) == 0) stop("No inputs provided in --inputs.", call. = FALSE)

parsed <- lapply(items, parse_item)
out_dir <- opt$`out-dir`
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

species_dfs <- list()
sizes_per_class <- c()

for (spec in parsed) {
  sp <- spec$name
  path <- spec$path
  message(sprintf("Reading species '%s' from: %s", sp, path))
  df <- safe_read(path)

  required_cols <- c("small_ds_seq","Local_Position","EditingLevel","mfe_struct")
  missing <- setdiff(required_cols, names(df))
  if (length(missing)) {
    warning(sprintf("[%s] Missing required columns: %s — skipping.",
                    sp, paste(missing, collapse=", ")))
    next
  }

  df_spl <- split_sequence(df, min_local_pos = opt$`min-local-pos`)
  if (nrow(df_spl) == 0) {
    warning(sprintf("[%s] No rows after Local_Position/sequence filtering — skipping.", sp))
    next
  }

  df_bal <- balance_data(df_spl, pos_thr = opt$`pos-threshold`, neg_thr = opt$`neg-threshold`, seed = opt$seed)
  if (is.null(df_bal) || nrow(df_bal) == 0) {
    warning(sprintf("[%s] No rows after class balancing — skipping.", sp))
    next
  }

  per_class_count <- sum(df_bal$EditingClass == "yes")
  species_dfs[[sp]] <- df_bal
  sizes_per_class <- c(sizes_per_class, per_class_count)
  message(sprintf("[%s] balanced per-class size: %d (total %d)", sp, per_class_count, 2 * per_class_count))
}

if (length(species_dfs) == 0) stop("No species remained after filtering/balancing.", call. = FALSE)

if (isTRUE(opt$`equalize-across`)) {
  target_per_class <- min(sizes_per_class)
  message(sprintf("Equalizing across species to per-class = %d (total per species = %d)",
                  target_per_class, 2 * target_per_class))
  species_dfs <- setNames(
    lapply(names(species_dfs), function(sp) final_downsample(species_dfs[[sp]], per_class = target_per_class, seed = opt$seed)),
    names(species_dfs)
  )
}

for (sp in names(species_dfs)) {
  write_species_outputs(sp, species_dfs[[sp]], out_dir = out_dir, train_frac = opt$`train-frac`, seed = opt$seed)
}

message("Done.")
