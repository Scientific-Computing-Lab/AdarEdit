#!/usr/bin/env Rscript

# Reference implementation of binary labelling and class balancing.
#
# For deterministic reconstruction across software versions, use
# select_published_site_pool.py and published_site_selection.tsv.gz.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(purrr)
  library(fs)
  library(tibble)
  library(argparse)
})

parser <- ArgumentParser(
  description = "Build balanced cross-tissue train/valid splits."
)
parser$add_argument("--data_dir", required = TRUE)
parser$add_argument("--output_dir", required = TRUE)
parser$add_argument("--seed", type = "integer", default = 42)
parser$add_argument("--yes_cutoff", type = "double", default = 15)
parser$add_argument("--no_cutoff", type = "double", default = 1)
parser$add_argument("--train_size", type = "integer", default = 19200)
parser$add_argument("--valid_size", type = "integer", default = 4800)
parser$add_argument("--structure_col", default = "structure")
parser$add_argument("--L_col", default = "L")
parser$add_argument("--R_col", default = "R")
parser$add_argument("--edit_col", default = "EditingIndex")
parser$add_argument(
  "--skip_if_insufficient", action = "store_true", default = FALSE
)
args <- parser$parse_args()

balanced_sample <- function(df, n_each) {
  bind_rows(
    df |> filter(yes_no == "yes") |> slice_sample(n = n_each),
    df |> filter(yes_no == "no") |> slice_sample(n = n_each)
  ) |> sample_frac(1)
}

check_enough <- function(df, label, need, who, skip = FALSE) {
  have <- df |> filter(yes_no == label) |> nrow()
  if (have < need) {
    msg <- paste0(
      "Not enough ", label, " in ", who, " (need ", need,
      ", have ", have, ")"
    )
    if (skip) {
      message(msg, " -- skipping.")
      return(FALSE)
    }
    stop(msg, call. = FALSE)
  }
  TRUE
}

set.seed(args$seed)
dir_create(args$output_dir)
csv_paths <- dir_ls(args$data_dir, glob = "*.csv")
if (length(csv_paths) == 0) {
  stop("No CSV files found in --data_dir", call. = FALSE)
}
tissue_names <- path_ext_remove(path_file(csv_paths))

tissue_dfs <- map(
  csv_paths,
  ~ suppressMessages(read_csv(.x, show_col_types = FALSE))
) |>
  set_names(tissue_names) |>
  map(\(.x) {
    .x |>
      mutate(
        key = paste(
          .data[[args$structure_col]],
          .data[[args$L_col]],
          .data[[args$R_col]],
          sep = "|"
        ),
        yes_no = case_when(
          .data[[args$edit_col]] >= args$yes_cutoff ~ "yes",
          .data[[args$edit_col]] < args$no_cutoff ~ "no",
          TRUE ~ NA_character_
        )
      ) |>
      filter(!is.na(yes_no))
  })

summary_log <- list()
train_each <- as.integer(args$train_size / 2)
valid_each <- as.integer(args$valid_size / 2)

for (train_tissue in tissue_names) {
  message("Building train set for: ", train_tissue)
  df_train_all <- tissue_dfs[[train_tissue]] |> sample_frac(1)
  if (!check_enough(
    df_train_all, "yes", train_each, paste0(train_tissue, " train"),
    args$skip_if_insufficient
  )) next
  if (!check_enough(
    df_train_all, "no", train_each, paste0(train_tissue, " train"),
    args$skip_if_insufficient
  )) next

  df_train_final <- balanced_sample(df_train_all, train_each)
  train_keys <- df_train_final$key
  outer_dir <- path(args$output_dir, train_tissue)
  dir_create(outer_dir)

  for (valid_tissue in tissue_names) {
    df_valid_raw <- tissue_dfs[[valid_tissue]]
    df_valid_clean <- df_valid_raw |> filter(!key %in% train_keys)
    enough_yes <- check_enough(
      df_valid_clean, "yes", valid_each,
      paste0(valid_tissue, " valid (after filtering)"),
      args$skip_if_insufficient
    )
    enough_no <- check_enough(
      df_valid_clean, "no", valid_each,
      paste0(valid_tissue, " valid (after filtering)"),
      args$skip_if_insufficient
    )
    if (!(enough_yes && enough_no)) next

    df_valid_final <- balanced_sample(df_valid_clean, valid_each)
    pair_dir <- path(
      outer_dir, paste0(train_tissue, "_", valid_tissue)
    )
    dir_create(pair_dir)
    retained <- c(
      args$structure_col, args$L_col, args$R_col, "yes_no"
    )
    write_csv(
      df_train_final |> select(all_of(retained)),
      path(pair_dir, paste0(train_tissue, "_train.csv"))
    )
    write_csv(
      df_valid_final |> select(all_of(retained)),
      path(pair_dir, paste0(valid_tissue, "_valid.csv"))
    )

    summary_log[[length(summary_log) + 1]] <- tibble(
      train_tissue = train_tissue,
      valid_tissue = valid_tissue,
      train_yes = sum(df_train_final$yes_no == "yes"),
      train_no = sum(df_train_final$yes_no == "no"),
      valid_yes = sum(df_valid_final$yes_no == "yes"),
      valid_no = sum(df_valid_final$yes_no == "no"),
      valid_filtered_total = nrow(df_valid_clean),
      overlap_removed = nrow(df_valid_raw) - nrow(df_valid_clean)
    )
  }
}

if (length(summary_log)) {
  write_csv(
    bind_rows(summary_log),
    path(args$output_dir, "cross_split_summary.csv")
  )
}
