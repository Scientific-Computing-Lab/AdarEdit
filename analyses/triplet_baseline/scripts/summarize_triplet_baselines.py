#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "analyses/triplet_baseline/results"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact GitHub-flavoured Markdown table without extra packages."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, float):
                rendered.append(f"{value:.10g}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize triplet-style baseline runs into CSV and Markdown tables.")
    parser.add_argument("--input_dir", default=str(RESULTS))
    parser.add_argument("--out_csv", default=str(RESULTS / "summary.csv"))
    parser.add_argument("--out_md", default=str(RESULTS / "summary.md"))
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.input_dir).glob("*.json")):
        payload = json.loads(path.read_text())
        row = {
            "tissue": path.stem.split("_")[0],
            "model": payload["variant"],
            "feature_radius": payload["feature_radius"],
            "best_C": payload["best_hyperparams"]["C"],
            "best_gamma": payload["best_hyperparams"]["gamma"],
            "train_size": payload["sizes"]["train"],
            "valid_size": payload["sizes"]["valid"],
            "test_size": payload["sizes"]["test"],
        }
        for prefix in ["valid", "test"]:
            for metric_name, metric_value in payload[f"{prefix}_metrics"].items():
                row[f"{prefix}_{metric_name}"] = metric_value
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["tissue", "model"]).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    best_by_tissue = (
        df.sort_values(["tissue", "test_f1", "test_auc"], ascending=[True, False, False])
        .groupby("tissue", as_index=False)
        .first()
    )
    lines = [
        "# Triplet Baseline Summary",
        "",
        "## All Runs",
        "",
        markdown_table(df),
        "",
        "## Best Per Tissue",
        "",
        markdown_table(best_by_tissue),
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
