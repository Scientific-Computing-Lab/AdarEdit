#!/usr/bin/env python3
"""
Cross-species sensitivity analysis on the region-disjoint species models.

For each species we take the held-out TEST split and the predictions for
the within-species model, for both the baseline and bio-aware architectures.
Each negative test adenosine is assigned a proximity = minimum bp distance to
the nearest edited (positive; >15% in the dataset-construction code) adenosine
in the same dsRNA region and substrand.
We then recompute tie-aware AUROC and F1 at the model's validation-selected
threshold after progressively excluding negatives that lie within 5, 10, and
30 bp of a positive.

Predictions are read directly from the shipped per-model test prediction CSVs.
Row indices and labels are verified against the ordered test JSONL.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "data"   # sibling of scripts/, so a copied tree writes to itself
OUT.mkdir(parents=True, exist_ok=True)
LINKER = "NNNNNNNNNN"

SPECIES = [
    {"name": "Octopus bimaculoides", "short": "Octopus", "dir": "Octopus"},
    {"name": "Ptychodera flava", "short": "Ptychodera", "dir": "Ptychodera"},
    {"name": "Strongylocentrotus purpuratus", "short": "S. purpuratus", "dir": "Strongylocentrotus"},
]


def parse_jsonl(path):
    recs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            content = d["messages"][1]["content"]
            label = d["messages"][2]["content"].strip().lower()
            L = content.split("L:", 1)[1].split(", A:", 1)[0]
            R = content.split(", R:", 1)[1].split(", Alu", 1)[0]
            full_seq = L + "A" + R
            pos = len(L) + 1
            lp = full_seq.find(LINKER)
            sub = 1 if lp == -1 or pos <= lp else 2
            recs.append({"full_seq": full_seq, "pos": pos, "sub": sub,
                         "label": 1 if label == "yes" else 0})
    return recs


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(model_key, species_dir, test):
    """Load and validate the canonical within-species test predictions."""
    prediction_path = (
        REPO
        / "checkpoints"
        / f"{model_key}_{species_dir}"
        / "test_predictions.csv"
    )
    rows = list(csv.DictReader(prediction_path.open()))
    if len(rows) != len(test):
        raise RuntimeError(
            f"{species_dir}/{model_key}: {len(rows)} predictions for "
            f"{len(test)} test records"
        )

    observed_indices = [int(row["source_row_index"]) for row in rows]
    if observed_indices != list(range(len(test))):
        raise RuntimeError(
            f"{species_dir}/{model_key}: prediction row indices are not ordered"
        )
    jsonl_labels = [record["label"] for record in test]
    prediction_labels = [int(float(row["label_from_loader"])) for row in rows]
    if prediction_labels != jsonl_labels:
        raise RuntimeError(
            f"{species_dir}/{model_key}: prediction/JSONL label-order mismatch"
        )
    if any(row["variant"] != model_key for row in rows):
        raise RuntimeError(f"{species_dir}/{model_key}: variant mismatch")
    if any(row["tissue"] != species_dir for row in rows):
        raise RuntimeError(f"{species_dir}/{model_key}: context mismatch")

    thresholds = {float(row["threshold"]) for row in rows}
    if len(thresholds) != 1:
        raise RuntimeError(
            f"{species_dir}/{model_key}: inconsistent prediction thresholds"
        )
    threshold = thresholds.pop()

    summary_path = (
        REPO / "checkpoints" / f"{model_key}_{species_dir}" / "summary.json"
    )
    summary = json.loads(summary_path.read_text())
    summary_threshold = float(summary["test_metrics"]["threshold"])
    if threshold != summary_threshold:
        raise RuntimeError(
            f"{species_dir}/{model_key}: prediction threshold {threshold} "
            f"!= summary threshold {summary_threshold}"
        )

    return {
        "probs": [float(row["prob"]) for row in rows],
        "threshold": threshold,
        "prediction_file": str(prediction_path.relative_to(REPO)),
        "prediction_sha256": sha256(prediction_path),
        "summary_file": str(summary_path.relative_to(REPO)),
        "summary_sha256": sha256(summary_path),
    }


def analyze(sp):
    d = REPO / "data/species" / sp["dir"]
    train = parse_jsonl(d / "train.jsonl")
    valid = parse_jsonl(d / "valid.jsonl")
    test = parse_jsonl(d / "test.jsonl")

    # yes-index over ALL splits (regions are split-disjoint, so same-region positives
    # of a test negative live in the test split; union is safe and complete)
    yes_idx = defaultdict(list)
    for rec in train + valid + test:
        if rec["label"] == 1:
            yes_idx[(rec["full_seq"], rec["sub"])].append(rec["pos"])
    yes_idx = {k: sorted(v) for k, v in yes_idx.items()}

    for rec in test:
        if rec["label"] == 0:
            pl = yes_idx.get((rec["full_seq"], rec["sub"]))
            rec["prox"] = min(abs(rec["pos"] - yp) for yp in pl) if pl else None
        else:
            rec["prox"] = 0

    results = {}
    prediction_sources = {}
    for model_key, model_name in [("baseline", "Baseline GAT"), ("bioaware", "Bio-aware GNN")]:
        source = load_predictions(model_key, sp["dir"], test)
        probs = source.pop("probs")
        for rec, p in zip(test, probs):
            rec[f"prob_{model_key}"] = float(p)

        thr = source["threshold"]
        prediction_sources[model_name] = source
        rows = []
        for excl, tag in [(None, "All sites"), (5, "Excl. neg ≤5 bp"),
                          (10, "Excl. neg ≤10 bp"), (30, "Excl. neg ≤30 bp")]:
            if excl is None:
                sub = test
            else:
                # A negative with prox is None has NO >15% site anywhere in its
                # region+substrand, i.e. it is maximally far from any positive.
                # A minimum-distance criterion must RETAIN those, not drop them.
                sub = [r for r in test if r["label"] == 1
                       or (r["label"] == 0 and (r["prox"] is None or r["prox"] > excl))]
            lab = [r["label"] for r in sub]
            pr = [r[f"prob_{model_key}"] for r in sub]
            rows.append({"filter": tag, "n": len(sub), "n_pos": sum(lab),
                         "n_neg": len(lab) - sum(lab), "threshold": thr,
                         "auroc": float(roc_auc_score(lab, pr)),
                         "f1": float(f1_score(lab, [p >= thr for p in pr]))})
        results[model_name] = rows
        a0 = rows[0]
        print(f"{sp['short']:14s} {model_name:14s} ALL n={a0['n']} (+{a0['n_pos']}/-{a0['n_neg']}) "
              f"AUROC={a0['auroc']:.3f} F1={a0['f1']:.3f}", flush=True)

    return {"name": sp["name"], "short": sp["short"], "results": results,
            "prediction_sources": prediction_sources}


def main():
    allr = [analyze(sp) for sp in SPECIES]
    (OUT / "sensitivity.json").write_text(json.dumps(allr, indent=2) + "\n")
    print("\nsaved", OUT / "sensitivity.json")


if __name__ == "__main__":
    main()
