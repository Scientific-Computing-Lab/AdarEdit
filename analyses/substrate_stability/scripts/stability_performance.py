#!/usr/bin/env python3
"""
Model performance stratified by substrate structural stability on the
pair-disjoint held-out test split.

Robust alignment (no prefix keys, no collisions): the graph builder skips any record
whose sequence length does not equal its dot-bracket length; removing exactly those
records from the ordered test.jsonl reproduces the prediction order 1:1 (verified by
label equality). Each prediction is therefore matched to its exact substrate, and the
substrate is identified by its FULL predicted structure.

Stability = paired fraction of the full RNAfold structure. Quartiles are computed over
the 884 unique substrates across the complete human benchmark; sites are assigned to
the low-stability (<=Q1) and high-stability (>=Q3) cohorts by their substrate. For each
cohort and architecture we report F1 (at the model's validation-selected threshold)
and tie-aware AUROC.
"""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

from sklearn.metrics import f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "checkpoints"
DATA = REPO / "data/human"
OUT = REPO / "analyses/substrate_stability/data"
OUT.mkdir(parents=True, exist_ok=True)
TISSUES = ["Artery", "Brain", "Liver", "MuscleSkeletal", "Combined"]


def parse(js):
    c = js["messages"][1]["content"]
    L = c.split("L:", 1)[1].split(", A:", 1)[0]
    R = c.split(", R:", 1)[1].split(", Alu", 1)[0] if ", R:" in c else ""
    seq = L + "A" + R
    struct = c.split("Structure:", 1)[1].strip()
    label = 1 if js["messages"][2]["content"].strip().lower() == "yes" else 0
    return seq, struct, label


def paired_frac(struct):
    return (struct.count("(") + struct.count(")")) / len(struct) if struct else 0.0


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(recs, thr):
    lab = [r["label"] for r in recs]; pr = [r["prob"] for r in recs]
    return {"n": len(recs), "n_pos": sum(lab),
            "f1": round(float(f1_score(lab, [p >= thr for p in pr])), 4),
            "auroc": round(float(roc_auc_score(lab, pr)), 4)}


# ---- GLOBAL substrate stability: unique Alu substrates over the whole dataset ----
# (structure is tissue-independent; use Combined train+valid+test for the full set)
global_pf = {}
for sp in ["train", "valid", "test"]:
    for line in open(DATA / "Combined" / f"{sp}.jsonl"):
        seq, struct, _ = parse(json.loads(line))
        if len(seq) == len(struct):
            global_pf[struct] = paired_frac(struct)
gvals = sorted(global_pf.values())
GQ1 = gvals[len(gvals) // 4]; GQ3 = gvals[3 * len(gvals) // 4]
print(f"GLOBAL substrates={len(global_pf)}  paired_frac: min={min(gvals):.3f} Q1={GQ1:.3f} "
      f"median={st.median(gvals):.3f} Q3={GQ3:.3f} max={max(gvals):.3f}", flush=True)

results = {
    "_global": {
        "n_substrates": len(global_pf),
        "q1": round(GQ1, 4),
        "q3": round(GQ3, 4),
        "median": round(st.median(gvals), 4),
        "paired_frac_values": [round(v, 4) for v in gvals],
        "cohort_definition": "global benchmark quartiles",
        "evaluation_split": "pair-disjoint test",
    }
}
for tissue in TISSUES:
    jl = [json.loads(l) for l in open(DATA / tissue / "test.jsonl")]
    # keep only graph-buildable records (len(seq) == len(struct)); preserves order
    kept = []
    for js in jl:
        seq, struct, label = parse(js)
        if len(seq) == len(struct):
            kept.append(
                {
                    "seq": seq,
                    "struct": struct,
                    "pf": paired_frac(struct),
                    "label": label,
                }
            )
    q1, q3 = GQ1, GQ3  # global stability thresholds for all tissues
    results[tissue] = {"n_sites": len(kept), "performance": {}}

    for variant, name in [("baseline", "Baseline GAT"), ("bioaware", "Bio-aware GNN")]:
        prediction_path = PKG / f"{variant}_{tissue}/test_predictions.csv"
        pred = list(csv.DictReader(open(prediction_path)))
        assert len(pred) == len(kept), f"{tissue} {variant}: {len(pred)} vs {len(kept)}"
        thresholds = {float(row["threshold"]) for row in pred}
        assert len(thresholds) == 1, f"{tissue} {variant}: inconsistent thresholds"
        thr = thresholds.pop()
        recs = []
        for k, p in zip(kept, pred):
            assert int(float(p["label_from_loader"])) == k["label"], "order mismatch"
            assert p["split"] == "test", "non-test prediction row"
            assert p["variant"] == variant, "variant mismatch"
            assert p["tissue"] == tissue, "context mismatch"
            prefix = p["pair_prefix"]
            assert prefix == k["seq"][:len(prefix)], "sequence-order mismatch"
            recs.append({"prob": float(p["prob"]), "label": k["label"], "pf": k["pf"]})
        allm = metrics(recs, thr)
        low = metrics([r for r in recs if r["pf"] <= q1], thr)
        high = metrics([r for r in recs if r["pf"] >= q3], thr)
        results[tissue]["performance"][name] = {
            "threshold": thr,
            "prediction_file": str(
                prediction_path.relative_to(REPO)
            ),
            "prediction_sha256": sha256(prediction_path),
            "all": allm,
            "low_stability": low,
            "high_stability": high,
        }
        print(f"{tissue:9s} {name:14s} thr={thr:.3f} | ALL n={allm['n']} F1={allm['f1']} AUROC={allm['auroc']}"
              f" | LOW<=Q1 n={low['n']} F1={low['f1']} AUROC={low['auroc']}"
              f" | HIGH>=Q3 n={high['n']} F1={high['f1']} AUROC={high['auroc']}", flush=True)

(OUT / "stability_performance.json").write_text(json.dumps(results, indent=2))
print("\nwrote", OUT / "stability_performance.json")
