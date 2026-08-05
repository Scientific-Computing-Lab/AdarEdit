#!/usr/bin/env python3
"""
SAFETY CHECK — run before training to verify that species RNA sequences
containing 'U' are encoded correctly.

Species sequences use 'U'; human sequences use 'T'. The graph builders must
treat them equivalently:
  - baseline gnnadar_verb_compact.py must map BOTH 'U' and 'T' to [0,0,0,1,0]
  - bio-aware bioaware_gnn.py must normalise with .replace("T","U")

Set ADAREDIT_MODEL_DIR to the model directory if it differs from `code/`, then run.
Exits non-zero (and prints FAIL) if anything is wrong.
"""
import ast
import os
import re
import sys
from pathlib import Path

# the directory the trainer inserts first on sys.path (train_strict_long.py: MODEL_ROOT)
MODEL_ROOT = Path(os.environ.get("ADAREDIT_MODEL_DIR",
                                 Path(__file__).resolve().parents[2] / "code"))

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"[{'OK ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail else ''}")
    ok = ok and cond

base = (MODEL_ROOT / "gnnadar_verb_compact.py")
bio  = (MODEL_ROOT / "bioaware_gnn.py")

check("baseline module exists at MODEL_ROOT", base.exists(), str(base))
check("bio-aware module exists at MODEL_ROOT", bio.exists(), str(bio))

if base.exists():
    src = base.read_text()
    tree = ast.parse(src, filename=str(base))
    base_map = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "BASE_MAP"
                for target in node.targets
            )
        ):
            base_map = ast.literal_eval(node.value)
            break
    expected_pyrimidine = [0, 0, 0, 1, 0]
    check(
        "baseline maps 'U' to the pyrimidine channel",
        base_map is not None and base_map.get("U") == expected_pyrimidine,
    )
    check(
        "baseline maps 'T' to the same pyrimidine channel",
        base_map is not None and base_map.get("T") == expected_pyrimidine,
    )
    check(
        "baseline parser accepts both 'U' and 'T'",
        base_map is not None and {"U", "T"}.issubset(base_map),
    )
    # Functional check: run the real parser on a species record; U must survive.
    _sp_jsonl = Path(__file__).resolve().parents[2] / "data" / "species" / "Octopus" / "train.jsonl"
    if _sp_jsonl.exists():
        try:
            sys.path.insert(0, str(MODEL_ROOT))
            import json as _json
            import gnnadar_verb_compact as _g
            _rec = _sp_jsonl.read_text().split("\n", 1)[0]
            _seq = _g.parse_openai_json(_json.loads(_rec))[0]
            check("parser preserves U on a real species record", "U" in _seq,
                  f"U in parsed sequence = {_seq.count('U')}")
        except Exception as _e:  # noqa: BLE001
            check("parser functional U check ran", False, f"{type(_e).__name__}: {_e}")

if bio.exists():
    src = bio.read_text()
    check("bio-aware normalises T->U  (.replace(\"T\",\"U\"))",
          'replace("T", "U")' in src or "replace('T', 'U')" in src)
    check("bio-aware BASES include 'U'", re.search(r"BASES\s*=\s*\[[^\]]*['\"]U['\"]", src) is not None)

# also confirm the actual exported species data really contains 'U'
EXPORT = Path(__file__).resolve().parent / "export" / "species"
for sp in ("Octopus", "Ptychodera", "Strongylocentrotus"):
    f = EXPORT / sp / "train.jsonl"
    if f.exists():
        first = f.read_text().split("\n", 1)[0]
        check(f"exported species {sp} sequence uses 'U' (RNA, no 'T')",
              ("U" in first) and ("T" not in first),
              f"U={first.count('U')} T={first.count('T')}")

print("\nVERDICT:", "SAFE TO TRAIN — U/T handled correctly" if ok else "DO NOT TRAIN — check module path")
sys.exit(0 if ok else 1)
