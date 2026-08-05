#!/usr/bin/env python3
"""Fail early unless the dedicated attention-analysis environment is active."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib
import numpy
import pandas
import PIL
import seaborn
import shap
import sklearn
import torch
import torch_geometric
import xgboost


ANALYSIS = Path(__file__).resolve().parents[1]
REQUIRED = {
    "python": "3.10",
    "torch": "1.11.0",
    "torch_geometric": "2.0.4",
    "xgboost": "1.7.5",
    "shap": "0.49.1",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "scikit_learn": "1.6.1",
    "matplotlib": "3.9.4",
    "pillow": "11.3.0",
    "seaborn": "0.13.2",
}


def base_version(value: str) -> str:
    return value.split("+", 1)[0]


def main() -> None:
    observed = {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "torch": base_version(torch.__version__),
        "torch_geometric": torch_geometric.__version__,
        "xgboost": xgboost.__version__,
        "shap": shap.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__,
        "seaborn": seaborn.__version__,
    }
    mismatches = {
        name: {"required": REQUIRED[name], "observed": observed[name]}
        for name in REQUIRED
        if observed[name] != REQUIRED[name]
    }
    record = {
        "environment_file": "environment.yml",
        "required": REQUIRED,
        "observed": observed,
        "status": "PASS" if not mismatches else "FAIL",
    }
    (ANALYSIS / "data" / "runtime_environment.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    if mismatches:
        details = "; ".join(
            f"{name}: required {item['required']}, observed {item['observed']}"
            for name, item in mismatches.items()
        )
        raise RuntimeError(
            "The dedicated attention environment is not active. " + details
        )
    print("PASS: dedicated attention-analysis environment verified")


if __name__ == "__main__":
    main()
