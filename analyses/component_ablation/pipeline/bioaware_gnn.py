#!/usr/bin/env python3
"""Compatibility import for the repository's canonical bio-aware implementation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


SOURCE = Path(__file__).resolve().parents[3] / "code" / "bioaware_gnn.py"
SPEC = spec_from_file_location("_adaredit_canonical_bioaware", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load canonical bio-aware implementation: {SOURCE}")
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

for NAME in dir(MODULE):
    if not NAME.startswith("_"):
        globals()[NAME] = getattr(MODULE, NAME)

__all__ = [name for name in dir(MODULE) if not name.startswith("_")]
