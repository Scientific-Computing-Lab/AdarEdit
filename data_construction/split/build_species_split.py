#!/usr/bin/env python3
"""Compatibility entry point for the canonical species benchmark builder.

The documented entry point is ``build_species_benchmark.py``. Both filenames
invoke the same region-disjoint dataset builder.
"""

from build_species_benchmark import main


if __name__ == "__main__":
    main()
