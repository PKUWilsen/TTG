#!/usr/bin/env python3
"""Placeholder visualization entry point.

For full evaluation visualizations, use scripts/eval_refine_phase.py with
``evaluation.num_samples_to_visualize`` set to a positive value in the config.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="TTG visualization helper.")
    parser.add_argument("--note", action="store_true", help="Print visualization instructions.")
    _ = parser.parse_args()
    print("Use scripts/eval_refine_phase.py and set evaluation.num_samples_to_visualize > 0 in configs/eval.yaml.")


if __name__ == "__main__":
    main()
