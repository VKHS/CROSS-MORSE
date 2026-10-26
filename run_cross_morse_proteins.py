#!/usr/bin/env python3
"""CROSS-MORSE entry point for ogbn-proteins."""
from run_cross_morse import main


if __name__ == "__main__":
    raise SystemExit(main(default_dataset='ogbn-proteins'))
