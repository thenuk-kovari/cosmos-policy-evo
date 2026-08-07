#!/usr/bin/env python3
"""Precompute the single towel instruction embedding used by this dataset."""

from __future__ import annotations

import argparse

from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings, save_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--task", default="fold the blue towel twice")
    args = parser.parse_args()
    save_embeddings(generate_t5_embeddings([args.task]), args.data_dir)


if __name__ == "__main__":
    main()
