#!/usr/bin/env python3
"""
Check vocabulary coverage of AdvancedRISCVTokenizer on the (fixed) dataset.

Author: ywangmu from HKUST

This script loads a TrainingSample pickle dataset, tokenizes all instructions
with AdvancedRISCVTokenizer, and reports how many tokens are out-of-vocabulary
with respect to tokenizer.vocab.
"""

import argparse
import os
import pickle
import sys
from collections import Counter
from typing import List

from tqdm import tqdm

# Make local modules importable (same pattern as other scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, "scripts"))

from prepare_balanced_dataset import TrainingSample  # noqa: F401
from advanced_tokenizer import AdvancedRISCVTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check AdvancedRISCVTokenizer vocabulary coverage on dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_data = os.path.join(
        script_dir,
        "dataset",
        "final_training_data_fp_fixed.pkl",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=default_data,
        help="Input dataset pickle (list[TrainingSample])",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to scan (for quick checks)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Show top-K most frequent OOV tokens",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    print("=" * 80)
    print("CHECKING TOKENIZER VOCABULARY COVERAGE")
    print("=" * 80)
    print(f"Dataset path: {args.data}")

    with open(args.data, "rb") as f:
        samples = pickle.load(f)

    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset is empty or not a list.")

    total_samples = len(samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print(f"Using first {len(samples)} / {total_samples} samples for coverage check.")
    else:
        print(f"Total samples: {total_samples}")

    tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=False)
    vocab = tokenizer.vocab

    total_tokens = 0
    oov_tokens = 0
    per_token_counts = Counter()
    oov_counts = Counter()

    for sample in tqdm(samples, desc="Scanning samples", unit="sample"):
        inst_list: List[str] = getattr(sample, "instructions", [])
        for inst in inst_list:
            tokens = tokenizer.tokenize_instruction(inst)
            for t in tokens:
                total_tokens += 1
                per_token_counts[t] += 1
                if t not in vocab:
                    oov_tokens += 1
                    oov_counts[t] += 1

    print("\n" + "=" * 80)
    print("COVERAGE SUMMARY")
    print("=" * 80)
    print(f"Total tokens (after tokenization): {total_tokens}")
    print(f"Total unique tokens:               {len(per_token_counts)}")
    print(f"Tokenizer vocab size:              {len(vocab)}")
    print(f"OOV token count:                   {oov_tokens}")
    oov_ratio = (oov_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0
    print(f"OOV token ratio:                   {oov_ratio:.6f}%")
    print(f"Unique OOV token types:            {len(oov_counts)}")

    if oov_tokens == 0:
        print("\nAll tokenizer output tokens are covered by the vocabulary.")
        return

    print("\nTop OOV tokens:")
    for tok, cnt in oov_counts.most_common(args.top_k):
        freq = cnt / total_tokens * 100.0
        print(f"  {tok:20s}  count={cnt:10d}  freq={freq:8.6f}%")


if __name__ == "__main__":
    main()


