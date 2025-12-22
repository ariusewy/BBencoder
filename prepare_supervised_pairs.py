#!/usr/bin/env python3
"""
Precompute supervised training pairs with BB similarity scores.

This script reuses the scoring logic from evaluate_bb_similarity.py to
generate (sample_a, sample_b, similarity) tuples and saves them for reuse
in supervised training. This avoids recomputing expensive LCS + dependency
scoring each time train.py is invoked.

Author: ywangmu from HKUST
"""

import argparse
import os
import pickle
import random
import sys
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

from advanced_tokenizer import AdvancedRISCVTokenizer
from prepare_balanced_dataset import TrainingSample
from evaluate_bb_similarity import (
    comp_ins,
    weighted_lcs,
    normalize_score,
    dep_consistency_bonus
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute supervised BB similarity pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'final_training_data.pkl')
    parser.add_argument('--data', type=str, default=default_data,
                        help='Input dataset pickle (list[TrainingSample])')
    parser.add_argument('--output', type=str, default='precomputed_pairs.pkl',
                        help='Output pickle path')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Subset of samples to use (None = all)')
    parser.add_argument('--num-pairs', type=int, default=5,
                        help='Pairs to generate per sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for partner selection')
    parser.add_argument('--normalize', type=str, default='maxlen',
                        choices=['maxlen', 'avglen', 'none'],
                        help='Normalization strategy for similarity scores')
    parser.add_argument('--enable-dep', action='store_true',
                        help='Enable dependency consistency bonus')
    parser.add_argument('--beta-dep', type=float, default=0.5,
                        help='Dependency bonus weight')
    parser.add_argument('--gamma-mis', type=float, default=0.05,
                        help='Dependency mismatch penalty')
    parser.add_argument('--dep-window', type=int, default=3,
                        help='Dependency matching window')
    parser.add_argument('--B-exact', type=float, default=2.0)
    parser.add_argument('--alpha-sub', type=float, default=0.7)
    parser.add_argument('--alpha-unit', type=float, default=0.45)
    parser.add_argument('--B-sameop', type=float, default=1.0)
    parser.add_argument('--B-sameimm', type=float, default=3.0)
    return parser.parse_args()


def load_samples(path: str, limit: int = None) -> List[TrainingSample]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Dataset is empty or malformed.")
    if limit and limit < len(samples):
        samples = samples[:limit]
    return samples


def score_pair(
    idx_a: int,
    idx_b: int,
    tokenized_instrs: List[List[List[str]]],
    tokenizer: AdvancedRISCVTokenizer,
    args,
    sidecars: List[Dict[str, Any]]
) -> float:
    toks_a = tokenized_instrs[idx_a]
    toks_b = tokenized_instrs[idx_b]

    def score_fn(i: int, j: int) -> float:
        return comp_ins(
            toks_a[i], toks_b[j], tokenizer,
            B_exact=args.B_exact,
            alpha_sub=args.alpha_sub,
            alpha_unit=args.alpha_unit,
            B_sameop=args.B_sameop,
            B_sameimm=args.B_sameimm
        )

    raw_score_seq, pairs = weighted_lcs(toks_a, toks_b, score_fn)
    dep_bonus = 0.0
    if args.enable_dep and len(pairs) > 0:
        side_a = sidecars[idx_a]
        side_b = sidecars[idx_b]
        dep_bonus = dep_consistency_bonus(
            pairs, side_a, side_b,
            beta_dep_total=args.beta_dep,
            gamma_mis=args.gamma_mis,
            dep_window=max(1, args.dep_window)
        )
    raw_score = raw_score_seq + dep_bonus
    norm_score = normalize_score(raw_score, len(toks_a), len(toks_b), args.normalize)
    return float(norm_score)


def main():
    args = parse_args()
    random.seed(args.seed)

    samples = load_samples(args.data, args.num_samples)
    print(f"Loaded {len(samples)} samples from {args.data}")

    tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=args.enable_dep)

    print("Tokenizing instructions...")
    tokenized_instrs: List[List[List[str]]] = []
    dependency_sidecars: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="Tokenizing", unit="bb"):
        instr_tokens = [tokenizer.tokenize_instruction(inst) for inst in sample.instructions]
        tokenized_instrs.append(instr_tokens)
        if args.enable_dep:
            dependency_sidecars.append(tokenizer.build_dependency_sidecar(sample.instructions))
        else:
            dependency_sidecars.append({})

    print("Generating similarity pairs...")
    total_pairs = len(samples) * args.num_pairs
    pairs: List[Dict[str, Any]] = []
    for idx_a in tqdm(range(len(samples)), desc="Samples", unit="bb"):
        for _ in range(args.num_pairs):
            idx_b = random.randrange(0, len(samples))
            while idx_b == idx_a:
                idx_b = random.randrange(0, len(samples))
            sim = score_pair(idx_a, idx_b, tokenized_instrs, tokenizer, args, dependency_sidecars)
            pairs.append({
                'idx_a': idx_a,
                'idx_b': idx_b,
                'similarity': sim
            })

    random.shuffle(pairs)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        'pairs': pairs,
        'num_samples': len(samples),
        'num_pairs_per_sample': args.num_pairs,
        'seed': args.seed,
        'normalize': args.normalize,
        'enable_dep': args.enable_dep,
        'beta_dep': args.beta_dep,
        'gamma_mis': args.gamma_mis,
        'dep_window': args.dep_window,
        'meta': {
            'data_path': args.data,
            'B_exact': args.B_exact,
            'alpha_sub': args.alpha_sub,
            'alpha_unit': args.alpha_unit,
            'B_sameop': args.B_sameop,
            'B_sameimm': args.B_sameimm
        },
        'tokenizer': tokenizer
    }

    with open(args.output, 'wb') as f:
        pickle.dump(payload, f)

    print(f"Saved {len(pairs)} precomputed pairs to {args.output}")


if __name__ == '__main__':
    main()

