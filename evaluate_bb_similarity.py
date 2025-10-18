#!/usr/bin/env python3
"""
BB Similarity Evaluation Script (Execution-Unit Aware + Optional Data-Dependency Consistency)
Compute similarity between two basic blocks (BBs) using:
- Execution-unit–aware instruction similarity
- Operand-structure matching
- Weighted LCS alignment at basic-block level
- Optional local data-dependency consistency bonus on aligned pairs
"""

import argparse
import pickle
import os
import sys
import random
from typing import List, Tuple, Optional, Dict, Any

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

from advanced_tokenizer import AdvancedRISCVTokenizer
from prepare_balanced_dataset import TrainingSample

# Default scoring params
B_EXACT_DEFAULT = 2.0
ALPHA_SUB_DEFAULT = 0.7
ALPHA_UNIT_DEFAULT = 0.45
B_SAMEOP_DEFAULT = 1.0
B_SAMEIMM_DEFAULT = 3.0

# Dependency scoring defaults
ENABLE_DEP_DEFAULT = False
BETA_DEP_DEFAULT = 0.5       # total per-instruction bonus if all checked sources match; we will split per source
GAMMA_MIS_DEFAULT = 0.1    # small penalty when one side is use and the other is not
DEP_WINDOW_DEFAULT = 1       # look back k aligned pairs (1=only previous aligned pair)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate similarity between two BBs using execution-unit–aware scoring (+optional dependency consistency)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_data_path = os.path.join(script_dir, 'dataset', 'final_training_data.pkl')
    parser.add_argument('--data', type=str, default=default_data_path, help='Path to training data (pickle)')
    parser.add_argument('--idx1', type=int, default=None, help='Index of first sample (overrides --random)')
    parser.add_argument('--idx2', type=int, default=None, help='Index of second sample (overrides --random)')
    parser.add_argument('--random', action='store_true', help='Pick two random samples if idx not specified')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--show-align', action='store_true', help='Print alignment details')

    # Scoring parameters
    parser.add_argument('--B-exact', type=float, default=B_EXACT_DEFAULT, help='Base score for exact mnemonic')
    parser.add_argument('--alpha-sub', type=float, default=ALPHA_SUB_DEFAULT, help='Alpha for same subclass')
    parser.add_argument('--alpha-unit', type=float, default=ALPHA_UNIT_DEFAULT, help='Alpha for same unit (diff subclass)')
    parser.add_argument('--B-sameop', type=float, default=B_SAMEOP_DEFAULT, help='Bonus for matching operand kind')
    parser.add_argument('--B-sameimm', type=float, default=B_SAMEIMM_DEFAULT, help='Extra bonus for equal concrete immediates (if applicable)')
    parser.add_argument('--normalize', choices=['maxlen', 'avglen', 'none'], default='maxlen',
                        help='Normalization method for BB similarity')

    # Dependency-consistency parameters
    parser.add_argument('--enable-dep', action='store_true', default=ENABLE_DEP_DEFAULT,
                        help='Enable dependency-consistency scoring on aligned pairs')
    parser.add_argument('--beta-dep', type=float, default=BETA_DEP_DEFAULT,
                        help='Total dependency bonus per matched instruction (will be split per source operand)')
    parser.add_argument('--gamma-mis', type=float, default=GAMMA_MIS_DEFAULT,
                        help='Small penalty when only one side is a use or references do not align')
    parser.add_argument('--dep-window', type=int, default=DEP_WINDOW_DEFAULT,
                        help='Look-back window over previous aligned pairs to match defs (>=1)')

    return parser.parse_args()


def load_dataset(data_path: str) -> List[TrainingSample]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    with open(data_path, 'rb') as f:
        samples = pickle.load(f)
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Dataset is empty or invalid format")
    return samples


def get_opcode(tokens: List[str]) -> Optional[str]:
    if len(tokens) >= 2 and tokens[0] == '<s>':
        return tokens[1].lower()
    return None


def operand_kinds(tokens: List[str]) -> List[str]:
    kinds = []
    in_dsts = False
    in_srcs = False
    in_mem = False
    in_addr = False

    for t in tokens:
        if t == '<dsts>':
            in_dsts, in_srcs, in_mem, in_addr = True, False, False, False
            continue
        if t == '<srcs>':
            in_dsts, in_srcs, in_mem, in_addr = False, True, False, False
            continue
        if t == '<mem>':
            in_mem = True
            continue
        if t == '</mem>':
            in_mem = False
            continue
        if t == '<addr>':
            in_addr = True
            continue
        if t == '</addr>':
            in_addr = False
            continue
        if t in ('<s>', '</s>', ';', '<csr>'):
            continue

        if t == '<const>':
            kinds.append('addr' if in_addr else 'imm')
        else:
            if in_dsts:
                kinds.append('dst_reg')
            elif in_srcs and in_mem:
                kinds.append('mem_base')
            elif in_srcs:
                kinds.append('src_reg')
            else:
                kinds.append('src_reg')
    return kinds


def same_family_alpha(op1: str, op2: str, tokenizer: AdvancedRISCVTokenizer,
                      B_exact: float, alpha_sub: float, alpha_unit: float) -> float:
    if op1 is None or op2 is None:
        return 0.0
    if op1 not in tokenizer.instruction_formats or op2 not in tokenizer.instruction_formats:
        return 0.0
    if op1 == op2:
        return B_exact
    f1 = tokenizer.instruction_formats[op1]
    f2 = tokenizer.instruction_formats[op2]
    if f1.exec_unit == f2.exec_unit:
        if f1.exec_subclass == f2.exec_subclass:
            return alpha_sub * B_exact
        else:
            return alpha_unit * B_exact
    return 0.0


def comp_ins(tokens1: List[str], tokens2: List[str], tokenizer: AdvancedRISCVTokenizer,
             B_exact: float, alpha_sub: float, alpha_unit: float,
             B_sameop: float, B_sameimm: float) -> float:
    base = same_family_alpha(get_opcode(tokens1), get_opcode(tokens2), tokenizer,
                             B_exact, alpha_sub, alpha_unit)
    if base == 0.0:
        return 0.0

    kinds1 = operand_kinds(tokens1)
    kinds2 = operand_kinds(tokens2)

    bonus = 0.0
    for k1, k2 in zip(kinds1, kinds2):
        if k1 == k2:
            bonus += B_sameop
    # Note: if you later keep concrete immediates, add equality check here and +B_sameimm
    return base + bonus


def weighted_lcs(seqA: List[List[str]], seqB: List[List[str]],
                 score_fn) -> Tuple[float, List[Tuple[int, int]]]:
    n, m = len(seqA), len(seqB)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    prev = [[None]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for j in range(1, m+1):
            w = score_fn(i-1, j-1)
            match = dp[i-1][j-1] + w
            skip_a = dp[i-1][j]
            skip_b = dp[i][j-1]
            best = match
            back = ('match', i-1, j-1)
            if skip_a >= best:
                best = skip_a
                back = ('skip_a', i-1, j)
            if skip_b >= best:
                best = skip_b
                back = ('skip_b', i, j-1)
            dp[i][j] = best
            prev[i][j] = back

    # reconstruct
    i, j = n, m
    pairs = []
    while i > 0 and j > 0:
        tag, pi, pj = prev[i][j]
        if tag == 'match' and dp[i][j] > dp[pi][pj]:
            pairs.append((i-1, j-1))
            i, j = pi, pj
        elif tag == 'skip_a':
            i, j = pi, pj
        else:
            i, j = pi, pj
    pairs.reverse()
    return dp[n][m], pairs


def normalize_score(raw_score: float, lenA: int, lenB: int, method: str) -> float:
    if method == 'none':
        return raw_score
    if method == 'maxlen':
        denom = max(lenA, lenB) if max(lenA, lenB) > 0 else 1
    elif method == 'avglen':
        denom = (lenA + lenB) / 2.0 if (lenA + lenB) > 0 else 1
    else:
        denom = max(lenA, lenB) if max(lenA, lenB) > 0 else 1
    return raw_score / denom


def show_alignment(bb1, bb2, toks1, toks2, pairs, tokenizer, params_str, meta1, meta2):
    print("\n" + "="*80)
    print("BB Similarity Alignment")
    print("="*80)
    print(params_str)
    print("\nBB1 meta:")
    print(f"  sample_id={meta1.get('sample_id')}, file={meta1.get('source_file')}, "
          f"type={meta1.get('file_type')}, block_id={meta1.get('block_id')}, "
          f"base={meta1.get('base_address')}, len={len(bb1)} (declared={meta1.get('num_instructions')})")
    print("BB2 meta:")
    print(f"  sample_id={meta2.get('sample_id')}, file={meta2.get('source_file')}, "
          f"type={meta2.get('file_type')}, block_id={meta2.get('block_id')}, "
          f"base={meta2.get('base_address')}, len={len(bb2)} (declared={meta2.get('num_instructions')})")
    print("-"*80)
    print("Matches (i -> j):")
    for i, j in pairs:
        op1 = get_opcode(toks1[i]) or "?"
        op2 = get_opcode(toks2[j]) or "?"
        f1 = tokenizer.instruction_formats.get(op1, None)
        f2 = tokenizer.instruction_formats.get(op2, None)
        eu1 = f1.exec_unit if f1 else "?"
        eu2 = f2.exec_unit if f2 else "?"
        sub1 = f1.exec_subclass if f1 else "?"
        sub2 = f2.exec_subclass if f2 else "?"
        print(f"  {i:>3} ({op1:8s}) [{eu1}/{sub1}]  <->  {j:>3} ({op2:8s}) [{eu2}/{sub2}]")
    print("-"*80)
    print("BB1:")
    for idx, inst in enumerate(bb1):
        marker = " *" if any(i == idx for i,_ in pairs) else "  "
        print(f"{marker} [{idx:>3}] {inst}")
    print("\nBB2:")
    for idx, inst in enumerate(bb2):
        marker = " *" if any(j == idx for _,j in pairs) else "  "
        print(f"{marker} [{idx:>3}] {inst}")


# ---------------- Dependency consistency bonus (uses tokenizer sidecar) ----------------

def dep_consistency_bonus(pairs: List[Tuple[int,int]],
                          sideA: Dict[str,Any],
                          sideB: Dict[str,Any],
                          beta_dep_total: float,
                          gamma_mis: float,
                          dep_window: int) -> float:
    """
    Iterate aligned pairs; for each pair, check per-source slot dependency consistency:
    - If both sides are 'use' and each references a def coming from a previous aligned pair within window -> +beta_per_src
    - If only one side is 'use' (or references don't align) -> -gamma_mis (small)
    Notes:
      - sidecar item format (from tokenizer):
        {'opcode': str|None,
         'src_slots': [{'kind': 'use'|'ext'|'const_zero', 'reg': str, 'def_line': int}],
         'dst_regs': [str]}
      - We compare src slots positionally up to min length.
    """
    if beta_dep_total <= 0.0 and gamma_mis <= 0.0:
        return 0.0

    perA = sideA['per_inst']
    perB = sideB['per_inst']
    bonus_total = 0.0

    for pos, (i, j) in enumerate(pairs):
        srcA = perA[i]['src_slots'] if i < len(perA) else []
        srcB = perB[j]['src_slots'] if j < len(perB) else []
        kmax = min(len(srcA), len(srcB))
        if kmax == 0:
            continue

        beta_per_src = beta_dep_total / max(1, kmax)

        for k in range(kmax):
            a = srcA[k]
            b = srcB[k]
            a_is_use = (a.get('kind') == 'use')
            b_is_use = (b.get('kind') == 'use')

            if a_is_use and b_is_use:
                okA = False
                okB = False
                # look back over previous aligned pairs to see if their defs match our def_line references
                for back in range(1, dep_window+1):
                    prev_pos = pos - back
                    if prev_pos < 0:
                        break
                    prev_i, prev_j = pairs[prev_pos]
                    if a.get('def_line', -1) == prev_i:
                        okA = True
                    if b.get('def_line', -1) == prev_j:
                        okB = True
                    if okA and okB:
                        break
                if okA and okB:
                    bonus_total += beta_per_src
                else:
                    if gamma_mis > 0.0:
                        bonus_total -= gamma_mis
            elif a_is_use != b_is_use:
                if gamma_mis > 0.0:
                    bonus_total -= gamma_mis
            else:
                # ext/const_zero 等情况 -> 中性
                pass

    return bonus_total


def main():
    args = parse_args()
    random.seed(args.seed)

    samples = load_dataset(args.data)

    if args.idx1 is not None and args.idx2 is not None:
        idx1, idx2 = args.idx1, args.idx2
    elif args.random:
        idx1 = random.randrange(0, len(samples))
        idx2 = random.randrange(0, len(samples))
        while idx2 == idx1:
            idx2 = random.randrange(0, len(samples))
    else:
        idx1, idx2 = 0, 1

    bb1: TrainingSample = samples[idx1]
    bb2: TrainingSample = samples[idx2]

    print("\n" + "="*80)
    print("BB Similarity Evaluation")
    print("="*80)
    print(f"Dataset: {args.data}")
    print(f"Sample indices: idx1={idx1}, idx2={idx2}")

    # Use tokenizer with dependency sidecar enabled; tokenization输出不变
    tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=True)

    toks1 = [tokenizer.tokenize_instruction(inst) for inst in bb1.instructions]
    toks2 = [tokenizer.tokenize_instruction(inst) for inst in bb2.instructions]

    def score_fn(i: int, j: int) -> float:
        return comp_ins(
            toks1[i], toks2[j], tokenizer,
            B_exact=args.B_exact,
            alpha_sub=args.alpha_sub,
            alpha_unit=args.alpha_unit,
            B_sameop=args.B_sameop,
            B_sameimm=args.B_sameimm
        )

    raw_score_seq, pairs = weighted_lcs(toks1, toks2, score_fn)

    dep_bonus = 0.0
    if args.enable_dep and len(pairs) > 0:
        # Build dependency sidecars using tokenizer's built-in sidecar
        sideA = tokenizer.build_dependency_sidecar(bb1.instructions)
        sideB = tokenizer.build_dependency_sidecar(bb2.instructions)
        dep_bonus = dep_consistency_bonus(
            pairs, sideA, sideB,
            beta_dep_total=args.beta_dep,
            gamma_mis=args.gamma_mis,
            dep_window=max(1, args.dep_window)
        )

    raw_score = raw_score_seq + dep_bonus
    norm_score = normalize_score(raw_score, len(toks1), len(toks2), args.normalize)

    print("\nScoring parameters:")
    print(f"  B_exact={args.B_exact:.2f}, alpha_sub={args.alpha_sub:.2f}, alpha_unit={args.alpha_unit:.2f}, "
          f"B_sameop={args.B_sameop:.2f}, B_sameimm={args.B_sameimm:.2f}, normalize={args.normalize}")
    print(f"  enable_dep={args.enable_dep}, beta_dep={args.beta_dep:.2f}, gamma_mis={args.gamma_mis:.2f}, dep_window={args.dep_window}")

    print("\nResults:")
    print(f"  BB1 length: {len(toks1)} (declared {getattr(bb1, 'num_instructions', len(toks1))})")
    print(f"  BB2 length: {len(toks2)} (declared {getattr(bb2, 'num_instructions', len(toks2))})")
    print(f"  Raw LCS score (sequence only): {raw_score_seq:.4f}")
    if args.enable_dep:
        print(f"  Dependency bonus: {dep_bonus:.4f}")
    print(f"  Total raw score: {raw_score:.4f}")
    print(f"  Normalized score ({args.normalize}): {norm_score:.4f}")

    if args.show_align:
        meta1 = {
            'sample_id': getattr(bb1, 'sample_id', None),
            'source_file': getattr(bb1, 'source_file', None),
            'file_type': getattr(bb1, 'file_type', None),
            'block_id': getattr(bb1, 'block_id', None),
            'base_address': getattr(bb1, 'base_address', None),
            'num_instructions': getattr(bb1, 'num_instructions', None),
        }
        meta2 = {
            'sample_id': getattr(bb2, 'sample_id', None),
            'source_file': getattr(bb2, 'source_file', None),
            'file_type': getattr(bb2, 'file_type', None),
            'block_id': getattr(bb2, 'block_id', None),
            'base_address': getattr(bb2, 'base_address', None),
            'num_instructions': getattr(bb2, 'num_instructions', None),
        }
        params_str = (f"Params: B_exact={args.B_exact}, alpha_sub={args.alpha_sub}, "
                      f"alpha_unit={args.alpha_unit}, B_sameop={args.B_sameop}, "
                      f"B_sameimm={args.B_sameimm}, normalize={args.normalize}, "
                      f"enable_dep={args.enable_dep}, beta_dep={args.beta_dep}, "
                      f"gamma_mis={args.gamma_mis}, dep_window={args.dep_window}")
        show_alignment(bb1.instructions, bb2.instructions, toks1, toks2, pairs, tokenizer, params_str, meta1, meta2)

    return 0


if __name__ == '__main__':
    sys.exit(main())