#!/usr/bin/env python3
"""
Check FP register correctness in final_training_data.pkl.

Author: ywangmu from HKUST

This script scans the balanced training dataset and looks for floating-point
instructions where integer registers (x*/int ABI) are used in places that
should be floating-point registers (f*/FP ABI), and vice versa for some
mixed int/FP ops.

It prints a summary and a few example violations to help confirm whether
the dataset was generated with incorrect register types.
"""

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# Ensure local modules (including scripts/) are importable, so that
# pickle can resolve TrainingSample from prepare_balanced_dataset.
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, "scripts"))

from prepare_balanced_dataset import TrainingSample  # noqa: F401
from advanced_tokenizer import AdvancedRISCVTokenizer


def build_opcode_sets() -> Dict[str, set]:
    """
    Build opcode sets for various FP instruction categories.

    These sets mirror the groupings used in AdvancedRISCVTokenizer._define_instruction_formats.
    """
    fp_load_ops = {"flw", "fld"}
    fp_store_ops = {"fsw", "fsd"}

    # FP conversions
    f2i_ops = {
        "fcvt.w.s", "fcvt.wu.s", "fcvt.l.s", "fcvt.lu.s",
        "fcvt.w.d", "fcvt.wu.d", "fcvt.l.d", "fcvt.lu.d",
    }
    i2f_ops = {
        "fcvt.s.w", "fcvt.s.wu", "fcvt.s.l", "fcvt.s.lu",
        "fcvt.d.w", "fcvt.d.wu", "fcvt.d.l", "fcvt.d.lu",
    }

    # FMA
    fma_ops = {
        "fmadd.s", "fmsub.s", "fnmsub.s", "fnmadd.s",
        "fmadd.d", "fmsub.d", "fnmsub.d", "fnmadd.d",
    }

    # 3-operand FP arithmetic (all FP regs)
    fp3_ops = {
        "fadd.s", "fsub.s", "fmul.s", "fdiv.s",
        "fadd.d", "fsub.d", "fmul.d", "fdiv.d",
    }

    # 3-operand "normal" FP ops (all FP regs)
    fp3_norm_ops = {
        "fsgnj.s", "fsgnjn.s", "fsgnjx.s", "fmin.s", "fmax.s",
        "fsgnj.d", "fsgnjn.d", "fsgnjx.d", "fmin.d", "fmax.d",
    }

    # 2-operand FP ops (both FP regs)
    fp2_ops = {"fsqrt.s", "fsqrt.d", "fcvt.d.s", "fcvt.s.d"}

    # FP compares: int dest, FP srcs
    fp_cmp_ops = {"feq.s", "flt.s", "fle.s", "feq.d", "flt.d", "fle.d"}

    # FP move / classify
    fp_mv_x_ops = {"fmv.x.w", "fclass.s", "fclass.d", "fmv.x.d"}  # int dest, FP src
    fp_mv_f_ops = {"fmv.w.x", "fmv.d.x"}  # FP dest, int src

    return {
        "FP_LOAD": fp_load_ops,
        "FP_STORE": fp_store_ops,
        "F2I": f2i_ops,
        "I2F": i2f_ops,
        "FP3": fp3_ops,
        "FP_FMA": fma_ops,
        "FP3_NORM": fp3_norm_ops,
        "FP2": fp2_ops,
        "FP_CMP": fp_cmp_ops,
        "FP_MV_X": fp_mv_x_ops,
        "FP_MV_F": fp_mv_f_ops,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check FP register usage in final_training_data.pkl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_data = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset",
        "final_training_data.pkl",
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
        "--max-examples-per-op",
        type=int,
        default=10,
        help="Max example violations to print per opcode",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    print("=" * 80)
    print("CHECKING FP REGISTER USAGE IN DATASET")
    print("=" * 80)
    print(f"Dataset path: {args.data}")

    with open(args.data, "rb") as f:
        samples = pickle.load(f)

    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset is empty or not a list.")

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print(f"Using first {len(samples)} samples for checking.")
    else:
        print(f"Total samples: {len(samples)}")

    tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=False)
    int_abi = tokenizer._int_abi
    fp_abi = tokenizer._fp_abi

    def is_int_reg(tok: str) -> bool:
        t = tok.lower()
        return bool(re.match(r"^x\d+$", t)) or (t in int_abi)

    def is_fp_reg(tok: str) -> bool:
        t = tok.lower()
        return bool(re.match(r"^f\d+$", t)) or (t in fp_abi)

    opcode_sets = build_opcode_sets()

    total_fp_insts = 0
    total_issues = 0
    issues_per_op: Dict[str, int] = defaultdict(int)
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record_issue(
        opcode: str,
        sample: Any,
        inst_line: str,
        detail: str,
    ):
        nonlocal total_issues
        total_issues += 1
        issues_per_op[opcode] += 1
        if len(examples[opcode]) < args.max_examples_per_op:
            example: Dict[str, Any] = {
                "sample_id": getattr(sample, "sample_id", "<unknown>"),
                "block_id": getattr(sample, "block_id", None),
                "opcode": opcode,
                "instruction": inst_line.strip(),
                "detail": detail,
            }
            examples[opcode].append(example)

    # Main scan loop
    for sample in samples:
        insts: List[str] = getattr(sample, "instructions", [])
        for inst in insts:
            line = inst.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if not parts:
                continue
            opcode = parts[0].lower()
            operands_str = parts[1] if len(parts) > 1 else ""
            operands = [p.strip() for p in operands_str.split(",") if p.strip()]

            category = None
            for cat, ops in opcode_sets.items():
                if opcode in ops:
                    category = cat
                    break

            if category is None:
                continue

            total_fp_insts += 1

            # Helper to safely access nth operand
            def op_at(idx: int) -> str:
                return operands[idx].lower() if idx < len(operands) else ""

            if category == "FP_LOAD":
                # e.g., fld f1, 0(x2)  -> dest should be FP
                dst = op_at(0)
                if dst and is_int_reg(dst) and not is_fp_reg(dst):
                    record_issue(
                        opcode,
                        sample,
                        line,
                        f"FP load dest should be FP register, got integer register '{dst}'",
                    )
            elif category == "FP_STORE":
                # e.g., fsd f1, 0(x2)  -> first operand should be FP src
                src = op_at(0)
                if src and is_int_reg(src) and not is_fp_reg(src):
                    record_issue(
                        opcode,
                        sample,
                        line,
                        f"FP store src should be FP register, got integer register '{src}'",
                    )
            elif category in {"FP3", "FP_FMA", "FP3_NORM"}:
                # All explicit regs should be FP
                bad_regs: List[str] = []
                for op in operands:
                    tok = op.lower()
                    # Strip immediates like "0(x2)" when they appear; we only check plain regs
                    if "(" in tok and ")" in tok:
                        continue
                    if is_int_reg(tok) and not is_fp_reg(tok):
                        bad_regs.append(tok)
                if bad_regs:
                    record_issue(
                        opcode,
                        sample,
                        line,
                        f"Expected FP registers, found integer registers {bad_regs}",
                    )
            elif category == "FP2":
                # Both operands should be FP regs
                dst = op_at(0)
                src = op_at(1)
                bad: List[str] = []
                if dst and is_int_reg(dst) and not is_fp_reg(dst):
                    bad.append(f"dest='{dst}'")
                if src and is_int_reg(src) and not is_fp_reg(src):
                    bad.append(f"src='{src}'")
                if bad:
                    record_issue(
                        opcode,
                        sample,
                        line,
                        "2-op FP instruction should use FP registers, got " + ", ".join(bad),
                    )
            elif category == "F2I":
                # fcvt.w.s rd, rs1  -> rd int, rs1 FP
                rd = op_at(0)
                rs1 = op_at(1)
                bad: List[str] = []
                if rd and is_fp_reg(rd) and not is_int_reg(rd):
                    bad.append(f"dest should be int, got FP '{rd}'")
                if rs1 and is_int_reg(rs1) and not is_fp_reg(rs1):
                    bad.append(f"src should be FP, got int '{rs1}'")
                if bad:
                    record_issue(opcode, sample, line, "; ".join(bad))
            elif category == "I2F":
                # fcvt.s.w rd, rs1  -> rd FP, rs1 int
                rd = op_at(0)
                rs1 = op_at(1)
                bad = []
                if rd and is_int_reg(rd) and not is_fp_reg(rd):
                    bad.append(f"dest should be FP, got int '{rd}'")
                if rs1 and is_fp_reg(rs1) and not is_int_reg(rs1):
                    bad.append(f"src should be int, got FP '{rs1}'")
                if bad:
                    record_issue(opcode, sample, line, "; ".join(bad))
            elif category == "FP_CMP":
                # feq.s rd, rs1, rs2 -> rd int, rs1/rs2 FP
                rd = op_at(0)
                rs1 = op_at(1)
                rs2 = op_at(2)
                bad = []
                if rd and is_fp_reg(rd) and not is_int_reg(rd):
                    bad.append(f"dest should be int, got FP '{rd}'")
                if rs1 and is_int_reg(rs1) and not is_fp_reg(rs1):
                    bad.append(f"src1 should be FP, got int '{rs1}'")
                if rs2 and is_int_reg(rs2) and not is_fp_reg(rs2):
                    bad.append(f"src2 should be FP, got int '{rs2}'")
                if bad:
                    record_issue(opcode, sample, line, "; ".join(bad))
            elif category == "FP_MV_X":
                # fmv.x.w rd, rs1  -> rd int, rs1 FP
                rd = op_at(0)
                rs1 = op_at(1)
                bad = []
                if rd and is_fp_reg(rd) and not is_int_reg(rd):
                    bad.append(f"dest should be int, got FP '{rd}'")
                if rs1 and is_int_reg(rs1) and not is_fp_reg(rs1):
                    bad.append(f"src should be FP, got int '{rs1}'")
                if bad:
                    record_issue(opcode, sample, line, "; ".join(bad))
            elif category == "FP_MV_F":
                # fmv.w.x rd, rs1  -> rd FP, rs1 int
                rd = op_at(0)
                rs1 = op_at(1)
                bad = []
                if rd and is_int_reg(rd) and not is_fp_reg(rd):
                    bad.append(f"dest should be FP, got int '{rd}'")
                if rs1 and is_fp_reg(rs1) and not is_int_reg(rs1):
                    bad.append(f"src should be int, got FP '{rs1}'")
                if bad:
                    record_issue(opcode, sample, line, "; ".join(bad))

    print("\n" + "=" * 80)
    print("SCAN SUMMARY")
    print("=" * 80)
    print(f"Total FP-related instructions scanned: {total_fp_insts}")
    print(f"Total suspicious instructions found:   {total_issues}")

    if not total_issues:
        print("\nNo obvious FP/int register mismatches detected.")
        return

    print("\nIssues by opcode:")
    for op, cnt in sorted(issues_per_op.items(), key=lambda x: -x[1]):
        print(f"  {op:12s}: {cnt:6d}")

    print("\nExample violations (limited per opcode):")
    for op, ex_list in examples.items():
        print(f"\n--- {op} ---")
        for ex in ex_list:
            sid = ex.get("sample_id", "<unknown>")
            bid = ex.get("block_id", None)
            prefix = f"[sample={sid}]"
            if bid is not None:
                prefix += f"[block={bid}]"
            print(f"{prefix}  {ex['instruction']}")
            print(f"    -> {ex['detail']}")


if __name__ == "__main__":
    main()


