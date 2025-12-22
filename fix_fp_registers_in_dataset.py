#!/usr/bin/env python3
"""
Fix FP register usage in final_training_data.pkl by converting integer x*
registers to f* registers in floating-point instructions.

Author: ywangmu from HKUST

This script loads the balanced training dataset (list[TrainingSample]),
scans all instructions, and for opcodes that are known to be FP-related,
rewrites the operands so that FP positions use f* registers instead of x*.

A new, corrected dataset is written to a separate pickle file.
"""

import argparse
import os
import pickle
import re
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Ensure local modules are importable (same pattern as other scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, "scripts"))

from prepare_balanced_dataset import TrainingSample  # noqa: F401
from advanced_tokenizer import AdvancedRISCVTokenizer


def build_opcode_sets() -> Dict[str, set]:
    """
    Build opcode sets for various FP instruction categories.

    The groupings mirror AdvancedRISCVTokenizer._define_instruction_formats.
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


def make_reg_helpers(tokenizer: AdvancedRISCVTokenizer):
    int_abi = tokenizer._int_abi
    fp_abi = tokenizer._fp_abi

    def is_int_reg(tok: str) -> bool:
        t = tok.lower()
        return bool(re.match(r"^x\d+$", t)) or (t in int_abi)

    def is_fp_reg(tok: str) -> bool:
        t = tok.lower()
        return bool(re.match(r"^f\d+$", t)) or (t in fp_abi)

    def to_fp_reg(tok: str) -> str:
        """
        Convert xN -> fN when we expect an FP register.
        Non-x tokens are left unchanged (including ABI names).
        """
        t = tok.strip()
        lower = t.lower()
        if is_fp_reg(lower):
            return t
        m = re.match(r"^x(\d+)$", lower)
        if m:
            return f"f{m.group(1)}"
        return t

    # Currently we do not need to enforce int regs strictly; we just avoid
    # converting them to FP when they should stay as int.
    def to_int_reg(tok: str) -> str:
        return tok

    return is_int_reg, is_fp_reg, to_fp_reg, to_int_reg


def categorize_opcode(opcode: str, opcode_sets: Dict[str, set]) -> Optional[str]:
    op = opcode.lower()
    for cat, ops in opcode_sets.items():
        if op in ops:
            return cat
    return None


def rewrite_instruction(
    inst: str,
    opcode_sets: Dict[str, set],
    to_fp_reg,
) -> str:
    """
    Rewrite a single instruction string if it is FP-related.

    Only register names in FP positions are changed from xN -> fN.
    Other instructions are left unchanged.
    """
    line = inst.strip()
    if not line or line.startswith("#"):
        return inst

    parts = line.split(None, 1)
    if not parts:
        return inst

    opcode = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    category = categorize_opcode(opcode, opcode_sets)
    if category is None:
        return inst

    operands = [p.strip() for p in rest.split(",")] if rest else []

    def op_at(idx: int) -> str:
        return operands[idx] if idx < len(operands) else ""

    def set_op(idx: int, value: str):
        if idx < len(operands):
            operands[idx] = value

    # Now apply category-specific rewriting.
    if category == "FP_LOAD":
        # flw/fld rd, offset(base) -> rd should be FP
        if operands:
            set_op(0, to_fp_reg(op_at(0)))

    elif category == "FP_STORE":
        # fsw/fsd rs2, offset(base) -> rs2 should be FP
        if operands:
            set_op(0, to_fp_reg(op_at(0)))

    elif category in {"FP3", "FP_FMA", "FP3_NORM"}:
        # All explicit registers should be FP
        for i, op in enumerate(operands):
            # These opcodes normally have only register operands; convert all.
            operands[i] = to_fp_reg(op)

    elif category == "FP2":
        # 2-operand FP ops: both should be FP
        if operands:
            set_op(0, to_fp_reg(op_at(0)))
        if len(operands) >= 2:
            set_op(1, to_fp_reg(op_at(1)))

    elif category == "F2I":
        # fcvt.w.s rd, rs1 -> rd int (keep), rs1 FP
        if len(operands) >= 2:
            set_op(1, to_fp_reg(op_at(1)))

    elif category == "I2F":
        # fcvt.s.w rd, rs1 -> rd FP, rs1 int
        if operands:
            set_op(0, to_fp_reg(op_at(0)))

    elif category == "FP_CMP":
        # feq.s rd, rs1, rs2 -> rd int, rs1/rs2 FP
        if len(operands) >= 2:
            set_op(1, to_fp_reg(op_at(1)))
        if len(operands) >= 3:
            set_op(2, to_fp_reg(op_at(2)))

    elif category == "FP_MV_X":
        # fmv.x.w rd, rs1 -> rd int, rs1 FP
        if len(operands) >= 2:
            set_op(1, to_fp_reg(op_at(1)))

    elif category == "FP_MV_F":
        # fmv.w.x rd, rs1 -> rd FP, rs1 int
        if operands:
            set_op(0, to_fp_reg(op_at(0)))

    # Rebuild instruction string in a normalized format.
    new_rest = ", ".join(operands) if operands else ""
    if new_rest:
        return f"{opcode} {new_rest}"
    return opcode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix FP register usage in final_training_data.pkl",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_data = os.path.join(
        script_dir,
        "dataset",
        "final_training_data.pkl",
    )
    default_output = os.path.join(
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
        "--output",
        type=str,
        default=default_output,
        help="Output pickle path for fixed dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to process (for quick tests)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    print("=" * 80)
    print("FIXING FP REGISTER USAGE IN DATASET")
    print("=" * 80)
    print(f"Input dataset:  {args.data}")
    print(f"Output dataset: {args.output}")

    with open(args.data, "rb") as f:
        samples = pickle.load(f)

    if not isinstance(samples, list) or not samples:
        raise ValueError("Dataset is empty or not a list.")

    total_samples = len(samples)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print(f"Processing first {len(samples)} / {total_samples} samples")
    else:
        print(f"Processing all {total_samples} samples")

    tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=False)
    _, _, to_fp_reg, _ = make_reg_helpers(tokenizer)
    opcode_sets = build_opcode_sets()

    fixed_samples: List[TrainingSample] = []
    total_insts = 0
    modified_insts = 0
    modified_samples = 0

    for sample in tqdm(samples, desc="Fixing samples", unit="sample"):
        orig_insts: List[str] = getattr(sample, "instructions", [])
        new_insts: List[str] = []

        sample_modified = False
        for inst in orig_insts:
            total_insts += 1
            new_inst = rewrite_instruction(inst, opcode_sets, to_fp_reg)
            new_insts.append(new_inst)
            if new_inst != inst:
                modified_insts += 1
                sample_modified = True

        if sample_modified:
            modified_samples += 1

        fixed_sample = replace(sample, instructions=new_insts)
        fixed_samples.append(fixed_sample)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples processed:     {len(samples)}")
    print(f"Samples with modifications:  {modified_samples}")
    print(f"Total instructions seen:     {total_insts}")
    print(f"Instructions modified:       {modified_insts}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        pickle.dump(fixed_samples, f)

    print(f"\nFixed dataset saved to: {args.output}")


if __name__ == "__main__":
    main()


