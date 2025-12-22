#!/usr/bin/env python3
"""
BBTokenizerMLMDataset: Dataset adapter for RWKV7 dual-MLM training
Author: ywangmu from HKUST

This dataset wraps AdvancedRISCVTokenizer and produces batches compatible with
train_rwkv7_deepspeed.py dual-task format:
- batch['token_level'] = (inputs_dict, targets)
- batch['instruction_level'] = (inputs_dict, targets)
Where inputs_dict currently contains only one channel: 'asm'.

Notes:
- We keep sequence length fixed (pad/truncate to max_seq_len) so we can rely on
  default collate without a custom collate_fn. Final alignment to 16 tokens is
  still handled by the training script's adjust_batch_to_max_valid_length.
- Token-level MLM: random token masking within the sequence (exclude specials).
- Instruction-level MLM: mask entire instruction spans between <s> boundaries.
"""
import os
import random
from typing import List, Tuple, Dict, Optional, Union

import torch
from torch.utils.data import Dataset

from advanced_tokenizer import AdvancedRISCVTokenizer


def _gather_files_from_maybe_dirs(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for f in fnames:
                    full = os.path.join(root, f)
                    if os.path.isfile(full):
                        files.append(full)
    return sorted(files)


class BBTokenizerMLMDataset(Dataset):
    """
    A minimal dataset that reads each file as one testcase.
    Produces two tasks: token-level MLM and instruction-level MLM, both using 'asm'.
    """
    def __init__(
        self,
        max_seq_len: int,
        min_seq_len: int = 0,
        data_files: Optional[List[str]] = None,
        testcases: Optional[List[str]] = None,
        token_mask_prob: float = 0.15,
        instr_mask_prob: float = 0.15,
        rng_seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=False)
        self.vocab = self.tokenizer.vocab
        self.id_to_token = self.tokenizer.id_to_token
        self.vocab_size = {'asm': len(self.vocab)}

        # Special token ids
        self.PAD_ID = self.vocab.get('<PAD>')
        self.UNK_ID = self.vocab.get('<UNK>')
        self.CLS_ID = self.vocab.get('<s>')
        self.SEP_ID = self.vocab.get('</s>')
        self.MASK_ID = self.vocab.get('<MASK>')

        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.token_mask_prob = float(max(0.0, min(1.0, token_mask_prob)))
        self.instr_mask_prob = float(max(0.0, min(1.0, instr_mask_prob)))
        self.rng = random.Random(rng_seed)

        # Build sample texts list: prefer provided testcases; else load from files/dirs
        self.samples: List[str] = []
        if testcases is not None:
            for t in testcases:
                try:
                    token_ids = self.tokenizer.encode(t)
                    if self.min_seq_len > 0 and len(token_ids) < self.min_seq_len:
                        continue
                    if len(token_ids) == 0:
                        continue
                    self.samples.append(t)
                except Exception:
                    continue
        else:
            if not data_files:
                raise FileNotFoundError("BBTokenizerMLMDataset requires either testcases or data_files.")
            files = _gather_files_from_maybe_dirs(data_files)
            if len(files) == 0:
                raise FileNotFoundError("No valid data files found for BBTokenizerMLMDataset.")
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                        text = fh.read()
                    token_ids = self.tokenizer.encode(text)
                    if self.min_seq_len > 0 and len(token_ids) < self.min_seq_len:
                        continue
                    if len(token_ids) == 0:
                        continue
                    self.samples.append(text)
                except Exception:
                    continue
        if len(self.samples) == 0:
            raise RuntimeError("No valid samples remained after length filtering.")

    # Expose constants for training script compatibility
    @property
    def PAD_ID(self) -> int:
        return self._PAD_ID

    @PAD_ID.setter
    def PAD_ID(self, v: int) -> None:
        self._PAD_ID = v

    @property
    def CLS_ID(self) -> int:
        return self._CLS_ID

    @CLS_ID.setter
    def CLS_ID(self, v: int) -> None:
        self._CLS_ID = v

    @property
    def SEP_ID(self) -> int:
        return self._SEP_ID

    @SEP_ID.setter
    def SEP_ID(self, v: int) -> None:
        self._SEP_ID = v

    @property
    def MASK_ID(self) -> int:
        return self._MASK_ID

    @MASK_ID.setter
    def MASK_ID(self, v: int) -> None:
        self._MASK_ID = v

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text = self.samples[idx]

        # Tokens and ids
        tokens = self.tokenizer.tokenize_testcase(text)  # includes multiple <s> and one </s> at end
        ids = [self.vocab.get(tok, self.UNK_ID) for tok in tokens]

        # Build token-level MLM sample
        token_inputs_asm, token_targets = self._build_token_level(ids, tokens)
        # Build instruction-level MLM sample
        instr_inputs_asm, instr_targets = self._build_instruction_level(ids, tokens)

        # Pad/trim to fixed max length for both tasks
        token_inputs_asm, token_targets = self._pad_or_trim(token_inputs_asm, token_targets)
        instr_inputs_asm, instr_targets = self._pad_or_trim(instr_inputs_asm, instr_targets)

        token_inputs = {'asm': torch.tensor(token_inputs_asm, dtype=torch.long)}
        token_targets = torch.tensor(token_targets, dtype=torch.long)

        instr_inputs = {'asm': torch.tensor(instr_inputs_asm, dtype=torch.long)}
        instr_targets = torch.tensor(instr_targets, dtype=torch.long)

        return {
            'token_level': (token_inputs, token_targets),
            'instruction_level': (instr_inputs, instr_targets),
        }

    # ------------------------ helpers ------------------------
    def _eligible_token_positions(self, tokens: List[str]) -> List[int]:
        specials = {'<PAD>', '<s>', '</s>'}
        pos = []
        for i, t in enumerate(tokens):
            if t not in specials:
                pos.append(i)
        return pos

    def _build_token_level(self, ids: List[int], tokens: List[str]) -> Tuple[List[int], List[int]]:
        inp = list(ids)
        tgt = [-100] * len(ids)
        eligible = self._eligible_token_positions(tokens)
        if not eligible:
            return inp, tgt
        num_to_mask = max(1, int(round(len(eligible) * self.token_mask_prob)))
        mask_positions = self.rng.sample(eligible, k=min(num_to_mask, len(eligible)))
        for p in mask_positions:
            r = self.rng.random()
            original = inp[p]
            if r < 0.8:
                inp[p] = self.MASK_ID
            elif r < 0.9:
                # random id, avoid specials if possible
                inp[p] = self._random_non_special_id()
            else:
                # keep original
                pass
            tgt[p] = original
        return inp, tgt

    def _random_non_special_id(self) -> int:
        # try to avoid <PAD>, <s>, </s>, <MASK>
        forb = {self.PAD_ID, self.CLS_ID, self.SEP_ID, self.MASK_ID}
        while True:
            rid = self.rng.randrange(0, self.vocab_size['asm'])
            if rid not in forb:
                return rid

    def _find_instruction_spans(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Find spans [start_idx, end_idx] for each instruction:
        - start at each '<s>'
        - end at next '<s>' - 1 or the last index before '</s>'
        """
        starts = [i for i, t in enumerate(tokens) if t == '<s>']
        if not starts:
            return []
        # last content index before </s>
        end_limit = len(tokens) - 1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == '</s>':
                end_limit = i - 1
                break
        spans: List[Tuple[int, int]] = []
        for i, s in enumerate(starts):
            if i + 1 < len(starts):
                e = starts[i + 1] - 1
            else:
                e = end_limit
            if e >= s:
                spans.append((s, e))
        return [sp for sp in spans if sp[0] <= sp[1]]

    def _build_instruction_level(self, ids: List[int], tokens: List[str]) -> Tuple[List[int], List[int]]:
        inp = list(ids)
        tgt = [-100] * len(ids)
        spans = self._find_instruction_spans(tokens)
        if not spans:
            return inp, tgt
        num_to_mask = max(1, int(round(len(spans) * self.instr_mask_prob)))
        chosen = self.rng.sample(spans, k=min(num_to_mask, len(spans)))
        for s, e in chosen:
            for p in range(s, e + 1):
                # keep <s> token unmasked to preserve boundary
                if tokens[p] == '<s>':
                    continue
                tgt[p] = inp[p]
                inp[p] = self.MASK_ID
        return inp, tgt

    def _pad_or_trim(self, inp: List[int], tgt: List[int]) -> Tuple[List[int], List[int]]:
        if len(inp) > self.max_seq_len:
            inp = inp[:self.max_seq_len]
            tgt = tgt[:self.max_seq_len]
        elif len(inp) < self.max_seq_len:
            pad_len = self.max_seq_len - len(inp)
            inp = inp + [self.PAD_ID] * pad_len
            tgt = tgt + [-100] * pad_len
        return inp, tgt


