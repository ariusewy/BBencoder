#!/usr/bin/env python3
"""
Unified Training Script for BBencoder
Author: ywangmu from HKUST

Supports:
- Models: lstm, babygpt, gpt2-small, gpt2
- Tasks: mlm (masked language modeling), supervised (similarity learning)
- Modes: demo (debug with small data), train (full training)
"""

import os
import sys
import argparse
import pickle
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

# Import models
from models_arch.lstm_encoder import LightweightLSTMEncoder
from models_arch.babygpt import BabyGPT, BabyGPTConfig
from models_arch.gpt2 import GPT2Encoder, GPT2Config

# Import tokenizer and datasets
from advanced_tokenizer import AdvancedRISCVTokenizer
from bbtokenizer_mlm_dataset import BBTokenizerMLMDataset


# ============================================================================
# TrainingSample shim for pickle compatibility
# ============================================================================
try:
    from prepare_balanced_dataset import TrainingSample
except:
    class TrainingSample:
        def __init__(self, testcase, instructions, bb_id=None):
            self.testcase = testcase
            self.instructions = instructions
            self.bb_id = bb_id


# ============================================================================
# Supervised Dataset
# ============================================================================
class SupervisedDataset(Dataset):
    """
    Dataset for supervised similarity learning with BB similarity scores
    Uses the same BB similarity computation as lightweight_encoder_supervised.py
    """
    
    def __init__(
        self,
        samples: List[TrainingSample],
        tokenizer: AdvancedRISCVTokenizer,
        max_length: int = 1600,
        num_pairs: int = 5,
        normalize: str = 'maxlen',
        enable_dep: bool = False,
        beta_dep: float = 0.5,
        dep_window: int = 3,
        gamma_mis: float = 0.05,
        precomputed_pairs: Optional[List[Dict[str, Any]]] = None,
        global_indices: Optional[List[int]] = None,
        precomputed_source: Optional[str] = None
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_pairs = num_pairs
        self.pad_idx = self.tokenizer.vocab.get('<PAD>', 0)
        self.tokenized_samples: List[List[int]] = [
            self._encode_sample(sample.instructions) for sample in self.samples
        ]
        
        # BB similarity parameters (same as lightweight_encoder_supervised.py)
        self.bb_params = {
            'B_exact': 2.0,
            'alpha_sub': 0.7,
            'alpha_unit': 0.45,
            'B_sameop': 1.0,
            'B_sameimm': 3.0,
            'normalize': normalize,
            'enable_dep': enable_dep,
            'beta_dep': beta_dep,
            'gamma_mis': gamma_mis,
            'dep_window': dep_window
        }
        
        # Create BB tokenizer with dependency support if needed
        self.bb_tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=enable_dep)
        self.index_map = None
        if global_indices is not None:
            self.index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(global_indices)}
        
        # Precompute pairs and similarities
        print(f"Generating {num_pairs} pairs per sample...")
        print(f"Using BB similarity with weighted LCS + execution unit awareness")
        if enable_dep:
            print(f"  Dependency consistency enabled (beta={beta_dep}, window={dep_window})")
        self.pairs = []
        if precomputed_pairs is not None and self.index_map is not None:
            self._load_precomputed_pairs(precomputed_pairs, precomputed_source)
        else:
            self._generate_pairs()
    
    def _encode_sample(self, instructions: List[str]) -> List[int]:
        text = '\n'.join(instructions)
        ids = self.tokenizer.encode(text)
        return ids[:self.max_length]
    
    def _generate_pairs(self):
        """Generate random pairs and compute BB similarities"""
        # Import BB similarity functions
        try:
            from evaluate_bb_similarity import (
                comp_ins, weighted_lcs, normalize_score, dep_consistency_bonus
            )
        except ImportError:
            print("Warning: Could not import BB similarity functions, using simple fallback")
            self._generate_pairs_simple()
            return
        
        similarity_scores = []
        # Disable progress bar when stdout is not a TTY (e.g., when logging to file)
        disable_bar = not sys.stdout.isatty()
        for i, sample in enumerate(
            tqdm(self.samples, desc="Generating pairs", miniters=1000, disable=disable_bar)
        ):
            partner_indices = self._sample_partners(i)
            for j in partner_indices:
                partner = self.samples[j]
                sim_score = self._compute_bb_similarity(
                    sample.instructions,
                    partner.instructions,
                    comp_ins, weighted_lcs, normalize_score, dep_consistency_bonus
                )
                self.pairs.append((i, j, sim_score))
                similarity_scores.append(sim_score)
            if (i + 1) % 1000 == 0 or (i + 1) == len(self.samples):
                pct = (i + 1) / len(self.samples) * 100
                print(f"  Processed {i + 1}/{len(self.samples)} samples ({pct:.1f}%)")
        
        if similarity_scores:
            print("\n  BB Similarity Distribution (training split):")
            print(f"    Mean:   {np.mean(similarity_scores):.3f}")
            print(f"    Std:    {np.std(similarity_scores):.3f}")
            print(f"    Min:    {np.min(similarity_scores):.3f}")
            print(f"    Max:    {np.max(similarity_scores):.3f}")
            print(f"    Median: {np.median(similarity_scores):.3f}")
            
            # Bucketized statistics (match legacy lightweight_encoder_supervised.py)
            ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10)]
            print("\n  BB Similarity by Range (training split):")
            total = len(similarity_scores)
            for low, high in ranges:
                count = sum(1 for s in similarity_scores if low <= s < high)
                pct = (count / total * 100.0) if total > 0 else 0.0
                print(f"    [{low:.1f}, {high:.1f}): {count:6d} pairs ({pct:5.1f}%)")
    
    def _sample_partners(self, idx: int) -> List[int]:
        partners = set()
        attempts = 0
        while len(partners) < self.num_pairs and attempts < self.num_pairs * 20:
            j = random.randint(0, len(self.samples) - 1)
            if j != idx:
                partners.add(j)
            attempts += 1
        # Fallback if we still don't have enough partners
        while len(partners) < self.num_pairs:
            j = random.randint(0, len(self.samples) - 1)
            if j != idx:
                partners.add(j)
        return list(partners)
    
    def _compute_bb_similarity(
        self, 
        instrs_a: List[str], 
        instrs_b: List[str],
        comp_ins, weighted_lcs, normalize_score, dep_consistency_bonus
    ) -> float:
        """
        Compute BB similarity using weighted LCS + execution unit awareness
        Same method as lightweight_encoder_supervised.py
        """
        # Tokenize instructions
        toks_a = [self.bb_tokenizer.tokenize_instruction(inst) for inst in instrs_a]
        toks_b = [self.bb_tokenizer.tokenize_instruction(inst) for inst in instrs_b]
        
        # Define scoring function for instruction pairs
        def score_fn(i: int, j: int) -> float:
            return comp_ins(
                toks_a[i], toks_b[j], self.bb_tokenizer,
                B_exact=self.bb_params['B_exact'],
                alpha_sub=self.bb_params['alpha_sub'],
                alpha_unit=self.bb_params['alpha_unit'],
                B_sameop=self.bb_params['B_sameop'],
                B_sameimm=self.bb_params['B_sameimm']
            )
        
        # Compute weighted LCS
        raw_score_seq, pairs = weighted_lcs(toks_a, toks_b, score_fn)
        
        # Add dependency consistency bonus if enabled
        dep_bonus = 0.0
        if self.bb_params['enable_dep'] and len(pairs) > 0:
            try:
                side_a = self.bb_tokenizer.build_dependency_sidecar(instrs_a)
                side_b = self.bb_tokenizer.build_dependency_sidecar(instrs_b)
                dep_bonus = dep_consistency_bonus(
                    pairs, side_a, side_b,
                    beta_dep_total=self.bb_params['beta_dep'],
                    gamma_mis=self.bb_params['gamma_mis'],
                    dep_window=max(1, self.bb_params['dep_window'])
                )
            except Exception:
                pass  # Silently skip if dependency computation fails
        
        raw_score = raw_score_seq + dep_bonus
        
        # Normalize score
        norm_score = normalize_score(
            raw_score, 
            len(toks_a), 
            len(toks_b), 
            self.bb_params['normalize']
        )
        
        return norm_score
    
    def _generate_pairs_simple(self):
        """Fallback: Generate pairs with simple Jaccard similarity"""
        print("Using simple Jaccard similarity fallback...")
        disable_bar = not sys.stdout.isatty()
        for i, sample in enumerate(
            tqdm(self.samples, desc="Generating pairs", miniters=1000, disable=disable_bar)
        ):
            partner_indices = self._sample_partners(i)
            for j in partner_indices:
                partner = self.samples[j]
                
                # Simple token overlap similarity
                tokens_a = set()
                tokens_b = set()
                
                for instr in sample.instructions:
                    toks = self.tokenizer.tokenize_instruction(instr)
                    tokens_a.update(t for t in toks if t not in {'<s>', '</s>', '<PAD>', '<UNK>','<dsts>', '<srcs>', ';'})
                
                for instr in partner.instructions:
                    toks = self.tokenizer.tokenize_instruction(instr)
                    tokens_b.update(t for t in toks if t not in {'<s>', '</s>', '<PAD>', '<UNK>', '<dsts>', '<srcs>', ';'})
                
                if len(tokens_a | tokens_b) == 0:
                    sim_score = 0.0
                else:
                    sim_score = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                
                self.pairs.append((i, j, sim_score))

    def _load_precomputed_pairs(self, precomputed_pairs, source_path: Optional[str]):
        """Load precomputed similarity pairs and filter by current subset"""
        loaded = 0
        for entry in precomputed_pairs:
            idx_a = entry['idx_a']
            idx_b = entry['idx_b']
            sim = entry['similarity']
            if idx_a in self.index_map and idx_b in self.index_map:
                self.pairs.append((self.index_map[idx_a], self.index_map[idx_b], sim))
                loaded += 1
        if loaded == 0:
            print("Warning: no precomputed pairs matched this split; falling back to on-the-fly generation.")
            self.pairs = []
            self._generate_pairs()
            return
        src_msg = f" from {source_path}" if source_path else ""
        print(f"Loaded {loaded} precomputed pairs{src_msg}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j, similarity = self.pairs[idx]
        
        ids_a = self.tokenized_samples[i]
        ids_b = self.tokenized_samples[j]
        
        if len(ids_a) < self.max_length:
            ids_a = ids_a + [self.pad_idx] * (self.max_length - len(ids_a))
        else:
            ids_a = ids_a[:self.max_length]
        
        if len(ids_b) < self.max_length:
            ids_b = ids_b + [self.pad_idx] * (self.max_length - len(ids_b))
        else:
            ids_b = ids_b[:self.max_length]
        
        return {
            'seq_a': torch.tensor(ids_a, dtype=torch.long),
            'seq_b': torch.tensor(ids_b, dtype=torch.long),
            'similarity': torch.tensor(similarity, dtype=torch.float32)
        }


# ============================================================================
# Model Creation
# ============================================================================
def create_model(model_type: str, vocab_size: int, task: str, args) -> nn.Module:
    """
    Create model based on type and task
    
    Args:
        model_type: 'lstm', 'babygpt', 'gpt2-small', 'gpt2'
        vocab_size: Size of vocabulary
        task: 'mlm' or 'supervised'
        args: Command line arguments
    
    Returns:
        model: PyTorch model
    """
    print(f"\nCreating {model_type} model for {task} task...")
    
    architecture_details = ""
    
    if model_type == 'lstm':
        if task == 'mlm':
            model = LightweightLSTMEncoder(
                vocab_size=vocab_size,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_layers=2,
                dropout=args.dropout,
                pad_idx=0,
                task='mlm'
            )
            architecture_details = (
                f"LSTM Encoder | layers=2, embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}, "
                f"dropout={args.dropout}, task=mlm"
            )
        else:  # supervised
            model = LightweightLSTMEncoder(
                vocab_size=vocab_size,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,
                num_layers=2,
                dropout=args.dropout,
                pad_idx=0,
                task='supervised'
            )
            architecture_details = (
                f"LSTM Encoder | layers=2, embed_dim={args.embed_dim}, hidden_dim={args.hidden_dim}, "
                f"output_dim={args.output_dim}, dropout={args.dropout}, task=supervised"
            )
    
    elif model_type == 'babygpt':
        config = BabyGPTConfig(
            block_size=args.max_seq_len,
            vocab_size=vocab_size,
            n_layer=4,
            n_head=4,
            n_embd=256,
            dropout=args.dropout,
            bias=True,
            task=task,
            output_dim=args.output_dim if task == 'supervised' else 32,
            pad_idx=0
        )
        model = BabyGPT(config)
        architecture_details = (
            "BabyGPT | layers={layers}, heads={heads}, embed_dim={emb}, block_size={block}, "
            "dropout={dropout}, bias={bias}, task={task}, output_dim={output_dim}"
        ).format(
            layers=config.n_layer,
            heads=config.n_head,
            emb=config.n_embd,
            block=config.block_size,
            dropout=config.dropout,
            bias=config.bias,
            task=task,
            output_dim=(config.output_dim if hasattr(config, 'output_dim') else 'N/A')
        )
    
    elif model_type in ['gpt2-small', 'gpt2']:
        if model_type == 'gpt2-small':
            config = GPT2Config(
                block_size=args.max_seq_len,
                vocab_size=vocab_size,
                n_layer=8,
                n_head=12,
                n_embd=384,
                dropout=args.dropout,
                task=task,
                output_dim=args.output_dim if task == 'supervised' else 32,
                pad_idx=0
            )
        else:  # gpt2
            config = GPT2Config(
                block_size=args.max_seq_len,
                vocab_size=vocab_size,
                n_layer=12,
                n_head=12,
                n_embd=768,
                dropout=args.dropout,
                task=task,
                output_dim=args.output_dim if task == 'supervised' else 32,
                pad_idx=0
            )
        model = GPT2Encoder(config)
        architecture_details = (
            f"{model_type.upper()} | layers={config.n_layer}, heads={config.n_head}, embed_dim={config.n_embd}, "
            f"block_size={config.block_size}, dropout={config.dropout}, task={task}, "
            f"output_dim={config.output_dim if hasattr(config, 'output_dim') else 'N/A'}"
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print model info
    if architecture_details:
        print(f"  Architecture: {architecture_details}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} (~{trainable_params/1e6:.2f}M)")
    
    return model


# ============================================================================
# MLM Training
# ============================================================================
def train_mlm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    args,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None
) -> Dict[str, List[float]]:
    """Train MLM task"""
    
    print("\n" + "="*80)
    print("Starting MLM Training")
    print("="*80)
    
    history = {'train_loss': [], 'val_loss': [], 'token_loss': [], 'instr_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_losses = []
        token_losses = []
        instr_losses = []
        
        # Show tqdm only when running in an interactive TTY.
        disable_bar = not sys.stdout.isatty()
        num_train_batches = len(train_loader)
        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
            miniters=1000,
            disable=disable_bar,
        )
        for batch_idx, batch in enumerate(train_iter):
            # Rename batch keys for model input
            inputs = {
                'token_mlm': (batch['token_level'][0], batch['token_level'][1].to(device)),
                'instr_mlm': (batch['instruction_level'][0], batch['instruction_level'][1].to(device))
            }
            
            # Move token_level and instruction_level inputs to device
            for key in ['token_mlm', 'instr_mlm']:
                inputs_dict, targets = inputs[key]
                inputs[key] = (
                    {'asm': inputs_dict['asm'].to(device)},
                    targets
                )
            
            # Forward pass
            loss, outputs = model(inputs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            token_losses.append(outputs['token_loss'].item())
            instr_losses.append(outputs['instr_loss'].item())
            
            if not disable_bar:
                train_iter.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'token': f"{outputs['token_loss'].item():.4f}",
                    'instr': f"{outputs['instr_loss'].item():.4f}"
                })
            # Optional textual logging for non-TTY / logfile runs
            if (
                args.log_interval > 0
                and ((batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == num_train_batches)
            ):
                print(
                    f"[MLM][Train] Epoch {epoch+1}/{args.epochs} "
                    f"Batch {batch_idx+1}/{num_train_batches} "
                    f"loss={loss.item():.4f} "
                    f"token={outputs['token_loss'].item():.4f} "
                    f"instr={outputs['instr_loss'].item():.4f}"
                )
            
            # Debug mode: only train on first batch
            if args.mode == 'demo' and batch_idx >= 0:
                break
        
        avg_train_loss = np.mean(train_losses)
        avg_token_loss = np.mean(token_losses)
        avg_instr_loss = np.mean(instr_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            disable_bar = not sys.stdout.isatty()
            num_val_batches = len(val_loader)
            val_iter = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                miniters=500,
                disable=disable_bar,
            )
            for batch_idx, batch in enumerate(val_iter):
                inputs = {
                    'token_mlm': (batch['token_level'][0], batch['token_level'][1].to(device)),
                    'instr_mlm': (batch['instruction_level'][0], batch['instruction_level'][1].to(device))
                }
                
                for key in ['token_mlm', 'instr_mlm']:
                    inputs_dict, targets = inputs[key]
                    inputs[key] = (
                        {'asm': inputs_dict['asm'].to(device)},
                        targets
                    )
                
                loss, outputs = model(inputs)
                val_losses.append(loss.item())
                
                if not disable_bar:
                    val_iter.set_postfix({'val_loss': f"{loss.item():.4f}"})
                if (
                    args.log_interval > 0
                    and ((batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == num_val_batches)
                ):
                    print(
                        f"[MLM][Val] Epoch {epoch+1}/{args.epochs} "
                        f"Batch {batch_idx+1}/{num_val_batches} "
                        f"val_loss={loss.item():.4f}"
                    )
                
                # Debug mode: only validate on first batch
                if args.mode == 'demo' and batch_idx >= 0:
                    break
        
        avg_val_loss = np.mean(val_losses)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['token_loss'].append(avg_token_loss)
        history['instr_loss'].append(avg_instr_loss)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Token: {avg_token_loss:.4f}, Instr: {avg_instr_loss:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f"  ✓ New best model saved: {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Step LR scheduler (if configured) based on validation loss
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Early stopping
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    return history


# ============================================================================
# Supervised Training
# ============================================================================
def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    args,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None
) -> Dict[str, List[float]]:
    """Train supervised similarity learning task"""
    
    print("\n" + "="*80)
    print("Starting Supervised Training")
    print("="*80)
    
    history = {'train_loss': [], 'val_loss': [], 'mse_loss': [], 'cosine_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    cosine_criterion = nn.SmoothL1Loss()
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_losses = []
        mse_losses = []
        cosine_losses = []
        
        disable_bar = not sys.stdout.isatty()
        num_train_batches = len(train_loader)
        train_iter = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} [Train]",
            miniters=1000,
            disable=disable_bar,
        )
        for batch_idx, batch in enumerate(train_iter):
            seq_a = batch['seq_a'].to(device)
            seq_b = batch['seq_b'].to(device)
            similarity = batch['similarity'].to(device)
            
            # Forward pass
            emb_a = model(seq_a)  # (batch, output_dim), L2 normalized
            emb_b = model(seq_b)
            
            # Compute cosine similarity
            cosine_sim = (emb_a * emb_b).sum(dim=1)  # Already normalized
            
            # Normalize similarity targets to legacy scale
            similarity_norm = torch.clamp(
                similarity / max(1e-6, args.similarity_scale), 0.0, 1.0
            )
            target_cosine = args.cosine_target_min + (
                args.cosine_target_max - args.cosine_target_min
            ) * similarity_norm
            
            # Losses
            mse_loss = mse_criterion(cosine_sim, target_cosine)
            cosine_loss = cosine_criterion(cosine_sim, target_cosine)
            loss = mse_loss + args.cosine_loss_weight * cosine_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            mse_losses.append(mse_loss.item())
            cosine_losses.append(cosine_loss.item())
            
            if not disable_bar:
                train_iter.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'mse': f"{mse_loss.item():.4f}",
                    'cosine': f"{cosine_loss.item():.4f}"
                })
            if (
                args.log_interval > 0
                and ((batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == num_train_batches)
            ):
                print(
                    f"[SUP][Train] Epoch {epoch+1}/{args.epochs} "
                    f"Batch {batch_idx+1}/{num_train_batches} "
                    f"loss={loss.item():.4f} "
                    f"mse={mse_loss.item():.4f} "
                    f"cosine={cosine_loss.item():.4f}"
                )
            
            # Debug mode: only train on first batch
            if args.mode == 'demo' and batch_idx >= 0:
                break
        
        avg_train_loss = np.mean(train_losses)
        avg_mse_loss = np.mean(mse_losses)
        avg_cosine_loss = np.mean(cosine_losses)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            disable_bar = not sys.stdout.isatty()
            num_val_batches = len(val_loader)
            val_iter = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{args.epochs} [Val]",
                miniters=500,
                disable=disable_bar,
            )
            for batch_idx, batch in enumerate(val_iter):
                seq_a = batch['seq_a'].to(device)
                seq_b = batch['seq_b'].to(device)
                similarity = batch['similarity'].to(device)
                
                emb_a = model(seq_a)
                emb_b = model(seq_b)
                cosine_sim = (emb_a * emb_b).sum(dim=1)
                
                similarity_norm = torch.clamp(
                    similarity / max(1e-6, args.similarity_scale), 0.0, 1.0
                )
                target_cosine = args.cosine_target_min + (
                    args.cosine_target_max - args.cosine_target_min
                ) * similarity_norm
                
                mse_loss = mse_criterion(cosine_sim, target_cosine)
                cosine_loss = cosine_criterion(cosine_sim, target_cosine)
                loss = mse_loss + args.cosine_loss_weight * cosine_loss
                
                val_losses.append(loss.item())
                
                if not disable_bar:
                    val_iter.set_postfix({'val_loss': f"{loss.item():.4f}"})
                if (
                    args.log_interval > 0
                    and ((batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == num_val_batches)
                ):
                    print(
                        f"[SUP][Val] Epoch {epoch+1}/{args.epochs} "
                        f"Batch {batch_idx+1}/{num_val_batches} "
                        f"val_loss={loss.item():.4f}"
                    )
                
                # Debug mode: only validate on first batch
                if args.mode == 'demo' and batch_idx >= 0:
                    break
        
        avg_val_loss = np.mean(val_losses)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['mse_loss'].append(avg_mse_loss)
        history['cosine_loss'].append(avg_cosine_loss)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (MSE: {avg_mse_loss:.4f}, Cosine: {avg_cosine_loss:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f"  ✓ New best model saved: {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Step LR scheduler (if configured) based on validation loss
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Early stopping
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    return history


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Unified BBencoder Training Script')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True, 
                        choices=['lstm', 'babygpt', 'gpt2-small', 'gpt2'],
                        help='Model architecture')
    
    # Task selection
    parser.add_argument('--task', type=str, required=True,
                        choices=['mlm', 'supervised'],
                        help='Training task')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                        choices=['demo', 'train'],
                        help='Training mode: demo (debug) or train (full)')
    
    # DataBBencoder/dataset/final_training_data_fp_fixed.pkl
    parser.add_argument('--data', type=str, default='dataset/final_training_data_fp_fixed.pkl',
                        help='Path to training data (pickle file)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to use (None = all)')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--precomputed-pairs', type=str, default=None,
                        help='Path to precomputed supervised pair similarities (pickle)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping (0 = no clipping)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience based on validation loss (0 = disabled)')
    # Optional LR scheduler (useful especially for Stage-2 supervised fine-tuning)
    parser.add_argument('--lr-scheduler', type=str, default='none',
                        choices=['none', 'plateau'],
                        help="Learning rate scheduler: 'plateau' reduces LR on validation loss plateau")
    parser.add_argument('--lr-decay-factor', type=float, default=0.5,
                        help='Multiplicative factor for LR decay when using plateau scheduler')
    parser.add_argument('--lr-decay-patience', type=int, default=5,
                        help='Number of epochs with no val loss improvement before LR is decayed')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Lower bound on learning rate when using a scheduler')
    
    # Model hyperparameters
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='Embedding dimension (LSTM only)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension (LSTM only)')
    parser.add_argument('--output-dim', type=int, default=32,
                        help='Output dimension (supervised task only)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--max-seq-len', type=int, default=512,
                        help='Maximum sequence length')
    
    # MLM specific
    parser.add_argument('--token-mask-prob', type=float, default=0.15,
                        help='Token masking probability (MLM)')
    parser.add_argument('--instr-mask-prob', type=float, default=0.15,
                        help='Instruction masking probability (MLM)')
    
    # Supervised specific
    parser.add_argument('--num-pairs', type=int, default=5,
                        help='Number of pairs per sample (supervised)')
    parser.add_argument('--normalize', type=str, default='maxlen',
                        choices=['maxlen', 'avglen', 'none'],
                        help='BB similarity normalization (supervised)')
    parser.add_argument('--enable-dep', action='store_true',
                        help='Enable dependency consistency reward (supervised)')
    parser.add_argument('--beta-dep', type=float, default=0.5,
                        help='Dependency reward weight (supervised)')
    parser.add_argument('--gamma-mis', type=float, default=0.05,
                        help='Dependency mismatch penalty (supervised)')
    parser.add_argument('--dep-window', type=int, default=3,
                        help='Dependency matching window (supervised)')
    # Legacy supervised-loss controls (match lightweight_encoder_supervised.py behaviour):
    # similarity-scale: divides the raw BB similarity (typically 0~4) before mapping into cosine target range
    parser.add_argument('--similarity-scale', type=float, default=4.0,
                        help='Divisor applied to raw BB similarity prior to mapping into cosine target range')
    # cosine-target-min / max: define the cosine target range after normalization; avoids regressing directly to [-1, 1]
    parser.add_argument('--cosine-target-min', type=float, default=0.2,
                        help='Lower bound of the cosine target after normalization (legacy default 0.2)')
    parser.add_argument('--cosine-target-max', type=float, default=1.0,
                        help='Upper bound of the cosine target after normalization (legacy default 1.0)')
    # cosine-loss-weight: relative weight of the SmoothL1 term, mirroring legacy alpha=1, beta=0.5
    parser.add_argument('--cosine-loss-weight', type=float, default=0.5,
                        help='Weight of SmoothL1 component in the supervised loss (legacy beta)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: models_{task}_{model}_{mode})')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='How many batches between progress log lines (0 = disable batch logs)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Pretrained checkpoint (for fine-tuning)
    parser.add_argument('--pretrained-checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint for fine-tuning or warm start')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Auto-adjust settings for demo mode
    if args.mode == 'demo':
        print("\n" + "="*80)
        print("DEMO MODE - DEBUG SETTINGS")
        print("="*80)
        args.num_samples = args.num_samples or 20
        args.epochs = min(args.epochs, 3)
        args.batch_size = min(args.batch_size, 4)
        args.save_every = 1
        print(f"  Samples: {args.num_samples}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print("="*80)
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"models_{args.task}_{args.model}_{args.mode}_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    with open(args.data, 'rb') as f:
        # Make TrainingSample discoverable
        if 'TrainingSample' not in sys.modules['__main__'].__dict__:
            sys.modules['__main__'].TrainingSample = TrainingSample
        samples = pickle.load(f)
    
    print(f"  Loaded {len(samples)} samples")
    
    # Limit samples
    if args.num_samples and args.num_samples < len(samples):
        samples = samples[:args.num_samples]
        print(f"  Using {len(samples)} samples")
    
    precomputed_pairs = None
    precomputed_tokenizer = None
    precomputed_meta = None
    if args.precomputed_pairs:
        if not os.path.exists(args.precomputed_pairs):
            raise FileNotFoundError(f"Precomputed pairs file not found: {args.precomputed_pairs}")
        with open(args.precomputed_pairs, 'rb') as f:
            precomputed_data = pickle.load(f)
        precomputed_pairs = precomputed_data.get('pairs')
        precomputed_tokenizer = precomputed_data.get('tokenizer')
        precomputed_meta = precomputed_data.get('meta', {})
        if not precomputed_pairs:
            raise ValueError(f"Precomputed pairs file {args.precomputed_pairs} does not contain 'pairs'.")
        expected_samples = precomputed_data.get('num_samples')
        if expected_samples and expected_samples != len(samples):
            print(f"Warning: precomputed pairs expect {expected_samples} samples, "
                  f"but current run uses {len(samples)} samples.")
    
    # Split train/val
    split_idx = int(len(samples) * (1 - args.validation_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")
    global_indices = list(range(len(samples)))
    train_indices = global_indices[:split_idx]
    val_indices = global_indices[split_idx:]
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    if precomputed_tokenizer is not None:
        tokenizer = precomputed_tokenizer
        print("  Loaded tokenizer from precomputed pairs file")
    else:
        tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=False)
    vocab_size = len(tokenizer.vocab)
    print(f"  Vocabulary size: {vocab_size}")
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"  Saved tokenizer: {tokenizer_path}")
    
    # Create datasets
    if args.task == 'mlm':
        print("\nCreating MLM datasets...")
        # BBTokenizerMLMDataset expects testcases as text strings (one per line)
        # Convert instruction lists to newline-separated strings
        train_testcases = ['\n'.join(s.instructions) for s in train_samples]
        val_testcases = ['\n'.join(s.instructions) for s in val_samples]
        
        train_dataset = BBTokenizerMLMDataset(
            max_seq_len=args.max_seq_len,
            min_seq_len=0,
            testcases=train_testcases,
            token_mask_prob=args.token_mask_prob,
            instr_mask_prob=args.instr_mask_prob,
            rng_seed=args.seed
        )
        val_dataset = BBTokenizerMLMDataset(
            max_seq_len=args.max_seq_len,
            min_seq_len=0,
            testcases=val_testcases,
            token_mask_prob=args.token_mask_prob,
            instr_mask_prob=args.instr_mask_prob,
            rng_seed=args.seed + 1
        )
    else:  # supervised
        print("\nCreating supervised datasets...")
        train_dataset = SupervisedDataset(
            samples=train_samples,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            num_pairs=args.num_pairs,
            normalize=args.normalize,
            enable_dep=args.enable_dep,
            beta_dep=args.beta_dep,
            gamma_mis=args.gamma_mis,
            dep_window=args.dep_window,
            precomputed_pairs=precomputed_pairs,
            global_indices=train_indices,
            precomputed_source=args.precomputed_pairs
        )
        val_dataset = SupervisedDataset(
            samples=val_samples,
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
            num_pairs=max(1, args.num_pairs),  # Fewer pairs for validation
            normalize=args.normalize,
            enable_dep=args.enable_dep,
            beta_dep=args.beta_dep,
            gamma_mis=args.gamma_mis,
            dep_window=args.dep_window,
            precomputed_pairs=precomputed_pairs,
            global_indices=val_indices,
            precomputed_source=args.precomputed_pairs
        )
    
    print(f"  Train dataset size: {len(train_dataset)}")
    print(f"  Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = create_model(args.model, vocab_size, args.task, args)
    model = model.to(device)
    
    # Load pretrained checkpoint if provided
    if args.pretrained_checkpoint:
        if not os.path.isfile(args.pretrained_checkpoint):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
        
        print(f"\nLoading pretrained weights from {args.pretrained_checkpoint}...")
        try:
            checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        load_result = model.load_state_dict(state_dict, strict=False)
        
        missing = load_result.missing_keys if hasattr(load_result, 'missing_keys') else load_result[0]
        unexpected = load_result.unexpected_keys if hasattr(load_result, 'unexpected_keys') else load_result[1]
        
        print(f"  ✓ Loaded pretrained weights (strict=False)")
        if missing:
            print(f"    Missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"    Unexpected keys ({len(unexpected)}): {unexpected}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Optional LR scheduler (primarily useful for Stage-2 supervised fine-tuning)
    lr_scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau]
    if args.lr_scheduler == 'plateau':
        # Use a simple ReduceLROnPlateau scheduler without 'verbose' argument
        # for broader compatibility with different torch versions.
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            min_lr=args.min_lr,
        )
    else:
        lr_scheduler = None
    
    # Train
    print("\n" + "="*80)
    print(f"Training Configuration")
    print("="*80)
    print(f"  Model: {args.model}")
    print(f"  Task: {args.task}")
    print(f"  Mode: {args.mode}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print("="*80)
    
    if args.task == 'mlm':
        history = train_mlm(model, train_loader, val_loader, optimizer, args, device, scheduler=lr_scheduler)
    else:  # supervised
        history = train_supervised(model, train_loader, val_loader, optimizer, args, device, scheduler=lr_scheduler)
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\nSaved training history: {history_path}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"  Best validation loss: {min(history['val_loss']):.4f}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Best model: {os.path.join(args.output_dir, 'best_model.pt')}")
    print("="*80)


if __name__ == '__main__':
    main()

