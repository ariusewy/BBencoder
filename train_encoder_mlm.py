#!/usr/bin/env python3
"""
MLM Training Script - BBencoder
Author: ywangmu from HKUST

Train lightweight LSTM encoder using Masked Language Modeling (MLM) pre-training.

This script implements dual MLM objectives similar to RWKV7:
1. Token-level MLM: Predict randomly masked tokens
2. Instruction-level MLM: Predict entire masked instruction spans

Differences from supervised training:
- Self-supervised pre-training (no BB similarity labels required)
- Can leverage large unlabeled assembly datasets
- Learns structural patterns from masking task
- Suitable for transfer learning and downstream fine-tuning

Usage:
python3 train_encoder_mlm.py --num-samples 1000 --epochs 50 --max-seq-len 512
"""

import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List
import sys
import time
import os
from datetime import datetime
from tqdm import tqdm

# Add project paths dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

from prepare_balanced_dataset import TrainingSample
from bbtokenizer_mlm_dataset import BBTokenizerMLMDataset
from advanced_tokenizer import AdvancedRISCVTokenizer


class LightweightMLMEncoder(nn.Module):
    """
    Lightweight LSTM-based encoder with MLM prediction head
    
    Architecture:
    - Embedding layer
    - Bidirectional LSTM with attention
    - Dual prediction heads for token-level and instruction-level MLM
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, 
                 hidden_dim: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        
        # Shared embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Context attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Dual MLM prediction heads
        self.token_mlm_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.instr_mlm_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Loss function (ignore -100 targets)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'embedding' not in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, inputs, targets=None):
        """
        Forward pass for dual MLM tasks
        
        Args:
            inputs: dict with 'token_mlm' and 'instr_mlm' entries
                    Each entry: (inputs_dict, targets)
            targets: not used (targets are in inputs dict)
        
        Returns:
            loss: combined MLM loss
            outputs: dict with detailed losses and logits
        """
        token_inputs, token_targets = inputs['token_mlm']
        instr_inputs, instr_targets = inputs['instr_mlm']
        
        # Token-level MLM forward
        token_ids = token_inputs['asm']  # (batch, seq_len)
        token_emb = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        token_lstm_out, _ = self.lstm(token_emb)  # (batch, seq_len, hidden*2)
        token_logits = self.token_mlm_head(token_lstm_out)  # (batch, seq_len, vocab)
        
        # Compute token-level loss
        token_loss = self.criterion(
            token_logits.view(-1, self.vocab_size), 
            token_targets.view(-1)
        )
        
        # Instruction-level MLM forward
        instr_ids = instr_inputs['asm']
        instr_emb = self.embedding(instr_ids)
        instr_lstm_out, _ = self.lstm(instr_emb)
        instr_logits = self.instr_mlm_head(instr_lstm_out)
        
        # Compute instruction-level loss
        instr_loss = self.criterion(
            instr_logits.view(-1, self.vocab_size), 
            instr_targets.view(-1)
        )
        
        # Combined loss
        loss = token_loss + instr_loss
        
        return loss, {
            'token_outputs': token_logits,
            'instr_outputs': instr_logits,
            'token_loss': token_loss,
            'instr_loss': instr_loss
        }
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract sequence embeddings for downstream tasks
        
        Args:
            token_ids: (batch, seq_len)
        
        Returns:
            embeddings: (batch, hidden*2) normalized embeddings
        """
        embedded = self.embedding(token_ids)
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention pooling
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_out = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)
        
        # L2 normalization
        output = F.normalize(weighted_out, p=2, dim=1)
        return output
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())


def collate_mlm_batch(batch):
    """
    Custom collate function for MLM batches
    
    Handles the dual-task batch structure from BBTokenizerMLMDataset
    """
    # Extract token-level data
    token_inputs_list = [item['token_level'][0]['asm'] for item in batch]
    token_targets_list = [item['token_level'][1] for item in batch]
    
    # Extract instruction-level data
    instr_inputs_list = [item['instruction_level'][0]['asm'] for item in batch]
    instr_targets_list = [item['instruction_level'][1] for item in batch]
    
    # Stack into tensors (already same length from dataset)
    token_inputs = torch.stack(token_inputs_list)
    token_targets = torch.stack(token_targets_list)
    instr_inputs = torch.stack(instr_inputs_list)
    instr_targets = torch.stack(instr_targets_list)
    
    return {
        'token_level': ({'asm': token_inputs}, token_targets),
        'instruction_level': ({'asm': instr_inputs}, instr_targets)
    }


def train_mlm_encoder(
    testcases: List[str],
    output_dir: str = 'models_mlm',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    max_seq_len: int = 512,
    min_seq_len: int = 0,
    device: str = 'cuda',
    save_every: int = 10,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    debug: bool = False,
):
    """
    Train MLM encoder
    
    Args:
        testcases: List of assembly instruction sequences
        output_dir: Directory to save models
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        max_seq_len: Maximum sequence length
        min_seq_len: Minimum sequence length (filter)
        device: Training device ('cuda' or 'cpu')
        save_every: Save checkpoint every N epochs
        validation_split: Validation set ratio
        early_stopping_patience: Early stopping patience
    
    Returns:
        model: Trained model
        tokenizer: Tokenizer
    """
    import random
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split dataset
    print("\n[1/4] Creating training and validation datasets...")
    random.seed(42)
    shuffled_testcases = testcases.copy()
    random.shuffle(shuffled_testcases)
    
    split_idx = int(len(shuffled_testcases) * (1 - validation_split))
    train_testcases = shuffled_testcases[:split_idx]
    val_testcases = shuffled_testcases[split_idx:]
    
    print(f"  Dataset split:")
    print(f"    Training: {len(train_testcases):,} testcases ({len(train_testcases)/len(testcases)*100:.1f}%)")
    print(f"    Validation: {len(val_testcases):,} testcases ({len(val_testcases)/len(testcases)*100:.1f}%)")
    
    # Create datasets
    print(f"\n  Creating training MLM dataset...")
    train_dataset = BBTokenizerMLMDataset(
        testcases=train_testcases,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        token_mask_prob=0.15,
        instr_mask_prob=0.15,
        rng_seed=42
    )
    
    print(f"\n  Creating validation MLM dataset...")
    val_dataset = BBTokenizerMLMDataset(
        testcases=val_testcases,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        token_mask_prob=0.15,
        instr_mask_prob=0.15,
        rng_seed=43
    )
    
    print(f"\n  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    print(f"  Vocabulary size: {train_dataset.vocab_size['asm']:,}")
    
    # Debug: Show sample masking details
    if debug and len(train_dataset) > 0:
        print("\n" + "="*80)
        print("[DEBUG] Dataset Sample Masking Analysis")
        print("="*80)
        
        sample_idx = 0
        sample_bb = train_testcases[sample_idx]
        
        print(f"\n[Sample #{sample_idx}] Original BB Instructions:")
        print("-" * 80)
        inst_lines = sample_bb.split('\n')
        for i, inst in enumerate(inst_lines[:10]):
            print(f"  {i+1:2d}. {inst}")
        if len(inst_lines) > 10:
            remaining = len(inst_lines) - 10
            print(f"  ... ({remaining} more instructions)")
        
        # Tokenize the sample
        tokenizer = train_dataset.tokenizer
        tokens = tokenizer.tokenize_testcase(sample_bb)
        token_ids = tokenizer.encode(sample_bb)
        
        print(f"\n[Sample #{sample_idx}] Tokenization:")
        print("-" * 80)
        print(f"  Total instructions: {len(inst_lines)}")
        print(f"  Total tokens: {len(tokens)}")
        print(f"  Total token IDs: {len(token_ids)}")
        print(f"  First 30 tokens: {tokens[:30]}")
        print(f"  First 30 token IDs: {token_ids[:30]}")
        
        # Get dataset item to see masking
        dataset_item = train_dataset[sample_idx]
        token_inputs, token_targets = dataset_item['token_level']
        instr_inputs, instr_targets = dataset_item['instruction_level']
        
        print(f"\n[Sample #{sample_idx}] Dataset Item Shapes:")
        print("-" * 80)
        print(f"  Token-level inputs['asm']: {token_inputs['asm'].shape}")
        print(f"  Token-level targets: {token_targets.shape}")
        print(f"  Instruction-level inputs['asm']: {instr_inputs['asm'].shape}")
        print(f"  Instruction-level targets: {instr_targets.shape}")
        
        # Analyze token-level masking
        token_masked_positions = (token_targets != -100).nonzero(as_tuple=True)[0]
        token_masked_count = len(token_masked_positions)
        
        print(f"\n[Sample #{sample_idx}] Token-Level Masking Analysis:")
        print("-" * 80)
        print(f"  Total masked positions: {token_masked_count}/{token_targets.numel()}")
        print(f"  Masking ratio: {token_masked_count/token_targets.numel()*100:.2f}%")
        
        if token_masked_count > 0:
            print(f"\n  First 10 masked token positions:")
            for i, pos in enumerate(token_masked_positions[:10].tolist()):
                original_id = token_targets[pos].item()
                masked_id = token_inputs['asm'][pos].item()
                original_token = tokenizer.id_to_token.get(original_id, f"<id={original_id}>")
                masked_token = tokenizer.id_to_token.get(masked_id, f"<id={masked_id}>")
                
                mask_type = "MASK"
                if masked_id == train_dataset.MASK_ID:
                    mask_type = "MASK"
                elif masked_id == original_id:
                    mask_type = "KEEP"
                else:
                    mask_type = "RAND"
                
                print(f"    Pos {pos:3d}: '{original_token}' (id={original_id:3d}) -> '{masked_token}' (id={masked_id:3d}) [{mask_type}]")
        
        # Analyze instruction-level masking
        instr_masked_positions = (instr_targets != -100).nonzero(as_tuple=True)[0]
        instr_masked_count = len(instr_masked_positions)
        
        print(f"\n[Sample #{sample_idx}] Instruction-Level Masking Analysis:")
        print("-" * 80)
        print(f"  Total masked positions: {instr_masked_count}/{instr_targets.numel()}")
        print(f"  Masking ratio: {instr_masked_count/instr_targets.numel()*100:.2f}%")
        
        if instr_masked_count > 0:
            # Find masked instruction spans
            print(f"\n  Masked instruction spans:")
            in_span = False
            span_start = None
            span_count = 0
            
            for pos in range(len(instr_targets)):
                if instr_targets[pos] != -100 and not in_span:
                    # Start of masked span
                    span_start = pos
                    in_span = True
                elif (instr_targets[pos] == -100 or pos == len(instr_targets) - 1) and in_span:
                    # End of masked span
                    span_end = pos if instr_targets[pos] == -100 else pos + 1
                    span_count += 1
                    
                    # Show span details (limit to first 5 spans)
                    if span_count <= 5:
                        span_tokens = []
                        for p in range(span_start, min(span_end, span_start + 10)):
                            tid = instr_targets[p].item()
                            if tid != -100:
                                tok = tokenizer.id_to_token.get(tid, f"<id={tid}>")
                                span_tokens.append(tok)
                        
                        span_len = span_end - span_start
                        tokens_preview = ' '.join(span_tokens[:10])
                        if span_len > 10:
                            tokens_preview += " ..."
                        
                        print(f"    Span {span_count}: positions [{span_start:3d}, {span_end:3d}), length={span_len:3d}")
                        print(f"              Tokens: {tokens_preview}")
                    
                    in_span = False
            
            if span_count > 5:
                print(f"    ... ({span_count - 5} more masked spans)")
            
            print(f"\n  Total masked instruction spans: {span_count}")
            
            # Find <s> markers to estimate instruction boundaries
            s_positions = []
            for pos in range(len(token_inputs['asm'])):
                if token_inputs['asm'][pos].item() == train_dataset.CLS_ID:
                    s_positions.append(pos)
            
            print(f"  Total <s> markers (instruction boundaries): {len(s_positions)}")
            if len(s_positions) > 0:
                print(f"  Expected masking rate: 15% of {len(s_positions)} instructions = ~{int(len(s_positions) * 0.15)} instructions")
        
        print("="*80 + "\n")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_mlm_batch,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mlm_batch,
        num_workers=0
    )
    
    # Create model
    print("\n[2/4] Creating model...")
    model = LightweightMLMEncoder(
        vocab_size=train_dataset.vocab_size['asm'],
        embed_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.2,
        pad_idx=train_dataset.PAD_ID
    )
    model.to(device)
    
    num_params = model.get_num_params()
    print(f"  Model parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate/10
    )
    
    # Save tokenizer
    tokenizer = train_dataset.tokenizer
    tokenizer_path = f'{output_dir}/tokenizer.pkl'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"  Tokenizer saved to {tokenizer_path}")
    
    # Validation function
    def validate_model(model, val_dataloader, device):
        model.eval()
        val_loss = 0
        val_token_loss = 0
        val_instr_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                token_inputs, token_targets = batch['token_level']
                instr_inputs, instr_targets = batch['instruction_level']
                
                token_inputs['asm'] = token_inputs['asm'].to(device)
                token_targets = token_targets.to(device)
                instr_inputs['asm'] = instr_inputs['asm'].to(device)
                instr_targets = instr_targets.to(device)
                
                inputs = {
                    'token_mlm': (token_inputs, token_targets),
                    'instr_mlm': (instr_inputs, instr_targets)
                }
                
                loss, outputs = model(inputs)
                
                val_loss += loss.item()
                val_token_loss += outputs['token_loss'].item()
                val_instr_loss += outputs['instr_loss'].item()
                num_batches += 1
        
        return (
            val_loss / num_batches if num_batches > 0 else float('inf'),
            val_token_loss / num_batches if num_batches > 0 else float('inf'),
            val_instr_loss / num_batches if num_batches > 0 else float('inf')
        )
    
    # Training loop
    print("\n[3/4] Training with dual MLM objectives...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_token_loss = 0
        train_instr_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            token_inputs, token_targets = batch['token_level']
            instr_inputs, instr_targets = batch['instruction_level']
            
            token_inputs['asm'] = token_inputs['asm'].to(device)
            token_targets = token_targets.to(device)
            instr_inputs['asm'] = instr_inputs['asm'].to(device)
            instr_targets = instr_targets.to(device)
            
            inputs = {
                'token_mlm': (token_inputs, token_targets),
                'instr_mlm': (instr_inputs, instr_targets)
            }
            
            loss, outputs = model(inputs)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_token_loss += outputs['token_loss'].item()
            train_instr_loss += outputs['instr_loss'].item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'token': f"{outputs['token_loss'].item():.4f}",
                'instr': f"{outputs['instr_loss'].item():.4f}"
            })
        
        scheduler.step()
        
        avg_train_loss = train_loss / num_batches
        avg_train_token_loss = train_token_loss / num_batches
        avg_train_instr_loss = train_instr_loss / num_batches
        
        val_loss, val_token_loss, val_instr_loss = validate_model(model, val_dataloader, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} (tok={avg_train_token_loss:.4f}, ins={avg_train_instr_loss:.4f}) | "
              f"Val Loss: {val_loss:.4f} (tok={val_token_loss:.4f}, ins={val_instr_loss:.4f}) | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'vocab_size': train_dataset.vocab_size['asm'],
                'embed_dim': embedding_dim,
                'hidden_dim': hidden_dim,
            }, f'{output_dir}/best_model.pt')
            print(f"  → Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  → Early stopping triggered (patience: {early_stopping_patience})")
                break
        
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'vocab_size': train_dataset.vocab_size['asm'],
                'embed_dim': embedding_dim,
                'hidden_dim': hidden_dim,
            }, f'{output_dir}/checkpoint_epoch_{epoch+1}.pt')
    
    print("\n[4/4] Training completed!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Models saved to: {output_dir}/")
    
    return model, tokenizer


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train RISC-V testcase encoder with MLM pre-training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data related
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, 'dataset', 'final_training_data.pkl')
    parser.add_argument('--data', type=str, 
                       default=default_data_path,
                       help='Training data path')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to use (None=use all)')
    
    # Training config
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max-seq-len', type=int, default=512,
                       help='Maximum token sequence length')
    parser.add_argument('--min-seq-len', type=int, default=0,
                       help='Minimum sequence length filter')
    
    # Model config
    parser.add_argument('--embed-dim', type=int, default=192,
                       help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=384,
                       help='LSTM hidden dimension')
    
    # Device and save
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Training device')
    default_output_dir = os.path.join(script_dir, 'models_mlm')
    parser.add_argument('--output-dir', type=str, 
                       default=default_output_dir,
                       help='Model save directory')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation set ratio (0.0-1.0)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug output for masking details')
    
    return parser.parse_args()


def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 80)
    print(" " * 20 + "BBencoder - MLM Pre-training")
    print(" " * 10 + "Dual Masked Language Modeling for RISC-V Assembly")
    print("=" * 80)


def print_config(args):
    """Print training configuration"""
    print("\n" + "-" * 80)
    print("Training Configuration")
    print("-" * 80)
    
    print(f"\nData Configuration:")
    print(f"  Dataset path: {args.data}")
    print(f"  Sample count: {args.num_samples if args.num_samples else 'All'}")
    print(f"  Max sequence length: {args.max_seq_len} tokens")
    print(f"  Min sequence length: {args.min_seq_len} tokens")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    
    print(f"\nModel Configuration:")
    print(f"  Embedding dim: {args.embed_dim}")
    print(f"  LSTM hidden dim: {args.hidden_dim}")
    print(f"  Architecture: BiLSTM + Attention + Dual MLM Heads")
    
    print(f"\nMLM Configuration:")
    print(f"  Token masking probability: 15%")
    print(f"  Instruction masking probability: 15%")
    print(f"  Tasks: Token-level + Instruction-level MLM")
    
    print(f"\nSave Configuration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Checkpoint frequency: every {args.save_every} epochs")
    print(f"  Random seed: {args.seed}")
    
    print(f"\nValidation Configuration:")
    print(f"  Validation split: {args.validation_split*100:.1f}%")
    print(f"  Early stopping patience: {args.early_stopping_patience} epochs")
    print("-" * 80)


def load_dataset(data_path, num_samples=None):
    """Load dataset"""
    print(f"\n[1/3] Loading dataset...")
    print(f"  Path: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    start_time = time.time()
    with open(data_path, 'rb') as f:
        all_samples = pickle.load(f)
    load_time = time.time() - start_time
    
    # Select sample count
    if num_samples is not None:
        samples = all_samples[:num_samples]
        print(f"  ✓ Loaded: {len(all_samples):,} total samples")
        print(f"  → Using: {len(samples):,} samples")
    else:
        samples = all_samples
        print(f"  ✓ Loaded: {len(samples):,} samples")
    
    # Statistics
    rocket_count = sum(1 for s in samples if s.file_type == 'rocket')
    rlrtl_count = sum(1 for s in samples if s.file_type == 'rlrtl_rocket')
    
    print(f"  Type distribution:")
    print(f"    - rocket: {rocket_count:,} ({rocket_count/len(samples)*100:.1f}%)")
    print(f"    - rlrtl:  {rlrtl_count:,} ({rlrtl_count/len(samples)*100:.1f}%)")
    print(f"  Load time: {load_time:.1f}s")
    
    # Extract instruction sequences
    instructions = ['\n'.join(s.instructions) for s in samples]
    
    return instructions, samples


def check_device(device_name):
    """Check and setup device"""
    print(f"\n[2/3] Checking device...")
    
    if device_name == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"  ✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"  ⚠️  CUDA not available, switching to CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print(f"  ✓ Using CPU")
    
    return device


def estimate_training_time(num_samples, batch_size, epochs, device):
    """Estimate training time"""
    print(f"\n[3/3] Estimating training time...")
    
    # Calculate training steps
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * epochs
    
    print(f"  Training samples: {num_samples:,}")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total training steps: {total_steps:,}")
    
    # Time estimation (MLM is faster than supervised BB similarity computation)
    if device.type == 'cuda':
        time_per_step = 0.05  # seconds (much faster than supervised training)
        total_minutes = (total_steps * time_per_step) / 60
        print(f"  Estimated training time: ~{total_minutes:.0f} minutes ({total_minutes/60:.1f} hours)")
    else:
        time_per_step = 0.15
        total_minutes = (total_steps * time_per_step) / 60
        print(f"  Estimated training time: ~{total_minutes:.0f} minutes ({total_minutes/60:.1f} hours)")
    
    print(f"\n  ℹ️  MLM pre-training is much faster than supervised training")
    print(f"  (no BB similarity computation required)")


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print_banner()
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print config
    print_config(args)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    try:
        # 1. Load data
        instructions, samples = load_dataset(args.data, args.num_samples)
        
        # 2. Check device
        device = check_device(args.device)
        
        # 3. Estimate training time
        estimate_training_time(len(instructions), args.batch_size, args.epochs, device)
        
        # 4. Start training
        print(f"\n" + "=" * 80)
        
        start_time = time.time()
        
        model, tokenizer = train_mlm_encoder(
            testcases=instructions,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            embedding_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            max_seq_len=args.max_seq_len,
            min_seq_len=args.min_seq_len,
            device=str(device),
            save_every=args.save_every,
            validation_split=args.validation_split,
            early_stopping_patience=args.early_stopping_patience,
            debug=args.debug
        )
        
        total_time = time.time() - start_time
        
        # Training complete
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        
        print(f"\nTotal training time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"Average per epoch: {total_time/args.epochs:.1f} seconds")
        
        print(f"\nModel saved to: {args.output_dir}/")
        print(f"  - best_model.pt: Best model based on validation loss")
        print(f"  - tokenizer.pkl: Tokenizer")
        print(f"  - checkpoint_epoch_*.pt: Training checkpoints")
        
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nNext steps:")
        print(f"  1. Fine-tune on supervised task:")
        print(f"     python train_full_model_supervised.py \\")
        print(f"       --pretrained {args.output_dir}/best_model.pt \\")
        print(f"       --num-samples 1000 --epochs 20")
        print(f"\n  2. Or evaluate embeddings directly:")
        print(f"     python evaluate_embedding_correlation.py \\")
        print(f"       --model {args.output_dir}/best_model.pt \\")
        print(f"       --tokenizer {args.output_dir}/tokenizer.pkl")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saved checkpoints can be used to resume training")
        
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

