#!/usr/bin/env python3
"""
Unified MLM Training Script - BBencoder
Author: ywangmu from HKUST

Train various encoder architectures using Masked Language Modeling (MLM).

Supported models:
- lstm: Lightweight LSTM-based encoder
- babygpt: Small transformer (4 layers, 256 dim)
- gpt2-small: Medium transformer (6 layers, 384 dim)
- gpt2: Full GPT-2 size (12 layers, 768 dim)

Usage:
python trainers/train_mlm.py --model babygpt --num-samples 1000 --epochs 50
"""

import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import sys
import time
import os
from datetime import datetime
from tqdm import tqdm

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'scripts'))
sys.path.insert(0, os.path.dirname(project_root))  # Add parent dir for BBencoder imports

from scripts.prepare_balanced_dataset import TrainingSample
from bbtokenizer_mlm_dataset import BBTokenizerMLMDataset
from models_arch.lstm_encoder import LightweightMLMEncoder
from models_arch.babygpt import BabyGPT, BabyGPTConfig
from models_arch.gpt2 import GPT2Encoder, GPT2Config


def collate_mlm_batch(batch):
    """Custom collate function for MLM batches"""
    token_inputs_list = [item['token_level'][0]['asm'] for item in batch]
    token_targets_list = [item['token_level'][1] for item in batch]
    instr_inputs_list = [item['instruction_level'][0]['asm'] for item in batch]
    instr_targets_list = [item['instruction_level'][1] for item in batch]
    
    token_inputs = torch.stack(token_inputs_list)
    token_targets = torch.stack(token_targets_list)
    instr_inputs = torch.stack(instr_inputs_list)
    instr_targets = torch.stack(instr_targets_list)
    
    return {
        'token_level': ({'asm': token_inputs}, token_targets),
        'instruction_level': ({'asm': instr_inputs}, instr_targets)
    }


def create_model(model_type: str, vocab_size: int, max_seq_len: int, args):
    """Create model based on specified type"""
    
    if model_type == 'lstm':
        model = LightweightMLMEncoder(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            dropout=args.dropout,
            pad_idx=0,
            task='mlm'  # Explicit task specification
        )
    
    elif model_type == 'babygpt':
        config = BabyGPTConfig(
            block_size=max_seq_len,
            vocab_size=vocab_size,
            n_layer=4,
            n_head=4,
            n_embd=256,
            dropout=args.dropout,
            bias=False
        )
        model = BabyGPT(config)
    
    elif model_type == 'gpt2-small':
        config = GPT2Config(
            block_size=max_seq_len,
            vocab_size=vocab_size,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=args.dropout,
            bias=True
        )
        model = GPT2Encoder(config)
    
    elif model_type == 'gpt2':
        config = GPT2Config(
            block_size=max_seq_len,
            vocab_size=vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=args.dropout,
            bias=True
        )
        model = GPT2Encoder(config)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_mlm_encoder(
    testcases: List[str],
    output_dir: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq_len: int,
    min_seq_len: int,
    device: str,
    save_every: int,
    validation_split: float,
    early_stopping_patience: int,
    debug: bool,
    args,
):
    """Train MLM encoder"""
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
    print(f"    Training: {len(train_testcases):,} testcases")
    print(f"    Validation: {len(val_testcases):,} testcases")
    
    # Create datasets
    train_dataset = BBTokenizerMLMDataset(
        testcases=train_testcases,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        token_mask_prob=0.15,
        instr_mask_prob=0.15,
        rng_seed=42
    )
    
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
    print(f"\n[2/4] Creating {model_type} model...")
    model = create_model(model_type, train_dataset.vocab_size['asm'], max_seq_len, args)
    model.to(device)
    
    num_params = model.get_num_params() if hasattr(model, 'get_num_params') else sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Create optimizer
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
                'model_type': model_type,
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
                'model_type': model_type,
            }, f'{output_dir}/checkpoint_epoch_{epoch+1}.pt')
    
    print("\n[4/4] Training completed!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Models saved to: {output_dir}/")
    
    return model, tokenizer


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train RISC-V encoder with MLM using various architectures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'babygpt', 'gpt2-small', 'gpt2'],
                       help='Model architecture to use')
    
    # Data related
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_path = os.path.join(project_root, 'dataset', 'final_training_data.pkl')
    parser.add_argument('--data', type=str, default=default_data_path,
                       help='Training data path')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to use')
    
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
    
    # Model config (for LSTM)
    parser.add_argument('--embed-dim', type=int, default=192,
                       help='Embedding dimension (LSTM only)')
    parser.add_argument('--hidden-dim', type=int, default=384,
                       help='LSTM hidden dimension (LSTM only)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Device and save
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Training device')
    default_output_dir = os.path.join(project_root, 'models_mlm')
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                       help='Model save directory')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug output')
    
    return parser.parse_args()


def print_banner():
    print("\n" + "=" * 80)
    print(" " * 15 + "BBencoder - Unified MLM Pre-training")
    print(" " * 10 + "Multi-Architecture Masked Language Modeling")
    print("=" * 80)


def print_config(args):
    print("\n" + "-" * 80)
    print("Training Configuration")
    print("-" * 80)
    
    print(f"\nModel: {args.model}")
    print(f"  lstm: Lightweight BiLSTM encoder")
    print(f"  babygpt: Small transformer (4L, 256D)")
    print(f"  gpt2-small: Medium transformer (6L, 384D)")
    print(f"  gpt2: Full GPT-2 (12L, 768D)")
    
    print(f"\nData Configuration:")
    print(f"  Dataset path: {args.data}")
    print(f"  Sample count: {args.num_samples if args.num_samples else 'All'}")
    print(f"  Max sequence length: {args.max_seq_len} tokens")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Dropout: {args.dropout}")
    
    print("-" * 80)


def load_dataset(data_path, num_samples=None):
    print(f"\n[1/3] Loading dataset...")
    print(f"  Path: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    start_time = time.time()
    with open(data_path, 'rb') as f:
        all_samples = pickle.load(f)
    load_time = time.time() - start_time
    
    if num_samples is not None:
        samples = all_samples[:num_samples]
        print(f"  ✓ Loaded: {len(all_samples):,} total samples")
        print(f"  → Using: {len(samples):,} samples")
    else:
        samples = all_samples
        print(f"  ✓ Loaded: {len(samples):,} samples")
    
    rocket_count = sum(1 for s in samples if s.file_type == 'rocket')
    rlrtl_count = sum(1 for s in samples if s.file_type == 'rlrtl_rocket')
    
    print(f"  Type distribution:")
    print(f"    - rocket: {rocket_count:,} ({rocket_count/len(samples)*100:.1f}%)")
    print(f"    - rlrtl:  {rlrtl_count:,} ({rlrtl_count/len(samples)*100:.1f}%)")
    print(f"  Load time: {load_time:.1f}s")
    
    instructions = ['\n'.join(s.instructions) for s in samples]
    return instructions, samples


def check_device(device_name):
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


def main():
    args = parse_args()
    
    print_banner()
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print_config(args)
    
    torch.manual_seed(args.seed)
    
    try:
        instructions, samples = load_dataset(args.data, args.num_samples)
        device = check_device(args.device)
        
        print(f"\n[3/3] Training {args.model} model...")
        print("=" * 80)
        
        start_time = time.time()
        
        model, tokenizer = train_mlm_encoder(
            testcases=instructions,
            output_dir=args.output_dir,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_seq_len=args.max_seq_len,
            min_seq_len=args.min_seq_len,
            device=str(device),
            save_every=args.save_every,
            validation_split=args.validation_split,
            early_stopping_patience=args.early_stopping_patience,
            debug=args.debug,
            args=args
        )
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        
        print(f"\nTotal training time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"Average per epoch: {total_time/args.epochs:.1f} seconds")
        
        print(f"\nModel saved to: {args.output_dir}/")
        print(f"  - best_model.pt: Best model based on validation loss")
        print(f"  - tokenizer.pkl: Tokenizer")
        
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

