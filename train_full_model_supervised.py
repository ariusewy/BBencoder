#!/usr/bin/env python3
"""
Supervised Training Script - BBencoder
Author: ywangmu from HKUST

Train lightweight LSTM encoder using actual BB similarity scores as supervision.

Key improvements over contrastive learning:
1. Uses BB similarity scores instead of binary labels
2. Better alignment with evaluation metrics
3. Should improve correlation between embedding distance and BB similarity

Usage:
python3 train_full_model_supervised.py --num-samples 1000 --epochs 50 --num-pairs 5
"""

import argparse
import pickle
import torch
import sys
import time
import os
from datetime import datetime

# Add project paths dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

from prepare_balanced_dataset import TrainingSample
from lightweight_encoder_supervised import train_supervised_encoder


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train RISC-V testcase encoder with BB similarity supervision',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data related
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
    parser.add_argument('--max-length', type=int, default=1600,
                       help='Maximum token sequence length')
    parser.add_argument('--num-pairs', type=int, default=5,
                       help='Number of pairs to generate per testcase')
    
    # Model config
    parser.add_argument('--embed-dim', type=int, default=192,
                       help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=384,
                       help='LSTM hidden dimension')
    parser.add_argument('--output-dim', type=int, default=96,
                       help='Output embedding dimension')
    
    # BB similarity parameters
    parser.add_argument('--B-exact', type=float, default=2.0,
                       help='Base score for exact mnemonic match')
    parser.add_argument('--alpha-sub', type=float, default=0.7,
                       help='Alpha for same subclass')
    parser.add_argument('--alpha-unit', type=float, default=0.45,
                       help='Alpha for same execution unit')
    parser.add_argument('--B-sameop', type=float, default=1.0,
                       help='Bonus for matching operand kind')
    parser.add_argument('--B-sameimm', type=float, default=3.0,
                       help='Bonus for equal immediates')
    parser.add_argument('--normalize', choices=['maxlen', 'avglen', 'none'], default='maxlen',
                       help='BB similarity normalization method')
    parser.add_argument('--enable-dep', action='store_true', default=False,
                       help='Enable dependency-consistency scoring')
    parser.add_argument('--beta-dep', type=float, default=0.5,
                       help='Dependency bonus per matched instruction')
    parser.add_argument('--gamma-mis', type=float, default=0.1,
                       help='Penalty when only one side is a use')
    parser.add_argument('--dep-window', type=int, default=3,
                       help='Look-back window for dependency matching')
    
    # Device and save
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Training device')
    default_output_dir = os.path.join(script_dir, 'models_supervised')
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
    
    return parser.parse_args()


def print_banner():
    """Print startup banner"""
    print("\n" + "=" * 80)
    print(" " * 15 + "BBencoder - Supervised Training")
    print(" " * 10 + "RISC-V Testcase Similarity with BB Supervision")
    print("=" * 80)


def print_config(args):
    """Print training configuration"""
    print("\n" + "-" * 80)
    print("Training Configuration")
    print("-" * 80)
    
    print(f"\nData Configuration:")
    print(f"  Dataset path: {args.data}")
    print(f"  Sample count: {args.num_samples if args.num_samples else 'All'}")
    print(f"  Max sequence length: {args.max_length} tokens")
    print(f"  Pairs per sample: {args.num_pairs}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    
    print(f"\nModel Configuration:")
    print(f"  Embedding dim: {args.embed_dim}")
    print(f"  LSTM hidden dim: {args.hidden_dim}")
    print(f"  Output dim: {args.output_dim}")
    
    print(f"\nBB Similarity Parameters:")
    print(f"  B_exact: {args.B_exact}")
    print(f"  alpha_sub: {args.alpha_sub}")
    print(f"  alpha_unit: {args.alpha_unit}")
    print(f"  Normalize: {args.normalize}")
    print(f"  Enable dependency: {args.enable_dep}")
    if args.enable_dep:
        print(f"  Dependency window: {args.dep_window}")
    
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
    print(f"\n[1/4] Loading dataset...")
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
    print(f"\n[2/4] Checking device...")
    
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


def estimate_training_time(num_samples, batch_size, epochs, device, num_pairs):
    """Estimate training time"""
    print(f"\n[3/4] Estimating training time...")
    
    # Calculate training steps
    total_pairs = num_samples * num_pairs
    steps_per_epoch = total_pairs // batch_size
    total_steps = steps_per_epoch * epochs
    
    print(f"  Training pairs: ~{total_pairs:,}")
    print(f"  Steps per epoch: ~{steps_per_epoch:,}")
    print(f"  Total training steps: ~{total_steps:,}")
    
    # Time estimation
    if device.type == 'cuda':
        time_per_step = 0.3  # seconds (slightly slower due to BB similarity computation)
        total_minutes = (total_steps * time_per_step) / 60
        print(f"  Estimated training time: ~{total_minutes:.0f} minutes ({total_minutes/60:.1f} hours)")
    else:
        time_per_step = 0.35
        total_minutes = (total_steps * time_per_step) / 60
        print(f"  Estimated training time: ~{total_minutes:.0f} minutes ({total_minutes/60:.1f} hours)")
    
    # Dataset creation time warning
    print(f"\n  ⚠️  NOTE: Dataset creation will compute BB similarity for all pairs")
    print(f"  This may take additional time (~{num_samples * num_pairs / 100:.0f} minutes for {num_samples} samples)")


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
        estimate_training_time(len(instructions), args.batch_size, args.epochs, device, args.num_pairs)
        
        # 4. Start training
        print(f"\n[4/4] Starting supervised training...")
        print("=" * 80)
        
        # Prepare BB parameters
        bb_params = {
            'B_exact': args.B_exact,
            'alpha_sub': args.alpha_sub,
            'alpha_unit': args.alpha_unit,
            'B_sameop': args.B_sameop,
            'B_sameimm': args.B_sameimm,
            'normalize': args.normalize,
            'enable_dep': args.enable_dep,
            'beta_dep': args.beta_dep,
            'gamma_mis': args.gamma_mis,
            'dep_window': args.dep_window
        }
        
        start_time = time.time()
        
        model, tokenizer = train_supervised_encoder(
            testcases=instructions,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            embedding_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            max_length=args.max_length,
            num_pairs_per_sample=args.num_pairs,
            device=str(device),
            save_every=args.save_every,
            validation_split=args.validation_split,
            early_stopping_patience=args.early_stopping_patience,
            bb_params=bb_params
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
        print(f"  1. Evaluate correlation:")
        print(f"     python evaluate_embedding_correlation.py \\")
        print(f"       --model {args.output_dir}/best_model.pt \\")
        print(f"       --tokenizer {args.output_dir}/tokenizer.pkl \\")
        print(f"       --num-pairs 500")
        print(f"\n  2. Check positive pairs:")
        print(f"     python check_positive_pairs.py \\")
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

