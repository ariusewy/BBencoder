#!/usr/bin/env python3
"""
Evaluate Embedding-BB Similarity Correlation
Author: ywangmu from HKUST

This script evaluates whether the trained embedding model can effectively distinguish test cases.
Specifically, it checks if embedding distance correlates with BB similarity scores.

Usage:
    python evaluate_embedding_correlation.py \
  --model models_supervised/best_model.pt \
  --tokenizer models_supervised/tokenizer.pkl \
  --num-pairs 500
"""

import argparse
import pickle
import torch
import sys
import os
import random
import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, 'scripts'))

from prepare_balanced_dataset import TrainingSample
from lightweight_encoder_supervised import LightweightSequenceEncoder
from advanced_tokenizer import AdvancedRISCVTokenizer
from evaluate_bb_similarity import (
    comp_ins, weighted_lcs, normalize_score, get_opcode,
    dep_consistency_bonus
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate correlation between embedding distance and BB similarity',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and model paths
    default_data_path = os.path.join(script_dir, 'dataset', 'final_training_data.pkl')
    default_model_path = os.path.join(script_dir, 'models_supervised', 'best_model.pt')
    default_tokenizer_path = os.path.join(script_dir, 'models_supervised', 'tokenizer.pkl')
    
    parser.add_argument('--data', type=str, default=default_data_path,
                       help='Path to training data (pickle)')
    parser.add_argument('--model', type=str, default=default_model_path,
                       help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer', type=str, default=default_tokenizer_path,
                       help='Path to tokenizer pickle')
    
    # Evaluation settings
    parser.add_argument('--num-pairs', type=int, default=500,
                       help='Number of test case pairs to evaluate')
    parser.add_argument('--num-samples', type=int, default=2000,
                       help='Number of samples to use from dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for model inference')
    
    # BB similarity scoring parameters
    parser.add_argument('--B-exact', type=float, default=2.0,
                       help='Base score for exact mnemonic')
    parser.add_argument('--alpha-sub', type=float, default=0.7,
                       help='Alpha for same subclass')
    parser.add_argument('--alpha-unit', type=float, default=0.45,
                       help='Alpha for same unit (diff subclass)')
    parser.add_argument('--B-sameop', type=float, default=1.0,
                       help='Bonus for matching operand kind')
    parser.add_argument('--B-sameimm', type=float, default=3.0,
                       help='Extra bonus for equal concrete immediates')
    parser.add_argument('--normalize', choices=['maxlen', 'avglen', 'none'], default='maxlen',
                       help='Normalization method for BB similarity')
    parser.add_argument('--enable-dep', action='store_true', default=False,
                       help='Enable dependency-consistency scoring')
    parser.add_argument('--beta-dep', type=float, default=0.5,
                       help='Dependency bonus per matched instruction')
    parser.add_argument('--gamma-mis', type=float, default=0.1,
                       help='Penalty when only one side is a use')
    parser.add_argument('--dep-window', type=int, default=3,
                       help='Look-back window for dependency matching')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save plots to output directory')
    parser.add_argument('--show-examples', type=int, default=5,
                       help='Number of examples to show for different distance ranges')

    # Pairwise heatmap settings
    parser.add_argument('--pairwise-heatmap', action='store_true', default=False,
                       help='Enable computing NxN pairwise BB similarity and embedding distance heatmaps')
    parser.add_argument('--heatmap-N', type=int, default=64,
                       help='Number of samples N to randomly select for pairwise heatmaps')
    parser.add_argument('--heatmap-cmap', type=str, default='viridis',
                       help='Matplotlib colormap for heatmaps')
    parser.add_argument('--save-matrices', action='store_true', default=True,
                       help='Save computed pairwise matrices (NPZ)')
    
    return parser.parse_args()


def load_dataset(data_path: str, num_samples: int = None) -> List[TrainingSample]:
    """Load dataset from pickle file"""
    print(f"Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    with open(data_path, 'rb') as f:
        samples = pickle.load(f)
    
    if num_samples is not None and num_samples < len(samples):
        samples = samples[:num_samples]
    
    print(f"  Loaded {len(samples):,} samples")
    return samples


def load_model(model_path: str, tokenizer_path: str, device: torch.device):
    """Load trained model and tokenizer"""
    print(f"\nLoading model from: {model_path}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    # Create a new tokenizer instance (vocabulary is fixed for RISC-V)
    # This avoids pickle compatibility issues with older saved tokenizers
    print(f"  Creating new tokenizer instance (vocabulary is fixed)")
    tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=False)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model architecture parameters from checkpoint
    # Try to extract from the state dict
    state_dict = checkpoint['model_state_dict']
    
    # Infer architecture from saved weights
    vocab_size = state_dict['embedding.weight'].shape[0]
    embed_dim = state_dict['embedding.weight'].shape[1]
    output_dim = state_dict['output_proj.3.weight'].shape[0]
    hidden_dim = state_dict['output_proj.0.weight'].shape[0]
    
    print(f"  Model architecture:")
    print(f"    vocab_size: {vocab_size}")
    print(f"    embed_dim: {embed_dim}")
    print(f"    hidden_dim: {hidden_dim}")
    print(f"    output_dim: {output_dim}")
    
    # Create model
    model = LightweightSequenceEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        dropout=0.2
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully")
    
    return model, tokenizer


def get_embedding(model, tokenizer, testcase: str, device: torch.device, max_length: int = 1600) -> torch.Tensor:
    """Get embedding for a test case"""
    # Tokenize
    token_ids = tokenizer.encode(testcase)[:max_length]
    
    # Convert to tensor
    token_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
    length = torch.tensor([len(token_ids)])
    
    # Get embedding
    with torch.no_grad():
        embedding = model(token_tensor, length)
    
    return embedding.squeeze(0).cpu()


def compute_bb_similarity(sample1: TrainingSample, sample2: TrainingSample,
                         bb_tokenizer: AdvancedRISCVTokenizer, args) -> float:
    """Compute BB similarity score between two samples"""
    # Tokenize instructions
    toks1 = [bb_tokenizer.tokenize_instruction(inst) for inst in sample1.instructions]
    toks2 = [bb_tokenizer.tokenize_instruction(inst) for inst in sample2.instructions]
    
    # Define scoring function
    def score_fn(i: int, j: int) -> float:
        return comp_ins(
            toks1[i], toks2[j], bb_tokenizer,
            B_exact=args.B_exact,
            alpha_sub=args.alpha_sub,
            alpha_unit=args.alpha_unit,
            B_sameop=args.B_sameop,
            B_sameimm=args.B_sameimm
        )
    
    # Compute weighted LCS
    raw_score_seq, pairs = weighted_lcs(toks1, toks2, score_fn)
    
    # Add dependency bonus if enabled
    dep_bonus = 0.0
    if args.enable_dep and len(pairs) > 0:
        sideA = bb_tokenizer.build_dependency_sidecar(sample1.instructions)
        sideB = bb_tokenizer.build_dependency_sidecar(sample2.instructions)
        dep_bonus = dep_consistency_bonus(
            pairs, sideA, sideB,
            beta_dep_total=args.beta_dep,
            gamma_mis=args.gamma_mis,
            dep_window=max(1, args.dep_window)
        )
    
    # Total raw score
    raw_score = raw_score_seq + dep_bonus
    
    # Normalize
    norm_score = normalize_score(raw_score, len(toks1), len(toks2), args.normalize)
    
    return norm_score


def evaluate_pairs(model, embed_tokenizer, bb_tokenizer, samples: List[TrainingSample],
                   num_pairs: int, args, device: torch.device) -> Dict:
    """Evaluate multiple test case pairs"""
    print(f"\nEvaluating {num_pairs:,} test case pairs...")
    
    results = {
        'embedding_distances': [],
        'cosine_similarities': [],
        'bb_similarities': [],
        'pair_indices': [],
        'lengths': []
    }
    
    # Generate random pairs
    for _ in tqdm(range(num_pairs), desc="Computing similarities"):
        idx1, idx2 = random.sample(range(len(samples)), 2)
        sample1 = samples[idx1]
        sample2 = samples[idx2]
        
        # Compute embedding distance
        testcase1 = '\n'.join(sample1.instructions)
        testcase2 = '\n'.join(sample2.instructions)
        
        emb1 = get_embedding(model, embed_tokenizer, testcase1, device)
        emb2 = get_embedding(model, embed_tokenizer, testcase2, device)
        
        # Cosine similarity (embeddings are already normalized)
        cosine_sim = (emb1 * emb2).sum().item()
        
        # Cosine distance (1 - cosine similarity)
        cosine_dist = 1.0 - cosine_sim
        
        # Compute BB similarity
        bb_sim = compute_bb_similarity(sample1, sample2, bb_tokenizer, args)
        
        # Store results
        results['embedding_distances'].append(cosine_dist)
        results['cosine_similarities'].append(cosine_sim)
        results['bb_similarities'].append(bb_sim)
        results['pair_indices'].append((idx1, idx2))
        results['lengths'].append((len(sample1.instructions), len(sample2.instructions)))
    
    # Convert to numpy arrays
    results['embedding_distances'] = np.array(results['embedding_distances'])
    results['cosine_similarities'] = np.array(results['cosine_similarities'])
    results['bb_similarities'] = np.array(results['bb_similarities'])
    
    return results


def analyze_correlation(results: Dict, args):
    """Analyze and print correlation statistics"""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    emb_dist = results['embedding_distances']
    bb_sim = results['bb_similarities']
    cos_sim = results['cosine_similarities']
    
    # Statistics
    print(f"\nEmbedding Distance Statistics:")
    print(f"  Mean: {emb_dist.mean():.4f}")
    print(f"  Std:  {emb_dist.std():.4f}")
    print(f"  Min:  {emb_dist.min():.4f}")
    print(f"  Max:  {emb_dist.max():.4f}")
    
    print(f"\nCosine Similarity Statistics (embedding-based):")
    print(f"  Mean: {cos_sim.mean():.4f}")
    print(f"  Std:  {cos_sim.std():.4f}")
    print(f"  Min:  {cos_sim.min():.4f}")
    print(f"  Max:  {cos_sim.max():.4f}")
    
    print(f"\nBB Similarity Statistics:")
    print(f"  Mean: {bb_sim.mean():.4f}")
    print(f"  Std:  {bb_sim.std():.4f}")
    print(f"  Min:  {bb_sim.min():.4f}")
    print(f"  Max:  {bb_sim.max():.4f}")
    
    # Correlation: embedding distance vs BB similarity
    # Note: We expect NEGATIVE correlation (high distance -> low similarity)
    pearson_dist, p_value_dist = pearsonr(emb_dist, bb_sim)
    spearman_dist, p_value_sp_dist = spearmanr(emb_dist, bb_sim)
    
    # Correlation: cosine similarity vs BB similarity
    # Note: We expect POSITIVE correlation (high cosine sim -> high BB sim)
    pearson_sim, p_value_sim = pearsonr(cos_sim, bb_sim)
    spearman_sim, p_value_sp_sim = spearmanr(cos_sim, bb_sim)
    
    print(f"\nCorrelation: Embedding Distance vs BB Similarity")
    print(f"  Pearson correlation:  {pearson_dist:.4f} (p-value: {p_value_dist:.2e})")
    print(f"  Spearman correlation: {spearman_dist:.4f} (p-value: {p_value_sp_dist:.2e})")
    print(f"  → Expected: NEGATIVE correlation (high distance = low similarity)")
    
    print(f"\nCorrelation: Embedding Cosine Similarity vs BB Similarity")
    print(f"  Pearson correlation:  {pearson_sim:.4f} (p-value: {p_value_sim:.2e})")
    print(f"  Spearman correlation: {spearman_sim:.4f} (p-value: {p_value_sp_sim:.2e})")
    print(f"  → Expected: POSITIVE correlation (high similarity = high similarity)")
    
    # Interpretation
    print(f"\n" + "-" * 80)
    print("INTERPRETATION")
    print("-" * 80)
    
    if abs(pearson_sim) > 0.7:
        quality = "STRONG"
    elif abs(pearson_sim) > 0.5:
        quality = "MODERATE"
    elif abs(pearson_sim) > 0.3:
        quality = "WEAK"
    else:
        quality = "VERY WEAK"
    
    print(f"\nEmbedding quality: {quality}")
    print(f"  The embedding model shows {quality.lower()} correlation with BB similarity.")
    
    if pearson_sim > 0.7:
        print(f"  ✓ Embeddings can effectively distinguish test cases!")
        print(f"  ✓ High embedding similarity → High BB similarity")
        print(f"  ✓ Low embedding similarity → Low BB similarity")
    elif pearson_sim > 0.5:
        print(f"  ~ Embeddings show reasonable correlation with BB similarity.")
        print(f"  ~ May be useful for coarse filtering.")
    else:
        print(f"  ✗ Embeddings do NOT correlate well with BB similarity.")
        print(f"  ✗ Model may need more training or different architecture.")
    
    return {
        'pearson_dist': pearson_dist,
        'pearson_sim': pearson_sim,
        'spearman_dist': spearman_dist,
        'spearman_sim': spearman_sim,
        'p_value_sim': p_value_sim
    }


def plot_results(results: Dict, stats: Dict, args):
    """Generate and save visualization plots"""
    print(f"\nGenerating plots...")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    emb_dist = results['embedding_distances']
    bb_sim = results['bb_similarities']
    cos_sim = results['cosine_similarities']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Cosine Similarity vs BB Similarity (scatter)
    ax1 = axes[0, 0]
    ax1.scatter(cos_sim, bb_sim, alpha=0.5, s=10, c='blue')
    ax1.set_xlabel('Embedding Cosine Similarity', fontsize=11)
    ax1.set_ylabel('BB Similarity Score', fontsize=11)
    ax1.set_title(f'Embedding vs BB Similarity\n(Pearson r={stats["pearson_sim"]:.3f})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal reference line
    lims = [
        max(ax1.get_xlim()[0], ax1.get_ylim()[0]),
        min(ax1.get_xlim()[1], ax1.get_ylim()[1])
    ]
    ax1.plot(lims, lims, 'r--', alpha=0.5, linewidth=1, label='y=x')
    ax1.legend()
    
    # Plot 2: Embedding Distance vs BB Similarity (scatter)
    ax2 = axes[0, 1]
    ax2.scatter(emb_dist, bb_sim, alpha=0.5, s=10, c='green')
    ax2.set_xlabel('Embedding Distance (1 - cosine)', fontsize=11)
    ax2.set_ylabel('BB Similarity Score', fontsize=11)
    ax2.set_title(f'Embedding Distance vs BB Similarity\n(Pearson r={stats["pearson_dist"]:.3f})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of Cosine Similarities
    ax3 = axes[1, 0]
    ax3.hist(cos_sim, bins=50, alpha=0.7, color='blue', label='Embedding Cosine Sim')
    ax3.hist(bb_sim, bins=50, alpha=0.7, color='orange', label='BB Similarity')
    ax3.set_xlabel('Similarity Score', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution of Similarity Scores', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Binned analysis
    ax4 = axes[1, 1]
    
    # Bin by embedding cosine similarity
    num_bins = 10
    bins = np.linspace(cos_sim.min(), cos_sim.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []
    
    for i in range(num_bins):
        mask = (cos_sim >= bins[i]) & (cos_sim < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(bb_sim[mask].mean())
            bin_stds.append(bb_sim[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    ax4.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5, linewidth=2)
    ax4.set_xlabel('Embedding Cosine Similarity (binned)', fontsize=11)
    ax4.set_ylabel('Mean BB Similarity Score', fontsize=11)
    ax4.set_title('Binned Analysis: Embedding Sim → BB Sim', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if args.save_plots:
        plot_path = os.path.join(output_dir, 'embedding_correlation_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {plot_path}")
    
    plt.close()


def compute_pairwise_matrices(model, embed_tokenizer, bb_tokenizer,
                              samples: List[TrainingSample], N: int, args, device: torch.device) -> Dict:
    """Compute NxN pairwise matrices for BB similarity and embedding distance.

    Author: ywangmu from HKUST

    Returns a dict with:
      - indices: list of selected dataset indices (length N)
      - bb_similarity: (N, N) numpy array
      - cosine_similarity: (N, N) numpy array
      - embedding_distance: (N, N) numpy array (1 - cosine)
    """
    total = len(samples)
    if total == 0:
        raise ValueError('No samples available for pairwise heatmap computation')
    N = max(1, min(N, total))

    print(f"\nSelecting {N} samples for pairwise heatmaps (from {total})...")
    selected_indices = random.sample(range(total), N)
    selected_samples = [samples[i] for i in selected_indices]

    # Prepare testcases strings once
    testcases = ['\n'.join(s.instructions) for s in selected_samples]

    # Compute embeddings for all N
    print("Computing embeddings for selected samples...")
    embeddings = []
    for tc in tqdm(testcases, desc='Embeddings'):
        emb = get_embedding(model, embed_tokenizer, tc, device)
        embeddings.append(emb.numpy())
    emb_mat = np.stack(embeddings, axis=0)  # (N, D)

    # Row-normalize for cosine sim (robust even if model output not normalized)
    norms = np.linalg.norm(emb_mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_unit = emb_mat / norms

    # Cosine similarity and distance
    cosine_similarity = emb_unit @ emb_unit.T
    embedding_distance = 1.0 - cosine_similarity
    np.fill_diagonal(embedding_distance, 0.0)

    # Compute BB similarity matrix (upper triangle, mirror to lower)
    print("Computing BB similarity matrix (NxN)...")
    bb_similarity = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc='BB Sim rows'):
        s_i = selected_samples[i]
        for j in range(i, N):
            s_j = selected_samples[j]
            score = compute_bb_similarity(s_i, s_j, bb_tokenizer, args)
            bb_similarity[i, j] = score
            bb_similarity[j, i] = score

    return {
        'indices': selected_indices,
        'bb_similarity': bb_similarity,
        'cosine_similarity': cosine_similarity,
        'embedding_distance': embedding_distance,
    }


def plot_pairwise_heatmaps(pairwise: Dict, args):
    """Plot and save NxN heatmaps for BB similarity and embedding distance."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    bb_mat = pairwise['bb_similarity']
    dist_mat = pairwise['embedding_distance']
    cos_mat = pairwise['cosine_similarity']
    N = bb_mat.shape[0]

    # Figure 1: BB similarity heatmap
    plt.figure(figsize=(8, 7))
    im = plt.imshow(bb_mat, cmap=args.heatmap_cmap, aspect='auto')
    plt.title(f'BB Similarity Heatmap (N={N})')
    plt.xlabel('Sample index (0..N-1)')
    plt.ylabel('Sample index (0..N-1)')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if args.save_plots:
        path_bb = os.path.join(output_dir, f'heatmap_bb_similarity_N{N}.png')
        plt.savefig(path_bb, dpi=300, bbox_inches='tight')
        print(f"  BB similarity heatmap saved to: {path_bb}")
    plt.close()

    # Figure 2: Embedding distance heatmap
    plt.figure(figsize=(8, 7))
    vmax = float(np.nanmax(dist_mat)) if np.isfinite(dist_mat).all() else None
    im = plt.imshow(dist_mat, cmap=args.heatmap_cmap, aspect='auto', vmin=0.0, vmax=vmax)
    plt.title(f'Embedding Distance Heatmap (1 - cosine, N={N})')
    plt.xlabel('Sample index (0..N-1)')
    plt.ylabel('Sample index (0..N-1)')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if args.save_plots:
        path_dist = os.path.join(output_dir, f'heatmap_embedding_distance_N{N}.png')
        plt.savefig(path_dist, dpi=300, bbox_inches='tight')
        print(f"  Embedding distance heatmap saved to: {path_dist}")
    plt.close()

    # Optional: cosine similarity heatmap for reference
    plt.figure(figsize=(8, 7))
    im = plt.imshow(cos_mat, cmap=args.heatmap_cmap, aspect='auto', vmin=-1.0, vmax=1.0)
    plt.title(f'Embedding Cosine Similarity Heatmap (N={N})')
    plt.xlabel('Sample index (0..N-1)')
    plt.ylabel('Sample index (0..N-1)')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if args.save_plots:
        path_cos = os.path.join(output_dir, f'heatmap_embedding_cosine_N{N}.png')
        plt.savefig(path_cos, dpi=300, bbox_inches='tight')
        print(f"  Embedding cosine similarity heatmap saved to: {path_cos}")
    plt.close()


def save_pairwise_matrices(pairwise: Dict, args):
    """Save pairwise matrices and indices to NPZ."""
    if not args.save_matrices:
        return
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'pairwise_matrices.npz')
    np.savez_compressed(
        out_path,
        indices=np.array(pairwise['indices'], dtype=np.int64),
        bb_similarity=pairwise['bb_similarity'],
        embedding_distance=pairwise['embedding_distance'],
        cosine_similarity=pairwise['cosine_similarity'],
    )
    print(f"Pairwise matrices saved to: {out_path}")


def show_examples(results: Dict, samples: List[TrainingSample], args):
    """Show examples from different distance/similarity ranges"""
    print("\n" + "=" * 80)
    print("EXAMPLE TEST CASE PAIRS")
    print("=" * 80)
    
    emb_dist = results['embedding_distances']
    bb_sim = results['bb_similarities']
    cos_sim = results['cosine_similarities']
    pair_indices = results['pair_indices']
    
    # Define ranges
    ranges = [
        ('Very High Similarity (Cosine Sim > 0.9)', cos_sim > 0.9),
        ('High Similarity (0.7 < Cosine Sim < 0.9)', (cos_sim > 0.7) & (cos_sim <= 0.9)),
        ('Medium Similarity (0.4 < Cosine Sim < 0.7)', (cos_sim > 0.4) & (cos_sim <= 0.7)),
        ('Low Similarity (Cosine Sim < 0.4)', cos_sim <= 0.4),
    ]
    
    for range_name, mask in ranges:
        if mask.sum() == 0:
            continue
        
        print(f"\n{range_name} ({mask.sum()} pairs)")
        print("-" * 80)
        
        # Sample up to show_examples pairs
        indices = np.where(mask)[0]
        sample_size = min(args.show_examples, len(indices))
        sampled_indices = np.random.choice(indices, size=sample_size, replace=False)
        
        for i, idx in enumerate(sampled_indices[:args.show_examples]):
            idx1, idx2 = pair_indices[idx]
            sample1 = samples[idx1]
            sample2 = samples[idx2]
            
            print(f"\n  Example {i+1}:")
            print(f"    Embedding Cosine Sim: {cos_sim[idx]:.4f}")
            print(f"    BB Similarity:        {bb_sim[idx]:.4f}")
            print(f"    Sample 1 (idx={idx1}, len={len(sample1.instructions)}):")
            print(f"      Type: {sample1.file_type}")
            print(f"      First 3 instructions: {sample1.instructions[:3]}")
            print(f"    Sample 2 (idx={idx2}, len={len(sample2.instructions)}):")
            print(f"      Type: {sample2.file_type}")
            print(f"      First 3 instructions: {sample2.instructions[:3]}")


def save_results(results: Dict, stats: Dict, args):
    """Save evaluation results to file"""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    results_path = os.path.join(output_dir, 'evaluation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nRaw results saved to: {results_path}")
    
    # Save statistics to text file
    stats_path = os.path.join(output_dir, 'statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("EMBEDDING-BB SIMILARITY CORRELATION EVALUATION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Number of pairs evaluated: {len(results['embedding_distances'])}\n\n")
        
        f.write("CORRELATION STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Pearson correlation (Cosine Sim vs BB Sim):  {stats['pearson_sim']:.4f}\n")
        f.write(f"Spearman correlation (Cosine Sim vs BB Sim): {stats['spearman_sim']:.4f}\n")
        f.write(f"P-value: {stats['p_value_sim']:.2e}\n\n")
        
        f.write(f"Pearson correlation (Distance vs BB Sim):  {stats['pearson_dist']:.4f}\n")
        f.write(f"Spearman correlation (Distance vs BB Sim): {stats['spearman_dist']:.4f}\n\n")
        
        f.write("EMBEDDING DISTANCE STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean: {results['embedding_distances'].mean():.4f}\n")
        f.write(f"Std:  {results['embedding_distances'].std():.4f}\n")
        f.write(f"Min:  {results['embedding_distances'].min():.4f}\n")
        f.write(f"Max:  {results['embedding_distances'].max():.4f}\n\n")
        
        f.write("BB SIMILARITY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean: {results['bb_similarities'].mean():.4f}\n")
        f.write(f"Std:  {results['bb_similarities'].std():.4f}\n")
        f.write(f"Min:  {results['bb_similarities'].min():.4f}\n")
        f.write(f"Max:  {results['bb_similarities'].max():.4f}\n")
    
    print(f"Statistics saved to: {stats_path}")


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Print header
    print("\n" + "=" * 80)
    print("EMBEDDING-BB SIMILARITY CORRELATION EVALUATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Model: {args.model}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Number of pairs: {args.num_pairs:,}")
    print(f"  Number of samples: {args.num_samples:,}")
    print(f"  Device: {args.device}")
    print(f"  Random seed: {args.seed}")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataset
    samples = load_dataset(args.data, args.num_samples)
    
    # Load model and tokenizers
    model, embed_tokenizer = load_model(args.model, args.tokenizer, device)
    bb_tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=args.enable_dep)
    
    # Optional: compute NxN pairwise heatmaps
    if args.pairwise_heatmap:
        pairwise = compute_pairwise_matrices(
            model, embed_tokenizer, bb_tokenizer,
            samples, args.heatmap_N, args, device
        )
        plot_pairwise_heatmaps(pairwise, args)
        save_pairwise_matrices(pairwise, args)

    # Evaluate pairs
    results = evaluate_pairs(model, embed_tokenizer, bb_tokenizer, samples, args.num_pairs, args, device)
    
    # Analyze correlation
    stats = analyze_correlation(results, args)
    
    # Plot results
    plot_results(results, stats, args)
    
    # Show examples
    show_examples(results, samples, args)
    
    # Save results
    save_results(results, stats, args)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - evaluation_results.pkl: Raw results")
    print(f"  - statistics.txt: Summary statistics")
    print(f"  - embedding_correlation_analysis.png: Visualization plots")
    if args.pairwise_heatmap:
        print(f"  - heatmap_bb_similarity_N{args.heatmap_N}.png: NxN BB similarity heatmap")
        print(f"  - heatmap_embedding_distance_N{args.heatmap_N}.png: NxN embedding distance heatmap")
        print(f"  - heatmap_embedding_cosine_N{args.heatmap_N}.png: NxN embedding cosine similarity heatmap")
        if args.save_matrices:
            print(f"  - pairwise_matrices.npz: Pairwise matrices and selected indices")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

