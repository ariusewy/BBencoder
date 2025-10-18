#!/usr/bin/env python3
"""
Supervised Lightweight Sequence Encoder for RISC-V Testcase Similarity
Author: ywangmu from HKUST

This version uses actual BB similarity scores as training targets,
instead of binary labels (similar/dissimilar).

Key changes:
1. Compute BB similarity for all pairs (both positive and negative)
2. Use regression loss (MSE) instead of contrastive loss
3. Model predicts similarity score, not just binary classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import random
import pickle
import time
import re
from tqdm import tqdm
from advanced_tokenizer import AdvancedRISCVTokenizer
from evaluate_bb_similarity import (
    comp_ins, weighted_lcs, normalize_score,
    dep_consistency_bonus
)


class LightweightSequenceEncoder(nn.Module):
    """
    Lightweight LSTM-based encoder for instruction sequences
    Same architecture as before
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, 
                 hidden_dim: int = 128, output_dim: int = 32,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, token_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = token_ids.size()
        
        embedded = self.embedding(token_ids)
        
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        weighted_out = (lstm_out * attn_weights).sum(dim=1)
        
        output = self.output_proj(weighted_out)
        
        # L2 normalization
        output = F.normalize(output, p=2, dim=1)
        
        return output
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SupervisedPairDataset(Dataset):
    """
    Dataset with actual BB similarity scores as targets
    """
    
    def __init__(self, testcases: List[str], tokenizer: AdvancedRISCVTokenizer,
                 bb_tokenizer: AdvancedRISCVTokenizer,
                 max_length: int = 256, 
                 num_pairs_per_sample: int = 5,
                 bb_params: dict = None):
        """
        Args:
            testcases: List of RISC-V assembly testcases
            tokenizer: Tokenizer for embedding
            bb_tokenizer: Tokenizer for BB similarity computation
            max_length: Maximum sequence length
            num_pairs_per_sample: Number of pairs to generate per testcase
            bb_params: Parameters for BB similarity computation
        """
        self.testcases = testcases
        self.tokenizer = tokenizer
        self.bb_tokenizer = bb_tokenizer
        self.max_length = max_length
        self.num_pairs_per_sample = num_pairs_per_sample
        
        if bb_params is None:
            bb_params = {
                'B_exact': 2.0,
                'alpha_sub': 0.7,
                'alpha_unit': 0.45,
                'B_sameop': 1.0,
                'B_sameimm': 3.0,
                'normalize': 'maxlen',
                'enable_dep': False, 
                'beta_dep': 0.5,
                'gamma_mis': 0.05,
                'dep_window': 3
            }
        self.bb_params = bb_params
        
        # Pre-tokenize all testcases
        self.tokenized_testcases = []
        for tc in testcases:
            tokens = tokenizer.encode(tc)
            self.tokenized_testcases.append(tokens[:max_length])
        
        # Generate training pairs with BB similarity scores
        print("  Generating pairs and computing BB similarity scores...")
        self.pairs = self._generate_pairs_with_scores()
        print(f"  Generated {len(self.pairs)} pairs")
    
    def _compute_bb_similarity(self, tc1: str, tc2: str) -> float:
        """Compute BB similarity score between two testcases"""
        insts1 = tc1.strip().split('\n')
        insts2 = tc2.strip().split('\n')
        
        toks1 = [self.bb_tokenizer.tokenize_instruction(inst) for inst in insts1]
        toks2 = [self.bb_tokenizer.tokenize_instruction(inst) for inst in insts2]
        
        def score_fn(i: int, j: int) -> float:
            return comp_ins(
                toks1[i], toks2[j], self.bb_tokenizer,
                B_exact=self.bb_params['B_exact'],
                alpha_sub=self.bb_params['alpha_sub'],
                alpha_unit=self.bb_params['alpha_unit'],
                B_sameop=self.bb_params['B_sameop'],
                B_sameimm=self.bb_params['B_sameimm']
            )
        
        raw_score_seq, pairs = weighted_lcs(toks1, toks2, score_fn)
        
        dep_bonus = 0.0
        if self.bb_params['enable_dep'] and len(pairs) > 0:
            sideA = self.bb_tokenizer.build_dependency_sidecar(insts1)
            sideB = self.bb_tokenizer.build_dependency_sidecar(insts2)
            dep_bonus = dep_consistency_bonus(
                pairs, sideA, sideB,
                beta_dep_total=self.bb_params['beta_dep'],
                gamma_mis=self.bb_params['gamma_mis'],
                dep_window=max(1, self.bb_params['dep_window'])
            )
        
        raw_score = raw_score_seq + dep_bonus
        norm_score = normalize_score(raw_score, len(toks1), len(toks2), self.bb_params['normalize'])
        
        return norm_score
    
    def _augment_testcase(self, testcase: str) -> str:
        """Register renaming augmentation"""
        lines = testcase.strip().split('\n')
        
        used_regs = set()
        for line in lines:
            tokens = re.findall(r'x\d+|sp|ra|gp|tp', line.lower())
            used_regs.update(tokens)
        
        available_regs = [f'x{i}' for i in range(1, 32)]
        random.shuffle(available_regs)
        
        mapping = {}
        for i, reg in enumerate(used_regs):
            if reg.startswith('x'):
                if i < len(available_regs):
                    mapping[reg] = available_regs[i]
        
        augmented_lines = []
        for line in lines:
            augmented_line = line
            for old_reg, new_reg in mapping.items():
                augmented_line = re.sub(rf'\b{old_reg}\b', new_reg, augmented_line, flags=re.IGNORECASE)
            augmented_lines.append(augmented_line)
        
        return '\n'.join(augmented_lines)
    
    def _generate_pairs_with_scores(self) -> List[Tuple[List[int], List[int], float]]:
        """
        Generate training pairs with BB similarity scores
        
        Strategy 1: Complete Random Pairing with Coverage Guarantee
        - Each sample is paired with K random samples
        - Ensures every sample appears at least K times
        - No artificial augmentation (register renaming)
        - All pairs have computed BB similarity scores
        """
        pairs = []
        
        print(f"  Strategy: Random pairing (K={self.num_pairs_per_sample} pairs per sample)")
        print(f"  Total pairs to generate: {len(self.testcases) * self.num_pairs_per_sample:,}")
        
        # For each testcase, pair with K random testcases
        for i, tc1 in enumerate(tqdm(self.testcases, desc="  Generating random pairs")):
            # Generate K random pairs for this sample
            paired_indices = set()
            
            for _ in range(self.num_pairs_per_sample):
                # Randomly select another testcase (avoid self-pairing and duplicates)
                attempts = 0
                while attempts < 100:  # Prevent infinite loop
                    j = random.randint(0, len(self.testcases) - 1)
                    if j != i and j not in paired_indices:
                        paired_indices.add(j)
                        break
                    attempts += 1
                
                # If we couldn't find a unique pair, allow duplicates
                if attempts >= 100:
                    j = random.randint(0, len(self.testcases) - 1)
                    while j == i:
                        j = random.randint(0, len(self.testcases) - 1)
                
                tc2 = self.testcases[j]
                
                # Compute BB similarity
                bb_sim = self._compute_bb_similarity(tc1, tc2)
                
                # Use pre-tokenized sequences
                tokens1 = self.tokenized_testcases[i]
                tokens2 = self.tokenized_testcases[j]
                
                pairs.append((tokens1, tokens2, bb_sim))
            
            # Progress update every 100 samples
            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(self.testcases)} samples ({(i+1)/len(self.testcases)*100:.1f}%)")
        
        # Shuffle all pairs
        print(f"  Shuffling {len(pairs):,} pairs...")
        random.shuffle(pairs)
        
        # Print statistics about BB similarity distribution
        bb_scores = [score for _, _, score in pairs]
        print(f"\n  BB Similarity Distribution (before training):")
        print(f"    Mean:   {np.mean(bb_scores):.3f}")
        print(f"    Std:    {np.std(bb_scores):.3f}")
        print(f"    Min:    {np.min(bb_scores):.3f}")
        print(f"    Max:    {np.max(bb_scores):.3f}")
        print(f"    Median: {np.median(bb_scores):.3f}")
        
        # Distribution by ranges
        ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10)]
        print(f"\n  BB Similarity by Range:")
        for low, high in ranges:
            count = sum(1 for s in bb_scores if low <= s < high)
            pct = count / len(bb_scores) * 100
            print(f"    [{low:.1f}, {high:.1f}): {count:6d} pairs ({pct:5.1f}%)")
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        tokens1, tokens2, bb_sim = self.pairs[idx]
        return {
            'tokens1': tokens1,
            'tokens2': tokens2,
            'bb_similarity': bb_sim
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    tokens1_list = [item['tokens1'] for item in batch]
    tokens2_list = [item['tokens2'] for item in batch]
    bb_similarities = torch.tensor([item['bb_similarity'] for item in batch], dtype=torch.float32)
    
    # Pad sequences
    tokens1_padded = nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in tokens1_list],
        batch_first=True,
        padding_value=0
    )
    tokens2_padded = nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in tokens2_list],
        batch_first=True,
        padding_value=0
    )
    
    # Get lengths
    lengths1 = torch.tensor([len(t) for t in tokens1_list])
    lengths2 = torch.tensor([len(t) for t in tokens2_list])
    
    return {
        'tokens1': tokens1_padded,
        'tokens2': tokens2_padded,
        'lengths1': lengths1,
        'lengths2': lengths2,
        'bb_similarities': bb_similarities
    }


class SimilarityRegressionLoss(nn.Module):
    """
    Regression loss for similarity prediction
    
    Combines:
    1. MSE loss between predicted and actual BB similarity
    2. Cosine similarity alignment (encourage similar embeddings for similar BBs)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        """
        Args:
            alpha: Weight for MSE loss
            beta: Weight for cosine alignment loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, bb_similarities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb1, emb2: (batch, embed_dim) - normalized embeddings
            bb_similarities: (batch,) - actual BB similarity scores
        
        Returns:
            loss: scalar
        """
        # Compute cosine similarity (embeddings are normalized)
        cos_sim = (emb1 * emb2).sum(dim=1)  # Range: [-1, 1]
        
        # Normalize BB similarities to similar range for better training
        # Based on random pairing: range is [0, ~4], normalize to [0, 1]
        # Using 4.0 as denominator to better match actual distribution
        bb_sim_normalized = bb_similarities / 4.0  # Adjust based on your BB similarity scale
        bb_sim_normalized = torch.clamp(bb_sim_normalized, 0.0, 1.0)
        
        # We want cosine similarity to correlate with BB similarity
        # Map [0, 1] to [0, 1] range (already there)
        # Or map to [-1, 1] to match cosine range: 2 * bb_sim - 1
        # Let's use [0, 1] -> [0.2, 1.0] to avoid negative values
        target_cos_sim = 0.2 + 0.8 * bb_sim_normalized
        
        # MSE loss between cosine similarity and target
        mse_loss = F.mse_loss(cos_sim, target_cos_sim)
        
        # Additional loss: encourage high similarity for high BB scores
        # Use smooth L1 loss for robustness
        smooth_l1_loss = F.smooth_l1_loss(cos_sim, target_cos_sim)
        
        # Combined loss
        loss = self.alpha * mse_loss + self.beta * smooth_l1_loss
        
        return loss


def train_supervised_encoder(
    testcases: List[str],
    output_dir: str = 'models_supervised',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    output_dim: int = 32,
    max_length: int = 256,
    num_pairs_per_sample: int = 5,
    device: str = 'cuda',
    save_every: int = 10,
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    bb_params: dict = None
):
    """
    Train the supervised encoder with BB similarity scores
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tokenizers
    print("\n[1/5] Creating tokenizers...")
    embed_tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=False)
    bb_tokenizer = AdvancedRISCVTokenizer(enable_dependency_sidecar=bb_params.get('enable_dep', False) if bb_params else False)
    print(f"  Vocabulary size: {embed_tokenizer.vocab_size}")
    
    # Save tokenizer
    tokenizer_path = f'{output_dir}/tokenizer.pkl'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(embed_tokenizer, f)
    print(f"  Tokenizer saved to {tokenizer_path}")
    
    # Split dataset
    print("\n[2/5] Creating training and validation datasets...")
    random.seed(42)
    shuffled_testcases = testcases.copy()
    random.shuffle(shuffled_testcases)
    
    split_idx = int(len(shuffled_testcases) * (1 - validation_split))
    train_testcases = shuffled_testcases[:split_idx]
    val_testcases = shuffled_testcases[split_idx:]
    
    print(f"  Dataset split:")
    print(f"    Training: {len(train_testcases):,} testcases ({len(train_testcases)/len(testcases)*100:.1f}%)")
    print(f"    Validation: {len(val_testcases):,} testcases ({len(val_testcases)/len(testcases)*100:.1f}%)")
    
    # Create datasets (this will take time as it computes BB similarities)
    print(f"\n  Creating training dataset (computing BB similarities)...")
    train_dataset = SupervisedPairDataset(
        train_testcases, embed_tokenizer, bb_tokenizer,
        max_length=max_length, 
        num_pairs_per_sample=num_pairs_per_sample,
        bb_params=bb_params
    )
    
    print(f"\n  Creating validation dataset (computing BB similarities)...")
    val_dataset = SupervisedPairDataset(
        val_testcases, embed_tokenizer, bb_tokenizer,
        max_length=max_length,
        num_pairs_per_sample=num_pairs_per_sample,
        bb_params=bb_params
    )
    
    print(f"\n  Training pairs: {len(train_dataset):,}")
    print(f"  Validation pairs: {len(val_dataset):,}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid pickling issues
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create model
    print("\n[3/5] Creating model...")
    model = LightweightSequenceEncoder(
        vocab_size=embed_tokenizer.vocab_size,
        embed_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2,
        dropout=0.2
    )
    model.to(device)
    
    num_params = model.get_num_params()
    print(f"  Model parameters: {num_params:,} (~{num_params/1e6:.2f}M)")
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = SimilarityRegressionLoss(alpha=1.0, beta=0.5)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate/10
    )
    
    # Validation function
    def validate_model(model, val_dataloader, criterion, device):
        model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                tokens1 = batch['tokens1'].to(device)
                tokens2 = batch['tokens2'].to(device)
                lengths1 = batch['lengths1']
                lengths2 = batch['lengths2']
                bb_similarities = batch['bb_similarities'].to(device)
                
                emb1 = model(tokens1, lengths1)
                emb2 = model(tokens2, lengths2)
                
                loss = criterion(emb1, emb2, bb_similarities)
                
                val_loss += loss.item()
                num_batches += 1
        
        return val_loss / num_batches if num_batches > 0 else float('inf')
    
    # Training loop
    print("\n[4/5] Training with BB similarity supervision...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        for batch in train_dataloader:
            tokens1 = batch['tokens1'].to(device)
            tokens2 = batch['tokens2'].to(device)
            lengths1 = batch['lengths1']
            lengths2 = batch['lengths2']
            bb_similarities = batch['bb_similarities'].to(device)
            
            emb1 = model(tokens1, lengths1)
            emb2 = model(tokens2, lengths2)
            
            loss = criterion(emb1, emb2, bb_similarities)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        avg_train_loss = train_loss / num_batches
        val_loss = validate_model(model, val_dataloader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
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
            }, f'{output_dir}/checkpoint_epoch_{epoch+1}.pt')
    
    print("\n[5/5] Training completed!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Models saved to: {output_dir}/")
    
    return model, embed_tokenizer


if __name__ == '__main__':
    print("This is a library module. Use train_full_model_supervised.py to train.")

