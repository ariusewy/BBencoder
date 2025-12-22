#!/usr/bin/env python3
"""
Unified LSTM Encoder Architecture for RISC-V Assembly
Author: ywangmu from HKUST

A flexible LSTM encoder supporting multiple training tasks:
- MLM (Masked Language Modeling): Self-supervised pre-training
- Supervised: Similarity learning with BB similarity scores
- Classification: Downstream classification tasks

The task is specified during initialization, and the model automatically
configures the appropriate output heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union


class LightweightLSTMEncoder(nn.Module):
    """
    Unified lightweight LSTM-based encoder for instruction sequences
    
    Supports multiple training tasks through different output heads:
    - 'mlm': Dual MLM heads for token and instruction-level prediction
    - 'supervised': Single projection head for similarity learning
    - 'classification': Classification head for downstream tasks
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        output_dim: Output dimension (for supervised/classification tasks)
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        pad_idx: Padding token index
        task: Training task ('mlm', 'supervised', 'classification')
        num_classes: Number of classes (for classification task)
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 64, 
        hidden_dim: int = 128, 
        output_dim: int = 32,
        num_layers: int = 2, 
        dropout: float = 0.2,
        pad_idx: int = 0,
        task: str = 'mlm',
        num_classes: Optional[int] = None
    ):
        super().__init__()
        
        assert task in ['mlm', 'supervised', 'classification'], \
            f"task must be 'mlm', 'supervised', or 'classification', got {task}"
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task = task
        self.pad_idx = pad_idx
        
        # ========== Shared Components (LSTM Backbone) ==========
        
        # Embedding layer
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
        
        # Attention mechanism for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # ========== Task-Specific Output Heads ==========
        
        if task == 'mlm':
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
            
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        elif task == 'supervised':
            # Projection head for similarity learning
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        
        elif task == 'classification':
            # Classification head
            assert num_classes is not None, "num_classes must be specified for classification task"
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
            self.criterion = nn.CrossEntropyLoss()
        
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
    
    def _encode(self, token_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Shared encoding logic: token_ids -> LSTM hidden states
        
        Args:
            token_ids: (batch, seq_len)
            lengths: (batch,) optional sequence lengths for packing
        
        Returns:
            lstm_out: (batch, seq_len, hidden_dim*2)
        """
        embedded = self.embedding(token_ids)
        
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, _ = self.lstm(embedded)
        
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        return lstm_out
    
    def _pool_sequence(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence using attention mechanism
        
        Args:
            lstm_out: (batch, seq_len, hidden_dim*2)
        
        Returns:
            pooled: (batch, hidden_dim*2)
        """
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim*2)
        return pooled
    
    def forward(
        self, 
        inputs: Union[torch.Tensor, Dict[str, Any]], 
        targets: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ):
        """
        Forward pass - behavior depends on task
        
        Args:
            inputs: 
                - For MLM: dict with 'token_mlm' and 'instr_mlm' entries
                - For supervised/classification: token_ids tensor (batch, seq_len)
            targets: 
                - For MLM: not used (targets in inputs dict)
                - For supervised: not used
                - For classification: class labels (batch,)
            lengths: Optional sequence lengths for packing (supervised/classification only)
        
        Returns:
            Depends on task:
            - MLM: (loss, outputs_dict)
            - Supervised: embeddings (batch, output_dim)
            - Classification: logits (batch, num_classes) or (loss, logits) if targets provided
        """
        
        if self.task == 'mlm':
            return self._forward_mlm(inputs)
        
        elif self.task == 'supervised':
            return self._forward_supervised(inputs, lengths)
        
        elif self.task == 'classification':
            return self._forward_classification(inputs, targets, lengths)
    
    def _forward_mlm(self, inputs: Dict[str, Any]):
        """Forward pass for MLM task"""
        token_inputs, token_targets = inputs['token_mlm']
        instr_inputs, instr_targets = inputs['instr_mlm']
        
        # Token-level MLM
        token_ids = token_inputs['asm']
        token_lstm_out = self._encode(token_ids)
        token_logits = self.token_mlm_head(token_lstm_out)
        token_loss = self.criterion(
            token_logits.view(-1, self.vocab_size), 
            token_targets.view(-1)
        )
        
        # Instruction-level MLM
        instr_ids = instr_inputs['asm']
        instr_lstm_out = self._encode(instr_ids)
        instr_logits = self.instr_mlm_head(instr_lstm_out)
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
    
    def _forward_supervised(self, token_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """Forward pass for supervised similarity learning"""
        lstm_out = self._encode(token_ids, lengths)
        pooled = self._pool_sequence(lstm_out)
        output = self.output_proj(pooled)
        
        # L2 normalization for similarity learning
        output = F.normalize(output, p=2, dim=1)
        return output
    
    def _forward_classification(
        self, 
        token_ids: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None
    ):
        """Forward pass for classification task"""
        lstm_out = self._encode(token_ids, lengths)
        pooled = self._pool_sequence(lstm_out)
        logits = self.classifier(pooled)
        
        if targets is not None:
            loss = self.criterion(logits, targets)
            return loss, logits
        return logits
    
    def get_embeddings(self, token_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract normalized sequence embeddings for any task
        
        Args:
            token_ids: (batch, seq_len)
            lengths: (batch,) optional
        
        Returns:
            embeddings: (batch, hidden_dim*2) normalized embeddings
        """
        lstm_out = self._encode(token_ids, lengths)
        pooled = self._pool_sequence(lstm_out)
        return F.normalize(pooled, p=2, dim=1)
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def switch_task(self, new_task: str, output_dim: Optional[int] = None, num_classes: Optional[int] = None):
        """
        Switch to a different task by replacing the output head
        
        Useful for transfer learning: MLM pre-train -> supervised fine-tune
        
        Args:
            new_task: New task ('mlm', 'supervised', 'classification')
            output_dim: Output dimension for supervised task
            num_classes: Number of classes for classification task
        """
        assert new_task in ['mlm', 'supervised', 'classification']
        
        # Remove old task-specific heads
        if hasattr(self, 'token_mlm_head'):
            del self.token_mlm_head
            del self.instr_mlm_head
        if hasattr(self, 'output_proj'):
            del self.output_proj
        if hasattr(self, 'classifier'):
            del self.classifier
        
        # Add new task-specific heads
        self.task = new_task
        dropout = 0.2  # default
        
        if new_task == 'mlm':
            self.token_mlm_head = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.vocab_size)
            )
            self.instr_mlm_head = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.vocab_size)
            )
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        elif new_task == 'supervised':
            assert output_dim is not None
            self.output_dim = output_dim
            self.output_proj = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, output_dim)
            )
        
        elif new_task == 'classification':
            assert num_classes is not None
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, num_classes)
            )
            self.criterion = nn.CrossEntropyLoss()
        
        # Reinitialize new heads
        self._init_weights()
        
        print(f"Switched to task: {new_task}")


# Backward compatibility aliases
LightweightMLMEncoder = lambda *args, **kwargs: LightweightLSTMEncoder(*args, task='mlm', **kwargs)
LightweightSupervisedEncoder = lambda *args, **kwargs: LightweightLSTMEncoder(*args, task='supervised', **kwargs)

