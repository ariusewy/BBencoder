#!/usr/bin/env python3
"""
GPT-2 Style Transformer for RISC-V Assembly
Author: ywangmu from HKUST

Full-featured GPT-2 architecture adapted for assembly code understanding.
Based on nanoGPT implementation by Andrej Karpathy.

Key adaptations:
- Bidirectional attention for MLM (not causal)
- Dual MLM heads for token and instruction level
- Optimized for assembly instruction sequences
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttentionPooling(nn.Module):
    """
    Self-Attention Pooling Layer
    
    Learns to weigh the importance of each token dynamically, addressing the
    limitations of mean-pooling which discards token order and fails to weigh
    critical tokens like opcodes.
    
    Reference: "Self-Attention Encoding and Pooling for Speaker Recognition"
               (Okabe et al., 2018)
    """
    
    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        # Fully connected layer for attention score computation
        self.W_a = nn.Linear(input_dim, input_dim, bias=bias)
        # Learnable context vector
        self.u_a = nn.Parameter(torch.randn(input_dim))
        
    def forward(self, H: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            H: Token hidden states (batch, seq_len, hidden_dim)
            padding_mask: Boolean mask (batch, seq_len), True for valid tokens
        
        Returns:
            v_BBE: Basic Block Embedding (batch, hidden_dim)
        """
        # Compute attention scores: e_i = u_a^T * tanh(W_a * h_i^T + b_a)
        scores = torch.tanh(self.W_a(H))  # (B, L, D)
        scores = torch.matmul(scores, self.u_a)  # (B, L)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            scores = scores.masked_fill(~padding_mask, float('-inf'))
        
        # Normalize to attention weights: alpha = softmax(e)
        alpha = F.softmax(scores, dim=1)  # (B, L)
        
        # Weighted sum: v_BBE = sum(alpha_i * h_i)
        v_BBE = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # (B, D)
        
        return v_BBE


class LayerNorm(nn.Module):
    """LayerNorm with optional bias"""
    
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
    
    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=False  # Bidirectional for MLM
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPT2Config:
    """Configuration for GPT-2 encoder"""
    block_size: int = 1024  # Maximum sequence length
    vocab_size: int = 350  # RISC-V assembly vocabulary
    n_layer: int = 12  # Number of transformer layers
    n_head: int = 12  # Number of attention heads
    n_embd: int = 768  # Embedding dimension
    dropout: float = 0.1  # Dropout rate
    bias: bool = True  # Use bias in layers
    task: str = 'mlm'  # Task type: 'mlm' or 'supervised'
    output_dim: int = 96  # Output dimension for supervised task
    pad_idx: int = 0  # Padding token index


class GPT2Encoder(nn.Module):
    """
    GPT-2 style transformer encoder for RISC-V assembly
    
    Supports two tasks:
    - MLM: Masked Language Modeling (pre-training)
    - Supervised: Similarity learning with Basic Block Embeddings
    
    Shared backbone with task-specific heads.
    """
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.task = config.task
        
        # Shared transformer backbone
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Task-specific heads
        if config.task == 'mlm':
            # Dual MLM heads
            self.token_mlm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.instr_mlm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            
            # Weight tying
            self.transformer.wte.weight = self.token_mlm_head.weight
            
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        elif config.task == 'supervised':
            # Self-attention pooling for BB embedding
            self.attention_pooling = SelfAttentionPooling(config.n_embd, bias=config.bias)
            
            # Projection layer to output dimension
            self.output_proj = nn.Linear(config.n_embd, config.output_dim, bias=config.bias)
            
        else:
            raise ValueError(f"Unknown task: {config.task}")
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print(f"GPT-2 Encoder ({config.task}) parameters: {self.get_num_params()/1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Shared encoder backbone
        
        Args:
            token_ids: (batch, seq_len)
        
        Returns:
            hidden_states: (batch, seq_len, n_embd)
        """
        device = token_ids.device
        b, t = token_ids.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(token_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        return x
    
    def forward(self, inputs, targets=None):
        """
        Forward pass for either MLM or Supervised task
        
        Args:
            inputs: 
                - MLM: dict with 'token_mlm' and 'instr_mlm' entries
                - Supervised: token_ids (batch, seq_len)
            targets: 
                - MLM: None (targets embedded in inputs)
                - Supervised: similarity scores (batch,)
        
        Returns:
            loss: task-specific loss
            outputs: dict with detailed outputs
        """
        if self.task == 'mlm':
            return self._forward_mlm(inputs)
        elif self.task == 'supervised':
            return self._forward_supervised(inputs)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _forward_mlm(self, inputs):
        """Forward pass for dual MLM tasks"""
        token_inputs, token_targets = inputs['token_mlm']
        instr_inputs, instr_targets = inputs['instr_mlm']
        
        # Token-level MLM forward
        token_ids = token_inputs['asm']
        x = self._encode(token_ids)
        token_logits = self.token_mlm_head(x)
        token_loss = self.criterion(token_logits.view(-1, token_logits.size(-1)), token_targets.view(-1))
        
        # Instruction-level MLM forward
        instr_ids = instr_inputs['asm']
        x = self._encode(instr_ids)
        instr_logits = self.instr_mlm_head(x)
        instr_loss = self.criterion(instr_logits.view(-1, instr_logits.size(-1)), instr_targets.view(-1))
        
        # Combined loss
        loss = token_loss + instr_loss
        
        return loss, {
            'token_outputs': token_logits,
            'instr_outputs': instr_logits,
            'token_loss': token_loss,
            'instr_loss': instr_loss
        }
    
    def _forward_supervised(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for supervised similarity learning
        
        Args:
            token_ids: (batch, seq_len)
        
        Returns:
            embeddings: (batch, output_dim) - L2 normalized BB embeddings
        """
        # Encode sequence
        hidden_states = self._encode(token_ids)  # (B, L, D)
        
        # Create padding mask (True for non-padding tokens)
        padding_mask = (token_ids != self.config.pad_idx)  # (B, L)
        
        # Self-attention pooling
        pooled = self.attention_pooling(hidden_states, padding_mask)  # (B, D)
        
        # Project to output dimension
        embeddings = self.output_proj(pooled)  # (B, output_dim)
        
        # L2 normalization for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract sequence embeddings for downstream tasks
        
        Uses self-attention pooling if supervised task, otherwise mean pooling.
        """
        if self.task == 'supervised':
            return self._forward_supervised(token_ids)
        else:
            # For MLM task, use mean pooling
            x = self._encode(token_ids)
            embeddings = x.mean(dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with weight decay
        
        Separate weight decay for 2D parameters (weights) vs 1D parameters (biases, layernorms)
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")
        
        return optimizer

