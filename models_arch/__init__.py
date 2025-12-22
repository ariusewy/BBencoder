"""
Model Architecture Modules for BBencoder
Author: ywangmu from HKUST

This package contains various encoder architectures for RISC-V assembly encoding:
- lstm_encoder.py: Unified LSTM-based encoder (supports multiple tasks)
- babygpt.py: Small GPT-style transformer for assembly
- gpt2.py: Full GPT-2 style transformer

All models support the unified training interface and can be used for:
- MLM (Masked Language Modeling) pre-training
- Supervised similarity learning
- Classification tasks
"""

from .lstm_encoder import (
    LightweightLSTMEncoder, 
    LightweightMLMEncoder,  # Backward compatibility
    LightweightSupervisedEncoder  # Backward compatibility
)
from .babygpt import BabyGPT, BabyGPTConfig
from .gpt2 import GPT2Encoder, GPT2Config

__all__ = [
    'LightweightLSTMEncoder',
    'LightweightMLMEncoder',
    'LightweightSupervisedEncoder',
    'BabyGPT',
    'BabyGPTConfig',
    'GPT2Encoder',
    'GPT2Config',
]

