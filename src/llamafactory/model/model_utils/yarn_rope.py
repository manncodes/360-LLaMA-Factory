# Copyright 2024 360 LLaMA Factory Team & YaRN Authors
# Licensed under the Apache License, Version 2.0

"""
YaRN (Yet another RoPE extensioN) implementation for LLaMA Factory.
Based on the paper: "YaRN: Efficient Context Window Extension of Large Language Models"
Reference: https://arxiv.org/abs/2309.00071
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from ...extras import logging

logger = logging.get_logger(__name__)


def find_correction_factor(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """
    Calculate the correction factor for YaRN scaling.
    This adjusts the temperature based on the wavelength of RoPE dimensions.
    """
    return (1 + math.log(num_rotations) / math.log(max_position_embeddings / 2048)) * 0.1 + 1.0


def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """
    Find the range of dimensions that need correction based on wavelength.
    """
    low = math.floor(find_correction_factor(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_factor(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_mask(min_val, max_val, dim):
    """
    Create a linear ramp mask for smooth interpolation.
    """
    if min_val == max_val:
        return torch.ones(dim)
    
    ramp_func = torch.linspace(0, 1, max_val - min_val + 1)
    mask = torch.ones(dim)
    mask[min_val:max_val + 1] = ramp_func
    return mask


def get_mscale(scale=1, mscale=1):
    """
    Calculate the magnitude scaling factor for YaRN.
    Formula: mscale = 0.1 * ln(scale) + 1
    """
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YaRNScaledRotaryEmbedding(nn.Module):
    """
    YaRN-scaled Rotary Position Embedding implementation.
    
    This implements the YaRN method which combines:
    1. NTK-by-parts interpolation
    2. Temperature scaling
    3. Magnitude scaling correction
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        scale: float = 1.0,
        original_max_position_embeddings: Optional[int] = None,
        beta_fast: float = 32,
        beta_slow: float = 1,
        mscale: Optional[float] = None,
        mscale_all_dim: float = 0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings or max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale or get_mscale(scale, mscale_all_dim)
        self.mscale_all_dim = mscale_all_dim
        
        # Pre-compute the inverse frequencies with YaRN scaling
        self._compute_yarn_scaling()
        
    def _compute_yarn_scaling(self):
        """
        Compute YaRN scaling factors for each dimension.
        """
        if self.scale <= 1:
            # No scaling needed
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        else:
            # YaRN scaling
            # Calculate wavelengths for each dimension
            freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            wavelengths = 2 * math.pi / freq
            
            # Determine which dimensions to scale
            ratio = self.original_max_position_embeddings / wavelengths
            
            # Create scaling mask based on wavelength ratio
            # Dimensions with long wavelengths (low frequency) use linear interpolation
            # Dimensions with short wavelengths (high frequency) use NTK scaling
            mask = ratio > self.beta_slow
            
            # Apply different scaling strategies
            inv_freq_linear = freq / self.scale  # Linear interpolation
            inv_freq_ntk = freq / (self.scale * ((self.base * self.scale) ** (torch.arange(0, self.dim, 2).float() / self.dim)))  # NTK scaling
            
            # Combine using mask
            inv_freq = torch.where(mask, inv_freq_linear, inv_freq_ntk)
            
            # Apply ramp for smooth transition
            ramp_mask = linear_ramp_mask(
                min_val=int(self.beta_slow * self.dim / (2 * self.original_max_position_embeddings)),
                max_val=int(self.beta_fast * self.dim / (2 * self.original_max_position_embeddings)),
                dim=self.dim // 2
            )
            inv_freq = inv_freq * ramp_mask.to(inv_freq.device)
        
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply YaRN-scaled rotary embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
            position_ids: Position indices
            
        Returns:
            cos, sin: Rotary embedding components with YaRN scaling
        """
        # Compute the rotary embeddings
        seq_len = position_ids.shape[-1]
        t = position_ids.unsqueeze(-1).float()
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t.squeeze(), self.inv_freq)
        
        # Create rotation matrix components
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.mscale
        sin = emb.sin() * self.mscale
        
        return cos, sin


def configure_yarn(config, model_args, is_trainable: bool = False):
    """
    Configure YaRN RoPE scaling for a model.
    
    Args:
        config: Model configuration
        model_args: Model arguments containing YaRN parameters
        is_trainable: Whether the model is being trained
    """
    if not hasattr(model_args, "rope_scaling") or model_args.rope_scaling != "yarn":
        return
    
    logger.info_rank0("Configuring YaRN RoPE scaling")
    
    # Default YaRN parameters
    yarn_config = {
        "type": "yarn",
        "factor": getattr(model_args, "yarn_factor", 2.0),
        "original_max_position_embeddings": getattr(
            model_args, "yarn_original_max_position_embeddings", 
            config.max_position_embeddings
        ),
        "beta_fast": getattr(model_args, "yarn_beta_fast", 32),
        "beta_slow": getattr(model_args, "yarn_beta_slow", 1),
        "mscale": getattr(model_args, "yarn_mscale", None),
        "mscale_all_dim": getattr(model_args, "yarn_mscale_all_dim", 0),
    }
    
    # Set the configuration
    setattr(config, "rope_scaling", yarn_config)
    
    # Update max position embeddings if needed
    if hasattr(model_args, "model_max_length"):
        setattr(config, "max_position_embeddings", model_args.model_max_length)
    
    logger.info_rank0(f"YaRN configured with factor={yarn_config['factor']}, "
                     f"beta_fast={yarn_config['beta_fast']}, "
                     f"beta_slow={yarn_config['beta_slow']}")
    
    if is_trainable:
        logger.warning_rank0("YaRN scaling may require fine-tuning for optimal performance.")