# Copyright 2024 360 LLaMA Factory Team & Microsoft Research
# Licensed under the Apache License, Version 2.0

"""
LongRope implementation for LLaMA Factory.
Based on the paper: "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens"
Reference: https://arxiv.org/abs/2402.13753
"""

import math
from typing import Optional, Dict, List, Tuple
import json

import torch
import torch.nn as nn
import numpy as np

from ...extras import logging

logger = logging.get_logger(__name__)


def identify_key_positions(seq_len: int, ratio: float = 0.25) -> List[int]:
    """
    Identify key token positions for LongRope rescaling.
    Uses the first and last 25% of positions as key positions by default.
    """
    key_positions = []
    n_key = int(seq_len * ratio)
    
    # First 25% positions
    key_positions.extend(range(n_key))
    # Last 25% positions  
    key_positions.extend(range(seq_len - n_key, seq_len))
    
    return key_positions


def search_optimal_rescale_factors(
    dim: int,
    max_position_embeddings: int,
    target_length: int,
    base: float = 10000,
    population_size: int = 20,
    num_iterations: int = 10,
) -> np.ndarray:
    """
    Evolution search to find optimal rescale factors for each RoPE dimension.
    This is a simplified version of the search algorithm from the paper.
    """
    # Initialize population with random rescale factors
    population = np.random.uniform(0.5, 2.0, (population_size, dim // 2))
    
    # Target scaling ratio
    scale_factor = target_length / max_position_embeddings
    
    for iteration in range(num_iterations):
        # Evaluate fitness (simplified - actual implementation would use perplexity)
        fitness = []
        for individual in population:
            # Fitness based on how well the rescaling preserves relative distances
            # Lower wavelength dimensions should have smaller rescale factors
            wavelengths = 2 * np.pi * base ** (np.arange(0, dim, 2) / dim)
            rescaled_wavelengths = wavelengths * individual
            
            # Penalty for deviating too much from target scale
            scale_penalty = np.abs(np.mean(individual) - scale_factor)
            
            # Penalty for non-monotonic rescaling
            monotonic_penalty = np.sum(np.maximum(0, np.diff(individual)))
            
            # Combined fitness (lower is better)
            fit = scale_penalty + 0.1 * monotonic_penalty
            fitness.append(fit)
        
        # Selection and reproduction
        fitness = np.array(fitness)
        sorted_indices = np.argsort(fitness)
        
        # Keep best individuals
        new_population = population[sorted_indices[:population_size // 2]]
        
        # Crossover and mutation
        for i in range(population_size // 2):
            parent1 = new_population[np.random.randint(len(new_population))]
            parent2 = new_population[np.random.randint(len(new_population))]
            
            # Crossover
            crossover_point = np.random.randint(1, dim // 2)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
            # Mutation
            mutation_mask = np.random.random(dim // 2) < 0.1
            child[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
            child = np.clip(child, 0.1, 10.0)
            
            new_population = np.vstack([new_population, child])
        
        population = new_population[:population_size]
    
    # Return best individual
    best_idx = np.argmin([fitness[i] for i in range(len(population))])
    return population[best_idx]


class LongRopeScaledRotaryEmbedding(nn.Module):
    """
    LongRope Rotary Position Embedding implementation.
    
    This implements non-uniform rescaling of RoPE dimensions to minimize
    information loss during interpolation.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        scale_factor: float = 1.0,
        rescale_factors: Optional[List[float]] = None,
        short_factor: Optional[List[float]] = None,
        long_factor: Optional[List[float]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale_factor = scale_factor
        
        # Use provided rescale factors or search for optimal ones
        if rescale_factors is not None:
            self.rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32)
        elif short_factor is not None and long_factor is not None:
            # Use short and long factors for different context ranges
            self.short_factor = torch.tensor(short_factor, dtype=torch.float32)
            self.long_factor = torch.tensor(long_factor, dtype=torch.float32)
            self.rescale_factors = self.short_factor  # Default to short
        else:
            # Search for optimal factors
            optimal_factors = search_optimal_rescale_factors(
                dim=dim,
                max_position_embeddings=max_position_embeddings,
                target_length=int(max_position_embeddings * scale_factor),
                base=base
            )
            self.rescale_factors = torch.tensor(optimal_factors, dtype=torch.float32)
        
        self._compute_longrope_scaling()
    
    def _compute_longrope_scaling(self):
        """
        Compute LongRope non-uniform scaling for RoPE dimensions.
        """
        # Base frequencies
        base_inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Apply non-uniform rescaling
        if hasattr(self, 'rescale_factors'):
            # Ensure rescale_factors has correct shape
            if len(self.rescale_factors) != len(base_inv_freq):
                # Interpolate or pad if necessary
                rescale_factors = torch.ones_like(base_inv_freq)
                min_len = min(len(self.rescale_factors), len(base_inv_freq))
                rescale_factors[:min_len] = self.rescale_factors[:min_len]
                self.rescale_factors = rescale_factors
            
            inv_freq = base_inv_freq / self.rescale_factors
        else:
            inv_freq = base_inv_freq / self.scale_factor
        
        self.register_buffer("inv_freq", inv_freq)
        
        # Register short and long factors if available
        if hasattr(self, 'short_factor'):
            self.register_buffer("short_inv_freq", base_inv_freq / self.short_factor)
        if hasattr(self, 'long_factor'):
            self.register_buffer("long_inv_freq", base_inv_freq / self.long_factor)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: torch.LongTensor,
        use_long_factor: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply LongRope scaled rotary embeddings.
        
        Args:
            x: Input tensor
            position_ids: Position indices
            use_long_factor: Whether to use long-range factors
            
        Returns:
            cos, sin: Rotary embedding components with LongRope scaling
        """
        seq_len = position_ids.shape[-1]
        
        # Select appropriate inverse frequencies
        if use_long_factor and hasattr(self, 'long_inv_freq'):
            inv_freq = self.long_inv_freq
        elif not use_long_factor and hasattr(self, 'short_inv_freq'):
            inv_freq = self.short_inv_freq
        else:
            inv_freq = self.inv_freq
        
        # Compute frequencies
        t = position_ids.unsqueeze(-1).float()
        freqs = torch.einsum("i,j->ij", t.squeeze(), inv_freq)
        
        # Create rotation matrix components
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin


def load_longrope_config(config_path: str) -> Dict:
    """
    Load LongRope configuration from JSON file.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load LongRope config from {config_path}: {e}")
        return {}


def configure_longrope(config, model_args, is_trainable: bool = False):
    """
    Configure LongRope scaling for a model.
    
    Args:
        config: Model configuration
        model_args: Model arguments containing LongRope parameters
        is_trainable: Whether the model is being trained
    """
    if not hasattr(model_args, "rope_scaling") or model_args.rope_scaling != "longrope":
        return
    
    logger.info_rank0("Configuring LongRope scaling")
    
    # Default LongRope parameters
    longrope_config = {
        "type": "longrope",
        "factor": getattr(model_args, "longrope_factor", 8.0),
    }
    
    # Load rescale factors if provided
    if hasattr(model_args, "longrope_config_path"):
        loaded_config = load_longrope_config(model_args.longrope_config_path)
        if "rescale_factors" in loaded_config:
            longrope_config["rescale_factors"] = loaded_config["rescale_factors"]
        if "short_factor" in loaded_config:
            longrope_config["short_factor"] = loaded_config["short_factor"]
        if "long_factor" in loaded_config:
            longrope_config["long_factor"] = loaded_config["long_factor"]
    
    # Set the configuration
    setattr(config, "rope_scaling", longrope_config)
    
    # Update max position embeddings
    if hasattr(model_args, "model_max_length"):
        setattr(config, "max_position_embeddings", model_args.model_max_length)
    
    logger.info_rank0(f"LongRope configured with factor={longrope_config['factor']}")
    
    if longrope_config['factor'] > 32:
        logger.warning_rank0(
            f"LongRope factor {longrope_config['factor']} is very large. "
            "The paper recommends progressive extension: 8x -> 128x -> 2048x"
        )
    
    if is_trainable and longrope_config['factor'] > 8:
        logger.info_rank0(
            "For factors > 8x, consider progressive extension: "
            "First extend to 8x without fine-tuning, then fine-tune at target length."
        )