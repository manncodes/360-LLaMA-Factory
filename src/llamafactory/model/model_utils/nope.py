# Copyright 2024 360 LLaMA Factory Team & McGill NLP
# Licensed under the Apache License, Version 2.0

"""
NoPE (No Position Encoding) implementation for LLaMA Factory.
Based on the paper: "The Impact of Positional Encoding on Length Generalization in Transformers"
Reference: https://arxiv.org/abs/2305.19466
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from ...extras import logging

logger = logging.get_logger(__name__)


class NoPositionalEmbedding(nn.Module):
    """
    No Positional Embedding - relies solely on causal mask for position information.
    
    This implementation removes explicit position encodings and relies on the
    causal attention mask to provide implicit positional information.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize NoPE. Accepts any arguments for compatibility but doesn't use them.
        """
        super().__init__()
        # NoPE doesn't need any parameters
        
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return identity cos/sin for compatibility with RoPE interface.
        
        Args:
            x: Input tensor (not used)
            position_ids: Position indices (not used)
            
        Returns:
            cos, sin: Identity transformations (all ones and zeros)
        """
        # Return ones for cos and zeros for sin (identity transformation)
        device = x.device if hasattr(x, 'device') else torch.device('cpu')
        dtype = x.dtype if hasattr(x, 'dtype') else torch.float32
        
        # Get sequence length from position_ids or x
        if position_ids is not None:
            seq_len = position_ids.shape[-1]
            batch_size = position_ids.shape[0]
        else:
            batch_size, seq_len = x.shape[:2]
        
        # Create dummy cos/sin that won't affect the computation
        # cos = 1, sin = 0 means no rotation
        dim = x.shape[-1] if len(x.shape) > 2 else 128  # Default dim
        
        cos = torch.ones(batch_size, seq_len, dim, device=device, dtype=dtype)
        sin = torch.zeros(batch_size, seq_len, dim, device=device, dtype=dtype)
        
        return cos, sin


def apply_nope_to_attention(attention_module: nn.Module):
    """
    Modify an attention module to work without positional encodings.
    
    This function modifies the attention computation to rely solely on
    causal masking for position information.
    """
    
    class NoPEAttentionWrapper(nn.Module):
        """Wrapper to apply NoPE to existing attention modules."""
        
        def __init__(self, base_attention):
            super().__init__()
            self.base_attention = base_attention
            # Disable RoPE if it exists
            if hasattr(self.base_attention, 'rotary_emb'):
                self.base_attention.rotary_emb = NoPositionalEmbedding()
        
        def forward(self, *args, **kwargs):
            # Remove position_ids from kwargs if present
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'position_ids'}
            
            # Call the base attention without position information
            output = self.base_attention(*args, **kwargs_filtered)
            
            return output
    
    return NoPEAttentionWrapper(attention_module)


def configure_nope(config, model_args, is_trainable: bool = False):
    """
    Configure NoPE (No Position Encoding) for a model.
    
    Args:
        config: Model configuration
        model_args: Model arguments
        is_trainable: Whether the model is being trained
    """
    if not hasattr(model_args, "rope_scaling") or model_args.rope_scaling != "nope":
        return
    
    logger.info_rank0("Configuring NoPE (No Position Encoding)")
    
    # Disable RoPE and other position encodings
    nope_config = {
        "type": "nope",
        "disabled": True,  # Signal to disable position encoding
    }
    
    # Set the configuration
    setattr(config, "rope_scaling", nope_config)
    
    # Disable position embeddings if they exist
    if hasattr(config, "position_embedding_type"):
        setattr(config, "position_embedding_type", "none")
    
    # Ensure causal mask is enabled (critical for NoPE)
    if hasattr(config, "is_decoder"):
        setattr(config, "is_decoder", True)
    if hasattr(config, "use_causal_mask"):
        setattr(config, "use_causal_mask", True)
    
    # Update max position embeddings (can be arbitrary large without cost)
    if hasattr(model_args, "model_max_length"):
        setattr(config, "max_position_embeddings", model_args.model_max_length)
    
    logger.info_rank0("NoPE configured - relying on causal mask for position information")
    
    if not is_trainable:
        logger.warning_rank0(
            "NoPE requires the model to be trained without position encodings. "
            "Using NoPE on a model trained with position encodings may degrade performance."
        )
    
    logger.info_rank0(
        "NoPE advantages: No computational cost, theoretically infinite context, "
        "better length generalization for decoder-only models"
    )


class NoPEModel(nn.Module):
    """
    Helper class to convert a model to use NoPE.
    """
    
    @staticmethod
    def convert_model_to_nope(model: nn.Module) -> nn.Module:
        """
        Convert an existing model to use NoPE.
        
        Args:
            model: The model to convert
            
        Returns:
            Model with NoPE applied
        """
        # Find and replace all RoPE embeddings with NoPE
        for name, module in model.named_modules():
            if "rotary" in name.lower() or "rope" in name.lower():
                # Replace with NoPE
                parent_name = ".".join(name.split(".")[:-1])
                module_name = name.split(".")[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, module_name, NoPositionalEmbedding())
                
                logger.info(f"Replaced {name} with NoPE")
            
            # Also handle position embeddings
            elif "position" in name.lower() and "embed" in name.lower():
                # Disable position embeddings
                parent_name = ".".join(name.split(".")[:-1])
                module_name = name.split(".")[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    # Replace with identity or remove
                    setattr(parent, module_name, nn.Identity())
                
                logger.info(f"Disabled position embedding: {name}")
        
        return model