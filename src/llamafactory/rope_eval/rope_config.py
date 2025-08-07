"""RoPE configuration management for needle-in-haystack evaluation."""

import math
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field


@dataclass
class RoPEConfig:
    """Configuration for RoPE scaling."""
    
    rope_type: str = "linear"  # linear, dynamic, yarn, longrope, llama3
    factor: float = 2.0
    original_max_position_embeddings: Optional[int] = None
    
    # YARN specific
    attention_factor: Optional[float] = 1.0
    beta_fast: Optional[int] = 32
    beta_slow: Optional[int] = 1
    
    # LongRoPE specific  
    short_factor: Optional[List[float]] = None
    long_factor: Optional[List[float]] = None
    
    # Llama3 specific
    low_freq_factor: Optional[float] = 1.0
    high_freq_factor: Optional[float] = 4.0
    
    # RoPE theta (base frequency)
    rope_theta: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to HuggingFace compatible dictionary."""
        config = {"rope_type": self.rope_type}
        
        if self.rope_type in ["linear", "dynamic"]:
            config["factor"] = self.factor
            
        elif self.rope_type == "yarn":
            config.update({
                "rope_type": "yarn",
                "factor": self.factor,
                "original_max_position_embeddings": self.original_max_position_embeddings,
                "attention_factor": self.attention_factor,
                "beta_fast": self.beta_fast,
                "beta_slow": self.beta_slow
            })
            
        elif self.rope_type == "longrope":
            config.update({
                "rope_type": "longrope", 
                "factor": self.factor,
                "short_factor": self.short_factor or [1.0] * 32,
                "long_factor": self.long_factor or [1.0] * 32,
                "original_max_position_embeddings": self.original_max_position_embeddings
            })
            
        elif self.rope_type == "llama3":
            config.update({
                "rope_type": "llama3",
                "factor": self.factor,
                "low_freq_factor": self.low_freq_factor,
                "high_freq_factor": self.high_freq_factor,
                "original_max_position_embeddings": self.original_max_position_embeddings
            })
            
        return config


class RoPEManager:
    """Manages RoPE configurations for different techniques."""
    
    SUPPORTED_TECHNIQUES = ["baseline", "linear", "dynamic", "yarn", "longrope", "llama3"]
    
    def __init__(self, base_context_length: int = 4096):
        self.base_context_length = base_context_length
        
    def create_config(self, technique: str, target_context: int) -> Optional[RoPEConfig]:
        """Create RoPE configuration for a specific technique and context length."""
        
        if technique == "baseline":
            return None  # No RoPE scaling
            
        # Calculate scaling factor
        factor = math.ceil(target_context / self.base_context_length)
        
        if technique == "linear":
            return RoPEConfig(rope_type="linear", factor=factor)
            
        elif technique == "dynamic":
            return RoPEConfig(rope_type="dynamic", factor=factor)
            
        elif technique == "yarn":
            return RoPEConfig(
                rope_type="yarn",
                factor=factor,
                original_max_position_embeddings=self.base_context_length,
                attention_factor=1.0,
                beta_fast=32,
                beta_slow=1
            )
            
        elif technique == "longrope":
            # Default factors - can be tuned per model
            return RoPEConfig(
                rope_type="longrope",
                factor=factor,
                original_max_position_embeddings=self.base_context_length,
                short_factor=[1.0] * 32,
                long_factor=[1.0] * 32
            )
            
        elif technique == "llama3":
            return RoPEConfig(
                rope_type="llama3",
                factor=factor,
                original_max_position_embeddings=self.base_context_length,
                low_freq_factor=1.0,
                high_freq_factor=4.0
            )
            
        else:
            raise ValueError(f"Unsupported RoPE technique: {technique}")
            
    def get_optimal_theta(self, context_length: int) -> float:
        """Get optimal RoPE theta based on context length."""
        if context_length <= 4096:
            return 10000.0
        elif context_length <= 8192:
            return 50000.0
        elif context_length <= 32768:
            return 500000.0
        else:
            return 1000000.0