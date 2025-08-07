"""Model loading with RoPE configuration support."""

import torch
from typing import Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from .rope_config import RoPEConfig


class ModelLoader:
    """Handles model loading with RoPE configurations."""
    
    @staticmethod
    def load_model_and_tokenizer(
        model_name: str,
        rope_config: Optional[RoPEConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True
    ) -> Tuple[Any, Any]:
        """Load model and tokenizer with optional RoPE configuration."""
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model configuration
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Apply RoPE configuration if provided
        if rope_config:
            ModelLoader._apply_rope_config(config, rope_config)
            
        # Load model with configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=trust_remote_code
        )
        
        if device != "cuda":
            model = model.to(device)
            
        model.eval()
        
        return model, tokenizer
        
    @staticmethod
    def _apply_rope_config(config: Any, rope_config: RoPEConfig):
        """Apply RoPE configuration to model config."""
        
        if rope_config is None:
            return
            
        # Convert RoPE config to dict
        rope_dict = rope_config.to_dict()
        
        # Apply to model config
        if hasattr(config, "rope_scaling"):
            config.rope_scaling = rope_dict
        else:
            # Try to set it anyway for models that might support it
            setattr(config, "rope_scaling", rope_dict)
            
        # Update max position embeddings based on factor
        current_max = getattr(config, "max_position_embeddings", 4096)
        new_max = int(current_max * rope_config.factor)
        config.max_position_embeddings = new_max
            
        # Set RoPE theta if specified
        if rope_config.rope_theta:
            if hasattr(config, "rope_theta"):
                config.rope_theta = rope_config.rope_theta
            else:
                setattr(config, "rope_theta", rope_config.rope_theta)
                
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get information about a model's RoPE support."""
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        info = {
            "model_name": model_name,
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "rope_theta": getattr(config, "rope_theta", None),
            "has_rope_scaling": hasattr(config, "rope_scaling"),
            "current_rope_scaling": getattr(config, "rope_scaling", None)
        }
        
        # Detect model family
        model_type = getattr(config, "model_type", "").lower()
        
        if "llama" in model_type:
            if "llama-3" in model_name.lower():
                info["model_family"] = "llama3"
                info["default_rope_theta"] = 500000.0
                info["supported_rope_types"] = ["linear", "dynamic", "yarn", "llama3"]
            else:
                info["model_family"] = "llama2"
                info["default_rope_theta"] = 10000.0
                info["supported_rope_types"] = ["linear", "dynamic", "yarn"]
        elif "mistral" in model_type:
            info["model_family"] = "mistral"
            info["default_rope_theta"] = 10000.0
            info["supported_rope_types"] = ["linear", "dynamic"]
        elif "qwen" in model_type:
            info["model_family"] = "qwen"
            info["default_rope_theta"] = 10000.0
            info["supported_rope_types"] = ["linear", "dynamic", "yarn"]
        else:
            info["model_family"] = "unknown"
            info["supported_rope_types"] = ["linear", "dynamic"]
            
        return info