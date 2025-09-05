# Copyright 2024 LMSYS and the LlamaFactory team.
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
# This code is inspired by the LMSYS's FastChat library.
# https://github.com/lm-sys/FastChat/blob/v0.2.30/fastchat/train/train.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import TYPE_CHECKING

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def configure_rope(config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool) -> None:
    # Handle nested rope_scaling configuration
    if isinstance(model_args.rope_scaling, dict):
        # Nested format: rope_scaling: {type: yarn, factor: 4.0, alpha: 2.0, ...}
        nested_config = model_args.rope_scaling
        rope_type = nested_config.get('type')
        
        if rope_type is None:
            logger.warning_rank0("Nested rope_scaling configuration missing 'type' field.")
            return
        
        logger.info_rank0(f"Using nested RoPE scaling configuration: {nested_config}")
        
        # Extract parameters from nested config
        scaling_factor = nested_config.get('factor')
        yarn_alpha = nested_config.get('alpha')
        yarn_beta = nested_config.get('beta') 
        longrope_short_factor = nested_config.get('short_factor')
        longrope_long_factor = nested_config.get('long_factor')
        original_max_position = nested_config.get('original_max_position_embeddings')
        
    else:
        # Flat format: rope_scaling_type + rope_scaling_factor + yarn_alpha, etc.
        rope_type = model_args.rope_scaling_type or model_args.rope_scaling
        scaling_factor = model_args.rope_scaling_factor
        yarn_alpha = model_args.yarn_alpha
        yarn_beta = model_args.yarn_beta
        longrope_short_factor = model_args.longrope_short_factor
        longrope_long_factor = model_args.longrope_long_factor
        original_max_position = model_args.original_max_position
    
    if rope_type is None:
        return

    if not hasattr(config, "rope_scaling"):
        logger.warning_rank0("Current model does not support RoPE scaling.")
        return

    # Calculate scaling factor
    if model_args.model_max_length is not None:
        current_max_length = getattr(config, "max_position_embeddings", None)
        if current_max_length and model_args.model_max_length > current_max_length:
            logger.info_rank0(f"Enlarge max model length from {current_max_length} to {model_args.model_max_length}.")
            setattr(config, "max_position_embeddings", model_args.model_max_length)
            auto_scaling_factor = float(math.ceil(model_args.model_max_length / current_max_length))
        else:
            logger.warning_rank0("Input length is smaller than max length. Consider increase input length.")
            auto_scaling_factor = 1.0
    else:
        auto_scaling_factor = 2.0
    
    # Use explicit scaling factor if provided, otherwise use calculated
    final_scaling_factor = scaling_factor if scaling_factor is not None else auto_scaling_factor
    
    # Validate scaling factor
    if final_scaling_factor <= 0:
        logger.warning_rank0(f"ROPE scaling factor {final_scaling_factor} is non-positive. This may cause training issues.")
    elif final_scaling_factor > 100:
        logger.warning_rank0(f"ROPE scaling factor {final_scaling_factor} is very large. This may cause numerical instability.")
    
    # Add validation warning if there's a mismatch  
    if scaling_factor is not None and abs(final_scaling_factor - scaling_factor) > 0.001:
        logger.warning_rank0(
            f"ROPE scaling factor mismatch: requested {scaling_factor}, "
            f"using {final_scaling_factor}. Check your configuration."
        )
    
    # Build RoPE configuration based on type
    if rope_type in ["linear", "dynamic"]:
        # Simple RoPE scaling
        rope_config = {"type": rope_type, "factor": final_scaling_factor}
        
        if is_trainable and rope_type == "dynamic":
            logger.warning_rank0(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )
    
    elif rope_type == "yarn":
        # YaRN RoPE scaling for continual pretraining
        rope_config = {
            "type": "yarn",
            "factor": final_scaling_factor,
            "alpha": yarn_alpha,
            "beta": yarn_beta
        }
        
        if original_max_position is not None:
            rope_config["original_max_position_embeddings"] = original_max_position
        
        logger.info_rank0(f"Using YaRN scaling for continual pretraining: alpha={yarn_alpha}, beta={yarn_beta}")
    
    elif rope_type == "longrope":
        # LongRoPE scaling for extreme context extension
        rope_config = {
            "type": "longrope", 
            "factor": final_scaling_factor,
            "short_factor": longrope_short_factor,
            "long_factor": longrope_long_factor
        }
        
        if original_max_position is not None:
            rope_config["original_max_position_embeddings"] = original_max_position
        
        logger.info_rank0(f"Using LongRoPE scaling: short_factor={longrope_short_factor}, long_factor={longrope_long_factor}")
    
    else:
        raise ValueError(f"Unsupported RoPE scaling type: {rope_type}")
    
    # Apply RoPE configuration
    setattr(config, "rope_scaling", rope_config)
    logger.info_rank0(f"Applied {rope_type} RoPE scaling with factor {final_scaling_factor}")
    
    # Apply rope_theta if specified (check both flat and nested sources)
    rope_theta = None
    if isinstance(model_args.rope_scaling, dict):
        rope_theta = model_args.rope_scaling.get('rope_theta')
    if rope_theta is None:
        rope_theta = model_args.rope_theta
        
    if rope_theta is not None:
        if hasattr(config, "rope_theta"):
            setattr(config, "rope_theta", rope_theta)
            logger.info_rank0(f"Setting RoPE theta to {rope_theta}")
        else:
            logger.warning_rank0("Model does not support custom rope_theta parameter")
