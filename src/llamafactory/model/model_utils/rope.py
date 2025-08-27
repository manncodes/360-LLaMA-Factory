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
    # Determine which RoPE scaling to use (new rope_scaling_type or legacy rope_scaling)
    rope_type = model_args.rope_scaling_type or model_args.rope_scaling
    
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
    scaling_factor = model_args.rope_scaling_factor or auto_scaling_factor
    
    # Build RoPE configuration based on type
    if rope_type in ["linear", "dynamic"]:
        # Simple RoPE scaling
        rope_config = {"type": rope_type, "factor": scaling_factor}
        
        if is_trainable and rope_type == "dynamic":
            logger.warning_rank0(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )
    
    elif rope_type == "yarn":
        # YaRN RoPE scaling for continual pretraining
        rope_config = {
            "type": "yarn",
            "factor": scaling_factor,
            "alpha": model_args.yarn_alpha,
            "beta": model_args.yarn_beta
        }
        
        if model_args.original_max_position is not None:
            rope_config["original_max_position_embeddings"] = model_args.original_max_position
        
        logger.info_rank0(f"Using YaRN scaling for continual pretraining: alpha={model_args.yarn_alpha}, beta={model_args.yarn_beta}")
    
    elif rope_type == "longrope":
        # LongRoPE scaling for extreme context extension
        rope_config = {
            "type": "longrope", 
            "factor": scaling_factor,
            "short_factor": model_args.longrope_short_factor,
            "long_factor": model_args.longrope_long_factor
        }
        
        if model_args.original_max_position is not None:
            rope_config["original_max_position_embeddings"] = model_args.original_max_position
        
        logger.info_rank0(f"Using LongRoPE scaling: short_factor={model_args.longrope_short_factor}, long_factor={model_args.longrope_long_factor}")
    
    else:
        raise ValueError(f"Unsupported RoPE scaling type: {rope_type}")
    
    # Apply RoPE configuration
    setattr(config, "rope_scaling", rope_config)
    logger.info_rank0(f"Applied {rope_type} RoPE scaling with factor {scaling_factor}")
    
    # Apply rope_theta if specified
    if model_args.rope_theta is not None:
        if hasattr(config, "rope_theta"):
            setattr(config, "rope_theta", model_args.rope_theta)
            logger.info_rank0(f"Setting RoPE theta to {model_args.rope_theta}")
        else:
            logger.warning_rank0("Model does not support custom rope_theta parameter")
