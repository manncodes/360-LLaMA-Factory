# Copyright 2024 the LlamaFactory team.
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

import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from transformers import LlamaModel, LlamaConfig, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache

from ...extras import logging

logger = logging.get_logger(__name__)


class CustomSplitLLamaModel(LlamaModel):
    """
    Custom Split LLaMA Model implementing LLaMA Pro-like architecture.
    This model combines layers from two different LLaMA models (8B and 70B) 
    with an adapter layer in between.
    """
    config_class = LlamaConfig
    base_model_prefix = "model"
    
    def __init__(self, config):
        super().__init__(config)
        
        # Check if required paths are provided in config
        if not hasattr(config, "path8b") or not hasattr(config, "path70b"):
            raise ValueError("Config must include 'path8b' and 'path70b' for CustomSplitLLamaModel")
        
        self.mlp = getattr(config, "mlp", False)
        
        # Load the 8B and 70B models
        logger.info_rank0(f"Loading 8B model from: {config.path8b}")
        model8b = AutoModelForCausalLM.from_pretrained(config.path8b)
        
        logger.info_rank0(f"Loading 70B model from: {config.path70b}")
        model70b = AutoModelForCausalLM.from_pretrained(config.path70b) 
        
        h8, h70 = model8b.config.hidden_size, model70b.config.hidden_size
        assert h70 == 2 * h8, f"Expected h70 ({h70}) to be 2x h8 ({h8})"
        
        # Determine layer splits
        if hasattr(config, 'num_layers_8'):
            half8 = config.num_layers_8 
            half70 = config.num_layers_70
        else:
            half8 = 16
            half70 = 16

        # Initialize model components
        self.embed_tokens = model8b.model.embed_tokens
        self.layers_first = nn.ModuleList(model8b.model.layers[:half8])
        
        # Create adapter layers
        if self.mlp:
            self.adapter_linear_1 = nn.Linear(h8, h70, bias=False)
            self.adapter_linear_2 = nn.Linear(h70, h70, bias=False)
        else:
            self.adapter = nn.Linear(h8, h70, bias=False)
            
        self.layers_last = nn.ModuleList(model70b.model.layers[-half70:])
        self.norm = model70b.model.norm

        # Initialize weights and apply final processing
        self.gradient_checkpointing = False
        self.post_init()
        self.lm_head = model70b.lm_head

        # Clean up loaded models to save memory
        del model8b, model70b
        logger.info_rank0("Successfully initialized CustomSplitLLamaModel")
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        # Handle position_ids creation
        if attention_mask is not None and position_ids is None:
            device = input_ids.device
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)
        elif position_ids is None: 
            device = input_ids.device
            seq_length = input_ids.shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Set default values for optional parameters
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Validate input arguments
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # Handle gradient checkpointing warning
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # Handle cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # Create causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hs = inputs_embeds

        # Create position embeddings
        position_embeddings = self.rotary_emb(hs, position_ids)

        # Initialize output containers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_kv = []

        # Process through first layers (8B model layers)
        for layer in self.layers_first:
            if output_hidden_states:
                all_hidden_states += (hs,)
                
            if use_cache:
                hs, present = layer(
                    hs,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                next_kv.append(present)
            else:
                hs, *cache = layer(
                    hs,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

        # Apply adapter transformation
        if self.mlp:
            hs = torch.relu(self.adapter_linear_1(hs))
            hs = self.adapter_linear_2(hs)
        else:
            hs = self.adapter(hs)

        # Process through last layers (70B model layers)
        for layer in self.layers_last:
            if output_hidden_states:
                all_hidden_states += (hs,)
                
            if use_cache:
                hs, present = layer(
                    hs,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                next_kv.append(present)
            else:
                hs, *cache = layer(
                    hs,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

        # Apply final layer norm
        hs = self.norm(hs)

        # Add final hidden states
        if output_hidden_states:
            all_hidden_states += (hs,)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hs, 
            past_key_values=tuple(next_kv) if use_cache else None,
            hidden_states=all_hidden_states,
        )


def load_custom_split_llama_model(config, model_args):
    """
    Load a CustomSplitLLamaModel with the given configuration.
    
    Args:
        config: Model configuration
        model_args: Model arguments containing paths and other settings
        
    Returns:
        CustomSplitLLamaModel instance
    """
    # Add required paths to config if provided in model_args
    if hasattr(model_args, 'path8b') and hasattr(model_args, 'path70b'):
        config.path8b = model_args.path8b
        config.path70b = model_args.path70b
        
    if hasattr(model_args, 'num_layers_8') and hasattr(model_args, 'num_layers_70'):
        config.num_layers_8 = model_args.num_layers_8
        config.num_layers_70 = model_args.num_layers_70
        
    if hasattr(model_args, 'use_mlp_adapter'):
        config.mlp = model_args.use_mlp_adapter
    
    return CustomSplitLLamaModel(config)