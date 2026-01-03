"""
VLMCoconut wrapper for Qwen2.5 VL models.

Unlike LLaMA Vision which uses cross-attention for vision-language fusion,
Qwen2.5 VL uses merged embeddings where visual tokens are embedded directly
into the input sequence.

Key Architecture Difference:
- LLaMA Vision: Uses cross-attention states (can be reused across passes)
- Qwen2.5 VL: Vision tokens are INLINE in the sequence (preserved in KV cache)

Strategy:
1. First pass: Use input_ids + pixel_values + image_grid_thw (let Qwen handle vision merging)
2. Extract vision-merged embeddings from the first pass hidden states
3. Subsequent passes: Use inputs_embeds with updated latent token embeddings
   (Vision tokens are preserved in the KV cache from the first pass)
"""

import torch
import torch.nn as nn
from collections import namedtuple
from transformers import DynamicCache


Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "past_key_values"])


class QwenVLMCoconut(nn.Module):
    """
    Coconut wrapper for Qwen2.5 VL models.
    
    Key insight: Qwen merges vision embeddings into the sequence internally.
    We use input_ids + pixel_values in the first pass, then cache the 
    vision-merged embeddings for subsequent passes.
    """

    def __init__(
        self,
        base_causallm,
        processor,
        latent_token_id,
        eos_token_id,
    ):
        super(QwenVLMCoconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.processor = processor
        self.base_causallm.config.use_cache = True

        self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, **kwargs):
        """
        Forward pass with Coconut continuous thought mechanism.
        
        Strategy (mirroring vlmcoconut.py):
        1. First pass: Use input_ids + pixel_values (let Qwen handle vision merging internally)
        2. Extract vision-merged embeddings from hidden states
        3. Subsequent passes: Use inputs_embeds with updated latent token embeddings
        """
        input_ids = kwargs['input_ids']
        attention_mask = kwargs.get('attention_mask')
        pixel_values = kwargs.get('pixel_values')
        image_grid_thw = kwargs.get('image_grid_thw')
        
        logits = []

        # Find all latent token positions
        latent_indices = (input_ids == self.latent_token_id).nonzero()

        latent_lists = []
        for i in range(input_ids.shape[0]):
            lst = []
            for idx in latent_indices:
                if idx[0] == i:
                    lst.append(idx[1].item())
            latent_lists.append(lst)

        max_n_latents = max([len(l) for l in latent_lists])
        
        # Compute the range for the first forward pass
        # This should go from 0 to the first latent token (excluding latent tokens)
        if max_n_latents > 0:
            first_latent_pos = latent_indices[:, 1].min().item()
            next_compute_range = (0, first_latent_pos)
        else:
            # No latent tokens - process the full sequence
            next_compute_range = (0, input_ids.shape[1])

        # We'll store the vision-merged embeddings after first pass
        inputs_embeds = None

        kv_cache = None
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # First pass - use input_ids + pixel_values (like vlmcoconut.py)
                # This lets Qwen handle vision merging internally with proper RoPE positions
                forward_kwargs = {
                    "input_ids": input_ids[:, next_compute_range[0]:next_compute_range[1]],
                    "attention_mask": attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
                    "output_hidden_states": True,
                    "use_cache": True,
                    "return_dict": True,
                    "past_key_values": DynamicCache(),
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "cache_position": torch.arange(
                        next_compute_range[0], next_compute_range[1],
                        device=input_ids.device
                    ),
                }
                
                # Add position_ids
                if 'position_ids' in kwargs:
                    forward_kwargs['position_ids'] = kwargs['position_ids'][:, next_compute_range[0]:next_compute_range[1]]

                outputs = self.base_causallm(**forward_kwargs)
                hidden_states_offset = next_compute_range[0]
                
                # Extract vision-merged embeddings from hidden states for subsequent passes
                # hidden_states[0] contains the input embeddings with vision tokens merged
                first_pass_embeds = outputs.hidden_states[0]
                
                # Build full inputs_embeds by combining first pass embeds with remaining text embeddings
                # The remaining text embeddings (after next_compute_range[1]) need to be computed
                if next_compute_range[1] < input_ids.shape[1]:
                    remaining_text_embeds = self.embedding(input_ids[:, next_compute_range[1]:])
                    inputs_embeds = torch.cat([first_pass_embeds, remaining_text_embeds], dim=1)
                else:
                    inputs_embeds = first_pass_embeds
            else:
                # Subsequent passes - use inputs_embeds (consistent with first pass)
                legacy_kv_cache = kv_cache.to_legacy_cache()
                past_key_values = [
                    (
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :],
                    )
                    for k, v in legacy_kv_cache
                ]
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

                forward_kwargs = {
                    "inputs_embeds": inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    "attention_mask": attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
                    "position_ids": torch.arange(
                        next_compute_range[0],
                        next_compute_range[1],
                        dtype=torch.long,
                        device=input_ids.device
                    ).unsqueeze(0),
                    "past_key_values": past_key_values,
                    "output_hidden_states": True,
                    "use_cache": True,
                    "cache_position": torch.arange(
                        next_compute_range[0], next_compute_range[1],
                        device=input_ids.device
                    ),
                }

                outputs = self.base_causallm(**forward_kwargs)
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            # Update compute range for next iteration
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # Feedback continuous thoughts to input_embeds
            # Replace latent token embeddings with hidden states from previous position
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # Avoid in-place operations by creating tensor list
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # Replace latent tokens with continuous thoughts (hidden states)
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                source_idx = token_idx - 1 - hidden_states_offset
                vec = hidden_states[batch_idx, source_idx, :]
                tensor_list[batch_idx][token_idx] = vec

            # Reassemble inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # Final pass
        final_forward_kwargs = {
            "inputs_embeds": inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            "attention_mask": attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
            "position_ids": torch.arange(
                next_compute_range[0],
                next_compute_range[1],
                dtype=torch.long,
                device=input_ids.device
            ).unsqueeze(0),
            "output_hidden_states": True,
        }
        
        if kv_cache:
            legacy_kv_cache = kv_cache.to_legacy_cache()
            past_key_values = [
                (
                    k[:, :, :next_compute_range[0], :],
                    v[:, :, :next_compute_range[0], :],
                )
                for k, v in legacy_kv_cache
            ]
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            final_forward_kwargs['past_key_values'] = past_key_values

        outputs = self.base_causallm(**final_forward_kwargs)
        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1
        loss = None
        logits = torch.cat(logits, dim=-2)
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits, past_key_values=past_key_values)

    def eval(self):
        self.base_causallm.eval()

    def generate(self, **kwargs):
        """
        Generate text with Coconut continuous thought mechanism.
        
        Strategy:
        1. Run Coconut forward pass with input_ids + pixel_values to get latent-filled embeddings
        2. Use the full inputs_embeds (with vision merged) for autoregressive generation
        3. For subsequent generation steps, we use inputs_embeds to preserve vision context
        """
        self.gen_forward_cnt = 0

        input_ids = kwargs["input_ids"]
        assert input_ids.shape[0] == 1, "Only support batch_size == 1 now"
        device = input_ids.device

        # Unpack inputs
        pixel_values = kwargs.get("pixel_values", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        max_new_tokens = kwargs.get("max_new_tokens", 128)

        # Run Coconut forward to get latent-filled embeddings + logits for prefix
        B, S = input_ids.shape
        attention_mask = kwargs.get("attention_mask", torch.ones(B, S, device=device))
        position_ids = kwargs.get(
            "position_ids",
            torch.arange(S, device=device).unsqueeze(0).expand(B, -1),
        )

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        inputs_embeds = outputs.inputs_embeds
        logits = outputs.logits

        tokens = input_ids[0].detach().tolist()

        # First generated token (greedy)
        next_token = int(logits[0, -1].argmax(-1))
        if next_token == self.eos_token_id:
            return torch.tensor(tokens + [next_token], device=device).unsqueeze(0)

        tokens.append(next_token)
        next_embed = self.embedding(torch.tensor([[next_token]], device=device))
        inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

        # Extend attention mask
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(B, 1, dtype=attention_mask.dtype, device=device),
            ],
            dim=1,
        )

        # Autoregressive generation loop
        # We use inputs_embeds (with vision merged) for all subsequent steps
        # This ensures the model can still attend to the vision tokens via their embeddings
        # We now also pass position_ids to ensure correct M-RoPE computation
        for _ in range(max_new_tokens - 1):
            # Create position_ids for the current step
            # We need to pass positions for ALL tokens in inputs_embeds
            step_position_ids = torch.arange(
                0, inputs_embeds.shape[1], dtype=torch.long, device=device
            ).unsqueeze(0)
            
            step_out = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=step_position_ids,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )

            logits_step = step_out.logits
            next_token = int(logits_step[0, -1].argmax(-1))

            if next_token == self.eos_token_id:
                break

            tokens.append(next_token)
            next_embed = self.embedding(torch.tensor([[next_token]], device=device))
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(B, 1, dtype=attention_mask.dtype, device=device),
                ],
                dim=1,
            )

        return torch.tensor(tokens, device=device).unsqueeze(0)
