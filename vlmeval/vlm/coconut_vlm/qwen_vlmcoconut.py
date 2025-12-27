"""
VLMCoconut wrapper for Qwen2.5 VL models.

Unlike LLaMA Vision which uses cross-attention for vision-language fusion,
Qwen2.5 VL uses merged embeddings where visual tokens are embedded directly
into the input sequence.

IMPORTANT: We must compute vision-merged embeddings upfront and use them
consistently throughout all Coconut passes. Using text-only embeddings
after a vision-aware first pass causes embedding mismatches.
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
    We must replicate this to have consistent embeddings across all Coconut passes.
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
        # Get image token ID for vision embedding replacement
        if hasattr(self.base_causallm.config, 'image_token_id'):
            self.image_token_id = self.base_causallm.config.image_token_id
        else:
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    

    def _get_vision_merged_embeddings(self, input_ids, pixel_values, image_grid_thw):
        """
        Compute embeddings with vision tokens properly merged.
        
        This replicates what Qwen does internally so we have consistent
        embeddings across all Coconut passes.
        """
        # Get text embeddings
        inputs_embeds = self.embedding(input_ids)
        
        if pixel_values is None:
            return inputs_embeds
            
        # Get vision embeddings from the visual encoder
        # Qwen2.5 VL uses self.base_causallm.visual for vision encoding
        if hasattr(self.base_causallm, 'visual'):
            with torch.no_grad():
                # The visual module expects pixel_values and grid_thw
                image_embeds = self.base_causallm.visual(
                    pixel_values, 
                    grid_thw=image_grid_thw
                )
        else:
            # Fallback: return text-only embeddings
            print("[WARNING] No visual module found, using text-only embeddings")
            return inputs_embeds
        
        # Find image token positions and replace with vision embeddings
        # Only support batch_size=1 for now
        assert input_ids.shape[0] == 1, "Only batch_size=1 is supported"
        
        # Find positions of image pad tokens
        image_mask = (input_ids[0] == self.image_token_id)
        image_positions = image_mask.nonzero(as_tuple=True)[0]
        
        if len(image_positions) > 0:
            # Flatten vision embeddings to [num_tokens, hidden_dim]
            vision_embeds = image_embeds.view(-1, image_embeds.shape[-1])
            
            num_vision_tokens = min(len(image_positions), vision_embeds.shape[0])
            
            # Replace image pad tokens with vision embeddings
            inputs_embeds[0, image_positions[:num_vision_tokens]] = vision_embeds[:num_vision_tokens]
        
        return inputs_embeds

    def forward(self, **kwargs):
        """
        Forward pass with Coconut continuous thought mechanism.
        
        Uses vision-merged embeddings consistently across all passes.
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

        
        # Get vision-merged embeddings upfront - this is crucial!
        inputs_embeds = self._get_vision_merged_embeddings(input_ids, pixel_values, image_grid_thw)

        if max_n_latents == 0:
            # No latent tokens - just do regular forward pass
            forward_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "output_hidden_states": True,
                "return_dict": True,
            }
            outputs = self.base_causallm(**forward_kwargs)
            return Outputs(loss=None, inputs_embeds=inputs_embeds, logits=outputs.logits, past_key_values=None)

        # First compute range is up to the first latent token
        next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # First pass - use inputs_embeds (with vision already merged!)
                forward_kwargs = {
                    "inputs_embeds": inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    "attention_mask": attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
                    "output_hidden_states": True,
                    "use_cache": True,
                    "return_dict": True,
                    "past_key_values": DynamicCache(),
                    "cache_position": torch.arange(
                        next_compute_range[0], next_compute_range[1],
                        device=input_ids.device
                    ),
                }
                
                # Add position_ids
                if 'position_ids' in kwargs:
                    forward_kwargs['position_ids'] = kwargs['position_ids'][:, next_compute_range[0]:next_compute_range[1]]
                else:
                    forward_kwargs['position_ids'] = torch.arange(
                        next_compute_range[0],
                        next_compute_range[1],
                        dtype=torch.long,
                        device=input_ids.device
                    ).unsqueeze(0)

                outputs = self.base_causallm(**forward_kwargs)
                hidden_states_offset = next_compute_range[0]
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
        for _ in range(max_new_tokens - 1):
            step_out = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
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
