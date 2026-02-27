"""
VLMCoconut wrapper for LLaVA / LLaVA-Next models.

Architecture difference from Llama-Vision:
- Llama-Vision uses CROSS-ATTENTION layers for vision-language fusion.
  Vision features are passed as cross_attention_states to every forward call.
- LLaVA uses INLINE vision tokens: CLIP features are projected via an MLP
  and placed at <image> token positions in the embedding sequence. The
  language model processes them via standard self-attention.

COCONUT strategy (analogous to vlmcoconut.py for Llama-Vision):
1. Extract vision features ONCE via vision_tower + multi_modal_projector
2. Build inputs_embeds by injecting vision features at <image> positions
3. Run multi-pass COCONUT loop through the model using inputs_embeds
4. Vision tokens in the KV cache from the first pass ensure visual info persists
5. Autoregressive generation recomputes the full prefix each step (like vlmcoconut.py)

Why this avoids the Qwen approach's visual feature degradation:
- We explicitly extract and inject vision features (not relying on hidden_states[0])
- Standard RoPE positioning (no rope_deltas complications)
- All passes go through LlavaNextForConditionalGeneration with input_ids=None,
  pixel_values=None, inputs_embeds=segment -- the model skips its internal
  vision routing and passes our embeddings straight to the language model.
"""

import torch
import torch.nn as nn
from collections import namedtuple
from transformers import DynamicCache


Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "past_key_values"])


def _extract_vision_features(base_model, pixel_values, image_sizes=None):
    """
    Extract vision features from LLaVA's vision tower + MLP projector.

    Analogous to _compute_cross_attention_states() in vlmcoconut.py:
    compute vision representations once, then reuse across all COCONUT passes.

    Returns:
        Flat tensor of projected image features [total_patches, hidden_dim]
        ready for masked_scatter into inputs_embeds.
    """
    inner = base_model.model if hasattr(base_model, 'model') else base_model

    with torch.no_grad():
        if hasattr(inner, 'get_image_features') and image_sizes is not None:
            # LlavaNextModel.get_image_features handles AnyRes packing
            output = inner.get_image_features(
                pixel_values, image_sizes, return_dict=True
            )
            if hasattr(output, 'pooler_output') and output.pooler_output is not None:
                image_features = torch.cat(output.pooler_output, dim=0)
            else:
                image_features = output
        elif hasattr(inner, 'get_image_features'):
            # LlavaModel.get_image_features returns tensor directly
            image_features = inner.get_image_features(pixel_values)
            if isinstance(image_features, (list, tuple)):
                image_features = torch.cat(image_features, dim=0)
        else:
            # Manual fallback: run vision tower + projector
            vision_tower = inner.vision_tower
            projector = inner.multi_modal_projector

            vout = vision_tower(
                pixel_values, output_hidden_states=True, return_dict=True
            )

            if hasattr(inner.config, 'vision_feature_layer'):
                layer = inner.config.vision_feature_layer
                if isinstance(layer, int):
                    features = vout.hidden_states[layer]
                else:
                    features = torch.cat(
                        [vout.hidden_states[l] for l in layer], dim=-1
                    )
            else:
                features = vout.last_hidden_state

            if (hasattr(inner.config, 'vision_feature_select_strategy')
                    and inner.config.vision_feature_select_strategy == 'default'):
                features = features[:, 1:]

            image_features = projector(features)
            if image_features.ndim == 3:
                image_features = image_features.reshape(
                    -1, image_features.shape[-1]
                )

    return image_features


class LLaVAVLMCoconut(nn.Module):
    """
    Coconut continuous-thought wrapper for LLaVA / LLaVA-Next models.

    Mirrors VLMCoconut (vlmcoconut.py) with adaptations for LLaVA's
    inline vision-token architecture:

        Llama-Vision                         LLaVA (this class)
        ─────────────                        ──────────────────
        cross_attention_states computed once  vision features extracted once
        passed to every forward call         injected into inputs_embeds once
        vision via cross-attn layers         vision via self-attn (inline tokens)
        KV cache + cross_attention_states    KV cache (vision tokens included)
    """

    def __init__(self, base_model, processor, latent_token_id, eos_token_id):
        super(LLaVAVLMCoconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_model = base_model
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.processor = processor

        self.embedding = base_model.get_input_embeddings()

        config = base_model.config
        if hasattr(config, 'image_token_id') and config.image_token_id is not None:
            self.image_token_id = config.image_token_id
        elif hasattr(config, 'image_token_index') and config.image_token_index is not None:
            self.image_token_id = config.image_token_index
        else:
            self.image_token_id = None

    def _build_multimodal_embeds(self, input_ids, pixel_values, image_sizes=None):
        """
        Build inputs_embeds with vision features injected at <image> positions.

        The processor has already expanded <image> into the correct number of
        placeholder tokens in input_ids.  We embed everything, then overwrite
        the placeholder positions with projected vision features.
        """
        inputs_embeds = self.embedding(input_ids)

        if pixel_values is not None and self.image_token_id is not None:
            image_features = _extract_vision_features(
                self.base_model, pixel_values, image_sizes
            )
            image_features = image_features.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

            image_mask = (input_ids == self.image_token_id)
            n_image_tokens = image_mask.sum().item()
            n_features = image_features.shape[0]

            if n_image_tokens != n_features:
                raise ValueError(
                    f"Image token count ({n_image_tokens}) != vision feature "
                    f"count ({n_features}). The processor may not have created "
                    f"the right number of placeholder tokens."
                )

            image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask_3d, image_features
            )

        return inputs_embeds

    def forward(self, **kwargs):
        """
        Forward pass with Coconut continuous thought mechanism.

        Flow (mirrors vlmcoconut.py):
        1. Build multimodal inputs_embeds (vision features injected at <image>)
        2. Find latent token positions in input_ids
        3. Multi-pass loop: process segments with KV cache, update latent embeds
        4. Final pass: process remaining tokens after all latent passes
        """
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        image_sizes = kwargs.get('image_sizes')
        logits = []

        # ── find latent token positions ──────────────────────────────────
        latent_indices = (input_ids == self.latent_token_id).nonzero()

        latent_lists = []
        for i in range(input_ids.shape[0]):
            lst = []
            for idx in latent_indices:
                if idx[0] == i:
                    lst.append(idx[1].item())
            latent_lists.append(lst)

        max_n_latents = max([len(l) for l in latent_lists])

        # ── build multimodal embeddings (vision injected) ────────────────
        inputs_embeds = self._build_multimodal_embeds(
            input_ids, pixel_values, image_sizes
        )

        # ── no latent tokens → straight forward pass ────────────────────
        if max_n_latents == 0:
            forward_kwargs = {
                "input_ids": None,
                "inputs_embeds": inputs_embeds,
                "attention_mask": kwargs.get('attention_mask'),
                "output_hidden_states": True,
                "return_dict": True,
            }
            if 'position_ids' in kwargs:
                forward_kwargs['position_ids'] = kwargs['position_ids']

            outputs = self.base_model(**forward_kwargs)
            return Outputs(
                loss=None, inputs_embeds=inputs_embeds,
                logits=outputs.logits, past_key_values=None,
            )

        # ── COCONUT multi-pass loop ──────────────────────────────────────
        next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # First pass: process prefix up to first latent token.
                # This segment includes ALL vision tokens, so their K/V
                # entries are written into the cache for subsequent passes.
                forward_kwargs = {
                    "input_ids": None,
                    "inputs_embeds": inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :
                    ],
                    "attention_mask": kwargs['attention_mask'][
                        :, :next_compute_range[1]
                    ],
                    "position_ids": torch.arange(
                        next_compute_range[0], next_compute_range[1],
                        dtype=torch.long, device=input_ids.device,
                    ).unsqueeze(0),
                    "output_hidden_states": True,
                    "use_cache": True,
                    "return_dict": True,
                    "past_key_values": DynamicCache(),
                    "cache_position": torch.arange(
                        next_compute_range[0], next_compute_range[1],
                        device=input_ids.device,
                    ),
                }
                outputs = self.base_model(**forward_kwargs)
                hidden_states_offset = next_compute_range[0]

            else:
                # Subsequent passes: process latent-token segment with
                # KV cache.  Vision token K/V entries from the first pass
                # are preserved in the cache, so latent tokens can attend
                # to the full visual context.
                legacy_kv_cache = kv_cache.to_legacy_cache()
                past_key_values = [
                    (
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :],
                    )
                    for k, v in legacy_kv_cache
                ]
                past_key_values = DynamicCache.from_legacy_cache(
                    past_key_values
                )

                forward_kwargs = {
                    "input_ids": None,
                    "inputs_embeds": inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :
                    ],
                    "attention_mask": kwargs['attention_mask'][
                        :, :next_compute_range[1]
                    ],
                    "position_ids": torch.arange(
                        next_compute_range[0], next_compute_range[1],
                        dtype=torch.long, device=input_ids.device,
                    ).unsqueeze(0),
                    "past_key_values": past_key_values,
                    "output_hidden_states": True,
                    "use_cache": True,
                    "return_dict": True,
                    "cache_position": torch.arange(
                        next_compute_range[0], next_compute_range[1],
                        device=input_ids.device,
                    ),
                }
                outputs = self.base_model(**forward_kwargs)
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            # advance the window
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

            # ── feedback continuous thoughts to inputs_embeds ────────────
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # avoid in-place operations on the computation graph
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                vec = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
                tensor_list[batch_idx][token_idx] = vec

            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # ── final pass: remaining tokens after all latent passes ─────────
        final_forward_kwargs = {
            "input_ids": None,
            "inputs_embeds": inputs_embeds[
                :, next_compute_range[0]:next_compute_range[1], :
            ],
            "attention_mask": kwargs['attention_mask'][
                :, :next_compute_range[1]
            ],
            "position_ids": torch.arange(
                next_compute_range[0], next_compute_range[1],
                dtype=torch.long, device=input_ids.device,
            ).unsqueeze(0),
            "output_hidden_states": True,
            "return_dict": True,
            "cache_position": torch.arange(
                next_compute_range[0], next_compute_range[1],
                device=input_ids.device,
            ),
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

        outputs = self.base_model(**final_forward_kwargs)
        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1
        logits = torch.cat(logits, dim=-2)

        return Outputs(
            loss=None,
            inputs_embeds=inputs_embeds,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def eval(self):
        self.base_model.eval()

    def generate(self, **kwargs):
        """
        Generate text with Coconut continuous thought mechanism.

        Strategy (mirrors vlmcoconut.py):
        1. Run Coconut forward → latent-filled inputs_embeds + prefix logits
        2. Greedy-decode first token from prefix logits
        3. Autoregressive loop: full-prefix recomputation each step
           (vision features are embedded in inputs_embeds, always available)
        """
        self.gen_forward_cnt = 0

        input_ids = kwargs["input_ids"]
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        device = input_ids.device

        pixel_values = kwargs.get("pixel_values", None)
        image_sizes = kwargs.get("image_sizes", None)
        max_new_tokens = kwargs.get("max_new_tokens", 128)

        # 1) Run Coconut forward to get latent-filled embeddings + logits
        B, S = input_ids.shape
        attention_mask = kwargs.get(
            "attention_mask", torch.ones(B, S, device=device)
        )

        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )

        inputs_embeds = outputs.inputs_embeds
        logits = outputs.logits
        tokens = input_ids[0].detach().tolist()

        # 2) First generated token (greedy)
        next_token = int(logits[0, -1].argmax(-1))
        if next_token == self.eos_token_id:
            return torch.tensor(
                tokens + [next_token], device=device
            ).unsqueeze(0)

        tokens.append(next_token)
        next_embed = self.embedding(
            torch.tensor([[next_token]], device=device)
        )
        inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(B, 1, dtype=attention_mask.dtype, device=device),
            ],
            dim=1,
        )

        # 3) Autoregressive loop: full-prefix forward each step
        #    Vision features live inside inputs_embeds, so they are always
        #    available to self-attention without any extra mechanism.
        for _ in range(max_new_tokens - 1):
            step_out = self.base_model(
                input_ids=None,
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
            next_embed = self.embedding(
                torch.tensor([[next_token]], device=device)
            )
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        B, 1, dtype=attention_mask.dtype, device=device
                    ),
                ],
                dim=1,
            )

        return torch.tensor(tokens, device=device).unsqueeze(0)
