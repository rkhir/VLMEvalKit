"""
VLMCoconut wrapper for LLaVA / LLaVA-Next models.

Architecture difference from Llama-Vision:
- Llama-Vision uses CROSS-ATTENTION layers for vision-language fusion.
  Vision features are passed as cross_attention_states to every forward call.
- LLaVA uses INLINE vision tokens: CLIP features are projected via an MLP
  and placed at <image> token positions in the embedding sequence. The
  language model processes them via standard self-attention.

COCONUT strategy (analogous to vlmcoconut.py for Llama-Vision):
1. First COCONUT pass processes the prefix (everything before the first
   latent token) with input_ids + pixel_values, letting the model run its
   own vision pipeline (CLIP -> MLP projector -> masked_scatter).
2. hidden_states[0] from that pass gives vision-merged prefix embeddings;
   combined with self.embedding() for latent/suffix tokens to form full
   inputs_embeds -- no redundant full-sequence forward pass.
3. Vision token K/V entries in the cache from the first pass ensure visual
   info persists across all subsequent COCONUT passes.
4. Autoregressive generation recomputes the full prefix each step.

Why this avoids the Qwen approach's visual feature degradation:
- Vision features are computed via the model's own pipeline, not a separate
  extraction step, ensuring token counts and AnyRes handling are correct.
- Standard RoPE positioning (no rope_deltas complications).
- Subsequent COCONUT passes go through the language model with inputs_embeds
  and KV cache, preserving vision context from the first pass.
"""

import torch
import torch.nn as nn
from collections import namedtuple
from transformers import DynamicCache


Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "past_key_values"])


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
            print("WARNING: No image token id found in config")

    def forward(self, **kwargs):
        """
        Forward pass with Coconut continuous thought mechanism.

        Flow:
        1. Find latent token positions in input_ids
        2. No latents -> single forward with pixel_values and return
        3. First COCONUT pass processes the prefix with pixel_values so the
           model handles its own vision pipeline; hidden_states[0] gives
           vision-merged prefix embeddings, combined with self.embedding()
           for latent/suffix tokens to form full inputs_embeds
        4. Subsequent passes update latent embeddings via KV-cached forwards
        5. Final pass processes remaining tokens after all latent passes
        """
        input_ids = kwargs['input_ids']
        pixel_values = kwargs.get('pixel_values')
        image_sizes = kwargs.get('image_sizes')

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

        # ── no latent tokens -> single forward pass, done ────────────────
        if max_n_latents == 0:
            outputs = self.base_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=kwargs.get('attention_mask'),
                output_hidden_states=True,
                return_dict=True,
            )
            return Outputs(
                loss=None,
                inputs_embeds=outputs.hidden_states[0].detach().clone(),
                logits=outputs.logits, past_key_values=None,
            )

        # ── COCONUT multi-pass loop ──────────────────────────────────────
        next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None
        inputs_embeds = None
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # First pass: process prefix with pixel_values so the
                # model runs its own vision pipeline (CLIP -> projector ->
                # masked_scatter).  Vision token K/V entries are written
                # into the cache for all subsequent passes.
                forward_kwargs = {
                    "input_ids": input_ids[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
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

                # Build full-sequence inputs_embeds: vision-merged prefix
                # from hidden_states[0], plain embeddings for the rest
                # (latent + suffix tokens contain no image tokens).
                prefix_embeds = outputs.hidden_states[0].detach().clone()
                rest_embeds = self.embedding(
                    input_ids[:, next_compute_range[1]:]
                )
                inputs_embeds = torch.cat(
                    [prefix_embeds, rest_embeds], dim=1
                )

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

        self.gen_forward_cnt += max_n_latents + 1

        return Outputs(
            loss=None,
            inputs_embeds=inputs_embeds,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )

    def eval(self):
        self.base_model.eval()

    def generate(self, **kwargs):
        """
        Generate text with Coconut continuous thought mechanism.

        Strategy (mirrors vlmcoconut.py):
        1. Run Coconut forward -> latent-filled inputs_embeds + final logits
        2. Greedy-decode first token from final logits
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
