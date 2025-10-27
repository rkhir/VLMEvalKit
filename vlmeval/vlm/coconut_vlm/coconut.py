# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

MAX_N_LATENT = 8  # kept for FSDP sync padding if you need it


class Coconut(nn.Module):
    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama-family
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        # internal state used by generate()
        self._last_pkv = None
        self._last_len = None
        self.gen_forward_cnt = 0

    @torch.no_grad()
    def _fill_checks(self, rep, dst):
        # keep dtype/device aligned
        if rep.dtype != dst.dtype:
            rep = rep.to(dst.dtype)
        if rep.device != dst.device:
            rep = rep.to(dst.device)
        return rep

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        compute_loss=False,
        use_cache=True,
        **kwargs,
    ):
        B, T = input_ids.shape
        logits_chunks = []

        latent_mask = (input_ids == self.latent_token_id)
        latent_indices = latent_mask.nonzero(as_tuple=False)
        latent_lists = [[pos.item() for b, pos in latent_indices if b.item() == i] for i in range(B)]
        has_latents = any(len(l) > 0 for l in latent_lists)

        inputs_embeds = self.embedding(input_ids)  # [B, T, H]

        segments = []
        if has_latents:
            first_latent = latent_indices[:, 1].min().item()
            if first_latent > 0:
                segments.append((0, first_latent, "init"))
            prev_positions = sorted(set(pos - 1 for lst in latent_lists for pos in lst if pos > 0))
            for p in prev_positions:
                segments.append((p, p + 1, "prev"))
            last_end = segments[-1][1] if segments else 0
            if last_end < T:
                segments.append((last_end, T, "tail"))
        else:
            segments.append((0, T, "all"))

        kv_cache = None
        self._last_pkv = None
        self._last_len = None
        self._kv_seq_len = None  # NEW: will store the *true* cache length

        for (start, end, kind) in segments:
            if kv_cache is None:
                if pixel_values is not None and start == 0:
                    fkw = dict(
                        input_ids=input_ids[:, :end],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, :end],
                        pixel_values=pixel_values,
                        output_hidden_states=True,
                        use_cache=use_cache,
                        return_dict=True,
                    )
                    if aspect_ratio_ids is not None:
                        fkw["aspect_ratio_ids"] = aspect_ratio_ids
                    if aspect_ratio_mask is not None:
                        fkw["aspect_ratio_mask"] = aspect_ratio_mask
                    outputs = self.base_causallm(**fkw)
                else:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, :end, :],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, :end],
                        output_hidden_states=True,
                        use_cache=use_cache,
                        return_dict=True,
                    )
                hidden_states_offset = 0
            else:
                past_key_values = [(k[:, :, :start, :], v[:, :, :start, :]) for (k, v) in kv_cache]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, start:end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, start:end],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=use_cache,
                    return_dict=True,
                )
                hidden_states_offset = start

            logits_chunks.append(outputs.logits)
            kv_cache = outputs.past_key_values if use_cache else None

            hidden_states = outputs.hidden_states[-1]  # [B, L, H]
            seg_len = hidden_states.shape[1]

            if kind in ("init", "prev", "all"):
                new_embeds = inputs_embeds.clone()
                for b, lst in enumerate(latent_lists):
                    for token_idx in lst:
                        prev_global = token_idx - 1
                        hidden_idx = prev_global - hidden_states_offset
                        if 0 <= hidden_idx < seg_len:
                            rep = hidden_states[b, hidden_idx, :].detach()
                            if rep.dtype != new_embeds.dtype:
                                rep = rep.to(new_embeds.dtype)
                            new_embeds[b, token_idx, :] = rep
                inputs_embeds = new_embeds

        # --- Store cache & actual kv sequence length for generation ---
        self._last_pkv = kv_cache
        # True cache length: take K tensor shape [B, H, KV_LEN, D]
        self._kv_seq_len = (kv_cache[0][0].shape[2] if kv_cache is not None else inputs_embeds.shape[1])
        self._last_len = input_ids.shape[1]

        logits = torch.cat(logits_chunks, dim=-2)

        loss = None
        if compute_loss and labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self, mode: bool = True):
        self.base_causallm.train(mode)
        return super().train(mode)

    def eval(self):
        self.base_causallm.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs,
    ):
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        device = input_ids.device
        # Use the caller's mask for the prefill so padding (if any) is respected.
        # Fallback to ones if none is provided.
        am_prefill = attention_mask
        if am_prefill is None:
            am_prefill = torch.ones_like(input_ids, device=device)
        else:
            am_prefill = am_prefill.to(device)
        # 1) Run forward with cache so vision context is in PKV
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=am_prefill,#torch.ones_like(input_ids, device=device),
            labels=None,
            position_ids=torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).view(1, -1),
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            compute_loss=False,
            use_cache=True,
        )

        tokens = input_ids[0].tolist()
        logits = outputs.logits  # [1, T, V]
        next_token = int(torch.argmax(logits[0, -1]))
        tokens.append(next_token)

        past = self._last_pkv
        # >>> CRITICAL: use the TRUE KV length (includes vision expansion etc.)
        kv_len = self._kv_seq_len if self._kv_seq_len is not None else input_ids.shape[1]
        attn = torch.ones(1, kv_len, dtype=torch.long, device=device)  # start from cached length

        # 2) Decode step-by-step using PKV; let HF infer positions from attention_mask
        for _ in range(max_new_tokens - 1):
            in_ids = torch.tensor([[next_token]], device=device, dtype=torch.long)  # [1,1]
            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)

            out = self.base_causallm(
                input_ids=in_ids,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            self.gen_forward_cnt += 1

            next_token = int(torch.argmax(out.logits[0, -1]))
            tokens.append(next_token)
            if next_token == self.eos_token_id:
                break

        if synced_gpus:
            # simple padding for FSDP if you need equal #steps
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                _ = self.base_causallm(
                    input_ids=torch.tensor([[self.eos_token_id]], device=device),
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                self.gen_forward_cnt += 1

        if output_embedding:
            final_ids = torch.tensor(tokens, device=device, dtype=torch.long).view(1, -1)
            final_embeds = self.embedding(final_ids)
            return final_ids, final_embeds
        else:
            return torch.tensor(tokens, device=device, dtype=torch.long).view(1, -1)
