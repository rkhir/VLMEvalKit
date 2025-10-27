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
        """
        Computes hidden states in segments so that for every <|latent|> at t,
        we inject the hidden state at t-1 into inputs_embeds[t].
        """

        B, T = input_ids.shape
        logits_chunks = []

        # locate all latent positions per batch item
        latent_mask = (input_ids == self.latent_token_id)
        latent_indices = latent_mask.nonzero(as_tuple=False)  # [N_lat, 2]
        latent_lists = [[pos.item() for b, pos in latent_indices if b.item() == i] for i in range(B)]
        has_latents = any(len(l) > 0 for l in latent_lists)

        # base embeddings (we will overwrite some rows with "thoughts")
        inputs_embeds = self.embedding(input_ids)  # [B, T, H]

        # build segment schedule:
        segments = []
        if has_latents:
            first_latent = latent_indices[:, 1].min().item()
            # 1) initial wide segment [0:first_latent)
            if first_latent > 0:
                segments.append((0, first_latent, "init"))

            # 2) one segment for every preceding position (pos-1) we must fill
            prev_positions = sorted(set(pos - 1 for lst in latent_lists for pos in lst if pos > 0))
            for p in prev_positions:
                segments.append((p, p + 1, "prev"))

            # 3) tail after the last handled token
            last_end = segments[-1][1] if len(segments) else 0
            if last_end < T:
                segments.append((last_end, T, "tail"))
        else:
            # no latents: single pass
            segments.append((0, T, "all"))

        kv_cache = None
        self._last_pkv = None
        self._last_len = None

        for (start, end, kind) in segments:
            L = end - start
            assert L > 0, f"Empty segment {start}:{end}"

            if kv_cache is None:
                # First call: ensure the preceding states see FULL prefix up to `end`.
                if pixel_values is not None and start == 0:
                    forward_kwargs = dict(
                        input_ids=input_ids[:, :end],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, :end],
                        pixel_values=pixel_values,
                        output_hidden_states=True,
                        use_cache=use_cache,
                        return_dict=True,
                    )
                    if aspect_ratio_ids is not None:
                        forward_kwargs["aspect_ratio_ids"] = aspect_ratio_ids
                    if aspect_ratio_mask is not None:
                        forward_kwargs["aspect_ratio_mask"] = aspect_ratio_mask
                    outputs = self.base_causallm(**forward_kwargs)
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
                # Extend with the new span using PKV.
                past_key_values = [(k[:, :, :start, :], v[:, :, :start, :]) for (k, v) in kv_cache]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, start:end, :],  # only the new slice
                    attention_mask=attention_mask[:, :end],       # prefix + new
                    position_ids=position_ids[:, start:end],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=use_cache,
                    return_dict=True,
                )
                hidden_states_offset = start

            logits_chunks.append(outputs.logits)          # [B, L, V]
            kv_cache = outputs.past_key_values if use_cache else None

            # last layer hidden states for THIS segment only
            hidden_states = outputs.hidden_states[-1]      # [B, L, H]
            seg_len = hidden_states.shape[1]

            # inject "thoughts" for any latent whose preceding token lies inside this segment
            if kind in ("init", "prev", "all"):
                new_embeds = inputs_embeds.clone()
                for b in range(B):
                    for token_idx in latent_lists[b]:
                        prev_global = token_idx - 1
                        hidden_idx = prev_global - hidden_states_offset
                        if 0 <= hidden_idx < seg_len:
                            rep = hidden_states[b, hidden_idx, :].detach()  # detach to keep graph/memory sane
                            rep = self._fill_checks(rep, new_embeds[b, token_idx, :])
                            new_embeds[b, token_idx, :] = rep
                inputs_embeds = new_embeds  # [B, T, H]

        # cache for generate()
        self._last_pkv = kv_cache
        self._last_len = T

        # concat logits along time to [B, T, V]
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
        """
        Cache-aware continuation generation after forward() has injected thoughts.
        """
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        device = input_ids.device

        # 1) Run forward once with cache so vision/context is established
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, device=device),
            labels=None,
            position_ids=torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).view(1, -1),
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            aspect_ratio_mask=aspect_ratio_mask,
            compute_loss=False,
            use_cache=True,
        )
        inputs_embeds = outputs.inputs_embeds  # [1, T, H]
        logits = outputs.logits               # [1, T, V]

        # priming token
        tokens = input_ids[0].detach().tolist()
        next_token = int(torch.argmax(logits[0, -1]))
        if next_token == self.eos_token_id:
            return torch.tensor(tokens + [next_token]).view(1, -1)
        tokens.append(next_token)

        # prepare cache-aware decoding state
        past = self._last_pkv
        cur_len = self._last_len
        attn = torch.ones(1, cur_len, dtype=torch.long, device=device)
        pos = torch.arange(cur_len, device=device, dtype=torch.long).view(1, -1)

        # 2) step-wise decode using input_ids (simpler & robust for RoPE)
        for _ in range(max_new_tokens - 1):
            in_ids = torch.tensor([[next_token]], device=device, dtype=torch.long)  # [1,1]
            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=1)
            pos_next = pos[:, -1:] + 1
            pos = torch.cat([pos, pos_next], dim=1)

            out = self.base_causallm(
                input_ids=in_ids,
                attention_mask=attn,
                position_ids=pos[:, -1:],  # position for the new token
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            self.gen_forward_cnt += 1

            next_token = int(torch.argmax(out.logits[0, -1]))
            if next_token == self.eos_token_id:
                tokens.append(next_token)
                break
            tokens.append(next_token)

        # FSDP sync padding if needed
        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                _ = self.base_causallm(
                    input_ids=torch.tensor([[self.eos_token_id]], device=device),
                    attention_mask=attn,
                    position_ids=pos[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                self.gen_forward_cnt += 1

        if output_embedding:
            # Return final embedding stream too
            final_ids = torch.tensor(tokens, device=device, dtype=torch.long).view(1, -1)
            final_embeds = self.embedding(final_ids)
            return final_ids, final_embeds
        else:
            return torch.tensor(tokens, device=device, dtype=torch.long).view(1, -1)
