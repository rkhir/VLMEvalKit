
import torch
import torch.nn as nn
from collections import namedtuple
from transformers import DynamicCache


Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "past_key_values"])
MAX_N_LATENT = 8


def _compute_cross_attention_states(base_causallm, pixel_values, aspect_ratio_mask=None, aspect_ratio_ids=None):
    base = base_causallm
    if not hasattr(base, "vision_model"):
        raise RuntimeError("Base model missing vision_model; cannot compute cross_attention_states")

    with torch.no_grad():
        vout = base.vision_model(
            pixel_values=pixel_values,
            aspect_ratio_mask=aspect_ratio_mask,
            aspect_ratio_ids=aspect_ratio_ids,
            return_dict=True,
        )
        feats = vout.last_hidden_state  # could be 3D/4D/5D depending on version

        # Project to LM hidden size if projector exists
        if hasattr(base, "multi_modal_projector"):
            feats = base.multi_modal_projector(feats)

        # Normalize to [B, I, T, C]
        if feats.ndim == 5:
            # [B, I, tiles, tokens_per_tile, C] -> [B, I, tiles*tokens_per_tile, C]
            B, I, tiles, tok, C = feats.shape
            feats = feats.reshape(B, I, tiles * tok, C).contiguous()
        elif feats.ndim == 4:
            # Likely already [B, I, T, C]
            B, I, T, C = feats.shape
            feats = feats.contiguous()
        elif feats.ndim == 3:
            # [B, N, C], where N = I * T
            B, N, C = feats.shape
            I = pixel_values.shape[1]
            T = N // I
            feats = feats.view(B, I, T, C).contiguous()
        else:
            raise ValueError(f"Unexpected vision feature shape: {feats.shape}")

        return feats

class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        processor,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.processor = processor
        self.base_causallm.config.use_cache = True
        #self._kv_cache = DynamicCache()

        self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self,**kwargs):
        input_ids = kwargs['input_ids']
        logits = []


        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()


        latent_lists = []
        print('input shape >>>>>', input_ids.shape)
        for i in range(input_ids.shape[0]):
            lst = []
            for idx in latent_indices:
                print('latent_indices idx', idx)
                if idx[0] == i:
                    lst.append(idx[1].item())
            latent_lists.append(lst)

        max_n_latents = max([len(l) for l in latent_lists])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            # next compute is the first latent index (min index)
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position
        else:
          next_compute_range = (0, input_ids.shape[1])
          forward_kwargs = {
          "input_ids": kwargs['input_ids'][:, next_compute_range[0] : next_compute_range[1]],
          "attention_mask": kwargs['attention_mask'][:, : next_compute_range[1]],
          "output_hidden_states": True,
          "return_dict": True,
          "pixel_values": kwargs["pixel_values"],
          "aspect_ratio_mask": kwargs['aspect_ratio_mask'],
          "aspect_ratio_ids": kwargs['aspect_ratio_ids'],
          "cross_attention_mask": kwargs.get('cross_attention_mask')
          }
          if 'position_ids' in kwargs:
              forward_kwargs['position_ids'] = kwargs['position_ids'][:, next_compute_range[0] : next_compute_range[1]]
          else:
              # Generate position_ids for sliced sequence
              forward_kwargs['position_ids'] = torch.arange(
                  next_compute_range[0],
                  next_compute_range[1],
                  dtype=torch.long,
                  device=input_ids.device
              ).unsqueeze(0)

          outputs = self.base_causallm(**forward_kwargs)
          return Outputs(loss=None, inputs_embeds=inputs_embeds, logits=outputs.logits, past_key_values=None)

        vision_kwargs = {}
        if 'pixel_values' in kwargs and kwargs['pixel_values'] is not None:
            vision_kwargs['pixel_values'] = kwargs['pixel_values']
            cross_states = _compute_cross_attention_states(self.base_causallm,kwargs['pixel_values'],kwargs.get('aspect_ratio_mask'),kwargs.get('aspect_ratio_ids'))
            #vision_kwargs["cross_attention_states"] = cross_states  # [B, I, T, C]

        vision_kwargs["cross_attention_mask"] = kwargs.get('cross_attention_mask') #cross_attn_mask
        if 'aspect_ratio_ids' in kwargs and kwargs['aspect_ratio_ids'] is not None:
            vision_kwargs['aspect_ratio_ids'] = kwargs['aspect_ratio_ids']
        if 'aspect_ratio_mask' in kwargs and kwargs['aspect_ratio_mask'] is not None:
            vision_kwargs['aspect_ratio_mask'] = kwargs['aspect_ratio_mask']


        kv_cache = None
        for pass_idx in range(max_n_latents):
            print(pass_idx)
            if kv_cache is None:

                forward_kwargs = {
                "input_ids": kwargs['input_ids'][:, next_compute_range[0] : next_compute_range[1]],
                "attention_mask": kwargs['attention_mask'][:, : next_compute_range[1]],
                "output_hidden_states": True,
                "use_cache":True,
                "return_dict": True,
                "past_key_values":   DynamicCache(),
                "pixel_values": vision_kwargs["pixel_values"],
                #"cross_attention_mask": vision_kwargs['cross_attention_mask'][:, next_compute_range[0]:next_compute_range[1], :, :],
                "aspect_ratio_mask": vision_kwargs['aspect_ratio_mask'],
                "aspect_ratio_ids": vision_kwargs['aspect_ratio_ids'],
                "cache_position": torch.arange(
                        next_compute_range[0], next_compute_range[1],
                        device=input_ids.device
                        ).unsqueeze(0),
                }

                if 'position_ids' in kwargs:
                    forward_kwargs['position_ids'] = kwargs['position_ids'][:, next_compute_range[0] : next_compute_range[1]]
                else:
                    # Generate position_ids for sliced sequence
                    forward_kwargs['position_ids'] = torch.arange(
                        next_compute_range[0],
                        next_compute_range[1],
                        dtype=torch.long,
                        device=input_ids.device
                    ).unsqueeze(0)

                outputs = self.base_causallm(**forward_kwargs)

                hidden_states_offset = next_compute_range[0]
            else:
                print(f'\n<<<using cache while thinking>>>>\n')
                # extract kv cache to reuse
                legacy_kv_cache = kv_cache.to_legacy_cache()
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in legacy_kv_cache
                ]
                past_key_values= DynamicCache.from_legacy_cache(past_key_values)

                forward_kwargs = {"inputs_embeds": inputs_embeds[:, next_compute_range[0]: next_compute_range[1], :],
                                  "attention_mask": kwargs['attention_mask'][:, : next_compute_range[1]],
                                  "position_ids": torch.arange(
                                      next_compute_range[0],
                                      next_compute_range[1],
                                      dtype=torch.long,
                                      device=input_ids.device
                                  ).unsqueeze(0),
                                  "past_key_values": past_key_values, "output_hidden_states": True,
                                  "use_cache": True,
                                  "cache_position": torch.arange(
                                        next_compute_range[0], next_compute_range[1],
                                        device=input_ids.device
                                        ).unsqueeze(0),
                                  'cross_attention_states': cross_states}
                #forward_kwargs["cross_attention_mask"] = cross_attn_mask[:, next_compute_range[0]:next_compute_range[1], :, :]

                # always pass the image and cross attention
                outputs = self.base_causallm(**forward_kwargs)
                hidden_states_offset = next_compute_range[0]
            ###############
            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values



            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                vec = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]


                tensor_list[batch_idx][token_idx] = vec

                # replace it with the preceding last hidden states
                #tensor_list[batch_idx][token_idx] = hidden_states[
                #    batch_idx, token_idx - 1 - hidden_states_offset, :
                #]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        final_forward_kwargs = {"inputs_embeds": inputs_embeds[:, next_compute_range[0]: next_compute_range[1], :],
                                "attention_mask": kwargs['attention_mask'][:, : next_compute_range[1]],
                                "position_ids": torch.arange(
                                    next_compute_range[0],
                                    next_compute_range[1],
                                    dtype=torch.long,
                                    device=input_ids.device
                                ).unsqueeze(0), "output_hidden_states": True, 'cross_attention_states': cross_states}
        if kv_cache:
          print(f'\n<<<using cache final forward pass>>>>\n')

          legacy_kv_cache = kv_cache.to_legacy_cache()
          past_key_values = [
              (
                k[:, :, : next_compute_range[0], :],
                v[:, :, : next_compute_range[0], :],
              )
              for k, v in legacy_kv_cache
          ]

          past_key_values= DynamicCache.from_legacy_cache(past_key_values)
          final_forward_kwargs['past_key_values'] = past_key_values

        outputs = self.base_causallm(**final_forward_kwargs)
        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1
        loss = None
        logits = torch.cat(logits, dim=-2)

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits, past_key_values=past_key_values)

    #def train(self):
    #    self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(self, **kwargs):

      self.gen_forward_cnt = 0

      input_ids = kwargs["input_ids"]
      assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
      device = input_ids.device

      # Unpack image-related kwargs once
      pixel_values = kwargs.get("pixel_values", None)
      aspect_ratio_mask = kwargs.get("aspect_ratio_mask", None)
      aspect_ratio_ids = kwargs.get("aspect_ratio_ids", None)
      base_cam = kwargs.get("cross_attention_mask", None)
      max_new_tokens = kwargs.get("max_new_tokens", 16)
      cross_states = _compute_cross_attention_states(self.base_causallm,kwargs['pixel_values'],kwargs.get('aspect_ratio_mask'),kwargs.get('aspect_ratio_ids'))

      # 1) Run Coconut forward to get latent-filled embeddings + logits for prefix
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
          aspect_ratio_mask=aspect_ratio_mask,
          aspect_ratio_ids=aspect_ratio_ids,
          cross_attention_mask=base_cam,
          compute_loss=False,
      )

      inputs_embeds = outputs.inputs_embeds
      logits = outputs.logits

      tokens = input_ids[0].detach().tolist()

      # 2) First generated token (greedy for now)
      next_token = int(logits[0, -1].argmax(-1))
      if next_token == self.eos_token_id:
          return torch.tensor(tokens + [next_token], device=device).unsqueeze(0)

      tokens.append(next_token)
      next_embed = self.embedding(torch.tensor([[next_token]], device=device))
      inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

      # Extend attention mask (1 for the new token)
      attention_mask = torch.cat(
          [
              attention_mask,
              torch.ones(B, 1, dtype=attention_mask.dtype, device=device),
          ],
          dim=1,
      )

      # Helper: extend cross-attention mask ONLY along seq_length
      def extend_cam(_cam, new_seq_len):
          if _cam is None:
              return None
          bsz, old_S, I, T = _cam.shape
          if new_seq_len <= old_S:
              return _cam[:, :new_seq_len]
          # repeat last valid row for new tokens
          last = _cam[:, old_S - 1 : old_S, :, :]
          repeat = new_seq_len - old_S
          pad = last.expand(bsz, repeat, I, T)
          return torch.cat([_cam, pad], dim=1)

      # 3) Autoregressive loop: full-prefix forward each step (simple & correct)
      for _ in range(max_new_tokens - 1):
          curr_len = inputs_embeds.shape[1]

          cam = extend_cam(base_cam, curr_len)

          step_out = self.base_causallm(
              inputs_embeds=inputs_embeds,
              attention_mask=attention_mask,
              cross_attention_states=cross_states,
              aspect_ratio_mask=aspect_ratio_mask,
              aspect_ratio_ids=aspect_ratio_ids,
              cross_attention_mask=cam,
              use_cache=False,
              output_hidden_states=False,
              return_dict=True,
          )

          logits_step = step_out.logits
          next_token = int(logits_step[0, -1].argmax(-1))
          # Debug:
          #txt = self.processor.tokenizer.decode([next_token], skip_special_tokens=False)

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

