# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations and flamingo-pytorch in this library.
# Llama 2 is licensed under the LLAMA 2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.
import torch
from torch import nn, einsum
from transformers import LlamaForCausalLM, LlamaConfig
import torch.distributed
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from cache_utils import Cache, DynamicCache, StaticCache
from modeling_attn_mask_utils import AttentionMaskConverter
import torch.utils.checkpoint



def FeedForward(dim, inner_dim):
    return nn.Sequential(
        nn.LayerNorm(dim, dtype=torch.bfloat16),
        nn.Linear(dim, inner_dim, bias=False, dtype=torch.bfloat16),
        nn.SiLU(),
        nn.Linear(inner_dim, dim, bias=False, dtype=torch.bfloat16)
    )


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_x = nn.LayerNorm(dim, dtype=torch.bfloat16)
        self.norm_latents = nn.LayerNorm(dim, dtype=torch.bfloat16)

        self.to_q = nn.Linear(dim, inner_dim, bias=False, dtype=torch.bfloat16)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False, dtype=torch.bfloat16)
        self.to_out = nn.Linear(inner_dim, dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        n - sequence
        d - dimension
        """
        x = self.norm_x(x)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        # kv_input = torch.cat((x, latents), dim=1)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


class Compressor(nn.Module):
    def __init__(self, num_layers, dim, n_heads, dim_head, inner_dim, k):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim, dim_head, n_heads),
                FeedForward(dim=dim, inner_dim=inner_dim)
            ]))
        self.norm = nn.LayerNorm(dim, dtype=torch.bfloat16)
        self.pos_emb = nn.Parameter(torch.randn(dim, dim, dtype=torch.bfloat16))

    def forward(self, latents, seg_embeddings):
        embeddings = seg_embeddings + self.pos_emb[:seg_embeddings.size(1), :].repeat(seg_embeddings.size(0), 1, 1)

        for attn, ffw in self.layers:
            latents = attn(embeddings, latents) + latents
            latents = ffw(latents) + latents

        return self.norm(latents)


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim, dtype=torch.bfloat16)

        self.to_q = nn.Linear(dim, inner_dim, bias=False, dtype=torch.bfloat16)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False, dtype=torch.bfloat16)
        self.to_out = nn.Linear(inner_dim, dim, bias=False, dtype=torch.bfloat16)

    def forward(
        self,
        x,
        media
    ):
        x = self.norm(x)

        q = self.to_q(x)

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        n_heads,
        inner_dim
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_head=dim_head, heads=n_heads)
        self.attn_gate = nn.Parameter(torch.tensor(0.))

        '''FFW Reduction Here'''
        self.ff = FeedForward(dim=dim, inner_dim=inner_dim)
        self.ff_gate = nn.Parameter(torch.tensor(0.))

    def forward(
        self,
        x,
        media
    ):
        x = self.attn(x, media) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x


class CustomLlamaForCausalLM(nn.Module):
    def __init__(self, model_name, arr):
        super().__init__()
        self.config = LlamaConfig.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.model = LlamaForCausalLM.from_pretrained(model_name, config=self.config, torch_dtype=torch.bfloat16)

        self.Xattn_layers = nn.ModuleList(
            [GatedCrossAttentionBlock(self.config.hidden_size, int(self.config.hidden_size/self.config.num_attention_heads), self.config.num_attention_heads, self.config.intermediate_size) for _ in range(len(arr))]
        )

        self.arr = arr

        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            soft_prompt=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        i_Xattn = 0
        for i, decoder_layer in enumerate(self.model.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if i in self.arr and soft_prompt is not None:
                # if soft_prompt is not None:
                hidden_states = self.Xattn_layers[i_Xattn](hidden_states, soft_prompt)
                i_Xattn = i_Xattn + 1

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.model.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            outputs = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        else:
            outputs = BaseModelOutputWithPast(last_hidden_state=hidden_states,past_key_values=next_cache,hidden_states=all_hidden_states,attentions=all_self_attns,)

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.model.lm_head.weight.split(self.model.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.model.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LCIRC(nn.Module):
    def __init__(self, model_name, num_layers, dim, n_heads, dim_head, inner_dim, K, arr, cp_path=None):
        super().__init__()
        self.model = CustomLlamaForCausalLM(model_name, arr)
        self.compressor = Compressor(num_layers, dim, n_heads, dim_head, inner_dim, K)
        self.latent = nn.Parameter(torch.randn(K, dim, dtype=torch.bfloat16))

        if cp_path:
            checkpoint = torch.load(cp_path)
            self.model.Xattn_layers.load_state_dict(checkpoint['Xattn_layers_state_dict'])
            self.compressor.load_state_dict(checkpoint['compressor_state_dict'])
            with torch.no_grad():
                self.latent.copy_(checkpoint['latent_state_dict']['latent'])

        ## Make model frozen
        for name, param in self.model.named_parameters():
            if "Xattn_layers" not in name:
                param.requires_grad = False

    def forward(self, input_embedding, max_len, soft_prompt):
        output = self.model(input_ids=input_embedding, output_hidden_states=True, soft_prompt=soft_prompt)
        return output.logits[:, -max_len:-1]


class QDLCIRC(nn.Module):
    def __init__(self, model_name, num_layers, dim, n_heads, dim_head, inner_dim, K, arr, cp_path=None, rank=None, is_eval=False):
        super().__init__()
        self.model = CustomLlamaForCausalLM(model_name, arr)
        self.compressor = Compressor(num_layers, dim, n_heads, dim_head, inner_dim, K)
        self.latent = nn.Parameter(torch.randn(K, dim, dtype=torch.bfloat16))
        self.mixer = GatedCrossAttentionBlock(dim, dim_head, n_heads, inner_dim)

        if is_eval and cp_path:
            checkpoint = torch.load(cp_path, map_location='cuda:0')
            self.model.Xattn_layers.load_state_dict(checkpoint['Xattn_layers_state_dict'])
            self.compressor.load_state_dict(checkpoint['compressor_state_dict'])
            with torch.no_grad():
                self.latent.copy_(checkpoint['latent_state_dict']['latent'])
            self.mixer.load_state_dict(checkpoint['mixer_state_dict'])
        elif cp_path:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(cp_path, map_location=map_location)
            self.model.Xattn_layers.load_state_dict(checkpoint['Xattn_layers_state_dict'])
            self.compressor.load_state_dict(checkpoint['compressor_state_dict'])
            with torch.no_grad():
                self.latent.copy_(checkpoint['latent_state_dict']['latent'])

        ## Make model frozen
        for name, param in self.model.named_parameters():
            if "Xattn_layers" not in name:
                param.requires_grad = False

    def forward(self, input_embedding, max_len, soft_prompt):
        output = self.model(input_ids=input_embedding, output_hidden_states=True, soft_prompt=soft_prompt)
        return output.logits[:, -max_len:-1]