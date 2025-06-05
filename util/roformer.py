import math
import os
from typing import Optional

import numpy as np
import torch

from torch import nn

from transformers.activations import ACT2FN

from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from timm.layers import Mlp, DropPath

from timm.models.vision_transformer import Attention


# --------------------------------------------------------
# RoFormer position embedding
# TODO
# --------------------------------------------------------
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class RoFormerSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            use_bias=True,
            rotary_value=False,
            attention_probs_dropout_prob=0.25,
            # is_decoder=False,
            # has_relative_attention_bias=False,
                 ):
        super().__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=use_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=use_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=use_bias)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        # self.is_decoder = is_decoder
        self.rotary_value = rotary_value

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,  # this variable should be used as encoder input
        attention_mask=None,
        sinusoidal_pos=None, # output from RoFormerSinusoidalPositionalEmbedding
        head_mask=None,
        encoder_hidden_states=None,  #  this variable should be used as decoder output
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # rotary query
        query_layer = self.apply_rotary(query_layer, sinusoidal_pos)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None  # if cross attention, should be true

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            # rotary key_layer & value_layer
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos)
            if self.rotary_value:
                value_layer = self.apply_rotary(value_layer, sinusoidal_pos)
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            # rotary key_layer & value_layer
            key_layer = self.apply_rotary(key_layer, sinusoidal_pos)
            if self.rotary_value:
                value_layer = self.apply_rotary(value_layer, sinusoidal_pos)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RoFormerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        # if self.is_decoder:
            # outputs = outputs + (past_key_value,)
        return outputs
        # return context_layer

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)
    

class Norm(nn.Module):
    def __init__(self, eps = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        return x / torch.sqrt(variance + self.eps)


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->RoFormer
class RoFormerSelfOutput(nn.Module):
    def __init__(self,
                hidden_size,
                use_bias=True,
                layer_norm_eps=1e-12,
                norm_type="layer_norm",
                hidden_dropout_prob=0.0,
                ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if norm_type=="layer_norm" else Norm(eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RoFormerAttention(nn.Module):
    def __init__(self,
                hidden_size,
                num_attention_heads,
                use_bias=True,
                rotary_value=False,
                attention_probs_dropout_prob=0.0,
                # is_decoder=False,
                layer_norm_eps=1e-12,
                norm_type="layer_norm",
                hidden_dropout_prob=0.0,
                ):
        super().__init__()
        self.self = RoFormerSelfAttention(hidden_size=hidden_size, 
                                          num_attention_heads=num_attention_heads, 
                                          use_bias=use_bias, 
                                          rotary_value=rotary_value, 
                                          attention_probs_dropout_prob=attention_probs_dropout_prob
                                        )
        self.output = RoFormerSelfOutput(hidden_size=hidden_size, 
                                         use_bias=use_bias, 
                                         layer_norm_eps=layer_norm_eps,
                                         norm_type=norm_type,
                                         hidden_dropout_prob=hidden_dropout_prob
                                        )
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    # End Copy
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->RoFormer
class RoFormerIntermediate(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 hidden_act,
                 use_bias=True,
                ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->RoFormer
class RoFormerOutput(nn.Module):
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 use_bias=True,
                 layer_norm_eps=1e-12,
                 norm_type="layer_norm",
                 hidden_dropout_prob=0.0,
                ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if norm_type=="layer_norm" else Norm(eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RoFormerLayer(nn.Module):
    def __init__(self, 
                hidden_size,
                num_attention_heads,
                use_bias=True,
                rotary_value=False,
                attention_probs_dropout_prob=0.0,
                # is_decoder=False,
                layer_norm_eps=1e-12,
                norm_type="layer_norm",
                hidden_dropout_prob=0.0,
                chunk_size_feed_forward=0,
                is_decoder=True,
                add_cross_attention=False,
                intermediate_size=4 * 1024,
                ):
        super().__init__()
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RoFormerAttention(hidden_size=hidden_size,
                                           num_attention_heads=num_attention_heads,
                                           use_bias=use_bias,
                                           rotary_value=rotary_value,
                                           attention_probs_dropout_prob=attention_probs_dropout_prob,
                                           layer_norm_eps=layer_norm_eps,
                                           norm_type=norm_type,
                                        )
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = RoFormerAttention(
                                            hidden_size=hidden_size,
                                           num_attention_heads=num_attention_heads,
                                           use_bias=use_bias,
                                           rotary_value=rotary_value,
                                           attention_probs_dropout_prob=attention_probs_dropout_prob,
                                           layer_norm_eps=layer_norm_eps,
                                           norm_type=norm_type,
                                           )
        self.intermediate = RoFormerIntermediate(hidden_size=hidden_size,
                                                 intermediate_size=hidden_size * 4,
                                                 hidden_act="gelu",
                                                 use_bias=use_bias,
                                                 )
        self.output = RoFormerOutput(hidden_size,
                                    intermediate_size,
                                    use_bias=use_bias,
                                    layer_norm_eps=layer_norm_eps,
                                    norm_type=norm_type,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    )


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
                    "layers by setting `add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                sinusoidal_pos,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RopeBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # TODO
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(dim, dim)
        self.crossattn = RoFormerSelfAttention(dim, num_heads)
        self.selfattn = RoFormerSelfAttention(dim, num_heads)
        
        

    def forward(self, q: torch.Tensor, kv: torch.Tensor = None) -> torch.Tensor:
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        sinusoidal_pos = self.embed_positions(q.shape[1], kv.shape[1])[
            None, None, :, :
        ].chunk(2, dim=-1)
        q = self.norm1(q)
        kv = self.norm1(kv)
        x = self.crossattn(hidden_states=q, sinusoidal_pos=sinusoidal_pos, encoder_hidden_states=kv)[0]
        x = x + self.drop_path1(self.ls1(x))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x