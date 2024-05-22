from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv2
import warnings

import torch
from torch import Tensor
from torch.nn import functional as F

import open_clip
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from open_clip.transformer import _expand_token
from timm.layers import resample_abs_pos_embed


################################################################################
#                               Hooks utils                                    #
################################################################################

# ------------ Hooked Multi-Head Attention ------------
# from https://github.com/mlfoundations/open_clip/blob/73fa7f03a33da53653f61841eb6d69aef161e521/src/open_clip/transformer.py#L129
def hooked_attention_forward(self, x, x_k, x_v, attn_mask: Optional[torch.Tensor] = None, need_weights: bool=False,):
    L, N, C = x.shape
    q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
    q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
    k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
    v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

    head_dim = q.shape[-1]
    scale = float(head_dim) ** -0.5
    q = q * scale
    attn = torch.bmm(q, k.transpose(-1, -2))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        attn += attn_mask

    attn = attn.softmax(dim=-1)
    # Hook for attention maps
    self.attention_map = attn

    x = torch.bmm(attn, v)
    x = x.transpose(0, 1).reshape(L, N, C)
    x = self.out_proj(x)
    return x

def hooked_attention_forward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)  # 3, B, L, H, D
        q, k, v = qkv[0], qkv[1], qkv[2]

        print("testing if xops work")
        out = xops.memory_efficient_attention(
            q, k, v, scale=self.scale,
        )
        print("testing if xops work")
        output = self.dense(out.view(B, L, -1))
        output = self.output_dropout(output)
        return output
    


# ------------ Hooked Residual Transformer Block ------------
# from https://github.com/mlfoundations/open_clip/blob/73fa7f03a33da53653f61841eb6d69aef161e521/src/open_clip/transformer.py#L231
# def hooked_resblock_forward(
#         self,
#         q_x: torch.Tensor,
#         k_x: Optional[torch.Tensor] = None,
#         v_x: Optional[torch.Tensor] = None,
#         attn_mask: Optional[torch.Tensor] = None,
# ):
#     k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
#     v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

#     x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
#     # Hook for intermediate features post Attn
#     self.feat_post_attn = x
#     x = x + self.ls_2(self.mlp(self.ln_2(x)))
#     # Hook for intermediate features post MLP
#     self.feat_post_mlp = x
#     return x

def hooked_resblock_forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        
        hidden_states = attention_input + attention_output
        self.feat_post_attn = hidden.states # ADDED, IMPORTANT
        
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        
        self.feat_post_mlp = output # ADDED, IMPORTANT
        return output


# ------------ Hooked PyTorch's Multi-Head AttentionResidual ------------
# modified from PyTorch Library
# https://github.com/pytorch/pytorch/blob/8c8e4e31f2ddd8e59de18ac733c0c205c23d14ad/torch/nn/functional.py#L5178
def hooked_torch_multi_head_attention_forward(self, query, key, value, key_padding_mask=None,
                                              need_weights=True, attn_mask=None):
    r"""
Args:
    query, key, value: map a query and a set of key-value pairs to an output.
        See "Attention Is All You Need" for more details.
    key_padding_mask: if provided, specified padding elements in the key will
        be ignored by the attention. When given a binary mask and a value is True,
        the corresponding value on the attention layer will be ignored. When given
        a byte mask and a value is non-zero, the corresponding value on the attention
        layer will be ignored
    need_weights: output attn_output_weights.
    attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
        the batches while a 3D mask allows to specify a different mask for the entries of each batch.

Shape:
    - Inputs:
    - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
      the embedding dimension.
    - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
      the embedding dimension.
    - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
      the embedding dimension.
    - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
      If a ByteTensor is provided, the non-zero positions will be ignored while the position
      with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
      value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
    - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
      3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
      S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
      positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
      while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
      is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
      is provided, it will be added to the attention weight.

    - Outputs:
    - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
      E is the embedding dimension.
    - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
      L is the target sequence length, S is the source sequence length.
    """
    if not self._qkv_same_embed_dim:
        out, _attn_maps = hooked_torch_func_multi_head_attention_forward(
                            query, key, value, self.embed_dim, self.num_heads,
                            self.in_proj_weight, self.in_proj_bias,
                            self.bias_k, self.bias_v, self.add_zero_attn,
                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                            training=self.training,
                            key_padding_mask=key_padding_mask, need_weights=True,
                            attn_mask=attn_mask, use_separate_proj_weight=True,
                            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                            v_proj_weight=self.v_proj_weight,
            )
        # Hook for attention maps
        self.attention_maps = _attn_maps
        return out, _attn_maps
    else:
        out, _attn_maps = hooked_torch_func_multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=True,
            attn_mask=attn_mask,
        )
        # Hook for attention maps
        self.attention_maps = _attn_maps
        return out, _attn_maps

def hooked_torch_func_multi_head_attention_forward(query: Tensor,
                                              key: Tensor,
                                              value: Tensor,
                                              embed_dim_to_check: int,
                                              num_heads: int,
                                              in_proj_weight: Tensor,
                                              in_proj_bias: Tensor,
                                              bias_k: Optional[Tensor],
                                              bias_v: Optional[Tensor],
                                              add_zero_attn: bool,
                                              dropout_p: float,
                                              out_proj_weight: Tensor,
                                              out_proj_bias: Tensor,
                                              training: bool = True,
                                              key_padding_mask: Optional[Tensor] = None,
                                              need_weights: bool = True,
                                              attn_mask: Optional[Tensor] = None,
                                              use_separate_proj_weight: bool = False,
                                              q_proj_weight: Optional[Tensor] = None,
                                              k_proj_weight: Optional[Tensor] = None,
                                              v_proj_weight: Optional[Tensor] = None,
                                              static_k: Optional[Tensor] = None,
                                              static_v: Optional[Tensor] = None,
                                             ) -> Tuple[Tensor, Optional[Tensor]]:
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and F.has_torch_function(tens_ops):
            return F.handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    # # use hooks for the attention weights if necessary
    # self.attention_map = attn_output_weights
    # # if attention_probs_forward_hook is not None and attention_probs_backwards_hook is not None:
    # if attention_probs_forward_hook is not None:
    #     attention_probs_forward_hook(attn_output_weights)
    #     # attn_output_weights.register_hook(attention_probs_backwards_hook)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # --- Fix: removed the unnecessary average over heads, Why?
        # average attention weights over heads
        # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # return attn_output, attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None


# ------------ Hooked TimmModel's Residual Transformer Block ------------
def hooked_resblock_timm_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    self.feat_post_attn = x
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    self.feat_post_mlp = x
    return x


# ------------ Hooked TimmModel's Attentional Pooler ------------
def hooked_attentional_pooler_timm_forward(self, x):
    B, N, C = x.shape

    if self.pos_embed is not None:
        # FIXME interpolate
        x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

    q_latent = self.latent.expand(B, -1, -1)
    q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)

    q, k = self.q_norm(q), self.k_norm(k)

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    x = attn @ v

    # Hook to save attention map for explainability
    self.attn_probs = attn

    x = x.transpose(1, 2).reshape(B, self.latent_len, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    x = x + self.mlp(self.norm(x))

    # optional pool if latent seq_len > 1 and pooled output is desired
    if self.pool == 'token':
        x = x[:, 0]
    elif self.pool == 'avg':
        x = x.mean(1)
    return x


# ------------ OpenCLIP ViT forward with dynamic size ------------
def vit_dynamic_size_forward(self, x: torch.Tensor):
    # self -> self.model.vision
    
    print("self: ",self)
    x = self.patch_embedding.proj(x)  # shape = [*, width, grid, grid]
    print("Conv2d working")
    
    grid_h, grid_w = x.shape[2:]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]


    # class embeddings and positional embeddings
    x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
    # shape = [*, grid ** 2 + 1, width]
    if x.shape[1] != self.positional_embedding.shape[1]:
        self.positional_embedding.data = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
                                         new_size=[grid_h, grid_w],
                                         # old_size=list(self.grid_size),
                                         num_prefix_tokens=1,
                                         interpolation='bicubic',
                                         antialias=True)

    x = x + self.positional_embedding.to(x.dtype)

    x = self.patch_dropout(x)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    if self.attn_pool is not None:
        if self.attn_pool_contrastive is not None:
            # This is untested, WIP pooling that should match paper
            x = self.ln_post(x)  # TBD LN first or separate one after each pool?
            tokens = self.attn_pool(x)
            if self.attn_pool_type == 'parallel':
                pooled = self.attn_pool_contrastive(x)
            else:
                assert self.attn_pool_type == 'cascade'
                pooled = self.attn_pool_contrastive(tokens)
        else:
            # this is the original OpenCLIP CoCa setup, does not match paper
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
    elif self.final_ln_after_pool:
        pooled, tokens = self._global_pool(x)
        pooled = self.ln_post(pooled)
    else:
        x = self.ln_post(x)
        pooled, tokens = self._global_pool(x)

    if self.proj is not None:
        pooled = pooled @ self.proj

    if self.output_tokens:
        return pooled, tokens

    return pooled



################################################################################
#                               Visualization utils                            #
################################################################################

def min_max(logits):
    B, num_prompt = logits.shape[:2]
    logits_min = logits.reshape(B, num_prompt, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    logits_max = logits.reshape(B, num_prompt, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    logits = (logits - logits_min) / (logits_max - logits_min)
    return logits

def visualize(image, heatmaps, alpha=0.6, text_prompts: List=None, save_path: Optional=None):
    W, H = heatmaps.shape[-2:]
    if isinstance(image, Image.Image):
        image = image.resize((W, H))
    elif isinstance(image, torch.Tensor):
        if image.ndim > 3:
            image = image.squeeze(0)
        image_unormed = (image.detach().cpu() * torch.Tensor(OPENAI_DATASET_STD)[:, None, None]) \
                        + torch.Tensor(OPENAI_DATASET_MEAN)[:, None, None]  # undo the normalization
        image = Image.fromarray((image_unormed.permute(1, 2, 0).numpy() * 255).astype('uint8'))  # convert to PIL
    else:
        raise f'image should be either of type PIL.Image.Image or torch.Tensor but found {type(image)}'
    if text_prompts is None:
        text_prompts = [p for p in range(heatmaps.shape[0])]

    # plot image
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if heatmaps.ndim > 3:
        heatmaps = heatmaps.squeeze(0)
    heatmaps = heatmaps.detach().cpu().numpy()

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    heatmaps = (heatmaps * 255).astype('uint8')
    heat_maps = [cv2.applyColorMap(logit, cv2.COLORMAP_JET) for logit in heatmaps]

    vizs = [(1 - alpha) * img_cv + alpha * heat_map for heat_map in heat_maps]
    for i, viz in enumerate(vizs):
        viz = cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(viz)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            #plt.savefig(f'heatmap_{text_prompts[i]}.png')
            plt.savefig(f'/home/jwiers/CogVLM/LeGrad/outputs/first_image.png')


def list_pretrained():
    openclip_list_ = open_clip.list_pretrained()
    filtered_list = [(model_name, pretrained) for (model_name, pretrained) in openclip_list_ if model_name]
    unsupported_models = ['RN', 'convnext']  # legrad doesn't support CNN-based VLMs (for the moment)
    _str = ": ".join(['model_name' + " " * (25 - len('model_name')), 'pretrained']) + "\n"  # for nice display
    for (model_name, pretrained) in openclip_list_:
        for unsup_model in unsupported_models:
            if unsup_model in model_name:
                skip = True
                break
            else:
                skip = False
        if not skip:
            filtered_list.append((model_name, pretrained))
            _str += ": ".join([model_name + " " * (25 - len(model_name)), pretrained]) + "\n"  # for nice display

    print(_str)
    return filtered_list


if __name__ == '__main__':
    list_pretrained()