import math
import time
from functools import partial
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from timm.models.layers.helpers import to_3tuple
from timm.models.layers import PatchEmbed, DropPath, Mlp, trunc_normal_, lecun_normal_
from timm.models.helpers import named_apply
from timm.models.vision_transformer import Block as oldBlock

from .gumbel_tricks import gumbel_soft_topk, soft_topk, gen_dynamic_tokens
from .ms_deform_attn import MSDeformAttn

import pdb

def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6    # 768 //6 -> 128
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])  # [[1728,], [128,]] -> [1728, 128]: out_h[a,b] == grid_h.flatten()[a] * omega[b]
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)], dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed    # list[D,D,D] -> [1, D^3, C]

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def unshuffle(x_prune, x_keep, policy_list):
    x = torch.cat([x_prune, x_keep], dim=1)
    B, L, C = x.size()

    # compute unshuffle index
    shuffled_idx = []
    for policy in policy_list:
        Lv = int(policy.sum(dim=1)[0])
        vis_idx = torch.nonzero(policy, as_tuple=False).reshape(B, Lv, 2)[:, :, 1]  # [B, Lv]
        shuffled_idx.append(vis_idx)
    shuffled_idx = torch.cat(shuffled_idx, dim=1)  # [B, L]
    assert shuffled_idx.size(1) == L, "unmatched length of x and policy"

    # unshuffle
    unshuffle_idx = shuffled_idx.argsort(dim=1)
    x = x.gather(dim=1, index=unshuffle_idx[:, :, None].expand(-1, -1, C))
    return x

class MlpPredictor(nn.Module):
    """ Mlp Predictor
    """
    def __init__(self, embed_dim, logsoftmax=False):
        super().__init__()
        self.logsoftmax = logsoftmax
        self.in_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, x):
        x = self.in_mlp(x[:, 1:])
        B, L, C = x.size()
        local_x = x[:, :, :C // 2]
        global_x = x[:, :, C // 2:].mean(dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, L, C//2)], dim=-1)
        x = self.out_mlp(x)
        if self.logsoftmax:
            return F.logsigmoid(x.squeeze(dim=-1))    # 之前是这个
            # return torch.sigmoid(x.squeeze(dim=-1))
        else:
            return x.squeeze(dim=-1)

class AttPredictor(nn.Module):
    """ Att Predictor
    """
    def __init__(self, embed_dim, dim_reduce_ratio=6, logsoftmax=True):
        super().__init__()
        self.logsoftmax = logsoftmax
        att_dim = embed_dim // dim_reduce_ratio
        self.att_dim = att_dim
        self.scale = nn.Parameter(torch.ones(1) * att_dim ** -0.5)
        self.linear = nn.Linear(embed_dim, att_dim)

    def forward(self, x):
        x = self.linear(x) # of shape [B, Lx, att_dim]
        cls_token = x[:, 0:1] # of shape [B, 1, att_dim]
        patch_token = x[:, 1:] # of shape [B, L, att_dim]

        attn = (patch_token @ cls_token.transpose(-2, -1)) * self.scale
        if self.logsoftmax:
            return F.logsigmoid(attn.squeeze(dim=-1)) # of shape [B, L]
        else:
            return attn.squeeze(dim=-1) # of shape [B, L]

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple); 沿指定维度(0)进行切片

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Edge_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xkv):
        # xq = x[:, 0]
        B, N, C = xkv.shape
        # qk = self.qkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple); 沿指定维度(0)进行切片
        q = self.q(xq).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xkv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xkv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, torch.mean(attn_save, 1)

class Edge_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        # attns = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            # attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn

class Edge_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(Edge_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Edge_Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn2 = self.multihead_attn(tgt, memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # self.attn = nn.MultiheadAttention(dim, num_heads, attn_drop, batch_first=True)
        self.norm2 = norm_layer(dim)
        self.def_attn = MSDeformAttn()
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, ref=None, value=None):
        x_att = self.attn(self.norm1(x))
        # if value is not None:
        #     x_att = self.def_attn(self.norm2(x_att), ref, value)
        x = x + self.drop_path1(self.ls1(x_att))
        if value is not None:
            x_att = self.def_attn(self.norm2(x), ref, value)
            x = x + self.drop_path2(self.ls2(x_att))
        x = x + self.drop_path3(self.ls3(self.mlp(self.norm3(x))))
        return x


class Deformable_Encoder(nn.Module):

    def __init__(
            self, dim, mlp_ratio=4., drop=0., init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim)
        self.norm2 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, ref, value):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), ref, value)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MaskCompletion(nn.Module):
    def __init__(self, embed_dim, grid_size):
        super().__init__()
        grid_size = to_3tuple(grid_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                            embed_dim,
                                                            num_tokens=1)

    def unshuffle(self, x, mask_tokens, policy):
        B, Lv, C = x.size()
        Ln = mask_tokens.size(1)
        vis_idx = torch.nonzero(policy, as_tuple=False).reshape(B, Lv, 2)[:, :, 1] # [B, Lv]
        nonvis_idx = torch.nonzero(1 - policy, as_tuple=False).reshape(B, Ln, 2)[:, :, 1] # [B, Ln]

        concat_idx = torch.cat([vis_idx, nonvis_idx], dim=1) # [B, Lv+Ln]
        concat_x = torch.cat([x, mask_tokens], dim=1) # [B, Lm+Lq, C]

        unshuffle_idx = concat_idx.argsort(dim=1)
        concat_x = concat_x.gather(dim=1, index=unshuffle_idx[:, :, None].expand(-1, -1, C))
        return concat_x

    def forward(self, x, policy):
        # compute the length of visible tokens and non-visible tokens
        B, Lv, _ = x.size()
        L = policy.size(1)
        assert torch.all(policy.sum(dim=1) == Lv)
        Ln = L - Lv

        # unshuffle
        x = self.unshuffle(x, self.mask_token.expand(B, Ln, -1), policy)
        x = x + self.pos_embed

        # import pdb
        # pdb.set_trace()

        return x

class ConcatUnshuffle(nn.Module):
    def __init__(self, embed_dim, grid_size, num_layers):
        super().__init__()
        grid_size = to_3tuple(grid_size)
        self.layer_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, embed_dim)) for _ in range(num_layers)])
        if len(grid_size) == 3:
            self.pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                embed_dim,
                                                                num_tokens=1) # [1, L, C]
        elif len(grid_size) == 2:
            raise NotImplementedError("Do not support 2D task yet")
            # self.pos_embed = build_2d_sincos_position_embedding(grid_size,
            #                                                     embed_dim,
            #                                                     num_tokens=1)
        else:
            raise NotImplementedError("Do not support dimensions of none 2 and 3 yet")

    def unshuffle(self, x, policy_list):
        B, L, C = x.size()

        # compute unshuffle index
        shuffled_idx = []
        for policy in policy_list:
            Lv = int(policy.sum(dim=1)[0])
            vis_idx = torch.nonzero(policy, as_tuple=False).reshape(B, Lv, 2)[:, :, 1] # [B, Lv]
            shuffled_idx.append(vis_idx)
        shuffled_idx = torch.cat(shuffled_idx, dim=1) # [B, L]
        # shuffled_idx = torch.cat(policy_list, dim=1)
        # pdb.set_trace()
        assert shuffled_idx.size(1) == L, "unmatched length of x and policy"

        # unshuffle
        unshuffle_idx = shuffled_idx.argsort(dim=1)
        x = x.gather(dim=1, index=unshuffle_idx[:, :, None].expand(-1, -1, C))
        return x

    def forward(self, x_list, policy_list):
        # create layer tokens by expanding to match the shape
        layer_token_list = []
        for i, x in enumerate(x_list):
            B, Lv, _ = x.size()
            layer_token_list.append(self.layer_tokens[i].expand(B, Lv, -1))
        x = torch.cat(x_list, dim=1) # [B, L, C]
        layer_token = torch.cat(layer_token_list, dim=1) # [B, L, C]
        x += layer_token

        # unshuffle
        x = self.unshuffle(x, policy_list)

        # add positional embedding
        x = x + self.pos_embed

        return x

class CompletionNet(nn.Module):
    """ Completion net version2: MAE decoder style
    """
    def __init__(self, grid_size, embed_dim, enc_dim, depth, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.in_norm = norm_layer(enc_dim)
        self.in_linear = nn.Linear(enc_dim, embed_dim)

        self.mask_complection = MaskCompletion(embed_dim, grid_size)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads,
                qkv_bias=qkv_bias,
                act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(depth)])

        self.out_norm = norm_layer(embed_dim)
        self.out_linear = nn.Linear(embed_dim, enc_dim)

    def forward(self, x, policy):
        if isinstance(x, (tuple, list)):
            x = x[-1]
        if isinstance(policy, (tuple, list)):
            policy = policy[-1]
        x = self.in_linear(self.in_norm(x)) # [B, Lv, C]
        x = self.mask_complection(x, policy) # [B, L, C]

        for blk in self.blocks:
            x = blk(x)
        x = self.out_linear(self.out_norm(x))
        return x[:, 1:]

class FusionNet(nn.Module):
    """ Fusion net version3: Fusion features from different layers
    """
    def __init__(self, grid_size, embed_dim, enc_dim, num_layers, depth, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.pre_linear_list = nn.ModuleList([nn.Linear(enc_dim, embed_dim) for _ in range(num_layers)])
        self.embed_dim = embed_dim
        self.post_linear = nn.Linear(embed_dim+1, embed_dim)
        self.score_linear = nn.Linear(1, embed_dim)
        self.score_norm = norm_layer(embed_dim)

        self.completion = ConcatUnshuffle(embed_dim, grid_size, num_layers)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads,
                qkv_bias=qkv_bias,
                act_layer=act_layer,
                norm_layer=norm_layer)
            for _ in range(depth)])

        self.out_norm = norm_layer(embed_dim)
        self.out_linear = nn.Linear(embed_dim, enc_dim)

    def forward(self, x_list, policy_list):
        post_x_list = []
        for x, pre_linear in zip(x_list, self.pre_linear_list):
            x = pre_linear(x)
            post_x_list.append(x)

        x = self.completion(post_x_list, policy_list) # [B, L, C]

        for blk in self.blocks:
            x = blk(x)
        x = self.out_linear(self.out_norm(x))
        return x[:, 1:]


class FusionNetV2(nn.Module):
    """ Fusion net version3: Fusion features from different layers
    """
    def __init__(self, grid_size, embed_dim, enc_dim, num_layers, depth, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.completion = ConcatUnshuffle(embed_dim, grid_size, num_layers)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads,
                qkv_bias=qkv_bias,
                act_layer=act_layer,
                norm_layer=norm_layer)
            for _ in range(depth)])

        self.out_norm = norm_layer(embed_dim)

    def forward(self, x_list, policy_list):
        x = self.completion(x_list, policy_list) # [B, L, C]
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        return x[:, 1:]

class VisionTransformerTokenPruningV1(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block,
            predictor=MlpPredictor, tp_loc=None, tp_ratio=None, tp_tau=1., distill=False, as_encoder=True, args=None):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.args = args

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.as_encoder = as_encoder
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.edge_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.edge_embed = nn.Linear(embed_dim, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.prune_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.policy = nn.Parameter(torch.ones(1, num_patches + self.num_tokens), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        # self.blks_enc = nn.ModuleList([
        #     block_fn(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(3)])
        # self.blks_sp = nn.ModuleList([
        #     block_fn(
        #         dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(9)])  # depth-tp_loc[0]-1
        # self.blks_def = nn.ModuleList([Deformable_Encoder(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
        #                  drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer) for i in range(3)])
        # self.trans = nn.Linear(embed_dim, 384)
        # self.trans_norm = norm_layer(384)
        # self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        # self.fusion_norm = norm_layer(embed_dim * 2)
        # self.token_predictor = nn.Linear(embed_dim, num_classes+1)
        self.cls_ref = nn.Parameter(torch.zeros(1, 1, 3))
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

        # score predictor for token prediction
        self.tp_loc = tp_loc
        self.tp_ratio = tp_ratio
        self.tp_tau = tp_tau
        self.distill = distill
        self.score_predictor_list = nn.ModuleList([
            predictor(embed_dim) for _ in range(len(tp_loc))
        ])
        # self.score_predictor = nn.Linear(embed_dim, 1)
        self.score_predictor = predictor(embed_dim)
        self.edge_attn = Edge_Attention(dim=embed_dim, num_heads=1, qkv_bias=qkv_bias, attn_drop=attn_drop_rate)
        decoder_layer = Edge_DecoderLayer(d_model=embed_dim, nhead=1, dim_feedforward=embed_dim * 4)
        self.edge_decoder = Edge_Decoder(decoder_layer, 3)
        self.mid_norm = norm_layer(embed_dim)
        self.mid_norm_list = nn.ModuleList([
            norm_layer(embed_dim) for _ in range(len(tp_loc))
        ]) if as_encoder else None

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def _reset_representation(self, representation_size):
        self.representation_size = representation_size
        if self.representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, self.representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.prune_embed, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token', 'prune_embed', 'edge_token'}

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None, representation_size=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        if representation_size is not None:
            self._reset_representation(representation_size)
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

    def get_num_layers(self):
        # return len(self.blocks)
        return 12

    def prune_token(self, x, policy, score, score_thred, time_meters=None):
        args = self.args
        B, Lx, _ = x.shape
        L = Lx - 1
        s_time = time.perf_counter()
        # pred_score = score_predictor(x[:, 1:]).reshape(B, L) # [B, L]. The [CLS] should be removed 用在Linear上
        # pred_score = score_predictor(x).reshape(B, L)
        # sigmoid_score = pred_score.sigmoid()
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters[f'{L}_scorepred'].append(time.perf_counter() - s_time)
        keep_k = int(L * (1 - self.tp_ratio))
        if self.training:
            # compute current policy
            if args.perturbation == 'gumbel':
                cur_policy = gumbel_soft_topk(score, keep_k, tau=self.tp_tau, hard=True) # cur_policy of size [B, L]
            elif args.perturbation == 'dynamic':
                # cur_policy = gen_dynamic_tokens(scores, args.score_thred=0.5, hard=True)
                score_hard = (score > score_thred).float()
                cur_policy = (score_hard - score).detach() + score
                keep_k = int(cur_policy.sum())
            elif args.perturbation == 'none':
                cur_policy = soft_topk(score, keep_k, tau=self.tp_tau, hard=True) # cur_policy of size [B, L]
            cur_policy = torch.cat([torch.ones(B, 1, device=cur_policy.device, dtype=cur_policy.dtype),
                                    cur_policy], dim=-1) # cur_policy of size [B, Lx=L+1]
            # pruning tokens of x according to cur_policy
            try:
                kept_x = x[cur_policy.bool()].reshape(B, keep_k + 1, -1) # x of size [B, keep_k+1, C]
            except:
                pdb.set_trace()
            # print(f"lenght of selected x is {x[cur_policy.bool()].shape}, reshape is [{B}, {keep_k + 1}]")
            if args.perturbation == 'dynamic':
                pruned_x = x[~cur_policy.bool()].unsqueeze(0)
            elif self.tp_ratio == 0:
                pruned_x = x[~cur_policy.bool()].unsqueeze(0).expand(B, -1, -1)
            else:
                pruned_x = x[~cur_policy.bool()].reshape(B, L - keep_k, -1) # x of size [B, L - keep_k, C]
            # update the global policy
            keep_policy, prune_policy = policy.clone(), policy.clone() # clone the policy
            keep_policy[keep_policy.bool()] = cur_policy.flatten() * keep_policy[keep_policy.bool()] # policy of size [B, N]
            prune_policy[prune_policy.bool()] = (1. - cur_policy.flatten()) * prune_policy[prune_policy.bool()] # pruned policy of size [B, N]
            # For backpropagating gradients
            # diff_keep_policy = keep_policy[keep_policy.bool()].reshape(B, keep_k + 1, 1)
            # diff_prune_policy = prune_policy[prune_policy.bool()].reshape(B, L - keep_k, 1)
            # kept_x = kept_x * diff_keep_policy
            # pruned_x = pruned_x * diff_prune_policy
        else:
            # current keep token indices
            if args.perturbation == 'dynamic':
                cur_policy = (score > score_thred).float()
                keep_k = int(cur_policy.sum())
            else:
                s_time = time.perf_counter()
                keep_noncls_idx = score.topk(keep_k, dim=-1)[1]
                if time_meters is not None:
                    torch.cuda.synchronize()
                    time_meters[f'{L}_topk'].append(time.perf_counter() - s_time)
                # build current policy
                s_time = time.perf_counter()
                cur_policy = torch.zeros_like(score).scatter_(-1, keep_noncls_idx, 1.0) # cur_policy of size [B, L]
                if time_meters is not None:
                    torch.cuda.synchronize()
                    time_meters[f'{L}_newtensor+scatter'].append(time.perf_counter() - s_time)
            s_time = time.perf_counter()
            cur_policy = torch.cat([torch.ones(B, 1, device=cur_policy.device, dtype=cur_policy.dtype),
                                    cur_policy], dim=-1) # cur_policy of size [B, Lx=L+1]
            if time_meters is not None:
                torch.cuda.synchronize()
                time_meters[f'{L}_cat'].append(time.perf_counter() - s_time)
            # pruning tokens of x according to cur_policy
            s_time = time.perf_counter()
            kept_x = x[cur_policy.bool()]
            pruned_x = x[~cur_policy.bool()]
            if time_meters is not None:
                torch.cuda.synchronize()
                time_meters[f'{L}_2idxsel'].append(time.perf_counter() - s_time)
            kept_x = kept_x.reshape(B, keep_k + 1, -1) # x of size [B, keep_k+1, 1]
            if args.perturbation == 'dynamic':
                pruned_x = pruned_x.unsqueeze(0)
            elif self.tp_ratio == 0:
                pruned_x = pruned_x.unsqueeze(0).expand(B, -1, -1)
            else:
                pruned_x = pruned_x.reshape(B, L - keep_k, -1) # x of size [B, L - keep_k, C]
            if time_meters is not None:
                torch.cuda.synchronize()
                time_meters[f'{L}_2reshape'].append(time.perf_counter() - s_time)
            # update the global policy
            s_time = time.perf_counter()
            keep_policy, prune_policy = policy.clone(), policy.clone() # clone the policy
            if time_meters is not None:
                torch.cuda.synchronize()
                time_meters[f'{L}_2clone'].append(time.perf_counter() - s_time)
            s_time = time.perf_counter()
            keep_policy[keep_policy.bool()] = cur_policy.flatten() * keep_policy[keep_policy.bool()] # policy of size [B, N]
            prune_policy[prune_policy.bool()] = (1. - cur_policy.flatten()) * prune_policy[prune_policy.bool()] # pruned policy of size [B, N]
            if time_meters is not None:
                torch.cuda.synchronize()
                time_meters[f'{L}_2lastidxset'].append(time.perf_counter() - s_time)
        return kept_x, pruned_x, keep_policy, prune_policy

    def forward_features(self, x, time_meters=None):
        args = self.args
        s_time = time.perf_counter()
        x = self.patch_embed(x)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters[f'patchembed'].append(time.perf_counter() - s_time)
        B, L, C = x.shape
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        s_time = time.perf_counter()
        policy = self.policy.expand(B, -1) # [B, L + 1]
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters[f'expand_policy'].append(time.perf_counter() - s_time)
        policy_list = []
        token_list = []
        accum_block_time = 0
        accum_tp_time = 0
        score_thred = [float(thred) for thred in args.score_thred.split('-')] if isinstance(args.score_thred, str) else [args.score_thred]
        prune_embed = self.prune_embed.expand(B, -1, -1)

        for i, blk in enumerate(self.blocks):
            s_time = time.perf_counter()
            x = blk(x)
            if time_meters is not None:
                torch.cuda.synchronize()
                duration = time.perf_counter() - s_time
                time_meters[f'{i}_block'].append(duration)
                accum_block_time += duration
            # print(f"num tokens after layer {i+1} is {x.size(1)}")
            if i in self.tp_loc:
                s_time = time.perf_counter()
                j = self.tp_loc.index(i)
                # score_predictor = self.score_predictor_list[j] if not args.share_pred else self.score_predictor_list[0]
                if j == 0:
                    pred_score = self.score_predictor(x).reshape(B, L)
                    # q = self.edge_token.expand(B, -1, -1)
                    # q = x[:, :1]
                    # q, attn = self.edge_decoder(q, x[:, 1:].transpose(0, 1))  # self.score_predictor(x).reshape(B, L)
                    # x[:, :1] = q
                    # cof = self.edge_embed(q)
                    # pred_score = (cof * attn).reshape(B, -1)
                    # pred_score = attn.reshape(B, -1)
                    score = pred_score.sigmoid()
                x, pruned_x, policy, prune_policy = self.prune_token(x, policy, score[policy[:, 1:].bool()].reshape(B, -1), score_thred[j], time_meters=time_meters)   # policy:值为1的表示保留，值为0的表示剪枝；prune_policy相反。
                x = self.pos_drop(x + prune_embed[policy.bool()].reshape(B, -1, C))
                # pos_embed = pos_embed[policy.bool()].reshape(B, -1, C)
                token_list.append(self.mid_norm_list[j](pruned_x))
                policy_list.append(prune_policy.detach())
                if time_meters is not None:
                    torch.cuda.synchronize()
                    duration = time.perf_counter() - s_time
                    time_meters[f'{j}_tp'].append(duration)
                    accum_tp_time += duration
        token_list.append(self.norm(x))
        policy_list.append(policy.detach())
        if time_meters is not None:
            time_meters['accum_blocks'].append(accum_block_time)
            time_meters['accum_tokenprune'].append(accum_tp_time)

        return token_list, policy_list, pred_score

    def forward_deformable_attention(self, x, time_meters=None):
        args = self.args
        n = int(x.shape[-1] // 8)
        s_time = time.perf_counter()
        x = self.patch_embed(x)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters[f'patchembed'].append(time.perf_counter() - s_time)
        B, L, _ = x.shape
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        s_time = time.perf_counter()
        policy = self.policy.expand(B, -1)  # [B, L + 1]
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters[f'expand_policy'].append(time.perf_counter() - s_time)
        policy_list = []
        token_list = []
        accum_block_time = 0
        accum_tp_time = 0
        ref_z, ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, n - 0.5, n, dtype=torch.float32, device=x.device),
                                             torch.linspace(0.5, n - 0.5, n, dtype=torch.float32, device=x.device),
                                             torch.linspace(0.5, n - 0.5, n, dtype=torch.float32, device=x.device))
        ref = (torch.stack((ref_x.reshape(-1), ref_y.reshape(-1), ref_z.reshape(-1)), -1)[None].repeat(B, 1, 1)) / n
        ref = torch.cat((self.cls_ref.expand(B, -1, -1), ref), dim=1)

        for i, blk in enumerate(self.blks_enc):
            s_time = time.perf_counter()
            x = blk(x)
            if time_meters is not None:
                torch.cuda.synchronize()
                duration = time.perf_counter() - s_time
                time_meters[f'{i}_block_enc'].append(duration)
                accum_block_time += duration

        # x_all = self.trans_norm(self.trans(x[:, 1:]))
        x_all = x[:, 1:]
        # score_predictor = self.score_predictor_list[0]
        score_predictor = self.score_predictor
        x, pruned_x, policy, prune_policy, pred_s = self.prune_token(x, policy, score_predictor, time_meters=time_meters)  # policy:值为1的表示保留，值为0的表示剪枝；prune_policy相反。
        token_list.append(self.mid_norm_list[0](pruned_x))
        policy_list.append(prune_policy.detach())

        ref = ref[policy.bool()].reshape(B, -1, 3)
        x_def = x
        # x, x_def = torch.split(x, [384, 384], -1)
        # for i in range(self.tp_loc[0]+1, len(self.blocks)):
        for i, blk in enumerate(self.blks_sp):
            s_time = time.perf_counter()
            x = blk(x)
            if time_meters is not None:
                torch.cuda.synchronize()
                duration = time.perf_counter() - s_time
                time_meters[f'{i}_block_sp'].append(duration)
                accum_block_time += duration
        # x_def = x
        for i, blk in enumerate(self.blks_def):
            s_time = time.perf_counter()
            x_def = blk(x_def, ref, x_all)
            if time_meters is not None:
                torch.cuda.synchronize()
                duration = time.perf_counter() - s_time
                time_meters[f'{i}_block_def'].append(duration)
                accum_block_time += duration
        # x = self.fusion(self.fusion_norm(torch.cat((x, x_def), -1)))
        token_list.append(self.norm(x+x_def))
        policy_list.append(policy.detach())
        if time_meters is not None:
            time_meters['accum_blocks'].append(accum_block_time)
            time_meters['accum_tokenprune'].append(accum_tp_time)

        return token_list, policy_list, pred_s

    def forward_sef_def_attention(self, x, time_meters=None):
        args = self.args
        n = int(x.shape[-1] // 8)
        s_time = time.perf_counter()
        x = self.patch_embed(x)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters[f'patchembed'].append(time.perf_counter() - s_time)
        B, L, C = x.shape
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        s_time = time.perf_counter()
        policy = self.policy.expand(B, -1) # [B, L + 1]
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters[f'expand_policy'].append(time.perf_counter() - s_time)
        policy_list = []
        token_list = []
        accum_block_time = 0
        accum_tp_time = 0
        score_thred = [float(thred) for thred in args.score_thred.split('-')] if isinstance(args.score_thred, str) else [args.score_thred]
        prune_embed = self.prune_embed.expand(B, -1, -1)

        ref_z, ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, n - 0.5, n, dtype=torch.float32, device=x.device),
                                             torch.linspace(0.5, n - 0.5, n, dtype=torch.float32, device=x.device),
                                             torch.linspace(0.5, n - 0.5, n, dtype=torch.float32, device=x.device))
        ref = (torch.stack((ref_x.reshape(-1), ref_y.reshape(-1), ref_z.reshape(-1)), -1)[None].repeat(B, 1, 1)) / n
        ref = torch.cat((self.cls_ref.expand(B, -1, -1), ref), dim=1)

        halting_layer = self.tp_loc[0] + 1
        j = 0
        for i, blk in enumerate(self.blocks[:halting_layer]):
            x = blk(x)
        x_all = x[:, 1:]

        s_time = time.perf_counter()
        pred_score = self.score_predictor(x).reshape(B, L)
        score = pred_score.sigmoid()
        x, pruned_x, policy, prune_policy = self.prune_token(x, policy, score[policy[:, 1:].bool()].reshape(B, -1), score_thred[j], time_meters=time_meters)   # policy:值为1的表示保留，值为0的表示剪枝；prune_policy相反。
        x = self.pos_drop(x + prune_embed[policy.bool()].reshape(B, -1, C))
        token_list.append(self.mid_norm_list[j](pruned_x))
        policy_list.append(prune_policy.detach())
        if time_meters is not None:
            torch.cuda.synchronize()
            duration = time.perf_counter() - s_time
            time_meters[f'{j}_tp'].append(duration)
            accum_tp_time += duration

        ref = ref[policy.bool()].reshape(B, -1, 3)
        for i, blk in enumerate(self.blocks[halting_layer:]):
            x = blk(x, ref, x_all)

        token_list.append(self.norm(x))
        policy_list.append(policy.detach())
        if time_meters is not None:
            time_meters['accum_blocks'].append(accum_block_time)
            time_meters['accum_tokenprune'].append(accum_tp_time)

        return token_list, policy_list, pred_score

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, time_meters=None):
        token_list, policy_list, scores = self.forward_sef_def_attention(x, time_meters=time_meters)
        if self.as_encoder:
            return token_list, policy_list, scores
        else:
            x = self.forward_head(token_list[-1])
            return x

# class VisionTransformerTokenPruningV2(nn.Module):
#     """ Vision Transformer
#     A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
#         - https://arxiv.org/abs/2010.11929
#     """

#     def __init__(
#             self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
#             embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
#             drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
#             embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block,
#             predictor=MlpPredictor, tp_loc=None, tp_ratio=None, tp_tau=1., distill=False, as_encoder=True, args=None):
#         super().__init__()
#         assert global_pool in ('', 'avg', 'token')
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         act_layer = act_layer or nn.GELU
#         self.args = args

#         self.num_classes = num_classes
#         self.global_pool = global_pool
#         self.as_encoder = as_encoder
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_tokens = 1

#         self.patch_embed = embed_layer(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
#         self.policy = nn.Parameter(torch.tensor(range(num_patches + self.num_tokens)), requires_grad=False)
#         # self.policy.requires_grad = False
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         self.blocks = nn.ModuleList([
#             block_fn(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(depth)])
#         use_fc_norm = self.global_pool == 'avg'
#         self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

#         # Representation layer. Used for original ViT models w/ in21k pretraining.
#         self.representation_size = representation_size
#         self.pre_logits = nn.Identity()
#         if representation_size:
#             self._reset_representation(representation_size)

#         # Classifier Head
#         self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
#         final_chs = self.representation_size if self.representation_size else self.embed_dim
#         self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

#         # score predictor for token prediction
#         self.tp_loc = tp_loc
#         self.tp_ratio = tp_ratio
#         self.tp_tau = tp_tau
#         self.distill = distill
#         self.score_predictor_list = nn.ModuleList([
#             predictor(embed_dim) for _ in range(len(tp_loc))
#         ])
#         self.mid_norm_list = nn.ModuleList([
#             norm_layer(embed_dim) for _ in range(len(tp_loc))
#         ]) if as_encoder else None

#         if weight_init != 'skip':
#             self.init_weights(weight_init)

#     def _reset_representation(self, representation_size):
#         self.representation_size = representation_size
#         if self.representation_size:
#             self.pre_logits = nn.Sequential(OrderedDict([
#                 ('fc', nn.Linear(self.embed_dim, self.representation_size)),
#                 ('act', nn.Tanh())
#             ]))
#         else:
#             self.pre_logits = nn.Identity()

#     def init_weights(self, mode=''):
#         assert mode in ('jax', 'jax_nlhb', 'moco', '')
#         head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
#         trunc_normal_(self.pos_embed, std=.02)
#         nn.init.normal_(self.cls_token, std=1e-6)
#         named_apply(get_init_weights_vit(mode, head_bias), self)

#     def _init_weights(self, m):
#         # this fn left here for compat with downstream users
#         init_weights_vit_timm(m)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token', 'dist_token'}

#     @torch.jit.ignore
#     def get_classifier(self):
#         return self.head

#     def reset_classifier(self, num_classes: int, global_pool=None, representation_size=None):
#         self.num_classes = num_classes
#         if global_pool is not None:
#             assert global_pool in ('', 'avg', 'token')
#             self.global_pool = global_pool
#         if representation_size is not None:
#             self._reset_representation(representation_size)
#         final_chs = self.representation_size if self.representation_size else self.embed_dim
#         self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

#     def get_num_layers(self):
#         return len(self.blocks)

#     def prune_token(self, x, score_predictor, global_kept_index=None, time_meters=None):
#         B, Lx, C = x.shape
#         L = Lx - 1
#         pred_score = score_predictor(x).reshape(B, L) # [B, L]. The [CLS] should be removed
#         keep_k = int(L * (1 - self.tp_ratio))
#         if self.training:
#             raise NotImplementedError("training phase for v2diff pruning is not yet implemented")
#             # # compute current policy
#             # cur_policy = gumbel_soft_topk(pred_score, keep_k, tau=self.tp_tau, hard=True) # cur_policy of size [B, L]
#             # cur_policy = torch.cat([torch.ones(B, 1, device=cur_policy.device, dtype=cur_policy.dtype),
#             #                         cur_policy], dim=-1) # cur_policy of size [B, Lx=L+1]
#             # # pruning tokens of x according to cur_policy
#             # kept_x = x[cur_policy.bool()].reshape(B, keep_k + 1, -1) # x of size [B, keep_k+1, C]
#             # pruned_x = x[~cur_policy.bool()].reshape(B, L - keep_k, -1) # x of size [B, L - keep_k, C]
#             # # update the global policy
#             # keep_policy, prune_policy = policy.clone(), policy.clone() # clone the policy
#             # keep_policy[keep_policy.bool()] = cur_policy.flatten() * keep_policy[keep_policy.bool()] # policy of size [B, N]
#             # prune_policy[prune_policy.bool()] = (1. - cur_policy.flatten()) * prune_policy[prune_policy.bool()] # pruned policy of size [B, N]
#             # # For backpropagating gradients
#             # diff_keep_policy = keep_policy[keep_policy.bool()].reshape(B, keep_k + 1, 1)
#             # diff_prune_policy = prune_policy[prune_policy.bool()].reshape(B, L - keep_k, 1)
#             # kept_x = kept_x * diff_keep_policy
#             # pruned_x = pruned_x * diff_prune_policy
#         else:
#             # current keep token indices
#             s_time = time.perf_counter()
#             sort_index = pred_score.sort(descending=True)[1]
#             keep_noncls_idx = sort_index[:, :keep_k] + 1
#             # keep_noncls_idx = pred_score.topk(keep_k, dim=-1)[1] + 1 # [B, keep_k]
#             kept_index = torch.cat([torch.zeros(B, 1, dtype=keep_noncls_idx.dtype, device=keep_noncls_idx.device),
#                                   keep_noncls_idx],
#                                   dim=1)
#             # pruned_index = (-pred_score).topk(L - keep_k, dim=-1)[1] + 1
#             pruned_index = sort_index[:, keep_k:] + 1
#             # pdb.set_trace()
#             if time_meters is not None:
#                 torch.cuda.synchronize()
#                 time_meters[f'{L}_2topk'].append(time.perf_counter() - s_time)
#             # build current policy
#             # cur_policy = torch.zeros_like(pred_score).scatter_(-1, keep_noncls_idx, 1.0) # cur_policy of size [B, L]
#             # s_time = time.perf_counter()
#             # cur_policy = torch.cat([torch.ones(B, 1, device=cur_policy.device, dtype=cur_policy.dtype),
#             #                         cur_policy], dim=-1) # cur_policy of size [B, Lx=L+1]
#             # if time_meters is not None:
#             #     time_meters[f'{L}_cat'].append(time.perf_counter() - s_time)
#             # pruning tokens of x according to cur_policy
#             # s_time = time.perf_counter()
#             # kept_index = (cur_policy.bool()).nonzero(as_tuple=False)[:, 1].view(B, keep_k + 1)
#             # pruned_index = ~(cur_policy.bool()).nonzero(as_tuple=False)[:, 1].view(B, L - keep_k)
#             # if time_meters is not None:
#             #     time_meters[f'{L}_2getidx'].append(time.perf_counter() - s_time)
#             s_time = time.perf_counter()
#             kept_x = torch.gather(x, dim=1, index=kept_index[:, :, None].expand(-1, -1, C))
#             pruned_x = torch.gather(x, dim=1, index=pruned_index[:, :, None].expand(-1, -1, C))
#             # pdb.set_trace()
#             # kept_x = x[cur_policy.bool()].reshape(B, keep_k + 1, -1) # x of size [B, keep_k+1, 1]
#             # pruned_x = x[~cur_policy.bool()].reshape(B, L - keep_k, -1) # x of size [B, L - keep_k, C]
#             if time_meters is not None:
#                 torch.cuda.synchronize()
#                 time_meters[f'{L}_2gather_x'].append(time.perf_counter() - s_time)
#             # update the global policy
#             s_time = time.perf_counter()
#             if global_kept_index is None:
#                 global_kept_index, global_pruned_index = kept_index, pruned_index
#             else:
#                 global_kept_index, global_pruned_index = torch.gather(global_kept_index, dim=1, index=kept_index), \
#                     torch.gather(global_kept_index, dim=1, index=pruned_index)
#             # keep_policy, prune_policy = policy.clone(), policy.clone() # clone the policy
#             # keep_policy[keep_policy.bool()] = cur_policy.flatten() * keep_policy[keep_policy.bool()] # policy of size [B, N]
#             # prune_policy[prune_policy.bool()] = (1. - cur_policy.flatten()) * prune_policy[prune_policy.bool()] # pruned policy of size [B, N]
#             if time_meters is not None:
#                 torch.cuda.synchronize()
#                 time_meters[f'{L}_2gather_index'].append(time.perf_counter() - s_time)
#             # pdb.set_trace()
#         return kept_x, pruned_x, global_kept_index, global_pruned_index

#     def forward_features(self, x, time_meters=None):
#         args = self.args
#         s_time = time.perf_counter()
#         x = self.patch_embed(x)
#         if time_meters is not None:
#             torch.cuda.synchronize()
#             time_meters['patchembed'].append(time.perf_counter() - s_time)
#         B, L, _ = x.shape

#         s_time = time.perf_counter()
#         x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
#         x = self.pos_drop(x + self.pos_embed)
#         if time_meters is not None:
#             torch.cuda.synchronize()
#             time_meters['cls+pe'].append(time.perf_counter() - s_time)

#         # policy = torch.ones(B, L + 1, dtype=x.dtype, device=x.device) # [B, L + 1]
#         s_time = time.perf_counter()
#         # policy = self.policy[None, :].expand(B, -1)
#         policy = None
#         if time_meters is not None:
#             torch.cuda.synchronize()
#             time_meters['init_policy'].append(time.perf_counter() - s_time)
#         # pdb.set_trace()
#         policy_list = []
#         token_list = []
#         for i, blk in enumerate(self.blocks):
#             s_time = time.perf_counter()
#             x = blk(x)
#             if time_meters is not None:
#                 torch.cuda.synchronize()
#                 time_meters[f'block{i+1}'].append(time.perf_counter() - s_time)
#             # print(f"num tokens after layer {i+1} is {x.size(1)}")
#             if i in self.tp_loc:
#                 s_time = time.perf_counter()
#                 j = self.tp_loc.index(i)
#                 score_predictor = self.score_predictor_list[j] if not args.share_pred else self.score_predictor_list[0]
#                 x, pruned_x, policy, prune_policy = self.prune_token(x, score_predictor, global_kept_index=policy, time_meters=None)
#                 token_list.append(self.mid_norm_list[j](pruned_x))
#                 policy_list.append(prune_policy.detach())
#                 if time_meters is not None:
#                     torch.cuda.synchronize()
#                     time_meters[f'{j}_tp'].append(time.perf_counter() - s_time)
#         token_list.append(self.norm(x))
#         policy_list.append(policy.detach())

#         return token_list, policy_list

#     def forward_head(self, x, pre_logits: bool = False):
#         if self.global_pool:
#             x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
#         x = self.fc_norm(x)
#         x = self.pre_logits(x)
#         return x if pre_logits else self.head(x)

#     def forward(self, x, time_meters=None):
#         token_list, policy_list = self.forward_features(x, time_meters=time_meters)
#         if self.as_encoder:
#             return token_list, policy_list
#         else:
#             x = self.forward_head(token_list[-1])
#             return x

if __name__ == "__main__":
    model = VisionTransformerTokenPruningV1(num_classes=0, global_pool='', tp_loc=[2, 5, 8], tp_ratio=0.25)
    input_tensor = torch.zeros(2, 3, 224, 224)
    # pdb.set_trace()
    output_tensor, policy_list = model(input_tensor)
    pdb.set_trace()