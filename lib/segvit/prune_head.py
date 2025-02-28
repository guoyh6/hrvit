import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
# from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy
from .atm_head import *


def convert_true_idx(orig, new):
    assert (~orig).sum() == len(
        new), "batch_idx and new pos mismatch!!! orig:{}, new:{} ".format((~orig).sum(), len(new))
    orig_new = torch.zeros_like(orig)
    orig_new[~orig] = new
    return orig_new


@HEADS.register_module()
class PruneHead(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            patch_size,
            in_channels,
            layers_per_decoder=2,
            num_heads=6,
            thresh=1.0,
            **kwargs,
    ):
        super(PruneHead, self).__init__(
            in_channels=in_channels, **kwargs)
        self.thresh = thresh
        self.image_size = img_size
        self.patch_size = patch_size
        nhead = num_heads
        dim = self.channels
        proj = nn.Linear(self.in_channels, dim)
        trunc_normal_(proj.weight, std=.02)
        norm = nn.LayerNorm(dim)
        decoder_layer = TPN_DecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
        decoder = TPN_Decoder(decoder_layer, layers_per_decoder)

        self.input_proj = proj
        self.proj_norm = norm
        self.decoder = decoder
        self.q = nn.Embedding(self.num_classes, dim)

        self.class_embed = nn.Linear(dim, 1 + 1)
        delattr(self, 'conv_seg')

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def per_img_forward(self, q, x):
        x = self.proj_norm(self.input_proj(x))
        q, attn = self.decoder(q, x.transpose(0, 1))    # q:[N_cls, B, 512], attn:[B, N_cls, N_token], attn是Q@K没做softmax的注意力图,表示每个cls与Tokens的相关性.
        cls = self.class_embed(q.transpose(0, 1))   # [B, N_cls, 2]
        pred = cls.softmax(dim=-1)[..., :-1] * attn.sigmoid()   # [B, N_cls, N_token] 每个cls的存在概率 乘上 与每个Token的相关性
        return attn, cls, pred

    def forward(self, inputs, inference=False, canvas=None):
        if inference:
            # x = self._transform_inputs(inputs)
            x = inputs
            canvas_copy = canvas.clone()
            if x.dim() == 4:
                x = self.d4_to_d3(x)
            B, hw, ch = x.shape

            q = self.q.weight.repeat(B, 1, 1).transpose(0, 1) # q.shape [cls, b, ch]
            attn, cls, pred = self.per_img_forward(q, x) # x.shape [b, hw, ch]
            if torch.any(torch.isnan(x)):
                print("x10 has Nan")
            canvas_copy = attn
            self.results = {"attn": canvas_copy}
            self.results.update({"pred_logits": cls})
            self.results.update({"pred": pred})

            return canvas_copy

        else:   # 在计算loss以及推理出原图mask的时候，取出上面预测的结果来用
            pred = self.results["pred"]
            pred = self.d3_to_d5(pred.transpose(-1, -2))
            pred = F.interpolate(pred, size=(self.image_size, self.image_size, self.image_size),
                                 mode='trilinear', align_corners=False)

            out = {"pred": pred}
            if self.training:
                out["pred_logits"] = self.results["pred_logits"]
                out["pred_masks"] = self.d3_to_d5(
                    self.results["attn"].transpose(-1, -2))
                return out
            else:
                return out["pred"]

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def d3_to_d5(self, t):
        n, hw, c = t.size()
        len_patch = self.image_size // self.patch_size
        return t.transpose(1, 2).reshape(n, c, len_patch, len_patch, len_patch)

    def d5_to_d3(self, t):
        return t.flatten(2).transpose(-1, -2)

    def loss_by_feat(self, seg_logit, seg_label):
        # seg_label = self._stack_batch_gt(batch_data_samples)
        # atm loss
        seg_label = seg_label.squeeze(1)    # [B,H,W]
        loss = self.loss_decode(
            seg_logit,
            seg_label,
            ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(seg_logit["pred"], seg_label, ignore_index=self.ignore_index)
        return loss
