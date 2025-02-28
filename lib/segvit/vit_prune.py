from .vit import VisionTransformer
from mmseg.registry import MODELS
import torch
# torch.autograd.set_detect_anomaly(True)
from torch import Tensor
from typing import Optional
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class Refine_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, batch_idx: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        outputs = []
        for x_, b_id in zip(tgt, batch_idx):
            x_ = x_.unsqueeze(0)
            hard_tokens = x_[:, ~b_id].transpose(0, 1)
            easy_tokens = x_[:, b_id].transpose(0, 1).detach()
            x_new = torch.zeros_like(x_)
            for mod in self.layers:
                hard_tokens = mod(hard_tokens, easy_tokens, tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)

            if self.norm is not None:
                hard_tokens = self.norm(hard_tokens)
            x_new[:, b_id] = x_[:, b_id].detach()
            x_new[:, ~b_id] = hard_tokens.transpose(0, 1)
            outputs.append(x_new)
        return torch.cat(outputs, dim=0)


class Refine_DecoderLayer(TransformerDecoderLayer):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


@MODELS.register_module()
class ViT_prune(VisionTransformer):
    def __init__(self,
                num_classes,
                freeze=False,
                **kwargs,
                 ):
        super(ViT_prune, self).__init__(
            **kwargs,
        )
        self.num_classes = num_classes
        # decoder_layer = Refine_DecoderLayer(
        #     d_model=1024, nhead=8, dim_feedforward=2048)
        # self.refine_decoder = Refine_Decoder(decoder_layer, num_layers=3)

    def forward(self, inputs, model):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)  # 用(Patch_size*Patch_size)的卷积核来得到Tokens
        if torch.any(torch.isnan(x)):
            print("x00 has Nan")

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)  # x + self.pos_embed
        if torch.any(torch.isnan(x)):
            print("x01 has Nan")
        
        hwd = hw_shape[0]*hw_shape[1]*hw_shape[2]
        batch_idx = torch.zeros(
            (B, hwd), device=x.device) != 0      # [B, hw] 每个Token是否已经pruning，false表示还没被剪枝
        canvas = torch.zeros_like(batch_idx).unsqueeze(
            1).repeat(1, self.num_classes, 1).float()       # [B, Nc, hw] 每个cls与Tokens的相关性
        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            # total += (~batch_idx).sum()
            if batch_idx[:, -hwd:].sum() == 0:
                x = layer(x)
            else:
                x = layer(x, batch_idx)
            if torch.any(torch.isnan(x)):
                print(f"x111_{i} has Nan")

            if i in self.out_indices:
                idx = self.out_indices.index(i)
                if i != self.out_indices[-1]:
                    canvas = model.auxiliary_head[idx](x, inference=True, canvas=canvas)
                else:
                    canvas = model.decode_head(x, inference=True, canvas=canvas)
                    if self.final_norm:
                        x = self.norm1(x)

                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x

                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], hw_shape[2],
                                  C).permute(0, 4, 1, 2, 3).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

                # if i == self.out_indices[-2]:
                #     x = self._pos_embeding(x, hw_shape, self.pos_embed[:, 1:])
                #     x = self.refine_decoder(x, batch_idx)

        return tuple(outs)