from functools import partial
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers.helpers import to_3tuple
from timm.models.layers import Mlp, DropPath

# from lib.networks.adavit import CompletionNet
import lib.networks as networks

from lib.models.unetr3d import PatchEmbed3D, PatchEmbed2P1D

import pdb

class AdaSeg3D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 encoder,
                 decoder,
                 args):
        super().__init__()
        self.args = args
        input_size = (args.roi_x, args.roi_y, args.roi_z)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        # build encoder
        tp_loc = [int(loc) for loc in args.tp_loc.split('-')]
        embed_layer = eval(args.patchembed)
        self.encoder = encoder(img_size=input_size,
                               patch_size=patch_size,
                               in_chans=args.in_chans,
                               num_classes=0,
                               global_pool='',
                               embed_dim=args.encoder_embed_dim,
                               depth=args.encoder_depth,
                               num_heads=args.encoder_num_heads,
                               drop_path_rate=args.drop_path,
                               embed_layer=embed_layer,
                               predictor=getattr(networks, args.score_predictor),
                               tp_loc=tp_loc,
                               tp_ratio=args.tp_ratio,
                               tp_tau=args.tp_tau,
                               args=args)
        if args.completion_net is None:
            completion_net = getattr(networks, 'CompletionNet')
        else:
            completion_net = getattr(networks, args.completion_net)
        self.completion_net = completion_net(grid_size=grid_size,
                                             embed_dim=args.compl_embed_dim,
                                             enc_dim=args.encoder_embed_dim,
                                             depth=args.compl_depth,
                                             num_heads=args.compl_num_heads,
                                             num_layers=len(tp_loc) + 1)
        # self.token_predictor = nn.Linear(args.compl_embed_dim, args.num_classes+1)
        self.decoder = decoder(in_channels=args.in_chans,
                               out_channels=args.num_classes,
                               img_size=input_size,
                               patch_size=patch_size,
                               feature_size=args.feature_size,
                               hidden_size=args.encoder_embed_dim,
                               spatial_dims=args.spatial_dim)

    def get_num_layers(self):
        return self.encoder.get_num_layers()

    def cal_true_edge_ratio(self, policy, target):
        with torch.no_grad():
            pooled_max = F.max_pool3d(target, (8, 8, 8))
            pooled_avg = F.avg_pool3d(target, (8, 8, 8))
            gt_score = (pooled_max != pooled_avg).float()

        gt_score = gt_score.reshape(1, -1)
        pred_score = 1 - policy[:, 1:]
        tp = torch.logical_and((gt_score == 1), (pred_score == 1))
        fp = torch.logical_and((gt_score == 0), (pred_score == 1))
        print("TP: {:.3f}".format((tp.sum() / gt_score.sum()).item()))
        print("FP: {:.3f}".format((fp.sum() / pred_score.sum()).item()))

    def dynamic_forward(self, x, target, return_score=False, vis_policy=False, time_meters=None):
        x_in = x
        x_list, scores_list, policy0_list, policy1_list = [], [], [], []
        for b, x_b in enumerate(x):
            # forward encoder
            s_time = time.perf_counter()
            token_list, policy_list, pred_s = self.encoder(x_b.unsqueeze(0), time_meters=time_meters)
            if time_meters is not None:
                torch.cuda.synchronize()  # CPU等待GPU计算完成
                time_meters['enc'].append(time.perf_counter() - s_time)
            if target is not None:
                self.cal_true_edge_ratio(policy_list[0], target[b])
            # forward completion and fusion
            # print(token_list[1].shape[1])
            x_b = self.completion_net(token_list, policy_list)
            x_list.append(x_b)
            scores_list.append(pred_s)
            policy0_list.append(policy_list[0])
            policy1_list.append(policy_list[1])

        x = torch.cat(x_list, dim=0)
        pred_score = torch.cat(scores_list, dim=0)
        policy0 = torch.cat(policy0_list, dim=0)
        policy1 = torch.cat(policy1_list, dim=0)
        # token_predict = self.token_predictor(x).permute(0, 2, 1).contiguous()
        # forward decoder
        x = self.decoder(x_in, x, [x, ] * self.args.encoder_depth)  # [B, gh*gw*gd, ph*pw*pd*C]

        if vis_policy:
            return x, [policy0, policy1]
        elif return_score:
            return x, pred_score  # , token_predict
        else:
            return x

    def forward(self, x, target=None, return_score=False, vis_policy=False, time_meters=None):
        if self.args.perturbation == 'dynamic':
            return self.dynamic_forward(x, target, return_score, vis_policy, time_meters)
        x_in = x    # [B, 1, 96, 96, 96]

        # forward encoder
        s_time = time.perf_counter()
        token_list, policy_list, pred_score = self.encoder(x, time_meters=time_meters)
        # pdb.set_trace()
        if time_meters is not None:
            torch.cuda.synchronize()    # CPU等待GPU计算完成
            time_meters['enc'].append(time.perf_counter() - s_time)

        # forward completion and fusion
        s_time = time.perf_counter()
        x = self.completion_net(token_list, policy_list)    # [B, N, C]
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['compl'].append(time.perf_counter() - s_time)

        # token_predict = self.token_predictor(x).permute(0, 2, 1).contiguous()

        # forward decoder
        s_time = time.perf_counter()
        x = self.decoder(x_in, x, [x, ] * self.args.encoder_depth)  # [B, Cls, 96, 96, 96]
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['dec'].append(time.perf_counter() - s_time)

        if vis_policy:
            return x, policy_list
        elif return_score:
            return x, pred_score  # , token_predict
        else:
            return x

# if __name__ == "__main__":
#     AdaSeg3D()