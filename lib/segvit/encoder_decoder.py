from typing import Any, Optional, List
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder as BaseEncoderDecoder
from mmengine.config import Config
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from fvcore.nn import FlopCountAnalysis
from mmseg.structures import SegDataSample
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from lib.segvit.vit_prune import ViT_prune
from lib.segvit.prune_head import PruneHead
from lib.segvit.atm_loss import ATMLoss


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


@MODELS.register_module()
class EncoderDecoderPrune(BaseEncoderDecoder):
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs, self)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   target: Tensor) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, target,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      target: Tensor) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, target, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, target,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def parse_losses(self, losses) -> Tensor:
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        sum_loss = sum(value for key, value in log_vars if 'loss' in key)
        return sum_loss

    def loss(self, inputs: Tensor, target: Tensor) -> dict:
        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, target)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, target)
            losses.update(loss_aux)

        sum_loss = self.parse_losses(losses)
        return sum_loss

    def predict(self, inputs: Tensor) -> Tensor:
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head(x)
        return seg_logits

    def forward(self,
                inputs: Tensor,
                target: Tensor = None,
                mode: str = 'tensor') -> ForwardResults:
        if self.training:
            return self.loss(inputs, target)
        else:
            return self.predict(inputs)


def main():
    cfg = Config.fromfile('/data/gyh/Dynamic/TokenSparse-for-MedSeg/lib/segvit/config_base_segvit_btcv.py')
    model = MODELS.build(cfg['model'])
    model = model.cuda()
    x = torch.rand(2, 1, 96, 96, 96).cuda()
    target = torch.randint(low=0, high=10, size=(2, 1, 96, 96, 96)).cuda()
    out = model(x, target, mode='loss')
    print(out)

if __name__ == '__main__':
    main()