#!/usr/bin/env python3
"""
ViT-related models
"""

import torch
import torch.nn as nn

from collections import OrderedDict
from torchvision import models

from .build_vit_backbone import build_vit_sup_models
from .mlp import MLP

class ViT(nn.Module):
    """ViT-related model."""
    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        self.froze_enc = True

        dap_cfg = cfg.MODEL.DAP

        self.build_backbone(
            cfg, dap_cfg, load_pretrain, vis=vis)
        self.cfg = cfg
        self.setup_head(cfg)

    def build_backbone(self, cfg, dap_cfg, load_pretrain, vis):
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, cfg.MODEL.MODEL_ROOT, dap_cfg, load_pretrain, cfg, vis, cfg.MODEL.TRANSFER_TYPE
        )

        for k, p in self.enc.named_parameters():
            if "dap" not in k:
                p.requires_grad = False

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES],
            special_bias=True
        )

    def forward(self, x, return_feature=False, task_id=None, is_train=None, cfg=None):
        if self.froze_enc and self.enc.training:
            self.enc.eval()

        x, reduce_sim, task_id_out = self.enc(x, task_id=task_id, is_train=is_train)

        if return_feature:
            return x, x
        x = self.head(x)

        if cfg.DATA.NAME == 'imagenet_r':
            # only for imagenet_r
            offset1 = task_id_out * cfg.CONTINUAL.INCREMENT
            offset2 = (task_id_out + 1) * cfg.CONTINUAL.INCREMENT
            if offset1 > 0:
                x[:, :offset1].data.fill_(-10e10)
            if offset2 < cfg.DATA.NUMBER_CLASSES:
                x[:, int(offset2):cfg.DATA.NUMBER_CLASSES].data.fill_(-10e10)

        return x, reduce_sim

    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x
