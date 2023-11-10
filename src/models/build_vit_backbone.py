#!/usr/bin/env python3
"""
call vit with DAP
"""

import numpy as np
import os

from .vit_backbones.vit import VisionTransformer
from .vit_dap.vit import ADPT_VisionTransformer


MODEL_ZOO = {
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
}

def build_vit_sup_models(
    model_type, crop_size, model_root=None, dap_cfg=None, load_pretrain=True, cfg=None, vis=False, transfer_type=None
):
    m2featdim = {
        "sup_vitb16_imagenet21k": 768,
    }
    model = ADPT_VisionTransformer(model_type, crop_size, num_classes=-1, dap_cfg=dap_cfg, model_root=model_root, total_cfg=cfg)

    if load_pretrain:
        model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]

