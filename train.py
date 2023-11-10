#!/usr/bin/env python3
"""
DAP: Generating Instance-level Prompts for Rehearsal-free Continual Learning (ICCV 2023 Oral)
main function
"""

import os
import torch
import warnings

import numpy as np
import random

from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser
warnings.filterwarnings("ignore")

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATA.NAME)

    if PathManager.exists(output_path):
        raise ValueError(f"Already run for {output_path}")

    PathManager.mkdirs(output_path)
    cfg.OUTPUT_DIR = output_path
    return cfg

def get_datasets(cfg):
    print("Loading training data...")
    train_dataset = data_loader._construct_dataset(cfg, split='train')
    print("Loading test data...")
    test_dataset = data_loader._construct_dataset(cfg, split='test')
    return train_dataset, test_dataset

def train(cfg):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    train_dataset, test_dataset = get_datasets(cfg)
    print("Constructing models...")
    model, cur_device = build_model(cfg)

    print("Setting up trainer...")
    trainer = Trainer(cfg, model, cur_device)
    trainer.train_classifier(train_dataset, test_dataset)

def main(args):
    cfg = setup(args)
    train(cfg)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
