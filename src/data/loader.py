#!/usr/bin/env python3
"""
data loader
"""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from src.data.tf_dataset import TFDataset

import numpy as np
from copy import deepcopy

def _construct_dataset(cfg, split):
    """Constructs the data loader for the given dataset."""
    # import the tensorflow here only if needed
    dataset = TFDataset(cfg, split)

    return dataset

def _construct_continual_loader(cfg, dataset, shuffle=False):
    sampler = None

    # In the case of ImageNet-R, we observed that runs with a sequential sampler show more consistent results.
    # When using a random sampler, the average performance across five runs is still ~70%; however, there is a variance in performance over runs.
    if cfg.DATA.NAME == 'imagenet_r':
        sampler = torch.utils.data.SequentialSampler(dataset)

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATA.BATCH_SIZE,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=False,
    )
    return loader

def _build_continual_dataset(cfg, dataset):
    prev_cls_increment = 0
    cls_increment = cfg.CONTINUAL.INITIAL
    scenario = []

    for i in range(cfg.CONTINUAL.N_TASKS):
        cls_less = np.where((np.asarray(dataset._targets) < cls_increment) & (prev_cls_increment <= np.asarray(dataset._targets)))[0]

        _labels = []
        _image_files = []
        for j in cls_less:
            _labels.append(dataset._targets[j])
            _image_files.append(dataset._image_tensor_list[j])

        cur_dataset = deepcopy(dataset)

        cur_dataset._targets = _labels
        cur_dataset._image_tensor_list = _image_files
        cur_dataset._class_ids = dataset._class_ids[prev_cls_increment:cls_increment]
        cur_dataset._class_ids_mask = dataset._class_ids_mask[prev_cls_increment:cls_increment]

        prev_cls_increment = cls_increment
        cls_increment += cfg.CONTINUAL.INCREMENT

        scenario.append(cur_dataset)

    return scenario

def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    # import the tensorflow here only if needed
    dataset = TFDataset(cfg, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_test_loader(cfg):
    """Test loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
