# coding=utf-8
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
load a built tf-dataset which can be downloaded https://drive.google.com/drive/folders/1bBqS8MuTQXUBV3DXJ_-YZyNOR4ejvm1O?usp=sharing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds

from src.data import base as base
from src.data.registry import Registry

TRAIN_SPLIT_PERCENT = 80
TEST_SPLIT_PERCENT = 20

@Registry.register("data", "class")
class Data(base.ImageTfdsData):
  def __init__(self, data_name=None, data_dir=None):
    dataset_name = data_name + ":*.*.*"
    dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
    dataset_builder.download_and_prepare()

    if data_name == "chestx" or data_name == "eurosat/rgb" or data_name == "isic" or data_name == "resisc45":
        num_examples = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
        train_count = num_examples * TRAIN_SPLIT_PERCENT // 100
        test_count = num_examples * TEST_SPLIT_PERCENT // 100

        tfds_splits = {
            "train":
                "train[:{}]".format(train_count),
            "test":
                "train[{}:]".format(train_count),
        }

        num_samples_splits = {
            "train": train_count,
            "test": test_count,
        }
    elif data_name == "imagenet_r":
        num_examples = dataset_builder.info.splits[tfds.Split.TEST].num_examples
        train_count = num_examples * TRAIN_SPLIT_PERCENT // 100
        test_count = num_examples * TEST_SPLIT_PERCENT // 100

        tfds_splits = {
            "train":
                "test[:{}]".format(train_count),
            "test":
                "test[{}:]".format(train_count),
        }

        num_samples_splits = {
            "train": train_count,
            "test": test_count,
        }
    else:
        train_count = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
        test_count = dataset_builder.info.splits[tfds.Split.TEST].num_examples

        num_samples_splits = {
            "train": train_count,
            "test": test_count,
        }

        tfds_splits = {
            "train": "train",
            "test": "test"
        }

    super(Data, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_preprocessing_threads=400,
        shuffle_buffer_size=20000,
        # Note: Rename tensors but keep their original types.
        base_preprocess_fn=base.make_get_and_cast_tensors_fn({
            "image": ("image", None),
            "label": ("label", None),
        }),
        image_key="image",
        num_channels=3,
        num_classes=dataset_builder.info.features["label"].num_classes)
