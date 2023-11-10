#!/usr/bin/env python3
"""
handle output of tf.data
"""

import functools
import tensorflow.compat.v1 as tf
import torch
import torch.utils.data
import numpy as np

from collections import Counter
from torch import Tensor

from src.data import tf_load
from src.data.registry import Registry

tf.config.experimental.set_visible_devices([], 'GPU')

class TFDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self.name = cfg.DATA.NAME

        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.get_data(cfg, split)

    def get_data(self, cfg, split):
        tf_data = build_tf_dataset(cfg, split)
        data_list = list(tf_data)

        self._image_tensor_list = [t[0].numpy().squeeze() for t in data_list]
        self._targets = [int(t[1].numpy()[0]) for t in data_list]
        
        if cfg.DATA.NAME == "imagenet_r":
            for index, target in enumerate(self._targets):
                self._targets[index] = IR_LABEL_MAP[target]
                
        self._class_ids = sorted(list(set(self._targets)))
        self._class_ids_mask = self._class_ids

        del data_list
        del tf_data

    def get_info(self):
        num_imgs = len(self._image_tensor_list)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return self.cfg.DATA.NUMBER_CLASSES

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        label = self._targets[index]
        im = to_torch_imgs(
            self._image_tensor_list[index], self.img_mean, self.img_std)

        sample = {
            "image": im,
            "label": label,
        }
        return sample

    def __len__(self):
        return len(self._targets)


def preprocess_fn(data, size=224, input_range=(0.0, 1.0)):
    image = data["image"]
    image = tf.image.resize(image, [size, size])

    image = tf.cast(image, tf.float32) / 255.0
    image = image * (input_range[1] - input_range[0]) + input_range[0]

    data["image"] = image
    return data


def build_tf_dataset(cfg, mode):
    """
    Builds a tf data instance, then transform to a list of tensors and labels
    """
    data_cls = Registry.lookup("data")
    vtab_tf_dataloader = data_cls(data_name=cfg.DATA.NAME, data_dir=cfg.DATA.DATAPATH)
    split_name_dict = {
        "dataset_train_split_name": 'train',
        "dataset_test_split_name": 'test'
    }

    def _dict_to_tuple(batch):
        return batch['image'], batch['label']

    return vtab_tf_dataloader.get_tf_data(
        batch_size=1,
        drop_remainder=False,
        split_name=split_name_dict[f"dataset_{mode}_split_name"],
        preprocess_fn=functools.partial(
            preprocess_fn,
            input_range=(0.0, 1.0),
            size=cfg.DATA.CROPSIZE,
            ),
        for_eval=mode != "train",
        shuffle_buffer_size=1000,
        prefetch=1,
        train_examples=None,
        epochs=1
    ).map(_dict_to_tuple)

def to_torch_imgs(img: np.ndarray, mean: Tensor, std: Tensor) -> Tensor:
    if len(img.shape) == 2:
        img = np.stack((img, img, img), axis=2)
    t_img: Tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    t_img -= mean
    t_img /= std

    return t_img
    

# mapping ImageNet-R, which labeled between 0-999, to 0-199 in a randomly shuffled order
IR_LABEL_MAP = {
    1: 162,
    2: 53,
    4: 4,
    6: 155,
    8: 54,
    9: 75,
    11: 79,
    13: 11,
    22: 46,
    23: 145,
    26: 186,
    29: 126,
    31: 29,
    39: 98,
    47: 84,
    63: 104,
    71: 39,
    76: 167,
    79: 7,
    84: 123,
    90: 139,
    94: 192,
    96: 19,
    97: 25,
    99: 164,
    100: 8,
    105: 47,
    107: 41,
    113: 67,
    122: 49,
    125: 35,
    130: 34,
    132: 69,
    144: 103,
    145: 187,
    147: 3,
    148: 22,
    150: 119,
    151: 60,
    155: 142,
    160: 38,
    161: 153,
    162: 9,
    163: 62,
    171: 90,
    172: 109,
    178: 72,
    187: 77,
    195: 0,
    199: 23,
    203: 146,
    207: 122,
    208: 94,
    219: 73,
    231: 16,
    232: 154,
    234: 45,
    235: 176,
    242: 17,
    245: 101,
    247: 143,
    250: 170,
    251: 78,
    254: 120,
    259: 59,
    260: 165,
    263: 86,
    265: 50,
    267: 51,
    269: 195,
    276: 64,
    277: 107,
    281: 111,
    288: 30,
    289: 156,
    291: 43,
    292: 114,
    293: 129,
    296: 74,
    299: 134,
    301: 68,
    308: 110,
    309: 42,
    310: 150,
    311: 161,
    314: 48,
    315: 1,
    319: 132,
    323: 121,
    327: 130,
    330: 85,
    334: 80,
    335: 108,
    337: 183,
    338: 116,
    340: 52,
    341: 168,
    344: 40,
    347: 97,
    353: 100,
    355: 21,
    361: 152,
    362: 157,
    365: 166,
    366: 180,
    367: 102,
    368: 131,
    372: 31,
    388: 44,
    390: 199,
    393: 174,
    397: 163,
    401: 196,
    407: 65,
    413: 6,
    414: 18,
    425: 135,
    428: 5,
    430: 33,
    435: 141,
    437: 99,
    441: 70,
    447: 13,
    448: 14,
    457: 149,
    462: 148,
    463: 198,
    469: 175,
    470: 136,
    471: 118,
    472: 125,
    476: 178,
    483: 159,
    487: 81,
    515: 71,
    546: 140,
    555: 179,
    558: 15,
    570: 158,
    579: 173,
    583: 127,
    587: 61,
    593: 128,
    594: 191,
    596: 193,
    609: 91,
    613: 106,
    617: 185,
    621: 147,
    629: 93,
    637: 105,
    657: 82,
    658: 144,
    701: 117,
    717: 92,
    724: 160,
    763: 32,
    768: 197,
    774: 76,
    776: 115,
    779: 112,
    780: 189,
    787: 184,
    805: 63,
    812: 95,
    815: 57,
    820: 177,
    824: 24,
    833: 27,
    847: 89,
    852: 96,
    866: 58,
    875: 194,
    883: 190,
    889: 55,
    895: 10,
    907: 137,
    928: 37,
    931: 124,
    932: 56,
    933: 66,
    934: 83,
    936: 138,
    937: 2,
    943: 171,
    945: 88,
    947: 188,
    948: 28,
    949: 151,
    951: 172,
    953: 36,
    954: 182,
    957: 169,
    963: 12,
    965: 26,
    967: 181,
    980: 20,
    981: 87,
    983: 113,
    988: 133
}
