#!/usr/bin/env python3
"""
launch helper functions
"""

import argparse
import os
import sys
import PIL
from collections import defaultdict
from tabulate import tabulate
from typing import Tuple

import torch

def collect_torch_env() -> str:
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()

def get_env_module() -> Tuple[str]:
    var_name = "ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def collect_env_info() -> str:
    data = []
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(get_env_module())
    data.append(("PyTorch", torch.__version__))
    data.append(("PyTorch Debug Build", torch.version.debug))

    has_cuda = torch.cuda.is_available()
    data.append(("CUDA available", has_cuda))
    if has_cuda:
        data.append(("CUDA ID", os.environ["CUDA_VISIBLE_DEVICES"]))
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))
    data.append(("Pillow", PIL.__version__))

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


def default_argument_parser():
    """
    create a simple parser to wrap around config file
    """
    parser = argparse.ArgumentParser(description="dap")
    parser.add_argument(
        "--config-file", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser

