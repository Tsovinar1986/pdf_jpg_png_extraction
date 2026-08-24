"""Minimal, importable wrapper around the vendored CRAFT model — a clean
Python API (PIL image in, boxes out) instead of the original repo's
CLI/file-I/O-oriented test.py script. Core detection logic (resize,
normalize, forward pass, post-process, rescale) is unchanged from
test.py's test_net(), just restructured as a function with no argparse,
no file writing, and no skimage dependency.

Vendored from https://github.com/clovaai/CRAFT-pytorch (MIT License,
Copyright (c) 2019-present NAVER Corp.) — see LICENSE in this directory.
Patched for modern torchvision (basenet/vgg16_bn.py no longer depends on
the removed torchvision.models.vgg.model_urls) and Python 3.12.
"""
from collections import OrderedDict
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from . import craft_utils, imgproc
from .craft import CRAFT


def _copy_state_dict(state_dict: dict) -> OrderedDict:
    """Strip a "module." prefix left behind by DataParallel checkpoints."""
    start_idx = 1 if list(state_dict.keys())[0].startswith("module") else 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[".".join(k.split(".")[start_idx:])] = v
    return new_state_dict


def load_model(weights_path: str, device: str = "cpu") -> CRAFT:
    net = CRAFT()
    state_dict = torch.load(weights_path, map_location=device)
    net.load_state_dict(_copy_state_dict(state_dict))
    net.to(device)
    net.eval()
    return net


def detect_boxes(
    net: CRAFT,
    image: "Image.Image",
    device: str = "cpu",
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
    canvas_size: int = 1280,
    mag_ratio: float = 1.5,
) -> List[Tuple[int, int, int, int]]:
    """Detect text regions in `image`, returned as (x0, y0, x1, y1) boxes
    in the image's own original pixel coordinates.
    """
    img = np.array(image.convert("RGB"))

    img_resized, target_ratio, _size_heatmap = imgproc.resize_aspect_ratio(
        img, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        y, _feature = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    boxes, _polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly=False
    )
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    out = []
    for box in boxes:
        xs = box[:, 0]
        ys = box[:, 1]
        out.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return out
