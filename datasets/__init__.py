# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.utils.data
import torchvision
from .torchvision_datasets import LvisDetection

from .coco import build as build_coco
from .lvis import build as build_lvis
from .distributed_weighted_sampler import DistributedWeightedSampler


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    elif isinstance(dataset, LvisDetection):
        return dataset.lvis



def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == "lvis":
        return build_lvis(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
