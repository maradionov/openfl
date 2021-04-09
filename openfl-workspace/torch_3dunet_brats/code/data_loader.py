# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import os
from tqdm import tqdm
import urllib.request
from hashlib import sha384
from os import path
import shutil
import tempfile
import sys
import json
import tarfile

import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.data import CacheDataset
from monai.data import Dataset 
from monai.data import (load_decathlon_datalist, load_decathlon_properties)
from monai.transforms import Randomizable
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
)
from monai.utils import set_determinism

import torch
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union, Dict
from openfl.federated import FederatedModel, FederatedDataSet
from openfl.utilities import TensorKey
from openfl.federated import PyTorchDataLoader

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d

class BraTSDataset(CacheDataset):
    """
    This dataset contains brain tumor 3d images for one collaborator train or val.
    Args:
        collaborator_count: total number of collaborators
        collaborator_num: number of current collaborator
        is_validation: validation option
        transform: transform function
    """

    def __init__(self, is_validation, shard_num, collaborator_count, **kwargs):
        train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=[128, 128, 64], random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
        ]
        )
        val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AsChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
        )
        self.is_validation = is_validation
        dataset_dir = './data/Task01_BrainTumour/' 

        self.indices: np.ndarray = np.array([])
        if is_validation:
            transform = val_transform
        else:
            transform = train_transform 
        data = self._generate_data_list(dataset_dir)
        data= data[shard_num :: collaborator_count]
        self.is_validation = is_validation
        assert(len(data) > 8)
        validation_size = len(data) // 8
        if is_validation:
            data= data[-validation_size:]
        else:
            data= data[: -validation_size]
        data = data[:3]
        super().__init__(data, transform, cache_num=1, num_workers=4)


    def _generate_data_list(self, dataset_dir: str) -> List[Dict]:
        datalist = load_decathlon_datalist(os.path.join(dataset_dir, "dataset.json"), True, "training")
        return datalist
    
    def __getitem__(self, index):
        tmp = super().__getitem__(index)
        return (tmp['image'], tmp['label'])



def my_hook(t):
    """Reporthook for urlretrieve."""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


def load_brats_dataset():
    """Load and untar brats dataset."""
    TAR_SHA384 = '049f8e1425d9e47a4cdabe03c5c2ff68aa01b6298a307'\
        '304638abd9b1341f0639d015357ca315d402984bc1cffa16bbf'
    # data_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    # filepath = './brats.tar'
    # with tqdm(unit='B', unit_scale=True, leave=True, miniters=1,
    #           desc='Downloading brats dataset: ') as t:
    #     urllib.request.urlretrieve(data_url, filename=filepath,
    #                                reporthook=my_hook(t), data=None)
    import shutil

    original = r'/home/maksim/tmp/brats.tar'
    target = r'./brats.tar'
    print('before copy')
    # shutil.copyfile(original, target)
    # assert sha384(open(filepath, 'rb').read(
    #     path.getsize(filepath))).hexdigest() == TAR_SHA384

    print('after copy')
    # with tarfile.open(filepath, "r:") as tar_ref:
    #     for member in tqdm(iterable=tar_ref.infolist(), desc='Untarring dataset'):
    #         tar_ref.extract(member, "./data")


class PyTorchBraTSDataLoader(PyTorchDataLoader):
    """PyTorch data loader for BraTS dataset."""

    def __init__(self, data_path, batch_size, **kwargs):
        """Instantiate the data object.

        Args:
            data_path: The file path to the data
            batch_size: The batch size of the data loader
            **kwargs: Additional arguments, passed to super
             init and load_mnist_shard
        """
        super().__init__(batch_size, **kwargs)

        load_brats_dataset()
        self.valid_dataset = BraTSDataset(True, shard_num=int(data_path), **kwargs)
        self.train_dataset = BraTSDataset(False, shard_num=int(data_path), **kwargs)
        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_valid_loader()
        self.batch_size = batch_size

    def get_valid_loader(self, num_batches=None):
        """Return validation dataloader."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def get_train_loader(self, num_batches=None):
        """Return train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_train_data_size(self):
        """Return size of train dataset."""
        return len(self.train_dataset)

    def get_valid_data_size(self):
        """Return size of validation dataset."""
        return len(self.valid_dataset)

    def get_feature_shape(self):
        """Return data shape."""
        return self.valid_dataset[0][0].shape
