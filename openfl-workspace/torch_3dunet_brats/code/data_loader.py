# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

import random
import urllib.request
from hashlib import sha384
import tarfile
import numpy as np

from torch.utils.data import DataLoader
from openfl.federated import PyTorchDataLoader

from tqdm import tqdm
import nibabel as nib
from skimage.transform import resize


class BraTSDataset():
    """
    This dataset contains brain tumor 3d images for one collaborator train or val.
    Args:
        data_list: list of image paths
    """

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        images = []
        for i in range(1, 5):
            img = nib.load(self.data_list[index]['image{}'.format(i)])
            img = np.asanyarray(img.dataobj)
            img = self.resize(img, (160, 160, 128))
            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = img.astype(np.float32)

        mask = nib.load(self.data_list[index]['label'])
        mask = np.asanyarray(mask.dataobj)
        mask = self.resize(mask, (160, 160, 128)).astype(np.uint8)
        mask = self.classify(mask)
        return (img, mask)

    def normalize(self, data):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data, sizes):
        data = resize(data, sizes, mode='edge',
                      anti_aliasing=False,
                      anti_aliasing_sigma=None,
                      preserve_range=True,
                      order=0)
        return data

    def classify(self, inputs):
        result = []
        # merge label 2 and label 3 to construct TC
        result.append(np.logical_or(inputs == 2, inputs == 3))
        # merge labels 1, 2 and 3 to construct WT
        result.append(
            np.logical_or(
                np.logical_or(inputs == 2, inputs == 3), inputs == 1
            )
        )
        # label 2 is ET
        result.append(inputs == 2)
        return np.stack(result, axis=0).astype(np.float32)


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
    # assert sha384(open(filepath, 'rb').read(
    #     path.getsize(filepath))).hexdigest() == TAR_SHA384

    # with tarfile.open(filepath, "r:") as tar_ref:
    #     for member in tqdm(iterable=tar_ref.infolist(), desc='Untarring dataset'):
    #         tar_ref.extract(member, "./data")


class PyTorchBraTSDataLoader(PyTorchDataLoader):
    """PyTorch data loader for BraTS dataset."""

    def __init__(self, data_path, batch_size, collaborator_count, **kwargs):
        """Instantiate the federated data object
        Args:
            collaborator_count: total number of collaborators
            collaborator_num: number of current collaborator
            batch_size:  the batch size of the data loader
            data_list: general list of all image paths, in current implementation 
                it should be created once so that each colaborator gets its own data
            **kwargs: additional arguments, passed to super init
        """
        super().__init__(batch_size, **kwargs)

        self.batch_size = batch_size
        self.train_list = self.generate_train_list(collaborator_count, int(data_path))
        self.val_list = self.generate_val_list(collaborator_count, int(data_path))
        self.training_set = BraTSDataset(self.train_list)
        self.valid_set = BraTSDataset(self.val_list)

        self.train_loader = self.get_train_loader()
        self.val_loader = self.get_valid_loader()

    def generate_name_list(self, collaborator_count, collaborator_num, is_validation):
        data_dir = './data/MICCAI_BraTS2020_TrainingData/'
        self.data_list = [
            {
                'image1': data_dir + 'BraTS20_Training_' + str(i)[1:] + '/BraTS20_Training_' + str(i)[1:] + '_flair.nii.gz',
                'image2': data_dir + 'BraTS20_Training_' + str(i)[1:] + '/BraTS20_Training_' + str(i)[1:] + '_t1ce.nii.gz',
                'image3': data_dir + 'BraTS20_Training_' + str(i)[1:] + '/BraTS20_Training_' + str(i)[1:] + '_t1.nii.gz',
                'image4': data_dir + 'BraTS20_Training_' + str(i)[1:] + '/BraTS20_Training_' + str(i)[1:] + '_t2.nii.gz',
                'label': data_dir + 'BraTS20_Training_' + str(i)[1:] + '/BraTS20_Training_' + str(i)[1:] + '_seg.nii.gz'
            } for i in range(1001, 1370)]
        random.seed(4)
        random.shuffle(self.data_list)

        # split all data for current collaborator
        data = self.data_list[collaborator_num:: collaborator_count]
        assert(len(data) > 7)
        validation_size = len(data) // 7
        if is_validation:
            data = data[-validation_size:]
        else:
            data = data[: -validation_size]
        return data

    def generate_train_list(self, collaborator_count, collaborator_num):
        return self.generate_name_list(collaborator_count, collaborator_num, False)

    def generate_val_list(self, collaborator_count, collaborator_num):
        return self.generate_name_list(collaborator_count, collaborator_num, True)

    def get_valid_loader(self, num_batches=None):
        return DataLoader(self.valid_set, num_workers=4, batch_size=self.batch_size)

    def get_train_loader(self, num_batches=None):
        return DataLoader(
            self.training_set, num_workers=4, batch_size=self.batch_size, shuffle=True
        )

    def get_train_data_size(self):
        return len(self.training_set)

    def get_valid_data_size(self):
        return len(self.valid_set)

    def get_feature_shape(self):
        return self.valid_set[0][0].shape
