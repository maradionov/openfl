# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey
from openfl.federated import PyTorchDataLoader

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
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

class DiceLossHeir(DiceLoss):
    __name__ = 'DiceLoss'

    def forward(self, output, target):
        return super().forward(input=output, target=target)

class PyTorchFederated3dUnet(PyTorchTaskRunner, UNet):
    """Simple Unet for segmentation."""

    def __init__(self, device='cuda', **kwargs):
        """Initialize.

        Args:
            device: The hardware device to use for training (Default = "cpu")
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(**kwargs)
        self.init_network(device=self.device, **kwargs)
        self._init_optimizer()
        self.loss_fn =DiceLossHeir(to_onehot_y=False, sigmoid=True, squared_pred=True)
        self.initialize_tensorkeys_for_functions()

    def _init_optimizer(self):
        """Initialize the optimizer."""
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def init_network(self,
                     device,
                     n_channels,
                     n_classes,
                     print_model=True,
                     **kwargs):
        """Create the network (model).

        Args:
            device: The hardware device to use for training
            n_channels: Number of input image channels
            n_classes: Number of output classes (1 for segmentation)
            print_model: Print the model topology (Default=True)
            **kwargs: Additional arguments to pass to the function

        """
        UNet.__init__(self,
            dimensions=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,)

        self.n_channels = n_channels
        self.n_classes = n_classes
        if print_model:
            print(self)
        print(device)
        self.to(device)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: Data input to the model for the forward pass
        """
        return UNet.forward(self, x)

    def validate(self, col_name, round_num, input_tensor_dict, use_tqdm=True, **kwargs):
        """Run validation of the model on the local data.

        Args:
            col_name:            Name of the collaborator
            round_num:           What round is it
            input_tensor_dict:   Required input tensors (for model)
            use_tqdm:     Use tqdm to print a progress bar (Default=True)

        Returns:
            global_output_dict:  Tensors to send back to the aggregator
            local_output_dict:   Tensors to maintain in the local TensorDB
        """
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        self.eval()
        self.to(self.device)
        val_score = 0
        total_samples = 0
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        post_trans = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
        )
        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="validate")

        with torch.no_grad():
            for data, target in loader:
                data, target = torch.tensor(data).to(self.device), torch.tensor(
                    target).to(self.device)
                output = self(data)
                output = post_trans(output)
                # get the index of the max log-probability
                value, not_nans = dice_metric(y_pred=output, y=target)

                # compute overall mean dice
                not_nans = not_nans.item()
                total_samples += not_nans
                val_score += value.item() * not_nans
        origin = col_name
        suffix = 'validate'
        if kwargs['apply'] == 'local':
            suffix += '_local'
        else:
            suffix += '_agg'
        tags = ('metric', suffix)
        # TODO figure out a better way to pass in metric for this pytorch
        #  validate function
        output_tensor_dict = {
            TensorKey('dice_coef', origin, round_num, True, tags):
                np.array(val_score / total_samples)
        }

        return output_tensor_dict, {}

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Resets the optimizer state variables.

        """
        self._init_optimizer()
