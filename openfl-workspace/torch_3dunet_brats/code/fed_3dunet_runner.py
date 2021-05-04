# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""
import numpy as np
import tqdm
import torch
import torch.optim as optim

from openfl.federated import PyTorchTaskRunner
from openfl.utilities import TensorKey

from .pt_3dunet_parts import soft_dice_coef
from .pt_3dunet_parts import soft_dice_loss
from .pt_3dunet_parts import DoubleConv
from .pt_3dunet_parts import Up
from .pt_3dunet_parts import Down
from .pt_3dunet_parts import Out


class PyTorchFederated3dUnet(PyTorchTaskRunner):
    """Simple Unet for segmentation."""

    def __init__(self, device='cpu', **kwargs):
        """Initialize.

        Args:
            device: The hardware device to use for training (Default = "cpu")
            **kwargs: Additional arguments to pass to the function

        """
        super().__init__(device, **kwargs)
        self.init_network(device=self.device, **kwargs)
        self._init_optimizer()
        self.loss_fn = soft_dice_loss
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
        self.in_channels = n_channels
        self.n_classes = n_classes
        depth_mult = 10

        self.conv = DoubleConv(self.in_channels, depth_mult)
        self.enc1 = Down(depth_mult, 2 * depth_mult)
        self.enc2 = Down(2 * depth_mult, 4 * depth_mult)
        self.enc3 = Down(4 * depth_mult, 8 * depth_mult)
        self.enc4 = Down(8 * depth_mult, 8 * depth_mult)

        self.dec1 = Up(16 * depth_mult, 4 * depth_mult)
        self.dec2 = Up(8 * depth_mult, 2 * depth_mult)
        self.dec3 = Up(4 * depth_mult, depth_mult)
        self.dec4 = Up(2 * depth_mult, depth_mult)
        self.out = Out(depth_mult, self.n_classes)
        if print_model:
            print(self)
        self.to(device)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        mask = torch.sigmoid(mask)

        return mask

    def validate(
        self, col_name, round_num, input_tensor_dict, use_tqdm=True, **kwargs
    ):
        """ Validate. Redifine function from PyTorchTaskRunner, to use our validation"""
        self.rebuild_model(round_num, input_tensor_dict, validation=True)
        loader = self.data_loader.get_valid_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="validate")
    # -------------Usual validation code---------------------------------------------------------------------------
        self.eval()
        self.to(self.device)
        metric = 0.0
        sample_num = 0

        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="validate")
        with torch.no_grad():
            for val_inputs, val_labels in loader:
                val_inputs = val_inputs.to(self.device)
                val_labels = val_labels.to(self.device)
                val_outputs = self(val_inputs)
                val_outputs = (val_outputs >= 0.5).float()
                value = soft_dice_coef(val_outputs, val_labels)
                sample_num += val_labels.shape[0]
                metric += value.cpu().numpy()

            metric = metric / sample_num
    # --------------------------------------------------------------------------

        origin = col_name
        suffix = "validate"
        if kwargs["apply"] == "local":
            suffix += "_local"
        else:
            suffix += "_agg"
        tags = ("metric", suffix)
        output_tensor_dict = {
            TensorKey("dice_coef", origin, round_num, True, tags): np.array(
                metric
            )
        }
        return output_tensor_dict, {}

    def reset_opt_vars(self):
        """Reset optimizer variables.

        Resets the optimizer state variables.

        """
        self._init_optimizer()
