"""
ResNet18 with row encoding
"""

import torch
import torch.nn as nn
from skema.img2mml.models.encoding.row_encoding import RowEncoding
from skema.img2mml.models.encoding.positional_features_for_cnn_encoder import (
    add_positional_features,
)


class ResNetBlock(nn.Module):  # res_block
    def __init__(self, in_channels, out_channels, stride, downsampling=False):
        super(ResNetBlock, self).__init__()

        self.stride = stride
        self.downsampling = downsampling
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.downsampling:
            res = self.downsample(x)
        else:
            res = x

        x = self.relu(self.bn(self.conv1(x)))
        x = self.bn(self.conv2(x))

        # residual connection
        x += res
        x = self.relu(x)

        return x


class ResNet18_Encoder(nn.Module):
    def __init__(self, img_channels, dec_hid_dim, dropout, device, res_block):
        super(ResNet18_Encoder, self).__init__()

        self.re = RowEncoding(device, dec_hid_dim, dropout)
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                img_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.resnet_encoder = nn.Sequential(
            self.initial_layer,
            res_block(64, 64, stride=1),
            res_block(64, 64, stride=1),
            res_block(64, 128, stride=2, downsampling=True),
            res_block(128, 128, stride=1),
            # res_block(128, 256, stride=2, downsampling=True),
            # res_block(256, 256, stride=1),
            # res_block(256, 512, stride=2, downsampling=True),
            # res_block(512, 512, stride=1),
        )

        self.linear = nn.Linear(128, dec_hid_dim)   # was 256, dec_hid
        self.init_weights()

    def init_weights(self):
        """
        initializing the model wghts with values
        drawn from normal distribution.
        else initialize them with 0.
        """
        for name, param in self.resnet_encoder.named_parameters():
            if "nn.Conv2d" in name or "nn.Linear" in name:
                if "weight" in name:
                    nn.init.normal_(param.data, mean=0, std=0.1)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
            elif "nn.BatchNorm2d" in name:
                if "weight" in name:
                    nn.init.constant_(param.data, 1)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, src, encoding_type=None):
        output = self.resnet_encoder(src)
        if encoding_type == "row_encoding":
            # output: (B, H*W, dec_hid_dim)
            # hidden, cell: [1, B, dec_hid_dim]
            output, hidden, cell = self.re(output)
            return output, hidden, cell

        else:
            output = torch.flatten(output, 2, -1)  # (B, 512, L=H*W)
            output = output.permute(0, 2, 1)  # (B, L, 512)
            if encoding_type == "positional_features":
                output += add_positional_features(output)  # (B, L, 512)
            return self.linear(output)  # (B, L, dec_hid_dim)
        