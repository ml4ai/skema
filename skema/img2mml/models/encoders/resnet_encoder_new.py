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
    def __init__(self, in_channels, out_channels, stride, downsampling=False, matching=False):
        super(ResNetBlock, self).__init__()

        self.stride = stride
        self.downsampling = downsampling
        self.matching = matching

        # if stride != 1:
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
        self.match_linear = nn.Linear(125,63)

    def forward(self, x):
        if self.downsampling:
            res = self.downsample(x)
        else:
            res = x

        x = self.relu(self.bn(self.conv1(x)))
        x = self.bn(self.conv2(x))

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

        self.init_layer = self.initial_layer
        self.l1 = res_block(64, 64, stride=1)
        self.l2 = res_block(64, 64, stride=1)
        self.l3 = res_block(64, 128, stride=1, downsampling=True)
        self.l4 = res_block(128, 128, stride=1)
        self.l5 = res_block(128, 256, stride=1, downsampling=True)
        self.l6 = res_block(256, 256, stride=1)
        self.l7 = res_block(256, 512, stride=1, downsampling=True)
        self.l8 = res_block(512, 512, stride=1, matching=True)
        self.linear1 = nn.Linear(125,63)
        self.linear2 = nn.Linear(512, dec_hid_dim)
        # self.init_weights()

    def forward(self, src, encoding_type=None):
        output = self.init_layer(src)
        # print("init: ", output.shape)
        output = self.l1(output)
        # print("l1: ", output.shape)
        output = self.l2(output)
        # print("l2: ", output.shape)
        output = self.l3(output)
        # print("l3: ", output.shape)
        output = self.l4(output)
        # print("l4: ", output.shape)
        output = self.l5(output)
        # print("l5: ", output.shape)
        output = self.l6(output)
        # print("l6: ", output.shape)
        output = self.l7(output)
        # print("l7: ", output.shape)
        output = self.l8(output)
        # print("l8: ", output.shape)
        output = self.linear1(output)

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
            return self.linear2(output)  # (B, L, dec_hid_dim)