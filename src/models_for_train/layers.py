# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def conv1x1(in_ch, out_ch, stride=1, bias=True):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, bias=bias, kernel_size=1, stride=stride)

def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def dyconv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return DyConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )

def subpel_dyconv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        DyConv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )

def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1), nn.PixelShuffle(r)
    )

def subpel_dyconv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        DyConv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )

class SubpelDyConv1x1(nn.Module):
    def __init__(self, in_ch, out_ch, r=1):
        super().__init__()
        self.dyconv = DyConv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0)
        self.up = nn.PixelShuffle(r)
    def forward(self, x, latent):
        return self.up(self.dyconv(x, latent))
    
class repconv3x3(nn.Module):
    def __init__(self, ch: int, *args):
        super().__init__()
        self.branch1 = nn.Sequential(conv3x3(ch, ch * 2),
                                     nn.BatchNorm2d(ch * 2),
                                     conv1x1(ch * 2, ch))
        self.branch2 = nn.Sequential(conv3x3(ch, ch * 2),
                                     nn.BatchNorm2d(ch * 2),
                                     conv1x1(ch * 2, ch))
    def forward(self, x):
        return x + self.branch1(x) + self.branch2(x)
    
class ChannelNorm(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, x):
        embedding = (x.pow(2).sum((2,3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        return x * gate

class DMBlock(nn.Module):
    def __init__(self, c):
        super(DMBlock, self).__init__()
        self.relu = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv3 = nn.Conv2d(c, c, 1)
        self.conv4 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv5 = nn.Conv2d(c * 4, c, 1)
    def forward(self, x):
        out1 = self.conv1(self.relu(x))
        out2 = self.conv2(self.relu(out1))
        out3 = self.conv3(self.relu(out2))
        out4 = self.conv4(self.relu(out3))
        out5 = self.conv5(torch.cat([out1, out2, out3, out4],dim=1))
        return out5 + x
    
class DyResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=1):
        super().__init__()
        self.conv1 = dyconv3x3(in_ch, out_ch, stride)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.relu2 = nn.SiLU(inplace=True)

        if stride != 1:
            self.adaptor = nn.Sequential(nn.AvgPool2d(3, stride, padding=1),
                                         conv1x1(in_ch, out_ch, 1))
        elif in_ch != out_ch:
            self.adaptor = conv1x1(in_ch, out_ch, 1)
        else:
            self.adaptor = None

    def forward(self, x, latent):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x, latent)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        out = out + identity
        return out
    
class DyResUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=2):
        super().__init__()
        self.subpel_conv = SubpelDyConv1x1(in_ch, in_ch//2, upsample)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.SiLU(inplace=True)
        self.conv2 = conv1x1(in_ch//2, out_ch)

        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x, latent):
        identity = x
        out = self.subpel_conv(x, latent)
        out = self.relu1(out)
        out = self.conv(out)
        out = self.relu2(out)
        out = self.conv2(out)

        identity = self.upsample(x)
        out = out + identity
        return out

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=1, dynamic=False):
        super().__init__()
        if dynamic:
            self.conv1 = dyconv3x3(in_ch, out_ch, stride)
        else:
            self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.relu2 = nn.SiLU(inplace=True)

        if stride != 1:
            self.adaptor = nn.Sequential(nn.AvgPool2d(3, stride, padding=1),
                                         conv1x1(in_ch, out_ch, 1))
        elif in_ch != out_ch:
            self.adaptor = conv1x1(in_ch, out_ch, 1)
        else:
            self.adaptor = None

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        out = out + identity
        return out

class RIR(nn.Module):
    def __init__(self, ch: int, dynamic=False):
        super().__init__()
        self.conv = nn.Sequential(ResBlock(ch, ch, dynamic=dynamic),
                                ResBlock(ch, ch, dynamic=dynamic))
    def forward(self, x):
        return x + self.conv(x)

class ResBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, rep=False):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.SiLU(inplace=True)
        if rep:
            self.conv2 = repconv3x3(in_ch//2)
        else:
            self.conv2 = conv3x3(in_ch//2, in_ch//2, stride)
        self.relu2 = nn.SiLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, out_ch)

        if stride != 1:
            self.adaptor = nn.Sequential(nn.AvgPool2d(3, stride, padding=1),
                                         conv1x1(in_ch, out_ch, 1))
        elif in_ch != out_ch:
            self.adaptor = conv1x1(in_ch, out_ch, 1)
        else:
            self.adaptor = None

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out

class ResUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=2, dynamic=False):
        super().__init__()
        if dynamic:
            self.subpel_conv = subpel_dyconv1x1(in_ch, in_ch//2, upsample)
        else:
            self.subpel_conv = subpel_conv1x1(in_ch, in_ch//2, upsample)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.SiLU(inplace=True)
        self.conv2 = conv1x1(in_ch//2, out_ch)

        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.relu1(out)
        out = self.conv(out)
        out = self.relu2(out)
        out = self.conv2(out)

        identity = self.upsample(x)
        out = out + identity
        return out
    
class ResUpsample2(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=2, dynamic=False):
        super().__init__()
        if dynamic:
            self.subpel_conv = subpel_dyconv1x1(in_ch, in_ch, upsample)
        else:
            self.subpel_conv = subpel_conv1x1(in_ch, in_ch, upsample)
        self.relu1 = nn.SiLU(inplace=True)
        self.conv = conv3x3(in_ch, in_ch)
        self.relu2 = nn.SiLU(inplace=True)
        self.conv2 = conv1x1(in_ch, out_ch)

        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.relu1(out)
        out = self.conv(out)
        out = self.relu2(out)
        out = self.conv2(out)

        identity = self.upsample(x)
        out = out + identity
        return out
    

def get_hyper_enc_dec_models(in_channel, y_channel, z_channel):
    hyper_enc = nn.Sequential(
        ResBlock(in_channel, in_channel),
        ResBlock(in_channel, in_channel, stride=2),
        RIR(in_channel),
        ResBlock(in_channel, z_channel, stride=2),
        RIR(z_channel),
    )
    hyper_dec = nn.Sequential(
        ResUpsample(z_channel, y_channel),
        ResUpsample(y_channel, y_channel),
        ResBottleneck(y_channel, y_channel),
    )

    return hyper_enc, hyper_dec