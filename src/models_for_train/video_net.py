import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layers import *


backward_grid = [{} for _ in range(9)]    # 0~7 for GPU, -1 for CPU


def add_grid_cache(flow):
    device_id = -1 if flow.device == torch.device('cpu') else flow.device.index
    if str(flow.size()) not in backward_grid[device_id]:
        N, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=flow.device, dtype=torch.float32).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=flow.device, dtype=torch.float32).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)


def torch_warp(feature, flow):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    add_grid_cache(flow)
    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = (backward_grid[device_id][str(flow.size())] + flow)
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


def flow_warp(im, flow):
    warp = torch_warp(im, flow)
    return warp


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)

    return outfeature


def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature


def loadweightformnp(layername):
    index = layername.find('modelL')
    model_path = 'src/models_for_test/flow_pretrain_np/'
    if index == -1:
        print('laod models error!!')
    else:
        name = layername[index:index + 11]
        modelweight = model_path + name + '-weight.npy'
        modelbias = model_path + name + '-bias.npy'
        weightnp = np.load(modelweight)
        # weightnp = np.transpose(weightnp, [2, 3, 1, 0])
        # print(weightnp)
        biasnp = np.load(modelbias)

        # init_weight = lambda shape, dtype: weightnp
        # init_bias   = lambda shape, dtype: biasnp
        # print('Done!')

        return torch.from_numpy(weightnp), torch.from_numpy(biasnp)
        # return init_weight, init_bias

class MEBasic(nn.Module):
    def __init__(self, layername):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        self.conv1.weight.data, self.conv1.bias.data = loadweightformnp(layername + '_F-1')
        self.conv2.weight.data, self.conv2.bias.data = loadweightformnp(layername + '_F-2')
        self.conv3.weight.data, self.conv3.bias.data = loadweightformnp(layername + '_F-3')
        self.conv4.weight.data, self.conv4.bias.data = loadweightformnp(layername + '_F-4')
        self.conv5.weight.data, self.conv5.bias.data = loadweightformnp(layername + '_F-5')

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([MEBasic('motion_estimation' + 'modelL' + str(intLevel + 1)) for intLevel in range(self.L)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                self.moduleBasic[level](torch.cat([im1_list[img_index],
                                                   flow_warp(im2_list[img_index], flow_up),
                                                   flow_up], 1))

        return flow


class MEBlock(nn.Module):
    def __init__(self, in_channel, dim=64, scale=4):
        super().__init__()
        self.motion_path = nn.Sequential(
            nn.Upsample(scale_factor=1. / scale, mode="bilinear", align_corners=False),
            ResBlock(in_channel, dim, stride=2),
            RIR(dim),
            ResBlock(dim, dim, stride=2),
            nn.Upsample(scale_factor=scale * 2, mode="bilinear", align_corners=False)
            )
        self.spatial_path = nn.Sequential(
            ResBlock(in_channel, dim, stride=2),
            RIR(dim),
        )
        self.upconv = ResUpsample(dim * 2, dim, upsample=2)
            
    def forward(self, x):
        x_m = self.motion_path(x)
        x_s = self.spatial_path(x)
        return self.upconv(torch.cat((x_m, x_s), dim=1))


def get_hyper_enc_dec_models(in_channel, y_channel, z_channel):
    # hyper_enc = nn.Sequential(
    #     ResBlock(in_channel, in_channel),
    #     ResBlock(in_channel, in_channel, stride=2),
    #     RIR(in_channel),
    #     ResBlock(in_channel, z_channel, stride=2),
    #     RIR(z_channel),
    # )
    hyper_enc = nn.Sequential(
        conv3x3(in_channel, in_channel),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, in_channel),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, in_channel, stride=2),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, in_channel),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, z_channel, stride=2)
    )
    hyper_dec = nn.Sequential(
        # ResUpsample(z_channel, y_channel),
        # ResUpsample(y_channel, y_channel),
        # ResBottleneck(y_channel, y_channel),
        ResUpsample2(z_channel, y_channel),
        ResUpsample2(y_channel, y_channel),
        ResBlock(y_channel, y_channel),
        # ResBlock(y_channel, y_channel),
    )

    return hyper_enc, hyper_dec


def get_hyper_enc_dec_models1(in_channel, y_channel, z_channel):
    # hyper_enc = nn.Sequential(
    #     ResBlock(in_channel, in_channel),
    #     ResBlock(in_channel, in_channel, stride=2),
    #     RIR(in_channel),
    #     ResBlock(in_channel, z_channel, stride=2),
    #     RIR(z_channel),
    # )
    hyper_enc = nn.Sequential(
        conv3x3(in_channel, in_channel),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, in_channel),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, in_channel, stride=2),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, in_channel),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, z_channel, stride=2)
    )
    hyper_dec = nn.Sequential(
        ResUpsample2(z_channel, y_channel),
        ResUpsample2(y_channel, y_channel),
        ResBottleneck(y_channel, y_channel),
    )

    return hyper_enc, hyper_dec

def get_hyper_enc_dec_models2(in_channel, y_channel, z_channel):
    # hyper_enc = nn.Sequential(
    #     ResBlock(in_channel, in_channel),
    #     ResBlock(in_channel, in_channel, stride=2),
    #     RIR(in_channel),
    #     ResBlock(in_channel, z_channel, stride=2),
    #     RIR(z_channel),
    # )
    hyper_enc = nn.Sequential(
        conv3x3(in_channel, in_channel),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, in_channel),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, in_channel, stride=2),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, in_channel),
        nn.SiLU(inplace=True),
        conv3x3(in_channel, z_channel, stride=2)
    )
    hyper_dec = nn.Sequential(
        # ResUpsample(z_channel, y_channel),
        # ResUpsample(y_channel, y_channel),
        ResBottleneck(z_channel, y_channel),
        ResBottleneck(y_channel, y_channel),
        ResBottleneck(y_channel, y_channel),
    )

    return hyper_enc, hyper_dec