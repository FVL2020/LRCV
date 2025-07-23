# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import numpy as np
import time

from .common_model import CompressionModel
from .layers import *
from ..utils.stream_helper import encode_i, decode_i, get_downsampled_shape, filesize


class IntraEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.Sequential(
            ResBlock(3, 192, stride=2),
            ResBlock(192, 192),
            ResBlock(192, 192, stride=2),
            ResBlock(192, 192),
            ResBlock(192, 192, stride=2),
            ResBlock(192, 192),
            conv3x3(192, 192, stride=2),
            )
        
    def forward(self, x):
        return self.enc(x)


class IntraDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dec1 = nn.Sequential(
            ResBlock(192, 192),
            )
        self.dec2 = nn.Sequential(
            ResUpsample(192, 192, upsample=2),
            ResBottleneck(192, 192),
            ResBottleneck(192, 192),
            ResBottleneck(192, 192),
            )
        self.dec3 = nn.Sequential(
            ResUpsample(192, 192, upsample=2),
            ResBottleneck(192, 192),
            ResBottleneck(192, 192),
            ResBottleneck(192, 192),
        )
        self.dec4 = nn.Sequential(
            ResUpsample(192, 192, upsample=2),
            ResBottleneck(192, 192),
            ResBottleneck(192, 192),
            ResBottleneck(192, 192),
            subpel_conv1x1(192, 3, 2)
        )

    def forward(self, y):
        f1 = self.dec1(y)
        f2 = self.dec2(f1)
        f3 = self.dec3(f2)
        x = self.dec4(f3)
        return x, [f3, f2, f1]


class IntraModel(CompressionModel):
    def __init__(self, N=192, ec_thread=False, stream_part=1):
        super().__init__(y_distribution='gaussian', z_channel=N,
                         ec_thread=ec_thread, stream_part=stream_part)

        self.enc = IntraEncoder()
        self.dec = IntraDecoder()

        self.hyper_enc = nn.Sequential(
            conv3x3(N, N),
            nn.SiLU(True),
            conv3x3(N, N),
            nn.SiLU(True),
            conv3x3(N, N, stride=2),
            nn.SiLU(True),
            conv3x3(N, N),
            nn.SiLU(True),
            conv3x3(N, N, stride=2)
            )
        self.hyper_dec = nn.Sequential(
            ResUpsample(N, N * 2),
            ResUpsample(N * 2, N * 2),
            ResBottleneck(N * 2, N * 2)
            )

        self.y_spatial_prior_adaptor_1 = conv1x1(N * 3, N * 2)
        self.y_spatial_prior_adaptor_2 = conv1x1(N * 3, N * 2)
        self.y_spatial_prior_adaptor_3 = conv1x1(N * 3, N * 2)
        self.y_spatial_prior = nn.Sequential(
            ResBottleneck(N * 2, N * 2),
            ResBottleneck(N * 2, N * 2),
            ResBottleneck(N * 2, N * 2),
        )

    def cal_time(self, t_list: list):
        torch.cuda.synchronize()
        t_now = time.time()
        t = t_now - sum(t_list)
        t_list.append(t)
        return t

    def forward(self, x):
        # torch.cuda.synchronize()
        # t_list = [time.time()]

        y = self.enc(x)
        z = self.hyper_enc(y)
        z_hat = self.quant(z, 'ste')
        # print('enc and henc:', self.cal_time(t_list))

        params = self.hyper_dec(z_hat)
        # print('hdec:', self.cal_time(t_list))
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        # print('forward_four_part_prior:', self.cal_time(t_list))

        x_hat, y_features = self.dec(y_hat)
        # print('dec:', self.cal_time(t_list))

        y_for_bit = y_res
        z_for_bit = z
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        N, _, H, W = x.size()
        pixel_num = N * H * W
        bpp_y = torch.sum(bits_y) / pixel_num
        bpp_z = torch.sum(bits_z) / pixel_num
        bits = torch.sum(bpp_y + bpp_z) * pixel_num
        bpp = bpp_y + bpp_z

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "y_feat": y_features,
            "bit": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }

    def encode_decode(self, x, output_path=None, pic_width=None, pic_height=None, **kwargs):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        if output_path is None:
            encoded = self.forward(x)
            result = {
                'bit': encoded['bit'].item(),
                'x_hat': encoded['x_hat'],
            }
            return result

        assert pic_height is not None
        assert pic_width is not None
        compressed = self.compress(x)
        bit_stream = compressed['bit_stream']
        encode_i(pic_height, pic_width, 1, 1, bit_stream, output_path)
        bit = filesize(output_path) * 8

        height, width, _, _, bit_stream = decode_i(output_path)
        decompressed = self.decompress(bit_stream, height, width)
        x_hat = decompressed['x_hat']
        # print(f'PSNR: {-10.0 * np.log10(torch.mean((x_hat - x) ** 2).item())}')
        result = {
            'bit': bit,
            'x_hat': x_hat,
        }
        return result

    def compress(self, x):

        torch.cuda.synchronize()
        start_time = time.time()
        y = self.enc(x)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat = torch.round(z)
        torch.cuda.synchronize()
        t1 = time.time() - start_time
        print('encode y and z:', t1)

        params = self.hyper_dec(z_hat)
        params = self.slice_to_y(params, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()

        x_hat, _ = self.dec(y_hat)
        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat.clamp_(0, 1),
        }
        return result

    def decompress(self, bit_stream, height, width):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        torch.cuda.synchronize()
        start_time = time.time()
        self.entropy_coder.set_stream(bit_stream)
        z_size = get_downsampled_shape(height, width, 64)
        y_height, y_width = get_downsampled_shape(height, width, 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)
        torch.cuda.synchronize()
        t1 = time.time() - start_time
        print('entropy decode z:', t1)

        params = self.hyper_dec(z_hat)
        torch.cuda.synchronize()
        t2 = time.time() - start_time - t1
        print('decode z:', t2)
        params = self.slice_to_y(params, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)
        torch.cuda.synchronize()
        t3 = time.time() - start_time - t1 - t2
        print('entropy decode y:', t3)

        x_hat, _ = self.dec(y_hat)
        torch.cuda.synchronize()
        print('decode y:', time.time() - start_time - t1 - t2 - t3)
        print()
        return {"x_hat": x_hat.clamp_(0, 1)}
