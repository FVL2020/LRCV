# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time

import torch
from torch import nn
from torchvision.ops import deform_conv2d, DeformConv2d
import numpy as np

from .image_model_quad import IntraModel
from .common_model import CompressionModel
from .video_net import ME_Spynet, get_hyper_enc_dec_models, flow_warp, bilineardownsacling
from .layers import *
    
class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, offset_channels):
        super().__init__()
        self.kernel_size = (3, 3)
        self.stride = 1
        self.padding = 1
        self.dilation = 1
        self.groups = 1
        self.deformable_groups = 16
        self.output_padding = 0

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.conv_offset = nn.Sequential(
            nn.Conv2d(offset_channels, out_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, offset_feat):
        out = self.conv_offset(offset_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            print('Offset mean is {}, larger than 100.'.format(offset_mean))
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)

class MultiScaleContextFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_down = ResBlock(64, 64, stride=2)
        self.conv2_down = ResBlock(128, 64, stride=2)
        self.conv3_out = nn.Sequential(
            ResBottleneck(192, 128),
            conv3x3(128, 128)
        )
        self.conv3_up = ResUpsample(128, 64, 2)
        self.conv2_up = ResUpsample(128, 64, 2)
        self.conv2_out = nn.Sequential(
            ResBottleneck(192, 64),
            conv3x3(64, 64)
        )
        self.conv1_out = nn.Sequential(
            ResBottleneck(128, 64),
            conv3x3(64, 64)
        )

    def forward(self, context1, context2, context3):
        context1_down = self.conv1_down(context1)
        context2_down = self.conv2_down(torch.cat((context1_down, context2), dim=1))
        context3_out = self.conv3_out(torch.cat((context2_down, context3), dim=1))
        context3_up = self.conv3_up(context3_out)
        context2_out = self.conv2_out(torch.cat((context3_up, context2, context1_down), dim=1))
        context2_up = self.conv2_up(torch.cat((context3_up, context2_out), dim=1))
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out

        return context1, context2, context3

class MvEncoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.mv_refine = nn.Sequential(
            ResBlock(8, channel, stride=2),
            RIR(channel),
            ResBlock(channel, channel, stride=2),
        )
        self.adaptor1 = ResBlock(8, channel, stride=2)
        self.adaptor2 = ResBlock(13, channel, stride=2)
        self.adaptor3 = ResBlock(18, channel, stride=2)
        self.adaptor = nn.Sequential(
            RIR(channel),
            ResBlock(channel, channel, stride=2),
        )
        self.unet = UNet(channel * 2, channel)
        self.est_offset = nn.Sequential(
            ResBlock(64 + 64 + 192, 128),
            ResBlock(128, channel),
            RIR(channel),
            conv3x3(channel, channel)
        )
        self.enc1 = nn.Sequential(
            ResBlock(channel + 64, channel, stride=1),
            ResBlock(channel, channel, stride=2),
            RIR(channel),
        )
        self.enc2 = nn.Sequential(
            ResBlock(channel + 64, channel, stride=1),
            ResBlock(channel, channel, stride=2),
            RIR(channel),
        )
        self.enc3 = nn.Sequential(
            ResBlock(channel + 128, channel, stride=1),
            RIR(channel),
            conv3x3(channel, channel)
        )

    def forward(self, feat, ref_y_feat_list, prior_list, frame_idx):
        if frame_idx == 1:
            prior_feat = self.adaptor1(torch.cat(prior_list[:-2], dim=1))
        elif frame_idx == 2:
            prior_feat = self.adaptor2(torch.cat(prior_list[:-2], dim=1))
        else:
            prior_feat = self.adaptor3(torch.cat(prior_list[:-2], dim=1))

        mv_feat = self.mv_refine(torch.cat(prior_list[-3:], dim=1))
        prior_feat = self.unet(torch.cat([mv_feat, self.adaptor(prior_feat)], dim=1))
        # TODO long term prior
        ref_y_feat_4x, ref_y_feat_8x, ref_y_feat_16x = ref_y_feat_list
        offset = self.est_offset(torch.cat([mv_feat, ref_y_feat_4x, feat], dim=1))
        out = self.enc1(torch.cat([offset, prior_feat], dim=1))
        out = self.enc2(torch.cat([out, ref_y_feat_8x], dim=1))
        out = self.enc3(torch.cat([out, ref_y_feat_16x], dim=1))
        return out

class MvDecoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.deform_conv1 = DeformConv(64, 64, channel)
        self.deform_conv2 = DeformConv(64, 64, channel)
        self.deform_conv3 = DeformConv(128, 128, channel)
        self.dec1 = nn.Sequential(
            RIR(channel),
        )
        self.dec2 = nn.Sequential(
            ResUpsample(channel, channel, upsample=2),
            RIR(channel),
        )
        self.dec3 = nn.Sequential(
            ResUpsample(channel, channel, upsample=2),
            RIR(channel),
        )
        self.context_fusion_net = MultiScaleContextFusion()

    def forward(self, x, ref_y_feat_list):
        ref_y_feat_4x, ref_y_feat_8x, ref_y_feat_16x = ref_y_feat_list
        offset_16x = self.dec1(x)
        context3 = self.deform_conv3(ref_y_feat_16x, offset_16x)
        offset_8x = self.dec2(offset_16x)
        context2 = self.deform_conv2(ref_y_feat_8x, offset_8x)
        offset_4x = self.dec3(offset_8x)
        context1 = self.deform_conv1(ref_y_feat_4x, offset_4x)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return [offset_4x, offset_8x, offset_16x], context1, context2, context3

class ContextualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            ResBlock(3, 192, stride=2),
            ResBottleneck(192, 192),
            ResBlock(192, 192, stride=2),
            )
        self.enc2 = nn.Sequential(
            ResBottleneck(256, 192),
            ResBottleneck(192, 192),
            ResBlock(192, 192, stride=2),
        )
        self.enc3 = nn.Sequential(
            ResBottleneck(256, 192),
            ResBottleneck(192, 192),
            ResBlock(192, 192, stride=2),
        )
        self.enc4 = nn.Sequential(
            ResBottleneck(320, 192),
            ResBottleneck(192, 192),
            ResBottleneck(192, 128),
            conv3x3(128, 128)
        )

    def feat_extractor(self, x):
        return self.enc1(x)

    def forward(self, feature, context1, context2, context3):
        feature = self.enc2(torch.cat([feature, context1], dim=1))
        # TODO high quality prior
        feature = self.enc3(torch.cat([feature, context2], dim=1))
        feature = self.enc4(torch.cat([feature, context3], dim=1))
        return feature

class ContextualDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec1 = nn.Sequential(
            ResBottleneck(256, 256),
            ResBottleneck(256, 256),
            ResBottleneck(256, 256),
            ResUpsample(256, 128, upsample=2),
        )
        self.dec2 = nn.Sequential(
            ResBottleneck(192, 192),
            ResUpsample(192, 128, upsample=2),
        )
        self.dec3 = nn.Sequential(
            ResBottleneck(192, 192),
            subpel_conv3x3(192, 3, 4),
        )
    def forward(self, x, context1, context2, context3):
        f1 = torch.cat([x, context3], dim=1)
        f2 = torch.cat([self.dec1(f1), context2], dim=1)
        f3 = torch.cat([self.dec2(f2), context1], dim=1)
        out = self.dec3(f3)
        return out, [f3, f2, f1]

class RTVC(CompressionModel):
    def __init__(self, ec_thread=False, stream_part=1, *args, **kwargs):
        super().__init__(y_distribution='laplace', z_channel=128, mv_z_channel=64,
                         ec_thread=ec_thread, stream_part=stream_part)

        channel_mv = 64
        mv_z_channel = 64
        self.image_model = IntraModel()

        # temporal prior
        self.feature_adaptor_I = nn.ModuleList([ResBottleneck(192, 64, 1), ResBottleneck(192, 64, 1), ResBottleneck(192, 128, 1)])
        self.feature_adaptor_P = nn.ModuleList([ResBottleneck(192, 64, 1), ResBottleneck(192, 64, 1), ResBottleneck(256, 128, 1)])
        self.temporal_fusion_adaptor = nn.ModuleList([ResBottleneck(128, 128, 1), \
                                                      ResBottleneck(128 * 2 + 64, 128, 1), ResBottleneck(128 * 4, 128, 1)])
        self.temporal_fusion = nn.Sequential(
            ResBottleneck(128, 128),
            ResBottleneck(128, 128),
        )

        # motion endecoder
        self.optic_flow = ME_Spynet()
        self.mv_encoder = MvEncoder(channel_mv)
        self.mv_decoder = MvDecoder(channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv + 128, channel_mv, mv_z_channel)

        self.mv_y_prior_fusion = nn.Sequential(
            ResBottleneck(channel_mv + 128, channel_mv * 3),
            ResBottleneck(channel_mv * 3, channel_mv * 3),
            ResBottleneck(channel_mv * 3, channel_mv * 2),
        )

        self.mv_y_spatial_prior_adaptor_1 = conv1x1(channel_mv * 3, channel_mv * 3)
        self.mv_y_spatial_prior_adaptor_2 = conv1x1(channel_mv * 3, channel_mv * 3)
        self.mv_y_spatial_prior_adaptor_3 = conv1x1(channel_mv * 3, channel_mv * 3)

        self.mv_y_spatial_prior = nn.Sequential(
            ResBottleneck(channel_mv * 3, channel_mv * 3),
            ResBottleneck(channel_mv * 3, channel_mv * 3),
            ResBottleneck(channel_mv * 3, channel_mv * 2),
        )

        # context endecoder
        self.contextual_encoder = ContextualEncoder()
        self.contextual_decoder = ContextualDecoder()
        self.contextual_hyper_prior_encoder, self.contextual_hyper_prior_decoder = \
            get_hyper_enc_dec_models(128 * 3, 128, 128)

        self.y_prior_fusion = nn.Sequential(
            ResBottleneck(128 * 3, 128 * 3),
            ResBottleneck(128 * 3, 128 * 3),
            ResBottleneck(128 * 3, 128 * 2),
        )

        self.y_spatial_prior_adaptor_1 = conv1x1(128 * 3, 128 * 3)
        self.y_spatial_prior_adaptor_2 = conv1x1(128 * 3, 128 * 3)
        self.y_spatial_prior_adaptor_3 = conv1x1(128 * 3, 128 * 3)

        self.y_spatial_prior = nn.Sequential(
            ResBottleneck(128 * 3, 128 * 3),
            ResBottleneck(128 * 3, 128 * 3),
            ResBottleneck(128 * 3, 128 * 2),
        )

    def mv_prior_param_decoder(self, mv_z_hat, temporal_prior):
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        mv_params = torch.cat((mv_params, temporal_prior), dim=1)
        mv_params = self.mv_y_prior_fusion(mv_params)
        return mv_params

    def ctx_prior_param_decoder(self, z_hat, context3, temporal_prior):
        ctx_params = self.contextual_hyper_prior_decoder(z_hat)
        params = torch.cat((ctx_params, context3, temporal_prior), dim=1)
        params = self.y_prior_fusion(params)
        return params

    def init_dpb(self, x):
        _, _, h, w = x.shape
        dpb = {}
        # enc
        dpb["prior_mv"] = [torch.empty(1, 3, h, w).cuda() for _ in range(2)]
        dpb["prior_frame"] = [torch.empty(1, 3, h, w).cuda() for _ in range(3)]
        dpb["prior_frame_gt"] = [torch.empty(1, 3, h, w).cuda() for _ in range(3)]
        dpb["ref_mv_feat"] = [torch.empty(1, 192, h//4, w//4).cuda(), torch.empty(1, 192, h//8, w//8).cuda(), \
                            torch.empty(1, 192, h//16, w//16).cuda()]
        # dec
        dpb["prior_mv_feat"] = [torch.empty(1, 64, h//16, w//16).cuda() for _ in range(2)]
        dpb["prior_y_feat"] = [torch.empty(1, 128, h//16, w//16).cuda() for _ in range(3)]
        dpb["ref_y_feat"] = [torch.empty(1, 128, h//4, w//4).cuda(), torch.empty(1, 192, h//8, w//8).cuda(), \
                            torch.empty(1, 192, h//16, w//16).cuda()]
        return dpb

    def update_dpb(self, dpb, x, x_hat, y_features, est_mv=None, mv_features=None, mv_y_hat=None):
        if est_mv is not None:
            dpb["prior_mv"] = dpb["prior_mv"][1:] + [est_mv]
            dpb["prior_mv_feat"] = dpb["prior_mv_feat"][1:] + [mv_y_hat]
            dpb["ref_mv_feat"] = mv_features
        dpb["prior_frame"] = dpb["prior_frame"][1:] + [x_hat]
        dpb["prior_frame_gt"] = dpb["prior_frame_gt"][1:] + [x]
        dpb["prior_y_feat"] = dpb["prior_y_feat"][1:] + [y_features[-1]]
        dpb["ref_y_feat"] = y_features

    def forward(self, x, dpb, frame_idx=0):
        if frame_idx == 0:
            dpb = self.init_dpb(x)
            out = self.image_model(x)
            self.update_dpb(dpb, x, out["x_hat"], out["y_feat"])
            out["dpb"] = dpb
            return out
        elif frame_idx == 1:
            ref_y_feat_list = [self.feature_adaptor_I[i](feat) for i, feat in enumerate(dpb["ref_y_feat"])]
            dpb["prior_y_feat"][-1] = ref_y_feat_list[-1]
            # TODO high quality prior
            # mv_encoder_prior_list = [dpb["prior_frame_gt"][-1]] + [x]
            mv_encoder_prior_list = [dpb["prior_frame"][-1]] + [x]
            temporal_prior_list = [dpb["prior_y_feat"][-1]]
            temporal_prior = self.temporal_fusion_adaptor[0](torch.cat(temporal_prior_list, dim=1))
        elif frame_idx == 2:
            ref_y_feat_list = [self.feature_adaptor_P[i](feat) for i, feat in enumerate(dpb["ref_y_feat"])]
            dpb["prior_y_feat"][-1] = ref_y_feat_list[-1]
            # TODO high quality prior
            # mv_encoder_prior_list = dpb["prior_frame_gt"][1:] + [x] + [dpb["prior_mv"][-1]]
            mv_encoder_prior_list = dpb["prior_frame"][1:] + [x] + [dpb["prior_mv"][-1]]
            temporal_prior_list = dpb["prior_y_feat"][1:] + [dpb["prior_mv_feat"][-1]]
            temporal_prior = self.temporal_fusion_adaptor[1](torch.cat(temporal_prior_list, dim=1))
        else:
            ref_y_feat_list = [self.feature_adaptor_P[i](feat) for i, feat in enumerate(dpb["ref_y_feat"])]
            dpb["prior_y_feat"][-1] = ref_y_feat_list[-1]
            # TODO high quality prior
            # mv_encoder_prior_list = dpb["prior_frame_gt"] + [x] + dpb["prior_mv"]
            mv_encoder_prior_list = dpb["prior_frame"] + [x] + dpb["prior_mv"]
            temporal_prior_list = dpb["prior_y_feat"] + dpb["prior_mv_feat"]
            temporal_prior = self.temporal_fusion_adaptor[2](torch.cat(temporal_prior_list, dim=1))

        temporal_prior = self.temporal_fusion(temporal_prior)

        est_mv = self.optic_flow(x, dpb["prior_frame"][-1])
        warp_frame = flow_warp(dpb["prior_frame"][-1], est_mv)
        mv_encoder_prior_list += [est_mv, dpb["prior_frame"][-1], warp_frame]
        y_feat = self.contextual_encoder.feat_extractor(x)
        mv_y = self.mv_encoder(y_feat, ref_y_feat_list, mv_encoder_prior_list, frame_idx)

        mv_z = self.mv_hyper_prior_encoder(torch.cat([mv_y, temporal_prior], dim=1))
        mv_z_hat = self.quant(mv_z, 'ste')
        mv_params = self.mv_prior_param_decoder(mv_z_hat, temporal_prior)
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_four_part_prior(
            mv_y, mv_params, self.mv_y_spatial_prior_adaptor_1, self.mv_y_spatial_prior_adaptor_2,
            self.mv_y_spatial_prior_adaptor_3, self.mv_y_spatial_prior)

        mv_features, context1, context2, context3 = self.mv_decoder(mv_y_hat, ref_y_feat_list)

        y = self.contextual_encoder(y_feat, context1, context2, context3)
        
        z = self.contextual_hyper_prior_encoder(torch.cat([y, temporal_prior, context3], dim=1))
        z_hat = self.quant(z, 'ste')
        params = self.ctx_prior_param_decoder(z_hat, context3, temporal_prior)
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat, y_features = self.contextual_decoder(y_hat, context1, context2, context3)

        self.update_dpb(dpb, x, x_hat, y_features, est_mv, mv_features, mv_y_hat)

        N, _, H, W = x.size()
        pixel_num = N * H * W

        y_for_bit = y_res
        mv_y_for_bit = mv_y_res
        z_for_bit = z
        mv_z_for_bit = mv_z
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y) / pixel_num
        bpp_z = torch.sum(bits_z) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = torch.sum(bpp) * pixel_num

        return {"x_hat": x_hat,
                "bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "dpb": dpb,
                "bit": bit,
                }
