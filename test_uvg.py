import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import math
import time
import random
import shutil
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models_for_train.video_model_dcn import RTVC
from src.datasets.dataset import Vimeo90KDataset, UVGDataset, ImageFolder, UVGDatasetFull


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        # self.lambda_I = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483]
        self.lambda_I = [117, 227, 435, 845, 1625, 3140]
        self.lambda_P = [85, 170, 380, 840]
    def forward(self, output, target, lmbda, rd_weight=1):
        out = {}
        # N, _, H, W = target.size()
        # num_pixels = N * H * W

        out["bpp_loss"] = output['bpp']
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["rd_loss"] = lmbda * out["mse_loss"] + out["bpp_loss"]
        out["loss"] = rd_weight * out["rd_loss"]

        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_uvg(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        loss = AverageMeter()
        psnr = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        for i, folder in enumerate(test_dataloader):
            bpp_per_folder = AverageMeter()
            psnr_per_folder = AverageMeter()
            print(f'Evaluating sequence {i + 1}: {folder["name"][0]}...', flush=True)
            dpb = None
            frames = folder["data"]
            for frame_idx, frame in enumerate(frames):
                frame = frame.to(device)
                # pad input image to be a multiple of window_size
                _, _, h, w = frame.size()
                mod_pad_h, mod_pad_w = 0, 0
                if h % 64 != 0:
                    mod_pad_h = 64 - h % 64
                if w % 64 != 0:
                    mod_pad_w = 64 - w % 64
                frame_pad = torch.nn.functional.pad(frame, (0, mod_pad_w, 0, mod_pad_h), "replicate")

                out = model(frame_pad, dpb, frame_idx % 32)
                dpb = out["dpb"]

                _, _, h, w = out["x_hat"].size()
                out["x_hat"] = out["x_hat"][:, :, 0:h - mod_pad_h, 0:w - mod_pad_w].clamp(0, 1)
                out_criterion = criterion(out, frame, 1024)
                psnr_per_folder.update(-10.0 * np.log10(out_criterion["mse_loss"].item()))
                bpp_per_folder.update(out_criterion["bpp_loss"].item())
                mse_loss.update(out_criterion["mse_loss"].item())
                loss.update(out_criterion["loss"].item())
                if frame_idx < 100:
                    print(f"Frame {frame_idx + 1}: PSNR={psnr_per_folder.val}, BPP={bpp_per_folder.val}")

            print(f'Result of sequence {folder["name"][0]}: PSNR={psnr_per_folder.avg:.4f}, BPP={bpp_per_folder.avg:.4f}', flush=True)
            psnr.update(psnr_per_folder.avg, len(frames))
            bpp_loss.update(bpp_per_folder.avg, len(frames))
        
        print(
            f"Test epoch {epoch}:"
            f"\tLoss: {loss.avg:.4f} |"
            f"\tPSNR: {psnr.avg:.4f} |"
            f"\tMSE loss: {mse_loss.avg:.6f} |"
            f"\tBpp loss: {bpp_loss.avg:.4f} |"
            , flush=True
        )

    return loss.avg

def test_uvg_full(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        loss = AverageMeter()
        psnr = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        for i, folder in enumerate(test_dataloader):
            bpp_per_folder = AverageMeter()
            psnr_per_folder = AverageMeter()
            print(f'Evaluating gop {i + 1}', flush=True)
            dpb = None
            frames = folder
            for frame_idx, frame in enumerate(frames):
                frame = frame.to(device)
                # pad input image to be a multiple of window_size
                _, _, h, w = frame.size()
                mod_pad_h, mod_pad_w = 0, 0
                if h % 64 != 0:
                    mod_pad_h = 64 - h % 64
                if w % 64 != 0:
                    mod_pad_w = 64 - w % 64
                frame_pad = torch.nn.functional.pad(frame, (0, mod_pad_w, 0, mod_pad_h), "replicate")

                out = model(frame_pad, dpb, frame_idx % 16)
                dpb = out["dpb"]

                _, _, h, w = out["x_hat"].size()
                out["x_hat"] = out["x_hat"][:, :, 0:h - mod_pad_h, 0:w - mod_pad_w]
                out_criterion = criterion(out, frame, 1024)
                psnr_per_folder.update(-10.0 * np.log10(out_criterion["mse_loss"].item()))
                bpp_per_folder.update(out_criterion["bpp_loss"].item())
                mse_loss.update(out_criterion["mse_loss"].item())
                loss.update(out_criterion["loss"].item())
                if frame_idx < 100:
                    print(f"Frame {frame_idx + 1}: PSNR={psnr_per_folder.val}, BPP={bpp_per_folder.val}")

            print(f'Result of gop {i + 1}: PSNR={psnr_per_folder.avg:.4f}, BPP={bpp_per_folder.avg:.4f}', flush=True)
            psnr.update(psnr_per_folder.avg, len(frames))
            bpp_loss.update(bpp_per_folder.avg, len(frames))
        
        print(
            f"Test epoch {epoch}:"
            f"\tLoss: {loss.avg:.4f} |"
            f"\tPSNR: {psnr.avg:.4f} |"
            f"\tMSE loss: {mse_loss.avg:.6f} |"
            f"\tBpp loss: {bpp_loss.avg:.4f} |"
            , flush=True
        )

    return loss.avg

def main(argv):
    test_dataset = UVGDataset(num_frame=96)
    # test_dataset = UVGDataset(root='data/ClassB', folderlist='data/ClassB.txt', num_frame=96)
    # test_dataset = UVGDataset(root='data/MCL-JCV/YUV_source/', folderlist='data/MCL-JCV.txt', num_frame=96)
    test = test_uvg
    # test_dataset = UVGDatasetFull(root='data/UVG_bt601/', num_frame=600)
    # test = test_uvg_full
    # test_dataset = UVGDatasetFull(root='data/ClassB_bt601', folderlist='data/ClassB.txt', num_frame=300)
    # test_dataset = UVGDatasetFull(root='data/MCL-JCV_bt601/', folderlist='data/MCL-JCV.txt', num_frame=300)
    # test = test_uvg_full
    net = RTVC()
    net = net.cuda()

    cp_list = ['checkpoints/cp_video_model_dcvc_lite_IPPPP.pth.tar']
    # cp_list = ['checkpoints/cp_video_model_dcn_85_IPPPPPP_best.pth.tar', 'checkpoints/cp_video_model_dcn_170_IPPPPPP_best.pth.tar', 
    #            'checkpoints/cp_video_model_dcn_380_IPPPPPP_best.pth.tar', 'checkpoints/cp_video_model_dcn_840_new_IPPPPPP_best.pth.tar']
    for cp in cp_list:
        checkpoint = torch.load(cp, map_location='cuda')
        if "state_dict" in checkpoint:
            ckpt = checkpoint["state_dict"]
        else:
            ckpt = checkpoint
        # model_dict = net.state_dict()
        # pretrained_dict = {k: v for k, v in ckpt.items() if
        #                     k in model_dict.keys() and v.shape == model_dict[k].shape}

        # model_dict.update(pretrained_dict)
        net.load_state_dict(ckpt)
        # net.update(force=True)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )
        
        criterion = RateDistortionLoss()
        t1 = time.time()
        loss = test(0, test_dataloader, net, criterion)
        t2 = time.time()

        print(f"loss: (now:{loss:.4f}), testing time: {t2 - t1:.2f}s\n")


if __name__ == "__main__":
    main(sys.argv[1:])
