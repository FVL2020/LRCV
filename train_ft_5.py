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
from src.models_for_train.image_model_quad import IntraModel
from src.datasets.dataset import Vimeo90KDataset, UVGDataset, ImageFolder


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        # self.lambda_I = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483]
        self.lambda_I = [845, 1625, 3140, 6060, 11705]
        self.lambda_P = [128, 256, 512, 1024, 2048, 4096]
    def forward(self, output, target, lmbda, rd_weight=1):
        out = {}
        # N, _, H, W = target.size()
        # num_pixels = N * H * W

        out["bpp_loss"] = output['bpp']
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["rd_loss"] = lmbda * out["mse_loss"] + out["bpp_loss"]
        out["loss"] = rd_weight * out["rd_loss"]
        # out["loss"] = out["loss"] / out["mse_loss"].detach()

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

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def reload_dataset(num_frames):
    train_dataset = Vimeo90KDataset(num_frame=num_frames)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
    )
    test_dataset = UVGDataset(num_frame=96)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader

def save_checkpoint(state, is_best, filename="cp"):
    filepath = f"checkpoints/{filename}.pth.tar"
    filepath_best = f"checkpoints/{filename}_best.pth.tar"
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, filepath_best)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=2,
        help="Number of training vimeo frame per sequence (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=False, help="Save model to disk"
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Checkpoint name"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--checkpoint-image-model", type=str, help="Path to a checkpoint")
    parser.add_argument('--retrain', action="store_true", help='load models from single rate and retrain')
    parser.add_argument('--freeze-flownet', action="store_true", help='freeze spynet')
    parser.add_argument('--freeze-image-model', action="store_true", help='freeze image model')
    args = parser.parse_args(argv)
    return args

def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, args):
    model.train()
    device = next(model.parameters()).device
    if args.num_frames == 2:
        rd_weight = [0, 1.0]
        lambda_I = [435, 845, 1625, 3140]
        lambda_P = [64, 128, 256, 512, 840]
        # lambda_P = [8192]
    elif args.num_frames == 3:
        rd_weight = [1.0, 0.5, 1.2]
        lambda_I = [435, 845, 1625, 3140]
        lambda_P = [64, 128, 256, 512, 840]
        # lambda_P = [4096]
    else:
        rd_weight = [1.0, 0.5, 1.2, 0.5, 0.9, 0.5, 1.1, 0.5]
        lambda_I = [435, 845, 1625, 3140, 6060]
        lambda_P = [64, 128, 256, 380, 840]
    # lambda_I = [117, 227, 435, 845, 1625, 3140, 6060, 11705]
    # lambda_P = [85, 170, 380, 840]
    # lambda_P = [256, 512, 1024, 2048]
    for i, d in enumerate(train_dataloader):
        optimizer.zero_grad()

        # frame = d[0].to(device)
        # out = model(frame)
        # out_criterion = criterion(out, frame, lambda_I[-1], rd_weight[0])
        # out_criterion["loss"].backward()
        # rd_loss_per_seq = out_criterion["loss"]
        rd_loss_per_seq = 0.
        for frame_idx, frame in enumerate(d):
            frame = frame.to(device)
            if frame_idx == 0:
                out = model(frame, dpb=None, frame_idx=0)
                dpb = out["dpb"]
                out_criterion = criterion(out, frame, lambda_I[-2], rd_weight[0])
                # ref_psnr = -10.0 * np.log10(out_criterion["mse_loss"].item())
            else:
                out = model(frame, dpb, frame_idx)
                dpb = out["dpb"]
                # rd_weight = (ref_psnr / (-10.0 * np.log10(out_criterion["mse_loss"].item()))) ** 4
                out_criterion = criterion(out, frame, lambda_P[-2], rd_weight[frame_idx])
            rd_loss_per_seq += out_criterion["loss"]
            if "offset_loss" in out and epoch <= 5:
                # print(out["offset_loss"], out_criterion["loss"])
                rd_loss_per_seq += out["offset_loss"]
        rd_loss_per_seq.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        if i % 1000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d[0])}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\t\tLoss: {rd_loss_per_seq.item():.4f} |'
                f'\tRD loss: {out_criterion["rd_loss"].item():.4f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
                , flush=True
            )

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
                out["x_hat"] = out["x_hat"][:, :, 0:h - mod_pad_h, 0:w - mod_pad_w]
                out_criterion = criterion(out, frame, 1024)
                psnr_per_folder.update(-10.0 * np.log10(out_criterion["mse_loss"].item()))
                bpp_per_folder.update(out_criterion["bpp_loss"].item())
                mse_loss.update(out_criterion["mse_loss"].item())
                loss.update(out_criterion["loss"].item())
                # if frame_idx % 32 < 7 or frame_idx % 32 >= 25:
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

def test_kodak(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        loss = AverageMeter()
        psnr = AverageMeter()
        bpp_loss = AverageMeter()
        mse_loss = AverageMeter()
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            out = model(img)
            out_criterion = criterion(out, img, 3140)
            loss.update(out_criterion["loss"].item())
            psnr.update(-10.0 * np.log10(out_criterion["mse_loss"].item()))
            mse_loss.update(out_criterion["mse_loss"].item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
        
        print(
            f"Test epoch {epoch}:"
            f"\tLoss: {loss.avg:.4f} |"
            f"\t\tPSNR: {psnr.avg:.4f} |"
            f"\t\tMSE loss: {mse_loss.avg:.6f} |"
            f"\t\tBpp loss: {bpp_loss.avg:.4f} |"
        )

    return loss.avg

def main(argv):
    args = parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    last_epoch = 0
    best_loss = float("inf")

    if args.num_frames == 1:
        train_dataset = Vimeo90KDataset(num_frame=1)
        test_dataset = ImageFolder(args.dataset)
        test = test_kodak
        net = IntraModel()
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10)
    elif args.num_frames == 14:
        train_dataset = Vimeo90KDataset(num_frame=7, flip_sequence=True)
        test_dataset = UVGDataset()
        test = test_uvg
        net = RTVC()
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.5)
    elif args.num_frames == 2:
        train_dataset = Vimeo90KDataset(num_frame=args.num_frames)
        test_dataset = UVGDataset(num_frame=args.num_frames if args.num_frames < 5 else 96)
        test = test_uvg
        net = RTVC()
        net = net.to(device)
        opt_params = []
        for n, p in net.named_parameters():
            if 'image_model' not in n and 'optic_flow' not in n:
                opt_params.append(p)
        optim_params = [
            {
                'params': opt_params,
                'lr': args.learning_rate
            },
        ]
        optimizer = optim.Adam(optim_params, lr=args.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.5)
    else:
        train_dataset = Vimeo90KDataset(num_frame=args.num_frames)
        test_dataset = UVGDataset(num_frame=args.num_frames if args.num_frames < 5 else 96)
        test = test_uvg
        net = RTVC()
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.5)
    
    for param in net.parameters():
        param.requires_grad = True

    if args.freeze_image_model:
        print("Freeze image model...")
        for n, p in net.named_parameters():
            if 'image_model' in n:
                p.requires_grad = False

    if args.freeze_flownet:
        print("Freeze flownet...")
        for n, p in net.named_parameters():
            if 'optic_flow' in n:
                p.requires_grad = False

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    if args.checkpoint:  # load from previous checkpoint
        if args.retrain:
            print("Loading single lambda pretrained checkpoint: ", args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if "state_dict" in checkpoint:
                ckpt = checkpoint["state_dict"]
            else:
                ckpt = checkpoint
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in ckpt.items() if
                               k in model_dict.keys() and v.shape == model_dict[k].shape}

            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
        else:
            print("Loading: ", args.checkpoint)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            last_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint["best_loss"]
            try:
                net.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
            except RuntimeError or ValueError:
                print("!!!Model does not match the checkpoint exactly, check it if necessary!!!")
                ckpt = checkpoint["state_dict"]
                model_dict = net.state_dict()
                pretrained_dict = {k: v for k, v in ckpt.items() if
                                k in model_dict.keys() and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                net.load_state_dict(model_dict)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if args.checkpoint_image_model:  # load from previous checkpoint
        print("Loading pretrained image model: ", args.checkpoint_image_model)
        checkpoint = torch.load(args.checkpoint_image_model, map_location=device)
        net.image_model.load_state_dict(checkpoint["state_dict"], strict=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    
    criterion = RateDistortionLoss()
    
    for epoch in range(0, args.epochs):
        # for n, p in net.named_parameters():
        #     if 'term_prior' in n:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False
        if epoch == 0:
            # if ('_teacher' in n):
            #     p.requires_grad = True
            # else:
            #     p.requires_grad = False
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00005
                # args.num_frames = 3
                # print(args.num_frames)
                # train_dataloader, test_dataloader = reload_dataset(num_frames=3)
        if epoch == 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00005
                # args.num_frames = 5
                # print(args.num_frames)
                # train_dataloader, test_dataloader = reload_dataset(num_frames=5)
        if epoch == 4:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.000025
        if epoch == 6:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001
            # print("Unfreeze image model...")
            # for n, p in net.named_parameters():
            #     if 'image_model' in n:
            #         p.requires_grad = True
        # if epoch == 8:
            # print("Unfreeze flownet...")
            # for n, p in net.named_parameters():
            #     if 'optic_flow' in n:
            #         p.requires_grad = True


        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        t1 = time.time()
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            args
        )
        t2 = time.time()
        # loss = test(epoch, test_dataloader, net, criterion)
        loss = 1
        t3 = time.time()
        if args.num_frames == 1:
            lr_scheduler.step(loss)
        else:
            lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        print(f"Epoch: {epoch}, loss: (now:{loss:.4f})/(best:{best_loss:.4f}), training time: {t2 - t1:.2f}s, testing time: {t3 - t2:.2f}s\n")

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "loss": loss,
					"best_loss": best_loss,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best, filename=args.name
            )


if __name__ == "__main__":
    main(sys.argv[1:])
