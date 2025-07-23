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
from src.datasets.dataset import UVGPatchDataset, UVGDataset, ImageFolder


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

def save_checkpoint(state, is_best, filename="cp"):
    filepath = f"checkpoints/{filename}.pth.tar"
    filepath_best = f"checkpoints/{filename}_best.pth.tar"
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, filepath_best)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--index",
        default=0,
        type=int,
        help="QP index (default: %(default)s)",
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
        "--gop",
        type=int,
        default=5,
        help="Number of training gop (default: %(default)s)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=96,
        help="Number of test UVG frames(default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--type", type=str, required=True, help="Dataset type for saving memory"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Scale for saving memory (default: %(default)s)",
    )
    parser.add_argument(
        "--method", type=str, required=True, help="Online updating method"
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
    args = parser.parse_args(argv)
    return args

def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, gop):
    MAX_BATCH = 4
    model.train()
    device = next(model.parameters()).device
    rd_weight = [1.0, 0.5, 1.2, 0.5, 0.9, 0.5, 1.1, 0.5]
    lambda_I = [845, 1625, 3140, 3140]
    lambda_P = [85, 170, 380, 840]
    for i, folder in enumerate(train_dataloader):
        if i != 5:
            continue
        print(f'Training sequence {i + 1}: {folder["name"][0]}...', flush=True)
        optimizer.zero_grad()
        rd_loss_per_seq = 0.
        frames = folder["data"]
        b = frames[0][0].shape[0]
        for n in range(math.ceil(b / MAX_BATCH)):
            for frame_idx, frame in enumerate(frames):
                if frame_idx % 32 >= gop:
                    continue
                frame = frame[0].to(device)
                frame_slice = frame[n * MAX_BATCH : (n + 1) * MAX_BATCH]
                if frame_idx % 32 == 0:
                    out = model(frame_slice, dpb=None, frame_idx=frame_idx % 32)
                    dpb = out["dpb"]
                    out_criterion = criterion(out, frame_slice, lambda_I[-1], rd_weight[0])
                    rd_loss_per_seq += out_criterion["loss"]
                    # ref_psnr = -10.0 * np.log10(out_criterion["mse_loss"].item())
                elif frame_idx % 32 == 4:
                    out = model(frame_slice, dpb, frame_idx=frame_idx % 32)
                    dpb = out["dpb"]
                    out_criterion = criterion(out, frame_slice, lambda_P[-1], rd_weight[frame_idx % 32])
                    rd_loss_per_seq += out_criterion["loss"]
                    # ref_psnr = -10.0 * np.log10(out_criterion["mse_loss"].item())
                    rd_loss_per_seq.backward()
                    if clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    rd_loss_per_seq = 0.
                else:
                    out = model(frame_slice, dpb, frame_idx % 32)
                    dpb = out["dpb"]
                    # rd_weight = (ref_psnr / (-10.0 * np.log10(out_criterion["mse_loss"].item()))) ** 4
                    out_criterion = criterion(out, frame_slice, lambda_P[-1], rd_weight[frame_idx % 32])
                    rd_loss_per_seq += out_criterion["loss"]
    # for i, folder in enumerate(train_dataloader):
    #     if i != 5:
    #         continue
    #     print(f'Training sequence {i + 1}: {folder["name"][0]}...', flush=True)
    #     optimizer.zero_grad()
    #     rd_loss_per_seq = 0.
    #     frames = folder["data"]
    #     b = frames[0][0].shape[0]
    #     for n in range(math.ceil(b / MAX_BATCH)):
    #         for frame_idx, frame in enumerate(frames):
    #             frame = frame[0].to(device)
    #             frame_slice = frame[n * MAX_BATCH : (n + 1) * MAX_BATCH]
    #             if frame_idx % gop == 0:
    #                 out = model(frame_slice, dpb=None, frame_idx=frame_idx % gop)
    #                 dpb = out["dpb"]
    #                 out_criterion = criterion(out, frame_slice, lambda_I[-1], rd_weight[0])
    #                 rd_loss_per_seq += out_criterion["loss"]
    #                 # ref_psnr = -10.0 * np.log10(out_criterion["mse_loss"].item())
    #             elif frame_idx % gop == gop - 1:
    #                 out = model(frame_slice, dpb, frame_idx=frame_idx % gop)
    #                 dpb = out["dpb"]
    #                 out_criterion = criterion(out, frame_slice, lambda_P[-1], rd_weight[frame_idx % gop])
    #                 rd_loss_per_seq += out_criterion["loss"]
    #                 # ref_psnr = -10.0 * np.log10(out_criterion["mse_loss"].item())
    #                 rd_loss_per_seq.backward()
    #                 if clip_max_norm > 0:
    #                     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #                 rd_loss_per_seq = 0.
    #             else:
    #                 out = model(frame_slice, dpb, frame_idx % gop)
    #                 dpb = out["dpb"]
    #                 # rd_weight = (ref_psnr / (-10.0 * np.log10(out_criterion["mse_loss"].item()))) ** 4
    #                 out_criterion = criterion(out, frame_slice, lambda_P[-1], rd_weight[frame_idx % gop])
    #                 rd_loss_per_seq += out_criterion["loss"]

        if i % 1 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
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
            # if i != 5:
            #     continue
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
                out_criterion = criterion(out, frame, 380)
                psnr_per_folder.update(-10.0 * np.log10(out_criterion["mse_loss"].item()))
                bpp_per_folder.update(out_criterion["bpp_loss"].item())
                mse_loss.update(out_criterion["mse_loss"].item())
                loss.update(out_criterion["loss"].item())
                print(f"Frame {frame_idx + 1}: PSNR={psnr_per_folder.val}, BPP={bpp_per_folder.val}", flush=True)

            print(f'Result of sequence {folder["name"][0]}: PSNR={psnr_per_folder.avg:.4f}, BPP={bpp_per_folder.avg:.4f}', flush=True)
            psnr.update(psnr_per_folder.avg, len(frames))
            bpp_loss.update(bpp_per_folder.avg, len(frames))
        
        torch.cuda.empty_cache()
        print(
            f"Test epoch {epoch}:"
            f"\tLoss: {loss.avg:.4f} |"
            f"\tPSNR: {psnr.avg:.4f} |"
            f"\tMSE loss: {mse_loss.avg:.6f} |"
            f"\tBpp loss: {bpp_loss.avg:.4f} |"
            , flush=True
        )

    return loss.avg

def train_oeu(model, criterion, train_dataloader, test_dataloader, epochs, clip_max_norm, train_gop, test_gop, refresh_cp, index):
    MAX_BATCH = 1
    loss = AverageMeter()
    psnr = AverageMeter()
    bpp_loss = AverageMeter()
    device = next(model.parameters()).device
    rd_weight = [1.0, 0.5, 1.2, 0.5, 0.9, 0.5, 1.1, 0.5]
    lambda_I = [845, 1625, 3140, 3140]
    lambda_P = [85, 170, 380, 840]

    # for i, folder, folder_test in enumerate(zip(train_dataloader, test_dataloader)):
    #     if i != 5:
    #         continue
    #     print(f'Training sequence {i + 1}: {folder["name"][0]}...', flush=True)
    #     print(f'Training sequence {i + 1}: {folder_test["name"][0]}...', flush=True)
    #     optimizer = refresh_cp(model)
    #     model.train()
    #     optimizer.zero_grad()
    #     rd_loss_per_seq = 0.
    #     frames = folder["data"]
    #     b = frames[0][0].shape[0]
    #     for n in range(math.ceil(b / MAX_BATCH)):
    #         for frame_idx, frame in enumerate(frames):
    #             if frame_idx % test_gop >= train_gop:
    #                 continue
    #             frame = frame[0].to(device)
    #             frame_slice = frame[n * MAX_BATCH : (n + 1) * MAX_BATCH]
    #             if frame_idx % test_gop == 0:
    #                 out = model(frame_slice, dpb=None, frame_idx=frame_idx % test_gop)
    #                 dpb = out["dpb"]
    #                 out_criterion = criterion(out, frame_slice, lambda_I[index], rd_weight[0])
    #                 rd_loss_per_seq += out_criterion["loss"]
    #                 # ref_psnr = -10.0 * np.log10(out_criterion["mse_loss"].item())
    #             elif frame_idx % test_gop == 4:
    #                 out = model(frame_slice, dpb, frame_idx=frame_idx % test_gop)
    #                 dpb = out["dpb"]
    #                 out_criterion = criterion(out, frame_slice, lambda_P[index], rd_weight[frame_idx % test_gop])
    #                 rd_loss_per_seq += out_criterion["loss"]
    #                 # ref_psnr = -10.0 * np.log10(out_criterion["mse_loss"].item())
    #                 rd_loss_per_seq.backward()
    #                 if clip_max_norm > 0:
    #                     torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #                 rd_loss_per_seq = 0.
    #             else:
    #                 out = model(frame_slice, dpb, frame_idx % test_gop)
    #                 dpb = out["dpb"]
    #                 # rd_weight = (ref_psnr / (-10.0 * np.log10(out_criterion["mse_loss"].item()))) ** 4
    #                 out_criterion = criterion(out, frame_slice, lambda_P[index], rd_weight[frame_idx % test_gop])
    #                 rd_loss_per_seq += out_criterion["loss"]

    for i, (folder, folder_test) in enumerate(zip(train_dataloader, test_dataloader)):
        # if i != 5:
        #     continue
        print(f'Training sequence {i + 1}: {folder["name"][0]}...', flush=True)
        optimizer = refresh_cp(model)
        optimizer.zero_grad()
        best_loss = [100, 0, 0]
        rd_loss_per_seq = 0.
        frames = folder["data"]
        b = frames[0][0].shape[0]

        for epoch in range(epochs):
            model.train()
            torch.cuda.synchronize()
            t1 = time.time()
            for n in range(math.ceil(b / MAX_BATCH)):
                for frame_idx, frame in enumerate(frames):
                    frame = frame[0].to(device)
                    frame_slice = frame[n * MAX_BATCH : (n + 1) * MAX_BATCH]
                    if frame_idx % train_gop == 0:
                        out = model(frame_slice, dpb=None, frame_idx=frame_idx % train_gop)
                        dpb = out["dpb"]
                        out_criterion = criterion(out, frame_slice, lambda_I[index], rd_weight[0])
                        rd_loss_per_seq += out_criterion["loss"]
                        # ref_psnr = -10.0 * np.log10(out_criterion["mse_loss"].item())
                    elif frame_idx % train_gop == train_gop - 1:
                        out = model(frame_slice, dpb, frame_idx=frame_idx % train_gop)
                        dpb = out["dpb"]
                        out_criterion = criterion(out, frame_slice, lambda_P[index], rd_weight[frame_idx % train_gop])
                        rd_loss_per_seq += out_criterion["loss"]
                        # ref_psnr = -10.0 * np.log10(out_criterion["mse_loss"].item())
                        rd_loss_per_seq.backward()
                        if clip_max_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        rd_loss_per_seq = 0.
                    else:
                        out = model(frame_slice, dpb, frame_idx % train_gop)
                        dpb = out["dpb"]
                        # rd_weight = (ref_psnr / (-10.0 * np.log10(out_criterion["mse_loss"].item()))) ** 4
                        out_criterion = criterion(out, frame_slice, lambda_P[index], rd_weight[frame_idx % train_gop])
                        rd_loss_per_seq += out_criterion["loss"]
            
            torch.cuda.synchronize()
            t2 = time.time()
            # print(
            #     f"Train epoch {epoch + 1}/{epochs}: "
            #     f'\tTime: {t2 - t1:.4f}'
            #     , flush=True
            # )

            model.eval()
            with torch.no_grad():
                loss_per_folder = AverageMeter()
                bpp_per_folder = AverageMeter()
                psnr_per_folder = AverageMeter()
                # print(f'Evaluating sequence {i + 1}: {folder_test["name"][0]}...', flush=True)
                dpb = None
                frames_test = folder_test["data"]
                for frame_idx, frame in enumerate(frames_test):
                    frame = frame.to(device)
                    # pad input image to be a multiple of window_size
                    _, _, h, w = frame.size()
                    mod_pad_h, mod_pad_w = 0, 0
                    if h % 64 != 0:
                        mod_pad_h = 64 - h % 64
                    if w % 64 != 0:
                        mod_pad_w = 64 - w % 64

                    frame_pad = torch.nn.functional.pad(frame, (0, mod_pad_w, 0, mod_pad_h), "replicate")
                    out = model(frame_pad, dpb, frame_idx % test_gop)
                    dpb = out["dpb"]

                    _, _, h, w = out["x_hat"].size()
                    out["x_hat"] = out["x_hat"][:, :, 0:h - mod_pad_h, 0:w - mod_pad_w].clamp_(0, 1)
                    out_criterion = criterion(out, frame, lambda_P[index])
                    psnr_per_folder.update(-10.0 * np.log10(out_criterion["mse_loss"].item()))
                    bpp_per_folder.update(out_criterion["bpp_loss"].item())
                    loss_per_folder.update(out_criterion["loss"].item())
                    # print(f"Frame {frame_idx + 1}: PSNR={psnr_per_folder.val}, BPP={bpp_per_folder.val}", flush=True)
    
            if loss_per_folder.avg < best_loss[0]:
                best_loss = [loss_per_folder.avg, psnr_per_folder.avg, bpp_per_folder.avg]
            print(f'Test epoch {epoch + 1}/{epochs}, Time: {t2 - t1:.4f}, Loss: {loss_per_folder.avg:.6f}/{best_loss[0]:.6f}, PSNR: {psnr_per_folder.avg:.4f}/{best_loss[1]:.4f}, BPP: {bpp_per_folder.avg:.4f}/{best_loss[2]:.4f}', flush=True)
            torch.cuda.empty_cache()

        loss.update(best_loss[0], len(frames_test))
        psnr.update(best_loss[1], len(frames_test))
        bpp_loss.update(best_loss[2], len(frames_test))
        
    print(
        f"OEU result:"
        f"\tLoss: {loss.avg:.6f} |"
        f"\tPSNR: {psnr.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        , flush=True
    )





def main(argv):
    args = parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataset = UVGPatchDataset(num_frame=args.num_frames, scale=args.scale, type=args.type)
    test_dataset = UVGDataset(num_frame=args.num_frames)
    test = test_uvg
    net = RTVC()
    net = net.to(device)

    online_update_params = []
    if args.method == 'oeu':
        # TODO
        for name, param in net.mv_encoder.named_parameters():
            online_update_params.append(param)
        for name, param in net.contextual_encoder.named_parameters():
            online_update_params.append(param)
    else:
        for name, param in net.mv_encoder.enc3[-1].named_parameters():
            online_update_params.append(param)
        for name, param in net.contextual_encoder.enc4[-1].named_parameters():
            online_update_params.append(param)
    optim_params = [
        {
            'params': online_update_params,
            'lr': args.learning_rate
        },
    ]
    optimizer = optim.Adam(optim_params, lr=args.learning_rate)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.1)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    
    def create_refresh_cp_func(checkpoint, optim_params, learning_rate, device):
        def refresh_cp(net: torch.nn.Module):
            print("Refresh: ", checkpoint)
            cp = torch.load(checkpoint, map_location=device)
            net.load_state_dict(cp["state_dict"])
            optimizer = optim.Adam(optim_params, lr=learning_rate)
            return optimizer

        return refresh_cp
    
    if args.checkpoint:  # load from previous checkpoint
        print("Loading: ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])


    refresh_cp = create_refresh_cp_func(args.checkpoint, optim_params, args.learning_rate, device)
        
    criterion = RateDistortionLoss()
    
    best_loss = 1
    best_loss = test(-1, test_dataloader, net, criterion)
    print(f'original loss: {best_loss:.4f}', flush=True)

    torch.cuda.synchronize()
    t1 = time.time()
    train_oeu(
        net,
        criterion,
        train_dataloader,
        test_dataloader,
        args.epochs,
        args.clip_max_norm,
        train_gop=args.gop,
        test_gop=args.num_frames,
        refresh_cp=refresh_cp,
        index=args.index
    )
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"OEU time: {t2 - t1:.2f}s\n", flush=True)
    
    
    # for epoch in range(0, args.epochs):
    #     # if args.checkpoint:  # load from previous checkpoint
    #     #     print("Loading: ", args.checkpoint)
    #     #     checkpoint = torch.load(args.checkpoint, map_location=device)
    #     #     try:
    #     #         net.load_state_dict(checkpoint["state_dict"])
    #     #         # TODO
    #     #         # optimizer.load_state_dict(checkpoint["optimizer"])
    #     #     except RuntimeError:
    #     #         print("!!!Model does not match the checkpoint exactly, check it if necessary!!!")
    #     #         ckpt = checkpoint["state_dict"]
    #     #         model_dict = net.state_dict()
    #     #         pretrained_dict = {k: v for k, v in ckpt.items() if
    #     #                         k in model_dict.keys() and v.shape == model_dict[k].shape}
    #     #         model_dict.update(pretrained_dict)
    #     #         net.load_state_dict(model_dict)
    #     print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    #     torch.cuda.synchronize()
    #     t1 = time.time()
    #     train_one_epoch(
    #         net,
    #         criterion,
    #         train_dataloader,
    #         optimizer,
    #         epoch,
    #         args.clip_max_norm,
    #         args.gop,
    #     )
    #     torch.cuda.synchronize()
    #     t2 = time.time()
    #     loss = test(epoch, test_dataloader, net, criterion)
    #     torch.cuda.synchronize()
    #     t3 = time.time()

    #     scheduler.step()
    #     best_loss = min(loss, best_loss)
    #     print(f"Epoch: {epoch}, loss: (now:{loss:.4f})/(best:{best_loss:.4f}), training time: {t2 - t1:.2f}s, testing time: {t3 - t2:.2f}s\n", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:])

'''
CUDA_VISIBLE_DEVICES=0 python test_oeu.py -e 10 -lr 1e-5 -n 1 --num-frames 50 --gop 5 \
--cuda --seed 1926 --clip_max_norm 1.0 --method oeu --type patch --scale 4 \
--checkpoint checkpoints/cp_video_model_dcn_840_IPPPPPP.pth.tar --index 3

CUDA_VISIBLE_DEVICES=0 python test_oeu.py -e 10 -lr 5e-5 -n 1 --num-frames 96 --gop 5 \
--cuda --seed 1926 --clip_max_norm 1.0 --method oeu --type resize --scale 4 \
--checkpoint checkpoints/cp_video_model_dcn_380_IPPPPPP_best.pth.tar --index 2
'''