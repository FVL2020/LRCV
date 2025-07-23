import os
import cv2
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .img_utils import imfrombytes, img2tensor, imresize
from .fileclient import FileClient


def mod_crop(img, scale):
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img

def paired_random_crop(img_gts, gt_patch_size):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_gt, w_gt = img_gts[0].shape[0:2]

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_gt - gt_patch_size)
    left = random.randint(0, w_gt - gt_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_gts = [v[:, :, top:top + gt_patch_size, left:left + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top:top + gt_patch_size, left:left + gt_patch_size, ...] for v in img_gts]

    return img_gts

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs

def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img

class ImageFolder(Dataset):

    def __init__(self, root='data/kodak'):
        splitdir = Path(root)
        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')
        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())
        self.transform = T.ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.samples)
    
class Vimeo90KDataset(Dataset):

    def __init__(self, root='data/vimeo_septuplet/', num_frame=7, flip_sequence=False):
        super(Vimeo90KDataset, self).__init__()
        self.file_client = FileClient()
        self.root = root
        self.flip_sequence = flip_sequence
        self.num_frame = num_frame
        self.seqlist = []
        self.neighbor_list = [1, 2, 3, 4, 5, 6, 7]
        self.gap_neighbor_list_2frame = [[1, 2, 3, 4, 5, 6, 7], [1, 3, 5, 7], [1, 4, 7], [2, 4, 6], [1, 5], [2, 6], [3, 7], [1, 6], [2, 7], [1, 7]]
        self.gap_neighbor_list_3frame = [[1, 2, 3, 4, 5, 6, 7], [1, 3, 5, 7], [1, 4, 7], [2, 4, 6]]

        data_path = root + "sequences"
        trainlist_path = root + "sep_trainlist.txt"
        testlist_path = root + "sep_testlist.txt"
        with open(trainlist_path) as f:
            train_seq = f.readlines()
        with open(testlist_path) as f:
            test_seq = f.readlines()
        for _, line in enumerate(train_seq + test_seq, 1):
            self.seqlist += [os.path.join(data_path, line.rstrip())]
        print("Training dataset find sequences:", len(self.seqlist), "train frames:", num_frame)

    def __len__(self):
        return len(self.seqlist)
    
    def __getitem__(self, index):
        neighbor_list = self.neighbor_list

        if self.num_frame == 2:
            neighbor_list = self.gap_neighbor_list_2frame[random.randint(0, 9)]
        if self.num_frame == 3:
            neighbor_list = self.gap_neighbor_list_3frame[random.randint(0, 3)]

        # random reverse
        if random.random() < 0.5:
            neighbor_list.reverse()

        seqpath = self.seqlist[index]
        start_idx = random.randint(0, len(neighbor_list) - self.num_frame)

        # get the neighboring GT frames
        img_gts = []
        for neighbor in neighbor_list[start_idx : start_idx + self.num_frame]:
            img_gt_path = seqpath + f'/im{neighbor}.png'
            img_gt = cv2.imread(img_gt_path)
            # img_bytes = self.file_client.get(img_gt_path, 'gt')
            # img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts = paired_random_crop(img_gts, gt_patch_size=256)
        # augmentation - flip, rotate
        img_results = augment(img_gts, hflip=True, rotation=True)
        img_results = img2tensor(img_results)
        if self.flip_sequence:  # flip the sequence: 7 frames to 14 frames
            img_results = img_results + img_results[::-1]

        # img_gts = torch.stack(img_results, dim=0)

        # if self.flip_sequence:  # flip the sequence: 7 frames to 14 frames
        #     img_gts = torch.cat([img_gts, img_gts.flip(0)], dim=0)

        return img_results

class UVGDataset(Dataset):

    def __init__(self, root='data/UVG/', folderlist='data/UVG/folders.txt', num_frame=96):
        super(UVGDataset, self).__init__()
        self.file_client = FileClient()
        self.root = root
        self.num_frame = num_frame
        self.folderlist = []

        with open(folderlist) as f:
            folders = f.readlines()
        for folder in folders:
            seq = folder.rstrip()
            frame_list = {"name": seq.split("_")[0],
                          "path": [os.path.join(root, seq, f'im{i + 1:05d}.png') for i in range(num_frame)]}
            self.folderlist.append(frame_list)
            # print(frame_list["name"], frame_list["path"][:3])
        print(f"Testing dataset contain {len(self.folderlist)} folders, {num_frame} frames / folder", )

    def __len__(self):
        return len(self.folderlist)
    
    def __getitem__(self, index):
        folder = self.folderlist[index]
        img_paths = folder["path"]
        folder_name = folder["name"]
        # get the neighboring GT frames
        img_gts = []
        for path in img_paths:
            img_bytes = self.file_client.get(path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=False)
            img_gts.append(img_gt)

        img_results = img2tensor(img_gts)

        return {"name": folder_name, "data": img_results}


class UVGDatasetFull(Dataset):

    def __init__(self, root='data/UVG_bt601/', folderlist='data/UVG/folders.txt', num_frame=600):
        super(UVGDatasetFull, self).__init__()
        self.file_client = FileClient()
        self.root = root
        self.num_frame = num_frame
        self.folderlist = []

        with open(folderlist) as f:
            folders = f.readlines()
        for folder in folders:
            seq = folder.rstrip()
            if seq.split("_")[0] == 'ShakeNDry':
                num_frame = 300
            else:
                num_frame = 600
            folderlist = [os.path.join(root, seq, f'im{i + 1:05d}.png') for i in range(96)]
            self.folderlist += [folderlist[i:i+32] for i in range(0, 96, 32)]
            # frame_list = {"name": seq.split("_")[0],
            #               "path": [os.path.join(root, seq, f'im{i + 1:05d}.png') for i in range(num_frame)]}
            # self.folderlist.append(frame_list)
            # print(frame_list["name"], frame_list["path"][:3])
        print(f"Testing dataset contain {len(self.folderlist)} GOPs")

    def __len__(self):
        return len(self.folderlist)
    
    def __getitem__(self, index):
        folder = self.folderlist[index]
        img_paths = folder
        # get the neighboring GT frames
        img_gts = []
        for path in img_paths:
            img_bytes = self.file_client.get(path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=False)
            img_gts.append(img_gt)

        img_results = img2tensor(img_gts)


        return img_results

class UVGPatchDataset(Dataset):

    def __init__(self, root='data/UVG/', folderlist='data/UVG/folders.txt', num_frame=96, scale=4, type='patch'):
        super(UVGPatchDataset, self).__init__()
        self.file_client = FileClient()
        self.root = root
        self.num_frame = num_frame
        self.folderlist = []
        self.type = type
        self.scale = scale
        with open(folderlist) as f:
            folders = f.readlines()
        for folder in folders:
            seq = folder.rstrip()
            frame_list = {"name": seq.split("_")[0],
                          "path": [os.path.join(root, seq, f'im{i + 1:05d}.png') for i in range(num_frame)]}
            self.folderlist.append(frame_list)
            # print(frame_list["name"], frame_list["path"][:3])
        print(f"Testing dataset contain {len(self.folderlist)} folders, {num_frame} frames / folder", )

    def __len__(self):
        return len(self.folderlist)
    
    def __getitem__(self, index):
        folder = self.folderlist[index]
        img_paths = folder["path"]
        folder_name = folder["name"]
        # get the neighboring GT frames
        img_gts = []
        h, w = 1080, 1920
        h_remainder, w_remainder = h % (64 * self.scale), w % (64 * self.scale)
        self.w = (w - w_remainder) // self.scale
        self.h = (h - h_remainder) // self.scale

        for path in img_paths:
            img_bytes = self.file_client.get(path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=False)
            img_gt = img_gt[:h - h_remainder, :w - w_remainder, ...]
            img_gts.append(img_gt)

        img_results = img2tensor(img_gts)
        if self.type == 'patch':
            img_results = [im.reshape(3, self.scale, self.h, self.scale, self.w).permute(1, 3, 0, 2, 4)\
                           .reshape(-1, 3, self.h, self.w) for im in img_results]
        else:
            img_results = [imresize(im, 1/self.scale).unsqueeze(0) for im in img_results]

        return {"name": folder_name, "data": img_results}
    
if __name__ == '__main__':
    # print(shape2coordinates((3,3)).shape)
    vimeo = UVGPatchDataset(num_frame=12, scale=2, type='resize')
    train_loader = DataLoader(vimeo, batch_size=1, shuffle=True)
    # kodak = ImageFolder()
    # train_loader = DataLoader(kodak, batch_size=1, shuffle=True)
    # uvg = UVGDataset()
    # train_loader = DataLoader(uvg, batch_size=1, shuffle=True)
    for features in train_loader:
        # print(features["name"])
        print(features["data"][0].shape)
        # print(features[0].shape)
        from torchvision.utils import save_image
        save_image(features["data"][0][0][0], '1.png')
        # save_image(features["data"][0][0][1], '2.png')
        # save_image(features["data"][0][0][2], '3.png')
        # save_image(features["data"][0][0][3], '4.png')
        # print(features[0].shape)
        # print(features[1].shape)
        break

