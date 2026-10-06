"""Image-ONLY student: UNet with a timm encoder, trained on teacher pseudo-masks.

Heads: `seg` (the mask) and `heat` (auxiliary: regress the teacher's text-derived
saliency map -- this is how extra text-derived signal reaches the image branch).
At test time only `seg` is used and no text is touched.
"""
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .common import IMAGENET_MEAN, IMAGENET_STD, PROJECT_ROOT


class ConvBlock(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        )


class UNet(nn.Module):
    def __init__(self, encoder="resnet34", pretrained=True):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        chs = self.enc.feature_info.channels()
        dec = [256, 128, 64, 32, 32, 32][: len(chs) - 1]
        self.blocks = nn.ModuleList()
        in_c = chs[-1]
        for skip_c, out_c in zip(reversed(chs[:-1]), dec):
            self.blocks.append(ConvBlock(in_c + skip_c, out_c))
            in_c = out_c
        self.seg = nn.Conv2d(in_c, 1, 1)
        self.heat = nn.Conv2d(in_c, 1, 1)

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.enc(x)
        y = feats[-1]
        for blk, skip in zip(self.blocks, reversed(feats[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        return self.seg(y), self.heat(y)


def load_student(ckpt_path, device, return_meta=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = UNet(ck["encoder"], pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    if return_meta:
        return model, ck["size"], {k: ck.get(k) for k in ("fold", "nfolds", "epoch", "val")}
    return model, ck["size"]


@torch.no_grad()
def predict_prob(model, img, size, device, tta=False):
    """img: HxWx3 uint8 -> HxW float probability map (image-only inference)."""
    H, W = img.shape[:2]
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
    p = torch.sigmoid(model(t)[0])
    if tta:
        p = (p + torch.sigmoid(model(t.flip(-1))[0]).flip(-1)) / 2
    p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
    return p[0, 0].cpu().numpy()


def load_manifest(path, min_iou=0.8, min_gain=0.0, min_area=0.003, max_area=0.5, min_peak=0.0):
    """Filter teacher output. Normal images (empty masks) are always kept."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    is_norm = df.kind == "normal"
    ok = (
        (df.status == "ok")
        & (num("sam_iou") >= min_iou)
        & (num("crop_gain") >= min_gain)
        & num("area_frac").between(min_area, max_area)
    )
    if min_peak > 0 and "heat_peak" in df:
        ok &= num("heat_peak") >= min_peak
    return df[is_norm | ok].reset_index(drop=True)


class PseudoDataset(Dataset):
    def __init__(self, df, size=384, train=True):
        self.df, self.size, self.train = df, size, train
        self.aug = None
        if train:
            import albumentations as A
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.85, 1.15), rotate=(-15, 15), translate_percent=(-0.1, 0.1), p=0.6),
                A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        S = self.size
        img = np.array(Image.open(PROJECT_ROOT / r.image_path).convert("RGB"))
        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR)
        if r.mask_path:
            m = np.array(Image.open(PROJECT_ROOT / r.mask_path).convert("L"))
            m = (cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
        else:
            m = np.zeros((S, S), np.float32)
        if r.heat_path:
            h = np.load(PROJECT_ROOT / r.heat_path).astype(np.float32)
            h = np.clip(cv2.resize(h, (S, S), interpolation=cv2.INTER_CUBIC), 0, 1)
            hv = 1.0
        elif r.kind == "normal":
            h, hv = np.zeros((S, S), np.float32), 1.0
        else:
            h, hv = np.zeros((S, S), np.float32), 0.0   # no heat target -> ignore heat loss
        if self.aug is not None:
            out = self.aug(image=img, mask=np.dstack([m, h]))
            img, mh = out["image"], out["mask"]
            m, h = mh[..., 0], mh[..., 1]
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        w = float(r.weight) if r.weight != "" else 1.0
        return (
            torch.from_numpy(x).permute(2, 0, 1),
            torch.from_numpy(np.stack([m, h])).float(),
            torch.tensor(w, dtype=torch.float32),
            torch.tensor(hv, dtype=torch.float32),
        )


def seg_loss(logit, target, w, bce_w=0.5):
    """Sample-weighted Dice + BCE (noise-tolerant vs pure CE)."""
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none").mean((1, 2, 3))
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return ((dice + bce_w * bce) * w).sum() / w.sum().clamp(min=1e-6)


def heat_loss(logit, target, w, hv):
    wt = w * hv
    mse = F.mse_loss(torch.sigmoid(logit), target, reduction="none").mean((1, 2, 3))
    return (mse * wt).sum() / wt.sum().clamp(min=1e-6)
