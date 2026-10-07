"""Shared paths and small helpers used by every script."""
from pathlib import Path
import random
import sys
import zlib

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDSAM_ROOT = PROJECT_ROOT / "MedSAM"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "medsam_vit_b.pth"

DATA = PROJECT_ROOT / "data"
ROCO_DIR = DATA / "roco"          # data/roco/<split>/images + metadata.csv
PSEUDO_DIR = DATA / "pseudo"      # data/pseudo/<split>/masks|heat + manifest_<split>.csv
EVAL_DIR = PROJECT_ROOT / "eval_sets"  # eval_sets/<name>/images|masks
RUNS = PROJECT_ROOT / "runs"

HF_DATASET = "eltorio/ROCOv2-radiology"
BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def add_medsam_to_path():
    p = str(MEDSAM_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


def set_seed(seed: int = 0):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rel(p) -> str:
    """Path relative to project root, posix style (stored in manifests)."""
    return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()


def fold_of(image_id, n_folds: int) -> int:
    """Stable fold assignment, used for cross-fitted self-training."""
    return zlib.crc32(str(image_id).encode()) % n_folds


def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def fit_mask(mask: np.ndarray, shape) -> np.ndarray:
    """Resize a boolean mask to (H, W) with nearest neighbour if sizes differ."""
    if mask.shape[:2] == tuple(shape[:2]):
        return mask
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(im.resize((shape[1], shape[0]), Image.NEAREST)) > 127


def dice_iou(pred, gt, eps=1e-6):
    """Dice and IoU for two boolean masks. Both empty counts as a perfect match."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float((pred & gt).sum())
    ps, gs = float(pred.sum()), float(gt.sum())
    if ps == 0 and gs == 0:
        return 1.0, 1.0
    return (2 * inter + eps) / (ps + gs + eps), (inter + eps) / (ps + gs - inter + eps)


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean (per-image scores are noisy on small eval sets)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def box_metrics(mask, boxes):
    """Mask vs ground-truth BOXES [[x0, y0, x1, y1], ...] (for box-only annotations).
    box_iou: IoU of the mask's bounding box with the best GT box (or the union of GT boxes); inside: fraction of
    mask pixels inside a GT box; cover: fraction of the best GT box covered; found: mask touches a GT box."""
    H, W = mask.shape
    gt = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in boxes:
        gt[int(y0):int(y1), int(x0):int(x1)] = True
    if not mask.any():
        return dict(box_iou=0.0, inside=0.0, cover=0.0, found=False)
    ys, xs = np.where(mask)
    mb = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]

    def biou(a, b):
        iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
        return iw * ih / max(u, 1e-6)
    union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    ious = [biou(mb, b) for b in boxes] + ([biou(mb, union)] if len(boxes) > 1 else [])
    best_box = boxes[int(np.argmax([biou(mb, b) for b in boxes]))]
    g1 = np.zeros((H, W), bool)
    g1[int(best_box[1]):int(best_box[3]), int(best_box[0]):int(best_box[2])] = True
    return dict(box_iou=float(max(ious)), inside=float((mask & gt).sum() / mask.sum()),
                cover=float((mask & g1).sum() / g1.sum()), found=bool((mask & gt).any()))
