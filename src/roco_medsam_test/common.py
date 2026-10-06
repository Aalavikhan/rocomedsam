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
