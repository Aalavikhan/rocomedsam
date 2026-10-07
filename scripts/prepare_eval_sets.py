"""Convert datasets with real masks into eval_sets/<name>/{images,masks}.

BUSI (Kaggle 'Breast Ultrasound Images Dataset'; benign+malignant only):
    uv run python scripts/prepare_eval_sets.py --kind busi --src D:/datasets/Dataset_BUSI_with_GT \
        --name busi --prompt ultrasound "breast lesion"

Any 'images dir + masks dir' dataset (Kvasir-SEG, COVID-QU-Ex infection masks, a CT lesion set, ...):
    uv run python scripts/prepare_eval_sets.py --kind folder --images D:/ds/images --masks D:/ds/masks \
        --name kvasir --prompt endoscopy polyp
Mask file may be '<stem>.png' or '<stem>_mask.png' (any extension). Masks are binarised
(>127 for 8-bit masks, so JPEG compression noise is ignored; >0 for 0/1 masks).
Only images that have a non-empty mask are copied (except --keep_empty).

Hugging Face mirrors (download with huggingface_hub.snapshot_download into data/external/<name>/ first):
    # BUSI parquet (MedOtter/BUSI, CC-BY-4.0): lesion images + the 133 'normal' images (empty masks, used to
    # measure false alarms; Dice is computed on lesion images only)
    uv run python scripts/prepare_eval_sets.py --kind busi_parquet --src data/external/BUSI --name busi \
        --prompt ultrasound "breast lesion"
    # BOX-ONLY annotations (COCO-style parquet or YOLO txt): written with boxes.json; 05_evaluate reports
    # box-level metrics for these sets instead of Dice against filled rectangles
    uv run python scripts/prepare_eval_sets.py --kind coco_parquet_boxes --split test \
        --src data/external/brain-tumor-image-dataset-semantic-segmentation --name brain_mri_box --prompt MRI "brain tumor"
    uv run python scripts/prepare_eval_sets.py --kind yolo_boxes --src data/external/Lung_Nodule_Segmentation \
        --name lung_ct_box --prompt CT "lung nodule"
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image
from tqdm import tqdm

from roco_medsam_test.common import EVAL_DIR

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_mask(p):
    """-> bool array. Handles gray, palette, RGB, RGBA (alpha ignored), 8/16-bit, 0/1 or 0/255."""
    im = Image.open(p)
    a = np.array(im.convert("RGB")).max(-1) if im.mode in ("RGB", "RGBA") else np.array(im)
    if a.ndim == 3:
        a = a.max(-1)
    return a > (127 if a.max() > 1 else 0)


def save_pair(out, stem, img_path, masks, keep_empty=False):
    m = np.zeros(masks[0].shape[:2], bool)
    for a in masks:
        if a.shape != m.shape:
            return False        # inconsistent multi-mask shapes: skip rather than guess
        m |= a
    if not m.any() and not keep_empty:
        return False
    Image.open(img_path).convert("RGB").save(out / "images" / f"{stem}.png")
    Image.fromarray(m.astype(np.uint8) * 255).save(out / "masks" / f"{stem}.png")
    return True


def write_boxes(out, boxes):
    """Box-only ground truth: boxes.json (used for metrics) + filled-box masks (only for visualisation)."""
    for stem, bb in boxes.items():
        W, H = Image.open(out / "images" / f"{stem}.png").size
        m = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in bb:
            m[int(max(0, y0)):int(min(H, y1)), int(max(0, x0)):int(min(W, x1))] = True
        Image.fromarray(m.astype(np.uint8) * 255).save(out / "masks" / f"{stem}.png")
    (out / "boxes.json").write_text(json.dumps(boxes))
    return len(boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["busi", "folder", "busi_parquet", "coco_parquet_boxes", "yolo_boxes"],
                    required=True)
    ap.add_argument("--split", default="test", help="coco_parquet_boxes: which parquet split")
    ap.add_argument("--name", required=True)
    ap.add_argument("--src")
    ap.add_argument("--images")
    ap.add_argument("--masks")
    ap.add_argument("--prompt", nargs=2, metavar=("MODALITY", "PHRASE"), help="enables --teacher baseline")
    args = ap.parse_args()

    out = EVAL_DIR / args.name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    n = 0

    if args.kind == "busi":
        for cls in ("benign", "malignant"):
            imgs = [p for p in sorted((Path(args.src) / cls).glob("*.png")) if "_mask" not in p.name]
            for ip in tqdm(imgs, desc=cls):
                masks = [read_mask(p) for p in sorted(ip.parent.glob(f"{ip.stem}_mask*.png"))]
                if masks:
                    n += save_pair(out, ip.stem.replace(" ", "_"), ip, masks)
    elif args.kind == "folder":
        mfiles = {p.stem.replace("_mask", ""): p for p in Path(args.masks).iterdir() if p.suffix.lower() in EXTS}
        for ip in tqdm(sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in EXTS)):
            if ip.stem in mfiles:
                n += save_pair(out, ip.stem, ip, [read_mask(mfiles[ip.stem])])

    elif args.kind == "busi_parquet":
        import pyarrow.parquet as pq
        df = pq.read_table(sorted(Path(args.src).glob("data/*.parquet"))[0]).to_pandas()
        for r in tqdm(df.itertuples(), total=len(df), desc="busi"):
            stem = f"{r.class_label}_{Path(r.image_id).stem}".replace(" ", "_")
            mask = read_mask(io.BytesIO(r.mask["bytes"]))
            if (r.class_label == "normal") == bool(mask.any()):
                continue            # label and mask disagree: skip rather than guess
            n += save_pair(out, stem, io.BytesIO(r.image["bytes"]), [mask], keep_empty=r.class_label == "normal")
    elif args.kind == "coco_parquet_boxes":
        import pyarrow.parquet as pq
        df = pq.read_table(sorted(Path(args.src).glob(f"data/{args.split}-*.parquet"))[0]).to_pandas()
        boxes = {}
        for r in tqdm(df.itertuples(), total=len(df), desc=args.split):   # one row per object
            stem = Path(r.file_name).stem.replace(".", "_")
            x, y, w, h = [float(v) for v in r.bbox]
            if w < 2 or h < 2:
                continue
            if stem not in boxes:
                boxes[stem] = []
                Image.open(io.BytesIO(r.image["bytes"])).convert("RGB").save(out / "images" / f"{stem}.png")
            boxes[stem].append([x, y, x + w, y + h])
        n = write_boxes(out, boxes)
    elif args.kind == "yolo_boxes":
        boxes = {}
        for ip in sorted(Path(args.src).rglob("images/*/*")):
            if ip.suffix.lower() not in EXTS:
                continue
            lp = ip.parent.parent.parent / "labels" / ip.parent.name / f"{ip.stem}.txt"
            if not lp.exists():
                continue
            im = Image.open(ip).convert("RGB")
            W, H = im.size
            bb = []
            for line in lp.read_text().splitlines():
                t = line.split()
                if len(t) == 5:     # class cx cy w h, normalised
                    cx, cy, w, h = (float(v) for v in t[1:])
                    bb.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            if bb:
                stem = ip.stem.split(".rf.")[0] + f"_{ip.parent.name}"
                im.save(out / "images" / f"{stem}.png")
                boxes[stem] = bb
        n = write_boxes(out, boxes)

    if args.prompt:
        (out / "prompt.txt").write_text("\n".join(args.prompt) + "\n")
    print(f"{n} image/mask pairs -> {out}")


if __name__ == "__main__":
    main()
