"""Step 4: refine pseudo-labels with the image-only students (self-training).

Each image is predicted by the fold-model that did NOT train on it (cross-fitting),
otherwise agreement with the teacher would just reflect memorisation.

Rules for images with a kept teacher mask:
  IoU(student, teacher) >= --replace_iou : use the student's (smoother) mask
  IoU < --drop_iou                       : drop the sample (likely teacher/caption noise)
  otherwise                              : keep the teacher mask
Optional --add_confident: also add teacher-rejected images where the student is very confident
(these have no text-derived heat target and get weight 0.5).

--reweight (used for the reported results): keep every teacher mask and scale its weight by the agreement
instead. Checked on the hand-boxed training images (data/dev/*.json) with ~0.24 pseudo-Dice fold students:
drop_iou=0.2 removed 76% of findings although half of the removed teacher masks were good, and replaced masks
were slightly worse than the teacher's (box IoU 0.675 -> 0.643); agreement itself is only weakly informative
(AUROC 0.66), which suits a soft weight but not a hard filter.

    uv run python scripts/04_self_train.py --ckpts runs/s_f0/best.pt runs/s_f1/best.pt
    uv run python scripts/03_train_student.py --out runs/student_r1 \
        --train_manifest data/pseudo/manifest_train_r1.csv
    uv run python scripts/03b_train_sam_lora.py --out runs/sam_lora_r1 \
        --train_manifest data/pseudo/manifest_train_r1.csv
Use the SAME filter flags here as in step 3. Self-training always uses the UNet folds: the MedSAM student
shares the teacher's model and would mostly confirm the teacher's own mistakes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

from roco_medsam_test.common import PROJECT_ROOT, PSEUDO_DIR, dice_iou, fold_of, load_mask, load_rgb, rel
from roco_medsam_test.student import add_filter_args, filter_kwargs, load_manifest, load_student, predict_prob


def largest_components(mask, keep_frac=0.2):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, np.arange(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= keep_frac * sizes.max()]
    return np.isin(lab, keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="fold models in fold order (0,1,...)")
    ap.add_argument("--manifest", default=str(PSEUDO_DIR / "manifest_train.csv"))
    ap.add_argument("--out_name", default="r1")
    ap.add_argument("--replace_iou", type=float, default=0.5)
    ap.add_argument("--drop_iou", type=float, default=0.2)
    ap.add_argument("--reweight", action="store_true",
                    help="no drop/replace: weight *= floor + (1-floor) * min(1, IoU/replace_iou)")
    ap.add_argument("--reweight_floor", type=float, default=0.5)
    ap.add_argument("--add_confident", action="store_true")
    ap.add_argument("--conf", type=float, default=0.9)
    add_filter_args(ap)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    nf = len(args.ckpts)
    models = []
    for i, c in enumerate(args.ckpts):
        m, size, meta = load_student(PROJECT_ROOT / c, device, return_meta=True)
        if meta["fold"] != i or meta["nfolds"] != nf:
            raise SystemExit(f"{c} was trained with fold={meta['fold']}, nfolds={meta['nfolds']}; "
                             f"expected fold={i}, nfolds={nf}. Pass checkpoints in fold order.")
        models.append((m, size))

    filt = filter_kwargs(args)
    full = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    kept_ids = set(load_manifest(args.manifest, **filt).image_id)
    out_masks = PSEUDO_DIR / f"train_{args.out_name}" / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)

    rows, stats = [], {"replaced": 0, "kept_teacher": 0, "dropped": 0, "added": 0, "reweighted": 0}
    for r in tqdm(full.itertuples(index=False), total=len(full), desc="refine"):
        d = r._asdict()
        if r.kind == "normal":
            rows.append(d)
            continue
        in_kept = r.image_id in kept_ids
        if not in_kept and not args.add_confident:
            continue
        model, size = models[fold_of(r.image_id, nf)]
        img = load_rgb(PROJECT_ROOT / r.image_path)
        prob = predict_prob(model, img, size, device, tta=True)
        sm = largest_components(prob > 0.5)
        area = float(sm.mean())
        conf = float(prob[sm].mean()) if sm.any() else 0.0
        area_ok = args.min_area <= area <= args.max_area

        if in_kept:
            _, iou = dice_iou(sm, load_mask(PROJECT_ROOT / r.mask_path))
            if args.reweight:   # keep every teacher mask; trust it less where the held-out student disagrees
                f = args.reweight_floor + (1 - args.reweight_floor) * min(1.0, iou / args.replace_iou)
                d["weight"] = str(round(float(d["weight"] or 1.0) * f, 4))
                stats["reweighted"] += 1
                rows.append(d)
                continue
            if iou < args.drop_iou:
                stats["dropped"] += 1
                continue
            if iou >= args.replace_iou and area_ok:
                p = out_masks / f"{r.image_id}.png"
                Image.fromarray(sm.astype(np.uint8) * 255).save(p)
                d["mask_path"] = rel(p)
                stats["replaced"] += 1
            else:
                stats["kept_teacher"] += 1
            rows.append(d)
        elif conf >= args.conf and area_ok:
            p = out_masks / f"{r.image_id}.png"
            Image.fromarray(sm.astype(np.uint8) * 255).save(p)
            d.update(mask_path=rel(p), heat_path="", status="ok", sam_iou="1.0", crop_gain="0.0", sep="9.9",
                     heat_peak="1.0", area_frac=str(area), weight="0.5")
            rows.append(d)
            stats["added"] += 1

    out = PSEUDO_DIR / f"manifest_train_{args.out_name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(stats, f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
