"""Step 5: evaluate an IMAGE-ONLY student (UNet from step 3, or MedSAM+LoRA from step 3b) on held-out
sets with real masks. The checkpoint type is detected automatically.

Layout:  eval_sets/<name>/images/*.png  and  eval_sets/<name>/masks/*.png (same stem)
Optional eval_sets/<name>/prompt.txt  (line 1 = modality, line 2 = finding phrase)
         -> enables --teacher baseline (text-conditioned reference).

    uv run python scripts/05_evaluate.py --ckpt runs/student_r1/best.pt --tta --save_vis
    uv run python scripts/05_evaluate.py --ckpt runs/sam_lora_r1/best.pt --tta --save_vis
Compare students with the summary.csv written next to each checkpoint (same sets, same metrics).

Headline number: student_dice@0.5 (fixed threshold) with a bootstrap 95% CI.
`oracle_thr_dice` picks the best threshold ON the eval set, so it is optimistic - a diagnostic only.
Trivial baselines (full-image mask, centered box) put the numbers in context.
Sets with boxes.json (box-only annotations) get box-level metrics instead of Dice. Images with an EMPTY mask
(e.g. BUSI 'normal') are excluded from Dice and give false_alarm_rate (any finding predicted).
pred_empty = fraction of lesion images where the student predicts no finding at all:
for the MedSAM student this is mostly the presence head (threshold chosen on ROCO validation in step 3b).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from roco_medsam_test.common import (EVAL_DIR, PROJECT_ROOT, bootstrap_ci, box_metrics, dice_iou, fit_mask,
                                     load_mask, load_rgb)
from roco_medsam_test.student import load_student, predict_prob


def load_predictor(ckpt_path, device, presence_thr=None):
    """-> (name, fn(img, tta) -> HxW probability map) for either student type."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("kind") == "sam_lora":
        from roco_medsam_test.sam_student import load_sam_student, sam_predict
        model, thr = load_sam_student(ckpt_path, device)
        if presence_thr is not None:     # diagnostic override of the threshold chosen on ROCO validation
            thr = presence_thr
        print(f"MedSAM+LoRA student (presence threshold {thr})")
        return "sam_lora", lambda img, tta: sam_predict(model, img, device, presence_thr=thr, tta=tta)[0]
    model, size = load_student(ckpt_path, device)
    print(f"UNet student ({ck.get('encoder')}, {size}px)")
    return "unet", lambda img, tta: predict_prob(model, img, size, device, tta=tta)

THRS = [0.3, 0.4, 0.5, 0.6, 0.7]


def center_box(h, w):
    m = np.zeros((h, w), bool)
    m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    return m


def save_vis(path, img, gt, prob):
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(img); ax[0].set_title("image")
    ax[1].imshow(img); ax[1].set_title("GT")
    ax[2].imshow(img); ax[2].set_title("student (image only)")
    if gt.any():
        ax[1].contour(gt.astype(float), levels=[0.5], colors="lime", linewidths=1)
    if (prob > 0.5).any():
        ax[2].contour((prob > 0.5).astype(float), levels=[0.5], colors="red", linewidths=1)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="*", default=None, help="names under eval_sets/ (default: all)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--teacher", action="store_true", help="also run text-conditioned teacher using prompt.txt")
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--presence_thr", type=float, default=None,
                    help="MedSAM-LoRA only: override the presence threshold stored in the checkpoint (0 = gate off)")
    ap.add_argument("--tag", default="", help="write results to eval_<tag>/ instead of eval/")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student, predict = load_predictor(PROJECT_ROOT / args.ckpt, device, args.presence_thr)
    teacher = None
    if args.teacher:
        from roco_medsam_test.teacher import Teacher
        teacher = Teacher(device)

    names = args.sets or sorted(p.name for p in EVAL_DIR.iterdir() if (p / "images").is_dir())
    out_dir = (PROJECT_ROOT / args.ckpt).parent / ("eval_" + args.tag if args.tag else "eval")
    out_dir.mkdir(exist_ok=True)
    summary, per_image = [], []

    for name in names:
        root = EVAL_DIR / name
        imgs = sorted((root / "images").glob("*.png"))
        prompt = None
        if teacher and (root / "prompt.txt").exists():
            prompt = [l.strip() for l in (root / "prompt.txt").read_text().splitlines() if l.strip()]
        boxes = json.loads((root / "boxes.json").read_text()) if (root / "boxes.json").exists() else None
        res = {k: [] for k in ["dice", "iou", "center_box", "full", "teacher", "empty", "fp", "fp_area",
                               "box_iou", "found", "cb_box_iou", "teacher_box_iou"] + [f"d@{t}" for t in THRS]}
        n_resized = 0
        vis_every = max(1, len(imgs) // 24)
        for i, ip in enumerate(tqdm(imgs, desc=name)):
            mp = root / "masks" / f"{ip.stem}.png"
            if not mp.exists():
                continue
            img, gt = load_rgb(ip), load_mask(mp)
            if gt.shape != img.shape[:2]:
                gt, n_resized = fit_mask(gt, img.shape), n_resized + 1
            prob = predict(img, args.tta)
            pred = prob > 0.5
            t_out = teacher.run(img, prompt[1], prompt[0]) if prompt and len(prompt) >= 2 else None
            row = {"set": name, "image": ip.name}
            if boxes is not None:                       # box-only ground truth: box-level metrics
                bb = boxes[ip.stem]
                r = box_metrics(pred, bb)
                res["box_iou"].append(r["box_iou"]); res["found"].append(r["found"])
                res["empty"].append(not pred.any())
                res["cb_box_iou"].append(box_metrics(center_box(*gt.shape), bb)["box_iou"])
                if prompt:
                    res["teacher_box_iou"].append(box_metrics(t_out["mask"], bb)["box_iou"] if t_out else 0.0)
                row.update(box_iou=r["box_iou"], found=r["found"])
                if prompt:
                    row["teacher_box_iou"] = res["teacher_box_iou"][-1]
            elif not gt.any():                          # normal image: any predicted finding is a false alarm
                res["fp"].append(bool(pred.any())); res["fp_area"].append(float(pred.mean()))
                row.update(normal=True, false_alarm=bool(pred.any()))
            else:
                d, j = dice_iou(pred, gt)
                res["empty"].append(not pred.any())
                res["dice"].append(d); res["iou"].append(j)
                for t in THRS:
                    res[f"d@{t}"].append(dice_iou(prob > t, gt)[0])
                res["center_box"].append(dice_iou(center_box(*gt.shape), gt)[0])
                res["full"].append(dice_iou(np.ones_like(gt), gt)[0])
                if prompt:
                    res["teacher"].append(dice_iou(t_out["mask"], gt)[0] if t_out else 0.0)
                row.update(dice=d, iou=j)
                if prompt:
                    row["teacher_dice"] = res["teacher"][-1]
            per_image.append(row)
            if args.save_vis and i % vis_every == 0:
                (out_dir / name).mkdir(exist_ok=True)
                save_vis(out_dir / name / f"{ip.stem}.png", img, gt, prob)
        if n_resized:
            print(f"{name}: WARNING {n_resized} masks had a different size than their image and were resized")
        rec = {"set": name, "student": student}
        if boxes is not None:
            lo, hi = bootstrap_ci(res["box_iou"])
            rec.update(gt="boxes", n=len(res["box_iou"]), box_iou=np.mean(res["box_iou"]), ci95_lo=lo, ci95_hi=hi,
                       **{"box_hit(iou>=0.5)": np.mean(np.array(res["box_iou"]) >= 0.5),
                          "found(touches box)": np.mean(res["found"]), "pred_empty": np.mean(res["empty"]),
                          "center_box_box_iou": np.mean(res["cb_box_iou"]),
                          "teacher_box_iou(text)": np.mean(res["teacher_box_iou"]) if res["teacher_box_iou"] else np.nan})
        elif res["dice"]:
            sweep = {t: np.mean(res[f"d@{t}"]) for t in THRS}
            lo, hi = bootstrap_ci(res["dice"])
            rec.update(gt="masks", n=len(res["dice"]), **{
                "student_dice@0.5": np.mean(res["dice"]), "ci95_lo": lo, "ci95_hi": hi,
                "student_iou@0.5": np.mean(res["iou"]),
                "hit_rate(dice>0.5)": np.mean(np.array(res["dice"]) > 0.5),
                "pred_empty": np.mean(res["empty"]),
                "oracle_thr_dice": max(sweep.values()), "oracle_thr": max(sweep, key=sweep.get),
                "center_box_dice": np.mean(res["center_box"]), "full_image_dice": np.mean(res["full"]),
                "teacher_dice(text)": np.mean(res["teacher"]) if res["teacher"] else np.nan})
        if res["fp"]:
            rec.update(n_normal=len(res["fp"]), false_alarm_rate=np.mean(res["fp"]),
                       normal_pred_area=np.mean(res["fp_area"]))
        if len(rec) > 2:
            summary.append(rec)
        else:
            print(f"{name}: no matched image/mask pairs")

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(per_image).to_csv(out_dir / "per_image.csv", index=False)
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
