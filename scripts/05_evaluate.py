"""Step 5: evaluate the IMAGE-ONLY student on held-out sets with real masks.

Layout:  eval_sets/<name>/images/*.png  and  eval_sets/<name>/masks/*.png (same stem)
Optional eval_sets/<name>/prompt.txt  (line 1 = modality, line 2 = finding phrase)
         -> enables --teacher baseline (text-conditioned reference).

    uv run python scripts/05_evaluate.py --ckpt runs/student_r1/best.pt --tta --save_vis

Headline number: student_dice@0.5 (fixed threshold) with a bootstrap 95% CI.
`oracle_thr_dice` picks the best threshold ON the eval set, so it is optimistic - a diagnostic only.
Trivial baselines (full-image mask, centered box) put the numbers in context.
"""
import argparse
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

from roco_medsam_test.common import (EVAL_DIR, PROJECT_ROOT, bootstrap_ci, dice_iou, fit_mask,
                                     load_mask, load_rgb)
from roco_medsam_test.student import load_student, predict_prob

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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_student(PROJECT_ROOT / args.ckpt, device)
    teacher = None
    if args.teacher:
        from roco_medsam_test.teacher import Teacher
        teacher = Teacher(device)

    names = args.sets or sorted(p.name for p in EVAL_DIR.iterdir() if (p / "images").is_dir())
    out_dir = (PROJECT_ROOT / args.ckpt).parent / "eval"
    out_dir.mkdir(exist_ok=True)
    summary, per_image = [], []

    for name in names:
        root = EVAL_DIR / name
        imgs = sorted((root / "images").glob("*.png"))
        prompt = None
        if teacher and (root / "prompt.txt").exists():
            prompt = [l.strip() for l in (root / "prompt.txt").read_text().splitlines() if l.strip()]
        res = {k: [] for k in ["dice", "iou", "center_box", "full", "teacher"] + [f"d@{t}" for t in THRS]}
        n_resized = 0
        for i, ip in enumerate(tqdm(imgs, desc=name)):
            mp = root / "masks" / f"{ip.stem}.png"
            if not mp.exists():
                continue
            img, gt = load_rgb(ip), load_mask(mp)
            if gt.shape != img.shape[:2]:
                gt, n_resized = fit_mask(gt, img.shape), n_resized + 1
            prob = predict_prob(model, img, size, device, tta=args.tta)
            d, j = dice_iou(prob > 0.5, gt)
            res["dice"].append(d); res["iou"].append(j)
            for t in THRS:
                res[f"d@{t}"].append(dice_iou(prob > t, gt)[0])
            res["center_box"].append(dice_iou(center_box(*gt.shape), gt)[0])
            res["full"].append(dice_iou(np.ones_like(gt), gt)[0])
            if prompt and len(prompt) >= 2:
                t_out = teacher.run(img, prompt[1], prompt[0])
                res["teacher"].append(dice_iou(t_out["mask"], gt)[0] if t_out else 0.0)
            per_image.append({"set": name, "image": ip.name, "dice": d, "iou": j})
            if args.save_vis and i < 12:
                (out_dir / name).mkdir(exist_ok=True)
                save_vis(out_dir / name / f"{ip.stem}.png", img, gt, prob)
        if not res["dice"]:
            print(f"{name}: no matched image/mask pairs"); continue
        if n_resized:
            print(f"{name}: WARNING {n_resized} masks had a different size than their image and were resized")
        sweep = {t: np.mean(res[f"d@{t}"]) for t in THRS}
        lo, hi = bootstrap_ci(res["dice"])
        summary.append({
            "set": name, "n": len(res["dice"]),
            "student_dice@0.5": np.mean(res["dice"]), "ci95_lo": lo, "ci95_hi": hi,
            "student_iou@0.5": np.mean(res["iou"]),
            "hit_rate(dice>0.5)": np.mean(np.array(res["dice"]) > 0.5),
            "oracle_thr_dice": max(sweep.values()), "oracle_thr": max(sweep, key=sweep.get),
            "center_box_dice": np.mean(res["center_box"]), "full_image_dice": np.mean(res["full"]),
            "teacher_dice(text)": np.mean(res["teacher"]) if res["teacher"] else np.nan,
        })

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(per_image).to_csv(out_dir / "per_image.csv", index=False)
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
