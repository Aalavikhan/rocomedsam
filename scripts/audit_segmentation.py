"""Audit the teacher's SEGMENTATION step (location -> mask) against hand-drawn finding boxes.

Detection (arrows + caption heatmap) is run once per image and cached; only Teacher.segment() is
re-evaluated, so detection is identical across compared variants.

    uv run python scripts/audit_segmentation.py              # dev + holdout: legacy vs current default
    uv run python scripts/audit_segmentation.py --vis        # + overlays in runs/audit/seg/
Metrics per image with a box (mask vs GT box; boxes are rough, +-10 px, so ~0.7 box IoU is "very good"):
    box_iou   IoU of the mask's bounding box with the best-matching GT box (or the union of GT boxes)
    inside    fraction of mask pixels that fall inside a GT box (low -> mask leaks / too big)
    cover     fraction of the GT box area covered by the mask (low -> mask too small; ~0.75 max for an ellipse)
    ok        box_iou >= 0.5
    found     mask touches a GT box at all (detection/location proxy; must not change between variants)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from roco_medsam_test.common import DATA, ROCO_DIR, box_metrics, load_rgb

# dev = used to choose the mask-selection rule; holdout = the 31 other preview_v2 images, only checked
SETS = {"dev": DATA / "dev" / "teacher_dev_gt.json", "holdout": DATA / "dev" / "seg_holdout_gt.json"}


def load_items(name):
    gt = json.loads(SETS[name].read_text())["images"]
    meta = pd.read_csv(ROCO_DIR / "train" / "metadata.csv", dtype=str, keep_default_na=False).set_index("image_id")
    return [(g["image_id"], g["boxes"], meta.loc[g["image_id"]]) for g in gt
            if g["boxes"] and g["image_id"] in meta.index]


def run_variants(teacher, items, variants, vis_dir=None):
    """variants: {name: dict of Teacher attribute overrides}. -> DataFrame (one row per image x variant)."""
    rows = []
    for iid, boxes, m in items:
        img = load_rgb(ROCO_DIR / "train" / "images" / f"{iid}.png")
        det = teacher.detect(img, m.phrase, m.modality, marked=m.marked == "True", caption=m.caption, term=m.term)
        for vname, over in variants.items():
            old = {k: getattr(teacher, k) for k in over}
            for k, v in over.items():
                setattr(teacher, k, v)
            try:
                best, used = teacher.segment(det)
            finally:
                for k, v in old.items():
                    setattr(teacher, k, v)
            mask = best["mask"] if best is not None else np.zeros(img.shape[:2], bool)
            r = dict(image_id=iid, variant=vname, arrow=used, area=float(mask.mean()), **box_metrics(mask, boxes))
            rows.append(r)
            if vis_dir is not None:
                save_vis(vis_dir / vname / f"{iid[-6:]}.png", det["work"], mask, boxes, r, f"{m.modality} | {m.phrase}")
    return pd.DataFrame(rows)


def save_vis(path, img, mask, boxes, r, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ov = img.astype(np.float32) / 255
    ov[mask] = 0.55 * ov[mask] + 0.45 * np.array([1, 0, 0])
    ax.imshow(ov)
    for x0, y0, x1, y1 in boxes:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.5))
    ax.set_title(f"{title[:60]}\nbox_iou={r['box_iou']:.2f} area={100 * r['area']:.1f}%", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def summarize(df):
    g = df.groupby("variant", sort=False)
    return pd.DataFrame({"n": g.size(), "found": g.found.mean(), "box_iou": g.box_iou.mean(),
                         "median_iou": g.box_iou.median(), "ok(iou>=.5)": g.box_iou.apply(lambda s: (s >= 0.5).mean()),
                         "inside": g.inside.mean(), "cover": g.cover.mean(), "area%": 100 * g.area.median()}).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev", "holdout"])
    ap.add_argument("--vis", action="store_true", help="save overlays to runs/audit/seg/<set>/<variant>/")
    args = ap.parse_args()
    from roco_medsam_test.common import RUNS
    from roco_medsam_test.teacher import Teacher
    teacher = Teacher("cuda" if torch.cuda.is_available() else "cpu")
    # "legacy" = the original mask choice; "default" = whatever teacher.py currently defaults to
    variants = {"legacy": {"select": "legacy", "clean_mask": False, "mask_max_area": 1.0}, "default": {}}
    for s in args.sets:
        df = run_variants(teacher, load_items(s), variants, RUNS / "audit" / "seg" / s if args.vis else None)
        print(f"\n== {s}\n{summarize(df).to_string()}")


if __name__ == "__main__":
    main()
