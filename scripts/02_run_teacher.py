"""Step 2: caption -> pseudo-mask (teacher). Resumable.

ALWAYS preview first and look at the overlays before the full run:
    uv run python scripts/02_run_teacher.py --split train --preview 40
    uv run python scripts/02_run_teacher.py --split train --preview 40 --only_marked   # captions that mention arrows etc.
Then the full run:
    uv run python scripts/02_run_teacher.py --split train
    uv run python scripts/02_run_teacher.py --split validation

When a caption says the image is marked and an arrow is found, the mask is anchored at the arrow tip and
the arrow is inpainted out; the arrow-free image is saved under pseudo/<split>/images_clean and used as
the training image (manifest image_path points to it), so the student never sees arrows.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from roco_medsam_test.common import PSEUDO_DIR, ROCO_DIR, load_rgb, rel, set_seed
from roco_medsam_test.teacher import Teacher

COLS = ["image_id", "split", "kind", "modality", "phrase", "status", "sam_iou", "heat_in", "heat_peak",
        "area_frac", "clip_margin", "crop_gain", "weight", "n_arrows", "arrow_used",
        "image_path", "mask_path", "heat_path"]


def save_preview(path, img, res, title):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
    ax[0].imshow(img)
    for x, y, dx, dy in res.get("arrow_tips", []):      # detected arrow tips (cyan) and pointing direction
        ax[0].plot([x], [y], "c+", ms=14, mew=2)
        ax[0].annotate("", xy=(x + 40 * dx, y + 40 * dy), xytext=(x, y),
                       arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5))
    ax[0].set_title(f"image | arrows found: {res.get('n_arrows', 0)}" + (" (used)" if res.get("arrow_used") else ""))
    base = res["clean_img"] if res.get("clean_img") is not None else img
    ax[1].imshow(base)
    ax[1].imshow(res["heat"], alpha=0.5, cmap="jet")
    ax[1].set_title(f"text-derived heat | peak={res['heat_peak']:.2f}")
    ov = base.copy().astype(np.float32) / 255
    ov[res["mask"]] = 0.55 * ov[res["mask"]] + 0.45 * np.array([1, 0, 0])
    ax[2].imshow(ov)
    ax[2].set_title(f"mask {100 * res['area_frac']:.1f}% of image | sam_iou={res['sam_iou']:.2f}")
    for a in ax:
        a.axis("off")
    fig.suptitle(title[:140], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0, help="save N overlays and exit (no manifest)")
    ap.add_argument("--only_marked", action="store_true", help="preview only captions that mention arrows/asterisks/circles")
    ap.add_argument("--cover_pow", type=float, default=0.15, help="higher -> larger masks preferred; 0 = tightest")
    ap.add_argument("--max_area", type=float, default=0.35, help="reject candidate masks larger than this fraction of the image")
    ap.add_argument("--no_arrows", action="store_true", help="disable arrow detection")
    args = ap.parse_args()

    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = pd.read_csv(ROCO_DIR / args.split / "metadata.csv", dtype=str, keep_default_na=False)
    if args.limit:
        meta = meta.head(args.limit)

    pdir = PSEUDO_DIR / args.split
    (pdir / "masks").mkdir(parents=True, exist_ok=True)
    (pdir / "heat").mkdir(parents=True, exist_ok=True)
    (pdir / "images_clean").mkdir(parents=True, exist_ok=True)
    man_path = PSEUDO_DIR / f"manifest_{args.split}.csv"

    if args.preview:
        (PSEUDO_DIR / "preview").mkdir(parents=True, exist_ok=True)
        f = meta[meta.kind == "finding"]
        if args.only_marked:
            f = f[f.marked.str.lower() == "true"]
        meta = f.sample(min(args.preview, len(f)), random_state=0)
    done = set()
    if man_path.exists() and not args.preview:
        done = set(pd.read_csv(man_path, dtype=str, keep_default_na=False).image_id)

    teacher = Teacher(device, cover_pow=args.cover_pow, max_area=args.max_area)
    buf = []

    def flush():
        nonlocal buf
        if buf and not args.preview:
            pd.DataFrame(buf, columns=COLS).to_csv(man_path, mode="a", header=not man_path.exists(), index=False)
        buf = []

    for r in tqdm(meta.itertuples(index=False), total=len(meta), desc=f"teacher {args.split}"):
        iid = str(r.image_id)
        if iid in done:
            continue
        img_path = ROCO_DIR / args.split / "images" / f"{iid}.png"
        row = {c: "" for c in COLS}
        row.update(image_id=iid, split=args.split, kind=r.kind, modality=r.modality,
                   phrase=r.phrase, image_path=rel(img_path), n_arrows=0, arrow_used=False)

        if r.kind == "normal":
            row.update(status="ok", weight=1.0)
            buf.append(row)
        else:
            img = load_rgb(img_path)
            marked = str(getattr(r, "marked", "")).lower() == "true"
            try:
                res = teacher.run(img, r.phrase, r.modality, marked=marked, use_arrows=not args.no_arrows)
                err = False
            except Exception as e:  # keep the long run alive, but record it
                print(f"\n{iid}: {e}")
                res, err = None, True
            if res is None:
                row.update(status="error" if err else "no_candidate")
                buf.append(row)
            else:
                if args.preview:
                    save_preview(PSEUDO_DIR / "preview" / f"{iid}.png", img, res,
                                 f"{r.modality} | {r.phrase} | {r.caption}")
                    continue
                mp, hp = pdir / "masks" / f"{iid}.png", pdir / "heat" / f"{iid}.npy"
                Image.fromarray(res["mask"].astype(np.uint8) * 255).save(mp)
                np.save(hp, cv2.resize(res["heat"], (56, 56), interpolation=cv2.INTER_AREA).astype(np.float16))
                if res["clean_img"] is not None:     # student trains on the arrow-free image
                    cp = pdir / "images_clean" / f"{iid}.png"
                    Image.fromarray(res["clean_img"]).save(cp)
                    row["image_path"] = rel(cp)
                # arrow-anchored masks are more trustworthy than a blurry heatmap alone: weight IoU higher
                a = 0.7 if res["arrow_used"] else 0.5
                row.update(status="ok", sam_iou=round(res["sam_iou"], 4), heat_in=round(res["heat_in"], 4),
                           heat_peak=round(res["heat_peak"], 4), area_frac=round(res["area_frac"], 5),
                           clip_margin=round(res["clip_margin"], 4), crop_gain=round(res["crop_gain"], 4),
                           weight=round(float(np.clip(a * res["sam_iou"] + (1 - a) * res["heat_in"], 0.05, 1)), 4),
                           n_arrows=res["n_arrows"], arrow_used=res["arrow_used"],
                           mask_path=rel(mp), heat_path=rel(hp))
                buf.append(row)
        if len(buf) >= 100:
            flush()
    flush()

    if args.preview:
        print(f"\nOverlays in {PSEUDO_DIR/'preview'}. Look at them before the full run.")
    else:
        m = pd.read_csv(man_path, dtype=str, keep_default_na=False)
        print(m.status.value_counts())
        print("images with arrow used:", int((m.arrow_used == "True").sum()))


if __name__ == "__main__":
    main()
