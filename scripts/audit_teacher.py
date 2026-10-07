"""Audit the teacher against the hand-made dev ground truth (data/dev/teacher_dev_gt.json).

Arrow detector: precision / recall of detected arrow tips (per arrow type).
Text heatmap:   does the caption-derived heat land on the annotated finding?
                pointing game (heat argmax inside the box), AUROC (in-box vs out-of-box pixels),
                lift (heat mass in box / box area fraction).

    uv run python scripts/audit_teacher.py --arrows
    uv run python scripts/audit_teacher.py --heat --vis
The dev set is for tuning the TEACHER only; never use it to select student models.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from roco_medsam_test.arrows import find_arrows, remove_annotations
from roco_medsam_test.captions import arrow_hints
from roco_medsam_test.teacher import ARROW_MIN_SCORE
from roco_medsam_test.common import DATA, ROCO_DIR, load_rgb

GT = DATA / "dev" / "teacher_dev_gt.json"            # train images: arrows + finding boxes (used for tuning)
HOLDOUT = DATA / "dev" / "arrows_holdout_gt.json"    # validation images: arrows only (NOT used for tuning)


def load_gt(path=GT):
    gt = json.loads(Path(path).read_text())["images"]
    metas = {}
    for g in gt:
        split = g.get("split", "train")
        if split not in metas:
            metas[split] = pd.read_csv(ROCO_DIR / split / "metadata.csv", dtype=str,
                                       keep_default_na=False).set_index("image_id")
        g["meta"] = metas[split].loc[g["image_id"]]
        g["img"] = load_rgb(ROCO_DIR / split / "images" / f"{g['image_id']}.png")
    return gt


def match_tips(det, gt_tips, tol):
    """Greedy nearest matching -> (matched gt indices, n unmatched detections)."""
    pairs = sorted((np.hypot(d[0] - t[0], d[1] - t[1]), i, j)
                   for i, d in enumerate(det) for j, t in enumerate(gt_tips))
    used_d, used_g = set(), set()
    for dist, i, j in pairs:
        if dist <= tol and i not in used_d and j not in used_g:
            used_d.add(i)
            used_g.add(j)
    return used_g, len(det) - len(used_d)


def audit_arrows(gt, verbose=False, hints=True, min_score=ARROW_MIN_SCORE):
    """Same call the teacher makes: caption hints (arrow colour / count) for marked captions."""
    per_type = defaultdict(lambda: [0, 0])
    tp = fp = n_gt = 0
    fp_unmarked = n_unmarked = 0
    for g in gt:
        H, W = g["img"].shape[:2]
        kw = {"min_score": min_score}
        if hints:
            h = arrow_hints(g["meta"].caption)
            kw.update(colors=h["colors"], max_arrows=h["max_n"])
        arrows, _ = find_arrows(g["img"], **kw)
        det = [a["tip"] for a in arrows]
        tips = [a["tip"] for a in g["arrows"]]
        hit, n_fp = match_tips(det, tips, tol=max(15.0, 0.04 * max(H, W)))
        for j, a in enumerate(g["arrows"]):
            per_type[a["type"]][0] += j in hit
            per_type[a["type"]][1] += 1
        tp += len(hit)
        fp += n_fp
        n_gt += len(tips)
        if g["meta"].marked != "True":
            n_unmarked += 1
            fp_unmarked += n_fp
        if verbose:
            print(f"{g['image_id'][-6:]}: gt={len(tips)} det={len(det)} hit={len(hit)} fp={n_fp}")
    print(f"\nARROWS  recall {tp}/{n_gt} = {tp / max(n_gt, 1):.2f} | precision {tp}/{tp + fp} = "
          f"{tp / max(tp + fp, 1):.2f} | false tips on {n_unmarked} unmarked images: {fp_unmarked}")
    for t, (h, n) in sorted(per_type.items()):
        print(f"   {t:8s} {h:3d}/{n:3d} = {h / n:.2f}")
    return tp / max(n_gt, 1), tp / max(tp + fp, 1)


def box_mask(boxes, H, W, pad=0.0):
    m = np.zeros((H, W), bool)
    p = pad * max(H, W)
    for x0, y0, x1, y1 in boxes:
        m[int(max(0, y0 - p)):int(min(H, y1 + p)), int(max(0, x0 - p)):int(min(W, x1 + p))] = True
    return m


def heat_scores(heat, boxes):
    H, W = heat.shape
    inside = box_mask(boxes, H, W)
    point = bool(box_mask(boxes, H, W, pad=0.03)[np.unravel_index(np.argmax(heat), heat.shape)])
    a, b = heat[inside], heat[~inside]
    rng = np.random.default_rng(0)
    a, b = rng.choice(a, min(len(a), 4000)), rng.choice(b, min(len(b), 4000))
    auroc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
    lift = float(heat[inside].sum() / max(heat.sum(), 1e-8) / inside.mean())
    return point, auroc, lift


def audit_heat(gt, method, prompt, vis_dir=None):
    import torch
    from roco_medsam_test.teacher import BiomedCLIPSaliency, build_prompts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = BiomedCLIPSaliency(device, method=method) if method != "none" else None
    rows = []
    for g in gt:
        if not g["boxes"]:
            continue
        m = g["meta"]
        img = g["img"]
        arrows, ann = find_arrows(img)
        work = remove_annotations(img, ann) if arrows else img
        if prompt == "center":          # control: no text at all, just a centred Gaussian
            H, W = work.shape[:2]
            peak = np.nan
            yy, xx = np.mgrid[:H, :W]
            heat = np.exp(-(((xx - W / 2) / (0.25 * W)) ** 2 + ((yy - H / 2) / (0.25 * H)) ** 2) / 2)
        else:
            src = m
            if prompt == "shuffled":    # control: another image's caption -> measures how much the TEXT matters
                src = gt[(gt.index(g) + len(gt) // 2) % len(gt)]["meta"]
            pos, negs = build_prompts(src.phrase, src.term, src.modality,
                                      mode="default" if prompt == "shuffled" else prompt)
            heat, _, _, peak = clip.heatmap(work, pos, negs)
        pt, auc, lift = heat_scores(heat, g["boxes"])
        rows.append({"image_id": g["image_id"], "point": pt, "auroc": auc, "lift": lift,
                     "peak": peak if prompt != "center" else np.nan})
        if vis_dir is not None:
            save_heat_vis(vis_dir / f"{g['image_id'][-6:]}.png", work, heat, g["boxes"], f"{m.modality} | {m.phrase}")
    df = pd.DataFrame(rows)
    print(f"HEAT [{method} / {prompt}] n={len(df)}  pointing {df.point.mean():.2f} | AUROC {df.auroc.mean():.3f}"
          f" | lift {df.lift.mean():.2f}")
    if df.peak.notna().any() and df.point.nunique() > 1:
        lo, hi = df[df.peak <= df.peak.median()], df[df.peak > df.peak.median()]
        print(f"   raw peak <= median ({df.peak.median():.3f}): pointing {lo.point.mean():.2f} | above: {hi.point.mean():.2f}")
    return df


def save_heat_vis(path, img, heat, boxes, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    ax[0].imshow(img)
    ax[1].imshow(img)
    ax[1].imshow(heat, alpha=0.5, cmap="jet", vmin=0, vmax=1)
    for a in ax:
        for x0, y0, x1, y1 in boxes:
            a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.5))
        a.axis("off")
    fig.suptitle(title[:120], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrows", action="store_true")
    ap.add_argument("--holdout", action="store_true", help="arrow audit on the untouched validation holdout")
    ap.add_argument("--heat", action="store_true")
    ap.add_argument("--methods", nargs="+", default=["default"])
    ap.add_argument("--prompts", nargs="+", default=["default"])
    ap.add_argument("--vis", action="store_true", help="save heat overlays to runs/audit/<method>_<prompt>/")
    ap.add_argument("--no_hints", action="store_true", help="ignore caption arrow colour/count hints")
    ap.add_argument("--min_score", type=float, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    gt = load_gt()
    akw = {"hints": not args.no_hints}
    if args.min_score is not None:
        akw["min_score"] = args.min_score
    if args.arrows:
        print("dev (train images):", end="")
        audit_arrows(gt, args.verbose, **akw)
    if args.holdout:
        print("holdout (validation images):", end="")
        audit_arrows(load_gt(HOLDOUT), args.verbose, **akw)
    if args.heat:
        from roco_medsam_test.common import RUNS
        for method in args.methods:
            for prompt in args.prompts:
                vis = None
                if args.vis:
                    vis = RUNS / "audit" / f"{method}_{prompt}".replace("/", "-")
                    vis.mkdir(parents=True, exist_ok=True)
                audit_heat(gt, method, prompt, vis)


if __name__ == "__main__":
    main()
