"""Find small annotation arrows in a radiology image (classical CV; TRAINING TIME ONLY).

An arrow = a small, solid, bright-white or strongly coloured shape that is longer than wide and
asymmetric along its axis (a wide head that ends abruptly at the base, tapering to a tip on the other
side). We return the tip, the pointing direction and the head length. The arrow pixels can then be
inpainted away so (a) CLIP/SAM do not segment the arrow itself and (b) the image-only student never
sees arrows and cannot learn an "arrow shortcut" that does not exist on public test sets.

Limitations: arrows touching large saturated-white regions merge with them and are missed; open
"V"-style arrowheads and near-equilateral triangles with no tail are rejected (direction ambiguous).
Always eyeball the previews.
"""
import cv2
import numpy as np


def _annotation_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    colored = (s > 150) & (v > 150)          # yellow / red / green / cyan markers
    white = (v >= 240) & (s < 40)            # white markers
    return (colored | white).astype(np.uint8)


def _analyse(m, big, KMAX=64):
    """m: boolean mask of ONE component -> arrow dict or None."""
    ys, xs = np.nonzero(m)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    area = len(pts)
    c = pts.mean(0)
    d = pts - c
    evals, evecs = np.linalg.eigh(d.T @ d / len(d))      # ascending
    l2, l1 = float(evals[0]), float(evals[1])
    if l2 <= 1e-6 or np.sqrt(l1 / l2) < 1.25:            # not elongated enough to have a direction
        return None
    u = evecs[:, 1]
    t = d @ u
    tmin, tmax = float(t.min()), float(t.max())
    L = tmax - tmin
    if L < 8 or L > 0.25 * big:
        return None
    # solid shapes only: hollow letters ("A", "B") have low solidity, arrows are filled
    hull = cv2.convexHull(pts.astype(np.float32))
    if area / max(cv2.contourArea(hull) + 0.5 * cv2.arcLength(hull, True), 1.0) < 0.45:
        return None
    # an arrowhead is a THICK filled shape: it survives a morphological opening that erases thin strokes
    # (text, letter markers like "L"/"R", thin tails). Kernel scales with image size.
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    sub = m[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
    ks = 3 if big < 500 else 5
    opened = cv2.morphologyEx(sub, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    if opened.sum() < 0.25 * area:
        return None

    # Width profile along the axis (~1 px bins). The head base is the ABRUPT jump in width; the tip side
    # tapers gradually. Windowed means (not single bins) so rasterisation spikes do not matter; zero
    # padding makes the shape's own ends count as jumps from/to 0.
    K = int(np.clip(round(L), 8, KMAX))
    w, _ = np.histogram(t, bins=K, range=(tmin, tmax))
    w = w.astype(float)
    mw = int(np.clip(round(0.12 * K), 2, 8))
    wp = np.concatenate([np.zeros(mw), w, np.zeros(mw)])
    cs = np.concatenate([[0.0], np.cumsum(wp)])
    js = np.arange(mw, len(wp) - mw + 1)                 # candidate boundaries (padded coordinates)
    diff = (cs[js + mw] - cs[js]) / mw - (cs[js] - cs[js - mw]) / mw
    peak = float(((cs[mw:] - cs[:-mw]) / mw).max())
    pos, neg = float(diff.max()), float(-diff.min())
    hi, lo = max(pos, neg), min(pos, neg)
    if hi < 0.4 * peak or hi < 1.4 * lo + 1e-6:          # no clear abrupt edge / symmetric blob -> not an arrow
        return None
    jb = int(js[np.argmax(np.abs(diff))]) - mw           # boundary bin index in unpadded coordinates
    t_base = tmin + jb * L / K
    dirv = u if pos > neg else -u                        # width grows toward +u at the jump -> head (and tip) on +u side
    tip_end = w[-mw:].mean() if pos > neg else w[:mw].mean()
    if tip_end > 0.5 * peak:                             # the tip side must taper to a point
        return None
    proj = d @ dirv
    t_tip = float(proj.max())
    tip = pts[proj >= t_tip - 1.5].mean(0)
    head_len = max(1.0, t_tip - (t_base if pos > neg else -t_base))
    return {"tip": (float(tip[0]), float(tip[1])), "dir": (float(dirv[0]), float(dirv[1])),
            "length": float(L), "head_len": float(head_len)}


def find_arrows(img, max_arrows=4, max_components=15):
    """img: HxWx3 uint8 -> (list of arrow dicts, uint8 mask of arrow pixels, slightly dilated)."""
    H, W = img.shape[:2]
    ann = _annotation_mask(img)
    n, lab, stats, centroids = cv2.connectedComponentsWithStats(ann, connectivity=8)
    min_area = max(30, int(5e-5 * H * W))
    max_area = int(0.01 * H * W)
    comps = [i for i in range(1, n) if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    empty = np.zeros((H, W), np.uint8)
    if not comps or len(comps) > max_components:         # many small bright blobs = coloured image, not sparse markers
        return [], empty
    arrows = []
    for i in comps:
        cx, cy = centroids[i]
        if not (0.06 * W < cx < 0.94 * W and 0.06 * H < cy < 0.94 * H):   # border markers ("L", "R", scale text)
            continue
        a = _analyse(lab == i, max(H, W))
        if a is not None:
            a["area"] = int(stats[i, cv2.CC_STAT_AREA])
            a["label"] = i
            arrows.append(a)
    arrows.sort(key=lambda a: -a["area"])
    arrows = arrows[:max_arrows]
    mask = empty.copy()
    for a in arrows:
        mask[lab == a["label"]] = 1
    if mask.any():
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return arrows, mask


def remove_annotations(img, ann_mask):
    """Inpaint the arrow pixels so they no longer look like image content."""
    if ann_mask is None or not ann_mask.any():
        return img
    return cv2.inpaint(img, ann_mask, 5, cv2.INPAINT_TELEA)
