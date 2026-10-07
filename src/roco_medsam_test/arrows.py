"""Find annotation arrows in a radiology image (classical CV; TRAINING TIME ONLY).

An arrow = a small, sharply bounded component of annotation "ink" (strongly coloured, near-white, or
near-black on a brighter background) whose width profile, measured from a sharp corner (the tip), grows
roughly linearly over the head and then either ends (a bare arrowhead) or drops abruptly to a narrow,
straight, centred shaft. Open "V" heads are accepted when they sit on a shaft. We return the tip, the
pointing direction and the head length. The arrow pixels can then be inpainted away so (a) CLIP/SAM do not
segment the arrow itself and (b) the image-only student never sees arrows and cannot learn an "arrow
shortcut" that does not exist on public test sets.

Limitations: arrows that touch structures of the same intensity (white arrow on bright bone, black arrow
running into air) merge with them and are missed; hollow/outlined arrows and bare chevrons (">") are not
detected. Audit with scripts/audit_teacher.py --arrows and eyeball the previews.
"""
import cv2
import numpy as np


def _channels(img):
    """Binary masks of candidate annotation ink: (name, uint8 mask)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    colored = (s > 110) & (v > 100)
    sat_frac = float(colored.mean())
    if sat_frac > 0.05:
        # tinted / pseudo-coloured image (cyan X-ray, PET-CT fusion): ink must also differ in HUE from the
        # dominant tint, otherwise the arrow merges with the background
        h = hsv[..., 0].astype(np.int16)
        hist = np.bincount(h[colored].ravel(), minlength=180)
        dom = int(np.argmax(np.convolve(np.concatenate([hist[-5:], hist, hist[:5]]), np.ones(11), "valid")))
        dh = np.abs(h - dom)
        colored &= np.minimum(dh, 180 - dh) > 15
    white = (v >= 220) & (s < 60)
    # black ink only counts where the neighbourhood is clearly brighter (a black arrow drawn on tissue),
    # otherwise every patch of air/background would be a candidate
    bg = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    black = (v <= 40) & (bg.astype(np.int16) - gray >= 70)
    # strict variants split pure ink from touching tissue of similar (but not identical) intensity,
    # e.g. a white arrow drawn onto a bright DWI lesion or a yellow arrow touching a PET hotspot
    strict = [("colored", (s > 180) & (v > 150)), ("white", (v >= 248) & (s < 30)), ("black", black & (v <= 12))]
    return [(n, m.astype(np.uint8)) for n, m in
            [("colored", colored), ("white", white), ("black", black)] + strict]


def _tip_candidates(pts, c):
    """Sharp convex-hull corners (possible arrow tips), sharpest first."""
    hull = cv2.convexHull(pts.astype(np.float32))
    poly = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)[:, 0, :].astype(np.float64)
    out = []
    if len(poly) >= 3:
        for i in range(len(poly)):
            p, a, b = poly[i], poly[i - 1], poly[(i + 1) % len(poly)]
            va, vb = a - p, b - p
            cos = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            if ang < 115:
                out.append((ang, p))
    if not out:  # degenerate hull (very thin shape): use both extremes of the principal axis
        d = pts - c
        u = np.linalg.eigh(d.T @ d)[1][:, 1]
        t = d @ u
        out = [(90.0, pts[np.argmin(t)]), (90.0, pts[np.argmax(t)])]
    return [p for _, p in sorted(out, key=lambda x: x[0])][:4]


def _fit_from_tip(pts, p, c):
    """Fit the arrow width model with the tip at hull corner p. -> dict or None."""
    u = c - p
    if np.linalg.norm(u) < 2:
        return None
    u /= np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    t, s = (pts - p) @ u, (pts - p) @ n
    L = float(t.max())
    if L < 7:
        return None
    # snap the tip to the actual pixels at the pointy end (hull corners can sit a pixel off)
    tip = pts[t <= min(1.5, 0.05 * L)].mean(0)
    bw = max(1.0, L / 160)       # ~1 px bins: small heads on long arrows need fine resolution
    K = int(L // bw) + 1
    idx = np.clip((np.maximum(t, 0) // bw).astype(int), 0, K - 1)
    smax = np.full(K, -np.inf)
    smin = np.full(K, np.inf)
    np.maximum.at(smax, idx, s)
    np.minimum.at(smin, idx, s)
    cnt = np.bincount(idx, minlength=K) / bw
    ok = np.isfinite(smax)
    if ok.mean() < 0.8:  # broken / dotted shape
        return None
    smax, smin = np.where(ok, smax, 0.0), np.where(ok, smin, 0.0)
    ext = np.where(ok, smax - smin + 1, 0.0)
    mid = (smax + smin) / 2
    ext = np.array([np.median(ext[max(0, k - 1):k + 2]) for k in range(K)])  # rasterisation noise
    tk = (np.arange(K) + 0.5) * bw
    total = ext.sum() + 1e-9

    # least-squares line e = a + b*t over every prefix [0, h) at once (closed form from cumulative sums)
    hs = np.arange(1, K + 1)
    S1, St, Stt = hs.astype(float), np.cumsum(tk), np.cumsum(tk * tk)
    Se, Ste = np.cumsum(ext), np.cumsum(tk * ext)
    den = S1 * Stt - St ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        bs = np.where(den > 1e-9, (S1 * Ste - St * Se) / den, 0.0)
    as_ = (Se - bs * St) / S1
    best = None
    for h in range(max(2, int(np.ceil(3 / bw))), K + 1):          # head length in bins (>= 3 px)
        b, a = bs[h - 1], as_[h - 1]
        Wh = a + b * h * bw
        if b <= 0 or Wh < 4 or a > 0.35 * Wh + 2:
            continue
        head_len = h * bw
        if not 0.45 <= head_len / Wh <= 2.6:
            continue
        th, eh = tk[:h], ext[:h]
        err = np.abs(eh - (a + b * th)).sum()
        shaft = K - h
        if shaft > 0:
            es = ext[h:]
            Ws = float(np.median(es))
            if Ws > 0.65 * Wh:
                continue
            # the head base is an ABRUPT step down to the shaft (ovals / bone fragments taper gradually)
            d = int(np.ceil(max(2.0, 0.25 * head_len) / bw))
            if ext[min(K - 1, h + d)] > Ws + 0.35 * (Wh - Ws):
                continue
            err += np.abs(es - Ws).sum()
            # straight, centred shaft
            ms = mid[h:]
            if np.std(ms) > 0.2 * Ws + 1.5 or abs(np.median(ms)) > 0.25 * Wh + 1:
                continue
            fill_s = float(np.median(cnt[h:] / np.maximum(ext[h:], 1)))
            if fill_s < 0.6:
                continue
        else:
            Ws = 0.0
            # a bare arrowhead ends at its full width (flat base), it does not round off
            if ext[max(0, K - int(np.ceil(2 / bw))):].mean() < 0.6 * Wh:
                continue
        fill_h = float(np.mean(cnt[:h] / np.maximum(ext[:h], 1)))
        if fill_h < (0.2 if shaft * bw >= 0.6 * head_len else 0.7):   # open V heads need a shaft
            continue
        if abs(np.mean(mid[:h])) > 0.2 * Wh + 1:                       # head symmetric about the axis
            continue
        nerr = err / total
        if best is None or nerr < best["err"]:
            best = {"err": float(nerr), "head_len": float(head_len), "head_w": float(Wh), "shaft_w": Ws,
                    "shaft_len": float(shaft * bw), "fill_h": fill_h}
    if best is None:
        return None
    best.update(tip=(float(tip[0]), float(tip[1])), dir=(float(-u[0]), float(-u[1])), length=L)
    return best


def _analyse(m, big):
    """m: boolean mask of ONE component -> arrow dict or None."""
    ys, xs = np.nonzero(m)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    if len(pts) < 15:
        return None
    c = pts.mean(0)
    best = None
    for p in _tip_candidates(pts, c):
        r = _fit_from_tip(pts, p, c)
        if r is None or r["length"] > 0.35 * big or r["head_w"] < max(5.0, 0.008 * big):
            continue
        # bare heads are easy to confuse with anatomy/letters: demand a much cleaner fit
        r["has_shaft"] = r["shaft_len"] >= 0.3 * r["head_len"]
        if r["err"] > (0.17 if r["has_shaft"] else 0.10):
            continue
        if best is None or r["err"] < best["err"]:
            best = r
    return best


def _contrast(name, gray, sat, lab, i, x, y, w, h):
    """(contrast, sharpness). contrast: how far the ink stands out from the ring 3-4 px around it.
    sharpness: fraction of that step already reached 2 px out. Drawn annotations have hard, ~1 px
    anti-aliased edges; bone/anatomy edges are blurred by reconstruction and partial volume."""
    H, W = gray.shape
    x0, y0, x1, y1 = max(0, x - 6), max(0, y - 6), min(W, x + w + 6), min(H, y + h + 6)
    m = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
    d1, d2, d4 = (cv2.dilate(m, np.ones((k, k), np.uint8)) > 0 for k in (3, 5, 9))
    near, far = d2 & ~d1, d4 & ~d2
    if not far.any() or not near.any():
        return 0.0, 0.0
    src = (sat if name == "colored" else gray)[y0:y1, x0:x1]
    v_in, v_near, v_far = float(np.median(src[m > 0])), float(np.median(src[near])), float(np.median(src[far]))
    d = v_in - v_far
    sharp = abs(v_in - v_near) / max(abs(d), 1e-6)
    return (-d if name == "black" else d), sharp


def _pure_ink(name, gray, lab, i, x, y, w, h):
    """Drawn annotations are (near) pure white / pure black; bright bone or dark anatomy usually is not."""
    if name == "colored":
        return True
    med = float(np.median(gray[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i]))
    return med >= 250 if name == "white" else med <= 8


def _text_like(stats, i, n):
    """Part of a row of similar-height glyphs (burned-in text such as dates, '512 x 512', labels)."""
    x, y, w, h = stats[i, :4]
    cy = y + h / 2
    k = 0
    for j in range(1, n):
        if j == i:
            continue
        xj, yj, wj, hj, aj = stats[j]
        if aj < 8 or not 0.6 * h <= hj <= 1.6 * h or abs(yj + hj / 2 - cy) > 0.35 * h:
            continue
        gap = max(xj - (x + w), x - (xj + wj))
        if gap <= 1.2 * h:
            k += 1
            if k >= 2:
                return True
    return False


def _score(a, big):
    """Confidence in [0, 1]: clean head(+shaft) fit, a head that is not tiny, strong ink contrast.
    (AUROC ~0.95 for true vs false detections on the dev and holdout sets.)"""
    fit = max(0.0, 1 - a["err"] / 0.17)
    size = min(1.0, a["head_w"] / (0.02 * big))
    con = min(1.0, a["contrast"] / 150)
    return float(fit * size * con * (1.0 if a["has_shaft"] else 0.7))


def find_arrows(img, max_arrows=6, max_components=400, min_contrast=70.0, colors=None, min_score=0.0):
    """img: HxWx3 uint8 -> (list of arrow dicts sorted by confidence, uint8 mask of arrow pixels, dilated).
    colors: optional subset of {"colored", "white", "black"} (e.g. from the caption: "red arrow")."""
    H, W = img.shape[:2]
    big = max(H, W)
    min_area = max(20, int(3e-5 * H * W))
    max_area = int(0.03 * H * W)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32)
    found = []
    for name, ch in _channels(img):
        if colors and name not in colors:
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ch, connectivity=8)
        comps = [i for i in range(1, n) if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
        comps = sorted(comps, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:max_components]
        for i in comps:
            x, y, w, h = stats[i, :4]
            if x <= 0 or y <= 0 or x + w >= W or y + h >= H:   # touches the border: frame/background
                continue
            sub = lab[y:y + h, x:x + w] == i
            a = _analyse(sub, big)
            if a is None:
                continue
            a["contrast"], a["sharpness"] = _contrast(name, gray, sat, lab, i, x, y, w, h)
            if a["contrast"] < (min_contrast if a["has_shaft"] else 1.5 * min_contrast):
                continue
            # an almost perfect head+shaft fit is strong evidence on its own (JPEG-softened or slightly
            # grey arrows); weaker fits must also look like drawn ink: pure colour and a hard edge
            strong = a["has_shaft"] and a["err"] <= 0.05
            if not strong and name != "colored" and (
                    a["sharpness"] < 0.9 or not _pure_ink(name, gray, lab, i, x, y, w, h)):
                continue
            if _text_like(stats, i, n):
                continue
            a["score"] = _score(a, big)
            if a["score"] < min_score:
                continue
            a["tip"] = (a["tip"][0] + x, a["tip"][1] + y)
            a.update(area=int(stats[i, cv2.CC_STAT_AREA]), channel=name,
                     pix=(sub, int(x), int(y)))
            found.append(a)
    # one detection per physical arrow (channels can overlap), most confident first
    found.sort(key=lambda a: -a["score"])
    arrows = []
    tol = max(6.0, 0.015 * big)
    for a in found:
        if all(np.hypot(a["tip"][0] - b["tip"][0], a["tip"][1] - b["tip"][1]) > tol for b in arrows):
            arrows.append(a)
    arrows = arrows[:max_arrows]
    mask = np.zeros((H, W), np.uint8)
    for a in arrows:
        sub, x, y = a.pop("pix")
        mask[y:y + sub.shape[0], x:x + sub.shape[1]] |= sub.astype(np.uint8)
    for a in found:
        a.pop("pix", None)
    if mask.any():
        k = 5 if big < 600 else 7   # cover anti-aliased / JPEG halo around the ink
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return arrows, mask


def remove_annotations(img, ann_mask):
    """Inpaint the arrow pixels so they no longer look like image content."""
    if ann_mask is None or not ann_mask.any():
        return img
    return cv2.inpaint(img, ann_mask, 5, cv2.INPAINT_TELEA)
