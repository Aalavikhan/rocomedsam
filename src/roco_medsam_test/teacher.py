"""Text-conditioned TEACHER (uses captions; training time only).

caption phrase --BiomedCLIP (SCLIP-style dense similarity, 448px)--> heatmap
heatmap --> candidate boxes --MedSAM--> candidate masks --> best one, with quality scores

If the caption says the image is marked (arrow, asterisk, ...) we look for arrows in the image. When
arrows are found, MedSAM is prompted with boxes anchored at the arrow tip (the arrow says WHERE the
finding is; the heatmap only helps rank). The arrows are inpainted away before CLIP/SAM see the
image and the arrow-free image is returned so the student never trains on arrows.
"""
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from .arrows import find_arrows, remove_annotations
from .captions import arrow_hints
from .common import BIOMEDCLIP, CHECKPOINT, add_medsam_to_path

ARROW_MIN_SCORE = 0.2   # minimum arrows.find_arrows confidence (tuned with scripts/audit_teacher.py)
# Dense CLIP method. Audited on data/dev/teacher_dev_gt.json (56 hand-boxed findings): heat peak inside the
# finding box 0.66 vs 0.16 for the original "maskclip@224/softmax" (a text-free centre prior scores 0.21).
CLIP_METHOD = "sclip@448/diff"
# How the mask is chosen among the MedSAM candidates of one location (see Teacher._score); audited with
# scripts/audit_segmentation.py.
SELECT = "stab_sep_heat"
UNION_RATIO = 0.5
CLEAN_MASK = True
MASK_MAX_AREA = 0.15   # the mask chosen at a location may not exceed this image fraction (location unaffected)

STAB_HI, STAB_LO = float(1 / (1 + np.exp(-1.0))), float(1 / (1 + np.exp(1.0)))   # sigmoid(+-1)

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


NEG_PROMPTS = ("a normal {modality} image", "normal anatomy", "background", "text, labels and annotations")


def build_prompts(phrase, term, modality, mode="default"):
    """-> (positive prompt(s), negative prompts). A list of positives is averaged into one text embedding."""
    negs = [n.format(modality=modality) for n in NEG_PROMPTS]
    if mode == "default":
        return f"a {modality} image showing {phrase}", negs
    term = term or phrase
    if mode == "term":
        return f"a {modality} image showing {term}", negs
    if mode == "ens":   # several phrasings of the finding; the averaged embedding is less template-sensitive
        pos = [f"a {modality} image showing {phrase}", f"{phrase}", f"{modality} of {phrase}",
               f"a {modality} image showing {term}", f"{term}"]
        return list(dict.fromkeys(pos)), negs
    raise ValueError(mode)


class BiomedCLIPSaliency:
    """Dense text-image similarity from BiomedCLIP.

    method = "<dense>@<res>/<score>", e.g. "maskclip@224/softmax" (original) or "clearclip@448/diff".
      dense: maskclip  - last block uses only the value path (MaskCLIP, Zhou et al. 2022)
             clearclip - last block keeps attention but with q-q similarity, no residual, no MLP
                         (ClearCLIP, Lan et al. 2024): suppresses the global/noisy residual stream
             sclip     - last block attention = softmax(qq^T) + softmax(kk^T) (SCLIP, Wang et al. 2024)
      res:   input resolution; the position embedding is interpolated, so 448 gives a 28x28 patch grid
      score: softmax - per-patch softmax over [positive, negatives] at CLIP temperature 100
             diff    - cos(patch, positive) - max cos(patch, negatives)  (no saturation)
    """

    def __init__(self, device, method="maskclip@224/softmax"):
        import open_clip
        self.device = device
        self.model = open_clip.create_model(BIOMEDCLIP).to(device).eval()
        self.tok = open_clip.get_tokenizer(BIOMEDCLIP)
        dense, rest = method.split("@")
        res, score = rest.split("/")
        assert dense in ("maskclip", "clearclip", "sclip") and score in ("softmax", "diff"), method
        self.dense, self.res, self.score = dense, int(res), score
        self._selfcheck()

    # ---- ViT internals (timm) ----
    @property
    def trunk(self):
        return self.model.visual.trunk

    def _embed_tokens(self, x):
        """Patch + position embedding for any input size (position embedding resized bicubically)."""
        t = self.trunk
        pe = t.patch_embed
        x = pe.proj(x)
        gh, gw = x.shape[-2:]
        x = pe.norm(x.flatten(2).transpose(1, 2))
        pos = t.pos_embed
        n_pre = 0 if getattr(t, "no_embed_class", False) else getattr(t, "num_prefix_tokens", 1)
        pre_pos, patch_pos = pos[:, :n_pre], pos[:, n_pre:]
        g0 = int(round(patch_pos.shape[1] ** 0.5))
        if (gh, gw) != (g0, g0):
            patch_pos = patch_pos.reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(gh, gw), mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        x = x + patch_pos
        cls = t.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls + pre_pos if n_pre else cls, x], 1)
        if hasattr(t, "patch_drop"):
            x = t.patch_drop(x)
        if hasattr(t, "norm_pre"):
            x = t.norm_pre(x)
        return x

    def _post_norm(self, v):
        """timm puts the final LayerNorm in `norm` (token pooling) or `fc_norm` (avg pooling)."""
        t = self.trunk
        v = t.norm(v)
        if hasattr(t, "fc_norm"):
            v = t.fc_norm(v)
        return v

    @torch.no_grad()
    def _selfcheck(self):
        """Verify that re-running the trunk by hand reproduces encode_image (CLS or mean pooling).
        Guards the MaskCLIP hook against timm/open_clip layout differences."""
        try:
            x = torch.randn(1, 3, 224, 224, device=self.device)
            tok = self._embed_tokens(x)
            for blk in self.trunk.blocks:
                tok = blk(tok)
            tok = self.trunk.norm(tok)
            fc = getattr(self.trunk, "fc_norm", torch.nn.Identity())
            ref = F.normalize(self.model.encode_image(x), dim=-1)
            sims = []
            for v in (fc(tok[:, 0]), fc(tok[:, 1:].mean(1))):
                sims.append(float((F.normalize(self.model.visual.head(v), dim=-1) @ ref.T).item()))
            if max(sims) < 0.98:
                warnings.warn(f"BiomedCLIP manual forward != encode_image (cos={sims}); "
                              "dense heatmaps may be wrong - inspect the previews carefully.")
            # the dense path must agree with the trunk's own last block when nothing is removed:
            # recompute block[-1] from qkv by hand (catches qkv layout / attention changes in timm)
            blk = self.trunk.blocks[-1]
            x0 = self._embed_tokens(x)
            for b in self.trunk.blocks[:-1]:
                x0 = b(x0)
            y = blk.norm1(x0)
            q, k, v = self._qkv(blk, y)
            a = (q @ k.transpose(-2, -1) * blk.attn.scale).softmax(-1)
            out = self._merge(blk, a @ v)
            manual = x0 + blk.ls1(out)
            manual = manual + blk.ls2(blk.mlp(blk.norm2(manual)))
            err = float((manual - blk(x0)).abs().max())
            if err > 1e-3:
                warnings.warn(f"BiomedCLIP dense path does not reproduce the last block (max err {err:.2e}); "
                              "dense heatmaps are likely wrong.")
        except Exception as e:  # never block the run, but tell the user
            warnings.warn(f"BiomedCLIP self-check could not run: {e}")

    # ---- encoders ----
    def prep(self, img, size=224):
        x = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (np.clip(x, 0, 1) - CLIP_MEAN) / CLIP_STD
        return torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)

    @torch.no_grad()
    def embed_text(self, texts):
        tok = self.tok(texts, context_length=256).to(self.device)
        return F.normalize(self.model.encode_text(tok), dim=-1)

    @torch.no_grad()
    def embed_image(self, img):
        return F.normalize(self.model.encode_image(self.prep(img)), dim=-1)

    @staticmethod
    def _qkv(blk, y):
        """(B, heads, N, head_dim) q, k, v from timm's fused qkv projection (layout: 3 x heads x head_dim)."""
        a = blk.attn
        B, N, _ = y.shape
        qkv = a.qkv(y).reshape(B, N, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        return a.q_norm(q), a.k_norm(k), v

    @staticmethod
    def _merge(blk, o):
        a = blk.attn
        o = o.transpose(1, 2).reshape(o.shape[0], o.shape[2], -1)
        return a.proj(a.norm(o) if hasattr(a, "norm") else o)

    @torch.no_grad()
    def patch_features(self, x):
        """Per-token CLIP-space embeddings (B, 1+N, D), L2-normalised. The last block is replaced by a
        dense-prediction variant (see class docstring); earlier blocks run unchanged."""
        x = self._embed_tokens(x)
        for blk in self.trunk.blocks[:-1]:
            x = blk(x)
        blk = self.trunk.blocks[-1]
        y = blk.norm1(x)
        q, k, v = self._qkv(blk, y)
        s = blk.attn.scale
        if self.dense == "maskclip":
            o = v                                                     # value path only
        elif self.dense == "clearclip":
            o = (q @ q.transpose(-2, -1) * s).softmax(-1) @ v
        else:  # sclip
            o = ((q @ q.transpose(-2, -1) * s).softmax(-1) + (k @ k.transpose(-2, -1) * s).softmax(-1)) @ v
        o = self._merge(blk, o)                                       # no residual, no MLP
        o = self._post_norm(o)
        return F.normalize(self.model.visual.head(o), dim=-1)

    @torch.no_grad()
    def heatmap(self, img, pos_text, neg_texts):
        """Returns (heat in [0,1] at image size, pos_emb, neg_emb, raw_peak).
        pos_text may be a list (prompt ensemble, embeddings averaged).
        raw_peak = max patch score before min-max normalisation (softmax: probability; diff: cosine
        margin); low values mean the map has no real peak (min-max would otherwise hide that)."""
        H, W = img.shape[:2]
        pf = self.patch_features(self.prep(img, self.res))[0, 1:]   # (N, D), drop CLS
        pos_list = [pos_text] if isinstance(pos_text, str) else list(pos_text)
        tp = F.normalize(self.embed_text(pos_list).mean(0, keepdim=True), dim=-1)
        tn = self.embed_text(list(neg_texts))
        sp, sn = (pf @ tp.T)[:, 0], pf @ tn.T
        if self.score == "softmax":
            s = (100.0 * torch.cat([sp[:, None], sn], 1)).softmax(-1)[:, 0]
        else:
            s = sp - sn.max(1).values
        g = int(round(s.numel() ** 0.5))
        h = s.reshape(g, g).float().cpu().numpy()
        heat = cv2.resize(h, (W, H), interpolation=cv2.INTER_CUBIC)
        heat = ndi.gaussian_filter(heat, sigma=0.01 * max(H, W))
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat.astype(np.float32), tp, tn, float(s.max())

    @torch.no_grad()
    def margin(self, img, pos_emb, neg_emb):
        e = self.embed_image(img)
        return float((e @ pos_emb.T).item() - (e @ neg_emb.T).mean().item())


class MedSAMBox:
    def __init__(self, device):
        add_medsam_to_path()
        from segment_anything import sam_model_registry
        self.device = device
        self.model = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT)).to(device).eval()

    @torch.no_grad()
    def embed(self, img):
        x = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        x = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)       # MedSAM min-max normalisation
        t = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)
        return self.model.image_encoder(t)

    @torch.no_grad()
    def predict(self, emb, box_xyxy, H, W):
        box = np.array([box_xyxy], dtype=np.float32) / np.array([W, H, W, H], dtype=np.float32) * 1024
        box_t = torch.as_tensor(box, dtype=torch.float32, device=self.device)[:, None, :]
        sparse, dense = self.model.prompt_encoder(points=None, boxes=box_t, masks=None)
        low, iou = self.model.mask_decoder(
            image_embeddings=emb,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        p = F.interpolate(torch.sigmoid(low), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        m = p > 0.5
        # SAM stability score: how much the mask changes when the logit threshold moves by +-1. A mask that
        # sits on real image edges barely moves; a mask "floating" in uniform tissue grows/shrinks a lot.
        stab = float((p > STAB_HI).sum()) / max(float((p > STAB_LO).sum()), 1.0)
        return m.cpu().numpy(), float(iou[0, 0]), stab


class Teacher:
    # box side as a fraction of min(H, W) for arrow-anchored candidates
    ARROW_SIZES = (0.07, 0.11, 0.17, 0.25, 0.35)

    def __init__(self, device="cuda", cover_pow=0.15, max_area=0.35, clip_method=CLIP_METHOD, select=SELECT,
                 union_ratio=UNION_RATIO, clean_mask=CLEAN_MASK, mask_max_area=MASK_MAX_AREA):
        """cover_pow: exponent on 'fraction of total heat captured by the mask' (legacy heat scoring only).
        max_area: candidates covering more than this fraction of the image are REJECTED outright
        (a mask that covers the picture makes segmentation pointless).
        select: how to pick among the candidate masks of one location (see _score).
        union_ratio: arrows whose best score is >= union_ratio * top score are merged into one mask."""
        self.clip = BiomedCLIPSaliency(device, method=clip_method)
        self.sam = MedSAMBox(device)
        self.cover_pow = cover_pow
        self.max_area = max_area
        self.select = select
        self.union_ratio = union_ratio
        self.clean_mask = clean_mask
        self.mask_max_area = mask_max_area

    # ---------------- candidate boxes ----------------
    @staticmethod
    def candidate_boxes(heat, thresholds=(0.6, 0.75, 0.9), min_area_frac=0.002, max_cands=6):
        H, W = heat.shape
        boxes = []
        for thr in thresholds:
            lab, n = ndi.label(heat > thr)
            if n == 0:
                continue
            sums = ndi.sum(heat, lab, index=np.arange(1, n + 1))
            slices = ndi.find_objects(lab)
            for k in np.argsort(sums)[::-1][:2]:
                sl = slices[int(k)]
                if (lab[sl] == k + 1).sum() < min_area_frac * H * W:
                    continue
                px, py = int(0.02 * W), int(0.02 * H)
                b = [max(0, sl[1].start - px), max(0, sl[0].start - py),
                     min(W - 1, sl[1].stop + px), min(H - 1, sl[0].stop + py)]
                if all(box_iou(b, o) < 0.8 for o in boxes):
                    boxes.append(b)
        return boxes[:max_cands]

    def arrow_boxes(self, arrow, H, W):
        """Square boxes of several sizes that start just beyond the arrow tip (arrow points at the
        near edge of the finding) or are centred just past the tip (tip inside the finding)."""
        tip, d = np.array(arrow["tip"]), np.array(arrow["dir"])
        gap = max(2.0, 0.3 * arrow["head_len"])
        base = min(H, W)
        boxes = []
        for f in self.ARROW_SIZES:
            s = f * base
            for center in (tip + d * (gap + 0.3 * s), tip + d * gap):   # box starts ~0.2*s BEFORE the tip, or is centred at it
                x0, y0 = center - s / 2
                x1, y1 = center + s / 2
                b = [max(0.0, x0), max(0.0, y0), min(W - 1.0, x1), min(H - 1.0, y1)]
                if b[2] - b[0] >= 8 and b[3] - b[1] >= 8:
                    boxes.append(b)
        return boxes

    # ---------------- candidate scoring ----------------
    def _eval(self, emb, box, H, W, heat, heat_total, gray=None):
        m, iou, stab = self.sam.predict(emb, box, H, W)
        area = float(m.mean())
        if area < 0.001 or area > self.max_area:
            return None
        c = {"mask": m, "sam_iou": iou, "stab": stab, "area_frac": area, "box": box,
             "heat_in": float(heat[m].mean()), "cover": float(heat[m].sum()) / heat_total, "sep": 0.0}
        if gray is not None:   # inside-vs-surrounding contrast (in units of the two regions' spread)
            inner = ndi.binary_erosion(m, iterations=3)
            inner = inner if inner.any() else m
            ring = ndi.binary_dilation(m, iterations=6) & ~m
            if ring.any():
                c["sep"] = float(abs(gray[inner].mean() - gray[ring].mean()) /
                                 (gray[inner].std() + gray[ring].std() + 1e-3))
        return c

    @staticmethod
    def _clean(m):
        """Fill holes; drop fragments smaller than 20% of the largest piece (MedSAM speckle)."""
        m = ndi.binary_fill_holes(m)
        lab, n = ndi.label(m)
        if n > 1:
            sizes = ndi.sum(m, lab, np.arange(1, n + 1))
            m = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.2 * sizes.max()))
        return m

    def _mask_score(self, c, kind):
        """Score used to choose the mask AT an already-decided location (never to choose the location).
        legacy:        the original hand-made formula (c["legacy"]; it differs for heat vs arrow candidates)
        stab:          MedSAM stability score only
        stab_sep_heat: stability x inside-vs-surround contrast x mean caption heat inside the mask
        self.select is one mode for both, or "arrow_mode/heat_mode" (kind = "arrow" or "heat")."""
        modes = self.select.split("/")
        mode = modes[0] if len(modes) == 1 or kind == "arrow" else modes[1]
        if mode == "legacy":
            return c["legacy"]
        if mode == "stab":
            return c["stab"]
        if mode == "stab_sep_heat":
            return c["stab"] * (c["sep"] + 0.1) * c["heat_in"]
        if mode == "stab_sep":
            return c["stab"] * (c["sep"] + 0.1)
        raise ValueError(self.select)

    def _best_from_heat(self, emb, boxes, H, W, heat, heat_total, gray=None):
        """WHERE: the legacy score picks the heat region (unchanged). WHAT: among the candidates that
        overlap that region's mask, the mask score picks the final mask."""
        cands = []
        for b in boxes:
            c = self._eval(emb, b, H, W, heat, heat_total, gray)
            if c is None:
                continue
            c["legacy"] = c["sam_iou"] * c["heat_in"] * (c["cover"] ** self.cover_pow) * (1 - c["area_frac"]) ** 2
            cands.append(c)
        if not cands:
            return None
        loc = max(cands, key=lambda c: c["legacy"])
        same = [c for c in cands if (c["mask"] & loc["mask"]).any()]
        same = [c for c in same if c["area_frac"] <= self.mask_max_area] or [min(same, key=lambda c: c["area_frac"])]
        best = max(same, key=lambda c: self._mask_score(c, "heat"))
        best["score"] = loc["legacy"]
        return best

    def _best_from_arrows(self, emb, arrows, H, W, heat, heat_total, gray=None):
        """Per arrow: candidates from boxes anchored at its tip. WHERE: the legacy score decides which arrows
        are used (union of up to 3 comparable ones; unchanged). WHAT: the mask score picks each arrow's mask."""
        tol = 0.04 * max(H, W)
        per_arrow = []
        for a in arrows:
            tx, ty = int(round(min(max(a["tip"][0], 0), W - 1))), int(round(min(max(a["tip"][1], 0), H - 1)))
            cands = []
            for b in self.arrow_boxes(a, H, W):
                c = self._eval(emb, b, H, W, heat, heat_total, gray)
                if c is None:
                    continue
                dist = float(ndi.distance_transform_edt(~c["mask"])[ty, tx])   # tip -> nearest mask pixel
                prox = 1.0 if dist <= tol else max(0.1, tol / dist)
                ys, xs = np.where(c["mask"])
                fill = ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / max(
                    1.0, (b[2] - b[0]) * (b[3] - b[1]))
                fill_pen = min(1.0, fill / 0.3)                                 # mask much smaller than box -> box too big
                # mask touching a side of the prompt box (not the image border) = box probably cuts the finding
                touch = sum([xs.min() <= b[0] + 2 and b[0] > 2, xs.max() >= b[2] - 2 and b[2] < W - 3,
                             ys.min() <= b[1] + 2 and b[1] > 2, ys.max() >= b[3] - 2 and b[3] < H - 3])
                c["legacy"] = (c["sam_iou"] * (0.5 + 0.5 * c["heat_in"]) * prox * fill_pen * (0.6 ** touch)
                               * (1 - c["area_frac"]) ** 2)
                cands.append(c)
            if cands:
                ok = [c for c in cands if c["area_frac"] <= self.mask_max_area] or [min(cands, key=lambda c: c["area_frac"])]
                best = max(ok, key=lambda c: self._mask_score(c, "arrow"))
                best["score"] = max(c["legacy"] for c in cands)     # arrow ranking stays on the legacy score
                best["anchors"] = [a]
                per_arrow.append(best)
        if not per_arrow:
            return None
        per_arrow.sort(key=lambda c: -c["score"])
        keep = [c for c in per_arrow[:3] if c["score"] >= self.union_ratio * per_arrow[0]["score"]]
        top = keep[0]
        if len(keep) == 1:
            return top
        m = np.any([c["mask"] for c in keep], axis=0)
        if float(m.mean()) > self.max_area:
            return top
        return {"mask": m, "anchors": [c["anchors"][0] for c in keep], "boxes": [c["box"] for c in keep],
                "sam_iou": float(np.mean([c["sam_iou"] for c in keep])), "area_frac": float(m.mean()),
                "box": top["box"], "heat_in": float(heat[m].mean()), "cover": float(heat[m].sum()) / heat_total,
                "stab": float(np.mean([c["stab"] for c in keep])), "sep": float(np.mean([c["sep"] for c in keep])),
                "score": top["score"]}

    # ---------------- main entry ----------------
    def detect(self, img, phrase, modality, marked=False, use_arrows=True, caption="", term=""):
        """WHERE: arrows (training-time annotation) and the caption heatmap. No masks yet."""
        arrows, ann = [], None
        if marked and use_arrows:
            hint = arrow_hints(caption)
            arrows, ann = find_arrows(img, max_arrows=hint["max_n"], colors=hint["colors"],
                                      min_score=ARROW_MIN_SCORE)
        work = remove_annotations(img, ann) if arrows else img
        pos, negs = build_prompts(phrase, term, modality)
        heat, pos_emb, neg_emb, peak = self.clip.heatmap(work, pos, negs)
        return {"work": work, "arrows": arrows, "heat": heat, "pos_emb": pos_emb, "neg_emb": neg_emb,
                "peak": peak, "emb": self.sam.embed(work)}

    def segment(self, det):
        """WHAT: detection -> best candidate mask dict (or None) and whether an arrow anchored it."""
        work, arrows, heat, emb = det["work"], det["arrows"], det["heat"], det["emb"]
        H, W = work.shape[:2]
        heat_total = float(heat.sum()) + 1e-8
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY).astype(np.float32) if "sep" in self.select else None
        best, used = None, False
        if arrows:
            best = self._best_from_arrows(emb, arrows, H, W, heat, heat_total, gray)
            used = best is not None
        if best is None:
            boxes = self.candidate_boxes(heat)
            if boxes:
                best = self._best_from_heat(emb, boxes, H, W, heat, heat_total, gray)
        if best is not None and self.clean_mask:
            best["mask"] = self._clean(best["mask"])
            best["area_frac"] = float(best["mask"].mean())
        return best, used

    def run(self, img, phrase, modality, marked=False, use_arrows=True, caption="", term=""):
        """None (no candidate) or dict with mask/heat, quality scores and arrow info.
        caption: full caption, used only for arrow hints (named arrow colour, singular/plural).
        Extra keys: n_arrows (detected), arrow_used (bool), arrow_tips [(x,y,dx,dy)],
        clean_img (arrow-free copy of img, or None when no arrow was found)."""
        det = self.detect(img, phrase, modality, marked, use_arrows, caption, term)
        best, used = self.segment(det)
        if best is None:
            return None
        work, arrows, heat = det["work"], det["arrows"], det["heat"]
        pos_emb, neg_emb, peak = det["pos_emb"], det["neg_emb"], det["peak"]

        ys, xs = np.where(best["mask"])
        py, px = int(0.1 * (ys.max() - ys.min() + 1)), int(0.1 * (xs.max() - xs.min() + 1))
        crop = work[max(0, ys.min() - py):ys.max() + py + 1, max(0, xs.min() - px):xs.max() + px + 1]
        m_crop = self.clip.margin(crop, pos_emb, neg_emb)
        m_full = self.clip.margin(work, pos_emb, neg_emb)
        best.update(clip_margin=m_crop, crop_gain=m_crop - m_full, heat=heat, heat_peak=peak,
                    n_arrows=len(arrows), arrow_used=used,
                    arrow_tips=[(a["tip"][0], a["tip"][1], a["dir"][0], a["dir"][1]) for a in arrows],
                    clean_img=work if arrows else None)
        return best
