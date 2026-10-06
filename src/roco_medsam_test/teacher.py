"""Text-conditioned TEACHER (uses captions; training time only).

caption phrase --BiomedCLIP (MaskCLIP-style dense similarity)--> heatmap
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
from .common import BIOMEDCLIP, CHECKPOINT, add_medsam_to_path

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


class BiomedCLIPSaliency:
    def __init__(self, device):
        import open_clip
        self.device = device
        self.model = open_clip.create_model(BIOMEDCLIP).to(device).eval()
        self.tok = open_clip.get_tokenizer(BIOMEDCLIP)
        self._selfcheck()

    # ---- ViT internals (timm) ----
    @property
    def trunk(self):
        return self.model.visual.trunk

    def _embed_tokens(self, x):
        t = self.trunk
        x = t._pos_embed(t.patch_embed(x))
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
        except Exception as e:  # never block the run, but tell the user
            warnings.warn(f"BiomedCLIP self-check could not run: {e}")

    # ---- encoders ----
    def prep(self, img):
        x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (np.clip(x, 0, 1) - CLIP_MEAN) / CLIP_STD
        return torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)

    @torch.no_grad()
    def embed_text(self, texts):
        tok = self.tok(texts, context_length=256).to(self.device)
        return F.normalize(self.model.encode_text(tok), dim=-1)

    @torch.no_grad()
    def embed_image(self, img):
        return F.normalize(self.model.encode_image(self.prep(img)), dim=-1)

    @torch.no_grad()
    def patch_features(self, x):
        """MaskCLIP trick: in the last block skip q-k attention and the MLP, use only the value
        path, then final norm + CLIP projection per patch token."""
        x = self._embed_tokens(x)
        for blk in self.trunk.blocks[:-1]:
            x = blk(x)
        blk = self.trunk.blocks[-1]
        y = blk.norm1(x)
        C = y.shape[-1]
        v = blk.attn.proj(blk.attn.qkv(y)[..., 2 * C:])
        v = self._post_norm(v)
        return F.normalize(self.model.visual.head(v), dim=-1)

    @torch.no_grad()
    def heatmap(self, img, pos_text, neg_texts):
        """Returns (heat in [0,1] at image size, pos_emb, neg_emb, raw_peak).
        raw_peak = max patch probability for the caption prompt before min-max normalisation;
        low values mean the map has no real peak (min-max would otherwise hide that)."""
        H, W = img.shape[:2]
        pf = self.patch_features(self.prep(img))[0, 1:]          # (N, D), drop CLS
        t = self.embed_text([pos_text] + list(neg_texts))        # (1+K, D)
        prob = (100.0 * pf @ t.T).softmax(-1)[:, 0]
        g = int(round(prob.numel() ** 0.5))
        h = prob.reshape(g, g).float().cpu().numpy()
        heat = cv2.resize(h, (W, H), interpolation=cv2.INTER_CUBIC)
        heat = ndi.gaussian_filter(heat, sigma=0.01 * max(H, W))
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat.astype(np.float32), t[0:1], t[1:], float(prob.max())

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
        p = F.interpolate(torch.sigmoid(low), size=(H, W), mode="bilinear", align_corners=False)
        return p[0, 0].cpu().numpy() > 0.5, float(iou[0, 0])


class Teacher:
    # box side as a fraction of min(H, W) for arrow-anchored candidates
    ARROW_SIZES = (0.07, 0.11, 0.17, 0.25, 0.35)

    def __init__(self, device="cuda", cover_pow=0.15, max_area=0.35):
        """cover_pow: exponent on 'fraction of total heat captured by the mask' in heat-candidate scoring.
        Higher -> prefers larger masks; 0 -> prefers the tightest, hottest spot.
        max_area: candidates covering more than this fraction of the image are REJECTED outright
        (a mask that covers the picture makes segmentation pointless)."""
        self.clip = BiomedCLIPSaliency(device)
        self.sam = MedSAMBox(device)
        self.cover_pow = cover_pow
        self.max_area = max_area

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
    def _eval(self, emb, box, H, W, heat, heat_total):
        m, iou = self.sam.predict(emb, box, H, W)
        area = float(m.mean())
        if area < 0.001 or area > self.max_area:
            return None
        return {"mask": m, "sam_iou": iou, "area_frac": area, "box": box,
                "heat_in": float(heat[m].mean()), "cover": float(heat[m].sum()) / heat_total}

    def _best_from_heat(self, emb, boxes, H, W, heat, heat_total):
        best = None
        for b in boxes:
            c = self._eval(emb, b, H, W, heat, heat_total)
            if c is None:
                continue
            c["score"] = c["sam_iou"] * c["heat_in"] * (c["cover"] ** self.cover_pow) * (1 - c["area_frac"]) ** 2
            if best is None or c["score"] > best["score"]:
                best = c
        return best

    def _best_from_arrows(self, emb, arrows, H, W, heat, heat_total):
        """Best candidate per arrow; union the (up to 3) arrows whose best score is comparable."""
        tol = 0.04 * max(H, W)
        per_arrow = []
        for a in arrows:
            tx, ty = int(round(min(max(a["tip"][0], 0), W - 1))), int(round(min(max(a["tip"][1], 0), H - 1)))
            best = None
            for b in self.arrow_boxes(a, H, W):
                c = self._eval(emb, b, H, W, heat, heat_total)
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
                c["score"] = (c["sam_iou"] * (0.5 + 0.5 * c["heat_in"]) * prox * fill_pen * (0.6 ** touch)
                              * (1 - c["area_frac"]) ** 2)
                if best is None or c["score"] > best["score"]:
                    best = c
            if best is not None:
                per_arrow.append(best)
        if not per_arrow:
            return None
        per_arrow.sort(key=lambda c: -c["score"])
        keep = [c for c in per_arrow[:3] if c["score"] >= 0.5 * per_arrow[0]["score"]]
        top = keep[0]
        if len(keep) == 1:
            return top
        m = np.any([c["mask"] for c in keep], axis=0)
        if float(m.mean()) > self.max_area:
            return top
        return {"mask": m, "sam_iou": float(np.mean([c["sam_iou"] for c in keep])), "area_frac": float(m.mean()),
                "box": top["box"], "heat_in": float(heat[m].mean()), "cover": float(heat[m].sum()) / heat_total,
                "score": top["score"]}

    # ---------------- main entry ----------------
    def run(self, img, phrase, modality, marked=False, use_arrows=True):
        """None (no candidate) or dict with mask/heat, quality scores and arrow info.
        Extra keys: n_arrows (detected), arrow_used (bool), arrow_tips [(x,y,dx,dy)],
        clean_img (arrow-free copy of img, or None when no arrow was found)."""
        H, W = img.shape[:2]
        arrows, ann = find_arrows(img) if (marked and use_arrows) else ([], None)
        work = remove_annotations(img, ann) if arrows else img

        pos = f"a {modality} image showing {phrase}"
        negs = [f"a normal {modality} image", "normal anatomy", "background",
                "text, labels and annotations"]
        heat, pos_emb, neg_emb, peak = self.clip.heatmap(work, pos, negs)
        heat_total = float(heat.sum()) + 1e-8
        emb = self.sam.embed(work)

        best, used = None, False
        if arrows:
            best = self._best_from_arrows(emb, arrows, H, W, heat, heat_total)
            used = best is not None
        if best is None:
            boxes = self.candidate_boxes(heat)
            if not boxes:
                return None
            best = self._best_from_heat(emb, boxes, H, W, heat, heat_total)
        if best is None:
            return None

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
