"""Image-ONLY student #2: MedSAM ViT-B + LoRA, prompt-free (trained on teacher pseudo-masks).

    image --MedSAM encoder (frozen, LoRA on qkv)--> embedding (256 x 64 x 64)
          --mask decoder (fine-tuned) + LEARNED constant prompt tokens--> mask logits (256 x 256)
          --presence head (pooled embedding)--> P(a finding exists); below threshold -> empty mask
          --heat head--> teacher heatmap (auxiliary, like the UNet's `heat` head)

No prompts and no text at inference. Preprocessing is MedSAM's own (1024 px, per-image min-max to [0, 1]),
NOT ImageNet normalisation. For deployment the LoRA weights are merged into qkv (`merge_lora`), so the saved
model is a plain MedSAM + two small heads.

Known risk: the teacher is also MedSAM, so this student may inherit the teacher's mistakes (confirmation
bias). Keep the UNet folds for self-training (step 4); train this student on the cleaned manifest.
"""
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

from .common import CHECKPOINT, PROJECT_ROOT, add_medsam_to_path

SAM_SIZE = 1024      # MedSAM input size (the ViT position embedding is fixed to a 64 x 64 grid)
LOW_RES = 256        # mask decoder output size
EMB = 64             # image embedding grid


def sam_preprocess(img):
    """HxWx3 uint8 -> 3x1024x1024 float32 in [0, 1] (identical to the teacher's MedSAMBox.embed)."""
    x = cv2.resize(img, (SAM_SIZE, SAM_SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
    return x.transpose(2, 0, 1)


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B(A(x)). B starts at zero, so training starts from plain MedSAM."""

    def __init__(self, base: nn.Linear, r=4, alpha=4):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin = lin.to(self.base.weight.device, self.base.weight.dtype)
        lin.weight.copy_(self.base.weight + (self.B @ self.A) * self.scale)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin


def build_medsam_vit_b(checkpoint_path=None):
    """MedSAM ViT-B. With checkpoint_path=None the weights are random (to be overwritten by load_state_dict);
    MedSAM's own builder always tries to open a checkpoint file, so the architecture is spelled out here
    (same hyper-parameters as segment_anything.build_sam.build_sam_vit_b)."""
    add_medsam_to_path()
    if checkpoint_path is not None:
        from segment_anything import sam_model_registry
        return sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    from functools import partial
    from segment_anything.modeling import (ImageEncoderViT, MaskDecoder, PromptEncoder, Sam,
                                           TwoWayTransformer)
    return Sam(
        image_encoder=ImageEncoderViT(
            depth=12, embed_dim=768, img_size=SAM_SIZE, mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=12, patch_size=16, qkv_bias=True,
            use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11], window_size=14, out_chans=256),
        prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(EMB, EMB),
                                     input_image_size=(SAM_SIZE, SAM_SIZE), mask_in_chans=16),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256),
        pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
    ).eval()


class SamStudent(nn.Module):
    def __init__(self, checkpoint_path=CHECKPOINT, lora_r=4, lora_alpha=4, grad_ckpt=True):
        super().__init__()
        sam = build_medsam_vit_b(checkpoint_path)
        self.encoder, self.decoder = sam.image_encoder, sam.mask_decoder
        self.grad_ckpt = grad_ckpt
        pe = sam.prompt_encoder
        # fixed positional encoding of the 64x64 embedding grid (not trained; same for every image)
        with torch.no_grad():
            self.register_buffer("image_pe", pe.get_dense_pe().clone())
            # learned constant prompts, initialised from MedSAM's embedding of a whole-image box prompt
            box = torch.tensor([[[0.0, 0.0, SAM_SIZE - 1.0, SAM_SIZE - 1.0]]])
            sparse, dense = pe(points=None, boxes=box, masks=None)
        self.sparse = nn.Parameter(sparse.clone())          # (1, 2, 256)
        self.dense = nn.Parameter(dense.clone())            # (1, 256, 64, 64)
        c = self.dense.shape[1]
        self.presence = nn.Sequential(nn.Linear(2 * c, 128), nn.ReLU(True), nn.Linear(128, 1))
        self.heat = nn.Sequential(nn.Conv2d(c, 64, 3, padding=1), nn.ReLU(True), nn.Conv2d(64, 1, 1))
        self.lora_r = lora_r
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        if lora_r > 0:
            for blk in self.encoder.blocks:
                blk.attn.qkv = LoRALinear(blk.attn.qkv, lora_r, lora_alpha)

    def encode(self, x):
        e = self.encoder
        if not (self.grad_ckpt and self.training):
            return e(x)
        x = e.patch_embed(x)
        if e.pos_embed is not None:
            x = x + e.pos_embed
        for blk in e.blocks:
            x = checkpoint(blk, x, use_reentrant=False)
        return e.neck(x.permute(0, 3, 1, 2))

    def forward(self, x):
        """x: (B, 3, 1024, 1024) in [0, 1] -> (mask logits B x 1 x 256 x 256, presence logit B, heat logit B x 1 x 64 x 64)"""
        emb = self.encode(x)
        B = emb.shape[0]
        low, _ = self.decoder(
            image_embeddings=emb,
            image_pe=self.image_pe,
            sparse_prompt_embeddings=self.sparse.expand(B, -1, -1),
            dense_prompt_embeddings=self.dense.expand(B, -1, -1, -1),
            multimask_output=False,
        )
        pooled = torch.cat([emb.mean((2, 3)), emb.amax((2, 3))], 1)
        return low, self.presence(pooled)[:, 0], self.heat(emb)

    def trainable_state(self):
        """Only what training changes (LoRA, decoder, prompts, heads): small enough to save every epoch."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    @torch.no_grad()
    def merge_lora(self):
        for blk in self.encoder.blocks:
            if isinstance(blk.attn.qkv, LoRALinear):
                blk.attn.qkv = blk.attn.qkv.merged()
        self.lora_r = 0
        return self


def load_sam_student(ckpt_path, device):
    """Deployment checkpoint (LoRA already merged) -> (model, presence_threshold)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SamStudent(checkpoint_path=None, lora_r=0, grad_ckpt=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), float(ck["presence_thr"])


@torch.no_grad()
def sam_predict(model, img, device, presence_thr=0.5, tta=False):
    """img: HxWx3 uint8 -> (HxW float probability map, presence probability). Image-only inference.
    If presence < presence_thr the map is all zeros (no finding)."""
    H, W = img.shape[:2]
    x = torch.from_numpy(sam_preprocess(img))[None].to(device)
    low, pres, _ = model(x)
    p, q = torch.sigmoid(low), torch.sigmoid(pres)
    if tta:
        low2, pres2, _ = model(x.flip(-1))
        p, q = (p + torch.sigmoid(low2).flip(-1)) / 2, (q + torch.sigmoid(pres2)) / 2
    q = float(q[0])
    if q < presence_thr:
        return np.zeros((H, W), np.float32), q
    p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
    return p[0, 0].cpu().numpy(), q


class SamPseudoDataset(Dataset):
    """Same manifest rows as student.PseudoDataset, MedSAM preprocessing. Returns
    (image 3x1024x1024, mask 1x256x256, heat 1x64x64, weight, heat_valid, has_finding)."""

    def __init__(self, df, train=True):
        self.df, self.train = df, train
        self.aug = None
        if train:
            import albumentations as A
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.85, 1.15), rotate=(-15, 15), translate_percent=(-0.1, 0.1), p=0.6),
                A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        S = SAM_SIZE
        img = cv2.resize(np.array(Image.open(PROJECT_ROOT / r.image_path).convert("RGB")), (S, S),
                         interpolation=cv2.INTER_CUBIC)
        if r.mask_path:
            m = np.array(Image.open(PROJECT_ROOT / r.mask_path).convert("L"))
            m = (cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
        else:
            m = np.zeros((S, S), np.float32)
        if r.heat_path:
            h = np.load(PROJECT_ROOT / r.heat_path).astype(np.float32)
            h = np.clip(cv2.resize(h, (S, S), interpolation=cv2.INTER_CUBIC), 0, 1)
            hv = 1.0
        elif r.kind == "normal":
            h, hv = np.zeros((S, S), np.float32), 1.0
        else:
            h, hv = np.zeros((S, S), np.float32), 0.0   # no heat target -> ignore heat loss
        if self.aug is not None:
            out = self.aug(image=img, mask=np.dstack([m, h]))
            img, mh = out["image"], out["mask"]
            m, h = mh[..., 0], mh[..., 1]
        x = (img.astype(np.float32) - img.min()) / max(float(img.max() - img.min()), 1e-8)
        m = cv2.resize(m, (LOW_RES, LOW_RES), interpolation=cv2.INTER_AREA) > 0.5
        h = cv2.resize(h, (EMB, EMB), interpolation=cv2.INTER_AREA)
        w = float(r.weight) if r.weight != "" else 1.0
        return (
            torch.from_numpy(x.transpose(2, 0, 1)),
            torch.from_numpy(m[None].astype(np.float32)),
            torch.from_numpy(h[None].astype(np.float32)),
            torch.tensor(w, dtype=torch.float32),
            torch.tensor(hv, dtype=torch.float32),
            torch.tensor(float(m.any()), dtype=torch.float32),
        )
