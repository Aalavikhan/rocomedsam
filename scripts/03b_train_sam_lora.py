"""Step 3b: train the image-only MedSAM + LoRA student on teacher pseudo-masks (alternative to the UNet).

    uv run python scripts/03b_train_sam_lora.py --out runs/sam_lora_r0
After self-training with the UNet folds (step 4), train on the cleaned manifest:
    uv run python scripts/03b_train_sam_lora.py --out runs/sam_lora_r1 \
        --train_manifest data/pseudo/manifest_train_r1.csv

Model AND presence-threshold selection use the ROCO *validation* pseudo-labels only. The public eval sets
are never used for selection. Use the SAME filter flags as in step 3 so both students see the same data.
Outputs: best_trainable.pt / last_trainable.pt (LoRA + trained parts, small) and best.pt (LoRA merged,
self-contained deployment checkpoint for scripts/05_evaluate.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from roco_medsam_test.common import PSEUDO_DIR, PROJECT_ROOT, set_seed
from roco_medsam_test.sam_student import SamPseudoDataset, SamStudent
from roco_medsam_test.student import add_filter_args, filter_kwargs, heat_loss, load_manifest, seg_loss

PRESENCE_THRS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def masked_seg_loss(low, m, w, has):
    """Dice+BCE (same as the UNet) on images that contain a finding only. Empty images are handled by the
    presence head; asking the SAM decoder to output nothing fights how it was trained."""
    if has.sum() == 0:
        return low.sum() * 0.0
    k = has > 0
    return seg_loss(low[k], m[k], w[k])


@torch.no_grad()
def validate(model, loader, device):
    """-> (best mean pseudo-dice over presence thresholds, that threshold, dice without gating, presence AUC)"""
    model.eval()
    dices, pres, has_f = [], [], []
    for x, m, _, _, _, has in loader:
        x, m = x.to(device), m.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            low, p, _ = model(x)
        pm = (torch.sigmoid(low.float()) > 0.5).float()
        inter = (pm * m).sum((1, 2, 3))
        den = pm.sum((1, 2, 3)) + m.sum((1, 2, 3))
        d_pred = torch.where(den == 0, torch.ones_like(den), (2 * inter + 1e-6) / (den + 1e-6))
        d_empty = (m.sum((1, 2, 3)) == 0).float()        # dice if we output an empty mask instead
        dices.append(torch.stack([d_pred, d_empty], 1).cpu())
        pres.append(torch.sigmoid(p.float()).cpu())
        has_f.append(has)
    d, q, h = torch.cat(dices).numpy(), torch.cat(pres).numpy(), torch.cat(has_f).numpy()
    scores = {t: float(np.where(q >= t, d[:, 0], d[:, 1]).mean()) for t in PRESENCE_THRS}
    thr = max(scores, key=scores.get)
    pos, neg = q[h > 0], q[h == 0]
    auc = float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()) if len(pos) and len(neg) else float("nan")
    return scores[thr], thr, float(d[:, 0].mean()), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", default=str(PSEUDO_DIR / "manifest_train.csv"))
    ap.add_argument("--val_manifest", default=str(PSEUDO_DIR / "manifest_validation.csv"))
    ap.add_argument("--out", default="runs/sam_lora_r0")
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=float, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=2, help="per step; 4 does not fit in 16 GB at 1024 px")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps (effective batch bs*accum)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--heat_w", type=float, default=0.5)
    ap.add_argument("--presence_w", type=float, default=0.5)
    add_filter_args(ap)
    ap.add_argument("--no_grad_ckpt", action="store_true", help="faster, needs more GPU memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    filt = filter_kwargs(args)
    tr = load_manifest(args.train_manifest, **filt)
    va = load_manifest(args.val_manifest, **filt)
    print(f"train {len(tr)} ({(tr.kind=='normal').sum()} normal) | val {len(va)} ({(va.kind=='normal').sum()} normal)")
    if len(tr) < args.bs or len(va) == 0:
        raise SystemExit("Too few samples after filtering - loosen --min_sep or run steps 1-2 on more data.")

    tl = DataLoader(SamPseudoDataset(tr, True), batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    # persistent workers: on Windows, re-spawning loader processes every epoch costs ~100 s
    vl = DataLoader(SamPseudoDataset(va, False), batch_size=args.bs, num_workers=args.workers,
                    persistent_workers=args.workers > 0)

    model = SamStudent(lora_r=args.lora_r, lora_alpha=args.lora_alpha, grad_ckpt=not args.no_grad_ckpt).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = max(1, args.epochs * (len(tl) // args.accum))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    if args.wandb:
        import wandb
        wandb.init(project="roco-medsam-student", name=Path(args.out).name, config=vars(args))

    meta = {"kind": "sam_lora", "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "size": 1024}
    best, step = -1.0, 0
    for ep in range(args.epochs):
        model.train()
        run = []
        opt.zero_grad(set_to_none=True)
        for it, (x, m, h, w, hv, has) in enumerate(tqdm(tl, desc=f"epoch {ep+1}/{args.epochs}", leave=False)):
            x, m, h, w, hv, has = (t.to(device) for t in (x, m, h, w, hv, has))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                low, pres, heat = model(x)
            low, pres, heat = low.float(), pres.float(), heat.float()
            loss = (masked_seg_loss(low, m, w, has)
                    + args.presence_w * F.binary_cross_entropy_with_logits(pres, has)
                    + args.heat_w * heat_loss(heat, h, w, hv))
            (loss / args.accum).backward()
            if (it + 1) % args.accum == 0 and step < steps:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
            run.append(loss.item())
        vd, thr, vd_raw, auc = validate(model, vl, device)
        print(f"epoch {ep+1}: loss {np.mean(run):.4f} | val pseudo-dice {vd:.4f} (presence thr {thr}) | "
              f"no gating {vd_raw:.4f} | presence AUC {auc:.3f}")
        if args.wandb:
            wandb.log({"loss": np.mean(run), "val_pseudo_dice": vd, "presence_thr": thr, "presence_auc": auc,
                       "epoch": ep + 1})
        ck = {**meta, "trainable": model.trainable_state(), "presence_thr": thr, "epoch": ep + 1, "val": vd}
        torch.save(ck, out / "last_trainable.pt")
        if vd > best:
            best = vd
            torch.save(ck, out / "best_trainable.pt")

    # deployment checkpoint: best epoch, LoRA merged into qkv, presence threshold chosen on validation
    ck = torch.load(out / "best_trainable.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["trainable"], strict=False)
    model.merge_lora()
    torch.save({**meta, "lora_r": 0, "lora_merged_from_r": args.lora_r, "model": model.state_dict(),
                "presence_thr": ck["presence_thr"], "epoch": ck["epoch"], "val": ck["val"]}, out / "best.pt")
    print(f"best val pseudo-dice {best:.4f} (epoch {ck['epoch']}, presence thr {ck['presence_thr']}) -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
