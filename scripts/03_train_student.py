"""Step 3: train the image-only student on teacher pseudo-masks.

    uv run python scripts/03_train_student.py --out runs/student_r0
Cross-fitted students for self-training (step 4):
    uv run python scripts/03_train_student.py --out runs/s_f0 --fold 0 --nfolds 2
    uv run python scripts/03_train_student.py --out runs/s_f1 --fold 1 --nfolds 2

Model selection uses the ROCO *validation* pseudo-labels only. The public
eval sets are never used for selection.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from roco_medsam_test.common import PSEUDO_DIR, PROJECT_ROOT, fold_of, set_seed
from roco_medsam_test.student import (PseudoDataset, UNet, add_filter_args, filter_kwargs, heat_loss,
                                      load_manifest, seg_loss)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    dices = []
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x)[0]) > 0.5).float()
        t = y[:, :1]
        inter = (p * t).sum((1, 2, 3))
        den = p.sum((1, 2, 3)) + t.sum((1, 2, 3))
        d = torch.where(den == 0, torch.ones_like(den), (2 * inter + 1e-6) / (den + 1e-6))
        dices += d.cpu().tolist()
    return float(np.mean(dices)) if dices else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", default=str(PSEUDO_DIR / "manifest_train.csv"))
    ap.add_argument("--val_manifest", default=str(PSEUDO_DIR / "manifest_validation.csv"))
    ap.add_argument("--out", default="runs/student_r0")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--heat_w", type=float, default=0.5)
    add_filter_args(ap)
    ap.add_argument("--fold", type=int, default=-1, help="hold out this fold (cross-fitting)")
    ap.add_argument("--nfolds", type=int, default=2)
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
    if args.fold >= 0:
        tr = tr[tr.image_id.map(lambda i: fold_of(i, args.nfolds) != args.fold)].reset_index(drop=True)
    print(f"train {len(tr)} ({(tr.kind=='normal').sum()} normal) | val {len(va)}")
    if len(tr) < args.bs or len(va) == 0:
        raise SystemExit("Too few samples after filtering - loosen --min_sep or run steps 1-2 on more data.")

    tl = DataLoader(PseudoDataset(tr, args.size, True), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    # persistent workers: on Windows, re-spawning loader processes every epoch costs ~100 s (10x the epoch itself)
    vl = DataLoader(PseudoDataset(va, args.size, False), batch_size=args.bs, num_workers=args.workers,
                    persistent_workers=args.workers > 0)

    model = UNet(args.encoder, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    if args.wandb:
        import wandb
        wandb.init(project="roco-medsam-student", name=Path(args.out).name, config=vars(args))

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        run = []
        for x, y, w, hv in tqdm(tl, desc=f"epoch {ep+1}/{args.epochs}", leave=False):
            x, y, w, hv = x.to(device), y.to(device), w.to(device), hv.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                seg, heat = model(x)
            seg, heat = seg.float(), heat.float()
            loss = seg_loss(seg, y[:, :1], w) + args.heat_w * heat_loss(heat, y[:, 1:], w, hv)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run.append(loss.item())
        vd = validate(model, vl, device)
        print(f"epoch {ep+1}: loss {np.mean(run):.4f} | val pseudo-dice {vd:.4f}")
        if args.wandb:
            wandb.log({"loss": np.mean(run), "val_pseudo_dice": vd, "epoch": ep + 1})
        ck = {"model": model.state_dict(), "encoder": args.encoder, "size": args.size, "epoch": ep + 1,
              "val": vd, "fold": args.fold, "nfolds": args.nfolds}
        torch.save(ck, out / "last.pt")
        if vd > best:
            best = vd
            torch.save(ck, out / "best.pt")
    print(f"best val pseudo-dice {best:.4f} -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
