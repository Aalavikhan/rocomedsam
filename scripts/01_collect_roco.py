"""Step 1: stream ROCOv2, keep images whose captions describe a localizable
finding (plus a few 'normal' ones), save images + metadata.csv.

    uv run python scripts/01_collect_roco.py --split train      --max_findings 20000
    uv run python scripts/01_collect_roco.py --split validation --max_findings 1500
"""
import argparse
import io
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from roco_medsam_test.captions import classify
from roco_medsam_test.common import HF_DATASET, ROCO_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--max_findings", type=int, default=20000)
    ap.add_argument("--normal_frac", type=float, default=0.15, help="normal images as fraction of max_findings")
    ap.add_argument("--max_per_modality", type=int, default=6000, help="avoid X-ray/CT dominating")
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--single_finding", action="store_true",
                    help="drop captions that mention several distinct findings (cleaner masks, fewer images)")
    args = ap.parse_args()

    ds = load_dataset(HF_DATASET, split=args.split, streaming=True)
    ds = ds.cast_column("image", HFImage(decode=False))  # don't decode images we will reject

    out = ROCO_DIR / args.split
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.csv"

    max_norm = int(args.normal_frac * args.max_findings)
    n_find = n_norm = 0
    per_mod = Counter()
    rows = []

    def dump():  # incremental, so a crash/Ctrl-C does not orphan the saved images
        pd.DataFrame(rows).to_csv(meta_path, index=False)

    for s in tqdm(ds, desc=f"scan {args.split}"):
        if n_find >= args.max_findings and n_norm >= max_norm:
            break
        info = classify(s["caption"])
        if info is None:
            continue
        if info["kind"] == "finding":
            if n_find >= args.max_findings or per_mod[info["modality"]] >= args.max_per_modality:
                continue
            if args.single_finding and info["n_terms"] > 1:
                continue
        elif n_norm >= max_norm:
            continue

        try:
            im = s["image"]
            if im.get("bytes"):
                img = Image.open(io.BytesIO(im["bytes"]))
            elif im.get("path"):
                img = Image.open(im["path"])
            else:
                continue
            img = img.convert("RGB")
        except Exception as e:
            print(f"skip {s['image_id']}: {e}")
            continue
        if max(img.size) > args.max_side:
            k = args.max_side / max(img.size)
            img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
        image_id = str(s["image_id"])
        img.save(out / "images" / f"{image_id}.png")

        rows.append({"image_id": image_id, "split": args.split, "caption": s["caption"], **info})
        if info["kind"] == "finding":
            n_find += 1
            per_mod[info["modality"]] += 1
        else:
            n_norm += 1
        if len(rows) % 500 == 0:
            dump()

    dump()
    if not rows:
        print("No usable captions found.")
        return
    df = pd.DataFrame(rows)
    print(f"\nsaved {len(df)} rows -> {meta_path}")
    print(df.groupby(["kind", "modality"]).size())
    fd = df[df.kind == "finding"]
    if len(fd):
        print("\nSpot-check some parsed phrases:")
        print(fd[["modality", "phrase"]].sample(min(15, len(fd)), random_state=0).to_string(index=False))


if __name__ == "__main__":
    main()
