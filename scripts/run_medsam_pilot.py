from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDSAM_ROOT = PROJECT_ROOT / "MedSAM"

sys.path.insert(0, str(MEDSAM_ROOT))

from segment_anything import (
    SamAutomaticMaskGenerator,
    sam_model_registry,
)


IMAGE_ROOT = PROJECT_ROOT / "roco_pilot"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "medsam_vit_b.pth"
OUTPUT_ROOT = PROJECT_ROOT / "medsam_results"


def load_image(path):
    return np.array(
        Image.open(path).convert("RGB")
    )


def normalize(image):
    image = image.astype(np.float32)

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    if high <= low:
        high = low + 1

    image = (image - low) / (high - low)

    return np.clip(image, 0, 1)


def overlay_masks(image, masks, max_masks=16):
    output = image.copy()

    masks = sorted(
        masks,
        key=lambda x: x["area"],
        reverse=True,
    )[:max_masks]

    rng = np.random.default_rng(42)

    for mask in masks:
        segmentation = mask["segmentation"]
        color = rng.random(3)

        output[segmentation] = (
            0.65 * output[segmentation]
            + 0.35 * color
        )

    return np.clip(output, 0, 1)


def main():

    print("====================================")
    print("        ROCO + MedSAM PILOT")
    print("====================================")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("\nLoading MedSAM...")

    model = sam_model_registry["vit_b"](
        checkpoint=str(CHECKPOINT)
    )

    model.to(device)
    model.eval()

    print("MedSAM loaded successfully.")

    # --------------------------------------------------------
    # IMPORTANT:
    # We intentionally use the automatic mask generator
    # only for this feasibility experiment.
    # --------------------------------------------------------

    generator = SamAutomaticMaskGenerator(
        model=model,

        points_per_side=16,
        points_per_batch=32,

        pred_iou_thresh=0.50,
        stability_score_thresh=0.65,

        stability_score_offset=1.0,

        box_nms_thresh=0.70,

        crop_n_layers=0,

        min_mask_region_area=500,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    modality_dirs = sorted(
        [
            p
            for p in IMAGE_ROOT.iterdir()
            if p.is_dir()
        ]
    )

    print(
        f"\nFound {len(modality_dirs)} modalities."
    )

    results = []

    for modality_dir in modality_dirs:

        images = sorted(
            [
                p
                for p in modality_dir.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in {".png", ".jpg", ".jpeg"}
            ]
        )

        if not images:
            print(
                f"\nSkipping {modality_dir.name}: no images"
            )
            continue

        image_path = images[0]

        print(
            f"\n[{modality_dir.name}]"
        )

        print(
            f"Image: {image_path.name}"
        )

        image = load_image(
            image_path
        )

        print(
            f"Image size: "
            f"{image.shape[1]} x {image.shape[0]}"
        )

        start = time.perf_counter()

        with torch.inference_mode():
            masks = generator.generate(
                image
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"Masks generated: {len(masks)}"
        )

        

        print(
            f"Runtime: {elapsed:.2f} seconds"
        )

        image_display = normalize(image)

        overlay = overlay_masks(
            image_display,
            masks,
        )

        output_dir = (
            OUTPUT_ROOT
            / modality_dir.name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / f"{image_path.stem}_result.png"
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 7),
        )

        axes[0].imshow(
            image_display
        )

        axes[0].set_title(
            "Original"
        )

        axes[0].axis("off")

        axes[1].imshow(
            overlay
        )

        axes[1].set_title(
            f"MedSAM masks "
            f"({len(masks)})"
        )

        axes[1].axis("off")

        fig.suptitle(
            f"{modality_dir.name} | "
            f"{image_path.name}"
        )

        fig.tight_layout()

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        results.append(
            {
                "modality":
                    modality_dir.name,
                "image":
                    image_path.name,
                "num_masks":
                    len(masks),
                "runtime_seconds":
                    elapsed,
                "output":
                    str(output_path),
            }
        )

        print(
            f"Saved: {output_path}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    import pandas as pd

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_ROOT / "pilot_results.csv",
        index=False,
    )

    print(
        "\n===================================="
    )
    print(
        "PILOT COMPLETE"
    )
    print(
        "===================================="
    )

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()