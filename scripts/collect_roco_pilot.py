from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import re

OUTPUT_DIR = Path("roco_pilot")
IMAGES_PER_MODALITY = 20

MODALITIES = {
    "xray": "X-Ray",
    "x-ray": "X-Ray",
    "radiograph": "X-Ray",
    "radiography": "X-Ray",
    "computed tomography": "CT",
    "ct": "CT",
    "magnetic resonance": "MRI",
    "mri": "MRI",
    "ultrasound": "Ultrasound",
    "sonography": "Ultrasound",
    "angiography": "Angiography",
    "pet": "PET",
    "positron emission": "PET",
    "pet/ct": "Combined",
    "pet-ct": "Combined",
}

def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower().strip())

def detect_modality(sample):
    """
    Use the ROCO sample's caption/CUI text to identify modality.

    We intentionally keep this conservative. If modality isn't
    confidently identified, the image is skipped.
    """

    text_parts = []

    if sample.get("caption"):
        text_parts.append(str(sample["caption"]))

    if sample.get("cui"):
        text_parts.append(str(sample["cui"]))

    text = normalize_text(" ".join(text_parts))

    # Check more specific combined modality first
    if "pet/ct" in text or "pet-ct" in text:
        return "Combined"

    # Longer phrases first
    for keyword in sorted(
        MODALITIES.keys(),
        key=len,
        reverse=True
    ):
        if keyword in text:
            return MODALITIES[keyword]

    return None


def main():

    print("Loading ROCOv2 train split in streaming mode...")

    dataset = load_dataset(
        "eltorio/ROCOv2-radiology",
        split="train",
        streaming=True,
    )

    counts = {
        modality: 0
        for modality in set(MODALITIES.values())
    }

    for modality in counts:
        (OUTPUT_DIR / modality).mkdir(
            parents=True,
            exist_ok=True
        )

    total_saved = 0

    print("\nCollecting images...\n")

    for sample in tqdm(dataset):

        if all(
            count >= IMAGES_PER_MODALITY
            for count in counts.values()
        ):
            break

        modality = detect_modality(sample)

        if modality is None:
            continue

        if counts[modality] >= IMAGES_PER_MODALITY:
            continue

        image = sample["image"]

        image_id = str(
            sample["image_id"]
        )

        output_path = (
            OUTPUT_DIR
            / modality
            / f"{image_id}.png"
        )

        try:
            image.convert("RGB").save(
                output_path
            )

            counts[modality] += 1
            total_saved += 1

        except Exception as e:
            print(
                f"\nCould not save {image_id}: {e}"
            )

    print("\n==============================")
    print("COLLECTION COMPLETE")
    print("==============================\n")

    for modality, count in counts.items():
        print(f"{modality:15s}: {count}")

    print(f"\nTotal saved: {total_saved}")

if __name__ == "__main__":
    main()