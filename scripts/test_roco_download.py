# from datasets import load_dataset

# print("Loading ROCOv2...")

# dataset = load_dataset(
#     "eltorio/ROCOv2-radiology",
#     split="train",
#     streaming=True,
# )

# print("Dataset connected successfully.")

# sample = next(iter(dataset))

# print("\nFields:")
# print(sample.keys())

# print("\nImage ID:")
# print(sample["image_id"])

# print("\nCaption:")
# print(sample["caption"])

# print("\nImage:")
# print(sample["image"])


from pathlib import Path

from datasets import load_dataset

OUTPUT = Path("roco_sample")
OUTPUT.mkdir(exist_ok=True)

print("Connecting to ROCOv2...")

dataset = load_dataset(
    "eltorio/ROCOv2-radiology",
    split="train",
    streaming=True,
)

sample = next(iter(dataset))

image = sample["image"]
image_id = sample["image_id"]

output_path = OUTPUT / f"{image_id}.png"

image.save(output_path)

print(f"Saved: {output_path}")
print(f"Caption: {sample['caption']}")