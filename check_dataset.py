"""
Sanity-check your dataset before training.

Prints:
  1. Total image count per class (real / fake) as ImageFolder sees them
     (this is what actually gets used for training)
  2. A breakdown per denomination subfolder, so you can catch missing
     files, empty folders, or class imbalance before you spend time training.

Usage:
    python check_dataset.py
    python check_dataset.py --data_dir data
"""

import argparse
import os

from torchvision import datasets

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def count_denomination_breakdown(data_dir: str):
    """Walk real/fake -> denomination subfolders and count images in each."""
    breakdown = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        breakdown[class_name] = {}
        for denom in sorted(os.listdir(class_path)):
            denom_path = os.path.join(class_path, denom)
            if not os.path.isdir(denom_path):
                continue
            count = sum(
                1 for f in os.listdir(denom_path)
                if f.lower().endswith(VALID_EXTENSIONS)
            )
            breakdown[class_name][denom] = count

    return breakdown


def main():
    parser = argparse.ArgumentParser(description="Verify dataset structure and counts")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[!] Missing folder: {args.data_dir}")
        return

    # What ImageFolder will actually load (this is the ground truth for training)
    try:
        dataset = datasets.ImageFolder(args.data_dir)
        print(f"ImageFolder classes: {dataset.classes}")
        print(f"Total images loaded by ImageFolder: {len(dataset)}")
        for class_name, idx in dataset.class_to_idx.items():
            n = sum(1 for _, label in dataset.samples if label == idx)
            print(f"  {class_name}: {n} images (label={idx})")
    except FileNotFoundError as e:
        print(f"[!] ImageFolder could not load this directory: {e}")
        return

    # Per-denomination breakdown for your own sanity check
    print("\nPer-denomination breakdown:")
    breakdown = count_denomination_breakdown(args.data_dir)
    for class_name, denoms in breakdown.items():
        print(f"  {class_name}/")
        if not denoms:
            print("    (no denomination subfolders found - images may be flat in this class folder)")
        for denom, count in denoms.items():
            flag = "  <-- EMPTY!" if count == 0 else ""
            print(f"    {denom}: {count} images{flag}")

    print("\nDone. If any counts look off (0 images, missing denomination, "
          "big imbalance between real/fake), fix the folders before training.")


if __name__ == "__main__":
    main()
