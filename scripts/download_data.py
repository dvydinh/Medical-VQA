"""
Download VQA-RAD dataset from Hugging Face and save to data/raw/.

Source: flaviagiammarino/vqa-rad (mirror of the original OSF release)
Reference: Lau et al. (2018), Scientific Data, 5(1), 180251.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main():
    raw_dir = Path("data/raw")
    img_dir = raw_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print("loading vqa-rad from huggingface...")
    ds = load_dataset("flaviagiammarino/vqa-rad")

    all_entries = []

    for split_name in ds.keys():
        split = ds[split_name]
        print(f"processing split: {split_name} ({len(split)} samples)")

        for idx, sample in enumerate(split):
            image: Image.Image = sample["image"]
            question = sample["question"]
            answer = sample["answer"]

            # save image
            img_filename = f"{split_name}_{idx:04d}.jpg"
            img_path = img_dir / img_filename

            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(img_path, "JPEG")

            entry = {
                "qid": f"{split_name}_{idx}",
                "image_name": img_filename,
                "question": question,
                "answer": answer,
                "split": split_name,
            }

            # preserve original metadata if available
            if "answer_type" in sample:
                entry["answer_type"] = sample["answer_type"]
            if "question_type" in sample:
                entry["question_type"] = sample["question_type"]

            all_entries.append(entry)

    output_path = raw_dir / "VQA_RAD_Dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\nsaved {len(all_entries)} QA pairs to {output_path}")
    print(f"saved images to {img_dir}")
    print(f"total images: {len(list(img_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
