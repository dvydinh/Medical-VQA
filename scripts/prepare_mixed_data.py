"""
Prepare mixed dataset for Medical VQA (VQA-RAD + SLAKE).

Downloads VQA-RAD and SLAKE datasets from Hugging Face.
Extracts English QA pairs, saves images to a unified directory,
and creates a single MedVQA_Mixed_Dataset.json file.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main():
    raw_dir = Path("data/raw")
    img_dir = raw_dir / "images_mixed"
    img_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    total_vqa_rad = 0
    total_slake = 0

    # 1. Process VQA-RAD
    print("loading vqa-rad from huggingface...")
    try:
        ds_vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
        for split_name in ds_vqa_rad.keys():
            split = ds_vqa_rad[split_name]
            print(f"processing vqa-rad split: {split_name} ({len(split)} samples)")

            for idx, sample in enumerate(split):
                image: Image.Image = sample["image"]
                question = sample["question"]
                answer = sample["answer"]

                img_filename = f"vqarad_{split_name}_{idx:04d}.jpg"
                img_path = img_dir / img_filename

                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_path, "JPEG")

                entry = {
                    "qid": f"vqarad_{split_name}_{idx}",
                    "image_name": img_filename,
                    "question": question,
                    "answer": str(answer),
                    # Keep original splits: "train" goes to train set, "test" goes to test set
                    "split": split_name,
                    "source": "vqa-rad"
                }

                if "answer_type" in sample:
                    entry["answer_type"] = sample["answer_type"]
                
                all_entries.append(entry)
                total_vqa_rad += 1
    except Exception as e:
        print(f"Error loading VQA-RAD: {e}")

    # 2. Process SLAKE
    print("\nloading SLAKE from huggingface (Keetawan/SLAKE)...")
    try:
        ds_slake = load_dataset("Keetawan/SLAKE")
        for split_name in ds_slake.keys():
            split = ds_slake[split_name]
            print(f"processing SLAKE split: {split_name} ({len(split)} samples)")

            for idx, sample in enumerate(split):
                # We only want English QA pairs
                if "q_lang" in sample and sample["q_lang"] != "en":
                    continue
                    
                image: Image.Image = sample["image"]
                question = sample["question"]
                answer = sample["answer"]

                img_filename = f"slake_{split_name}_{idx:05d}.jpg"
                img_path = img_dir / img_filename

                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_path, "JPEG")

                entry = {
                    "qid": f"slake_{split_name}_{idx}",
                    "image_name": img_filename,
                    "question": question,
                    "answer": str(answer),
                    # ALL SLAKE data goes to the "train" split to augment training.
                    # We ONLY evaluate on the VQA-RAD "test" split to maintain benchmark integrity.
                    "split": "train",
                    "source": "slake"
                }

                if "answer_type" in sample:
                    entry["answer_type"] = sample["answer_type"]
                
                all_entries.append(entry)
                total_slake += 1
    except Exception as e:
        print(f"Error loading SLAKE: {e}")

    # 3. Save JSON
    output_path = raw_dir / "MedVQA_Mixed_Dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\n=== SUMMARY ===")
    print(f"VQA-RAD samples: {total_vqa_rad}")
    print(f"SLAKE (English) samples: {total_slake}")
    print(f"Total QA pairs saved to {output_path}: {len(all_entries)}")
    print(f"Total images saved to {img_dir}: {len(list(img_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
