"""
Validate VQA-RAD dataset integrity and consistency.

Checks:
  1. Image file integrity (detect corrupted files)
  2. Label normalization (lowercase, strip whitespace)
  3. Answer type distribution (closed vs open-ended)
  4. Question type distribution
  5. Class balance analysis
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


def check_image_integrity(img_dir: Path) -> tuple[list[str], list[str]]:
    """Open every image with PIL to detect corrupted files."""
    valid = []
    corrupted = []

    img_files = sorted(img_dir.glob("*"))
    for img_path in img_files:
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()
            valid.append(img_path.name)
        except (OSError, SyntaxError) as e:
            print(f"  [corrupted] {img_path.name}: {e}")
            corrupted.append(img_path.name)

    return valid, corrupted


def normalize_answer(answer: str) -> str:
    """Normalize answer string: lowercase, strip whitespace."""
    return answer.strip().lower()


def classify_answer_type(answer: str) -> str:
    """Classify as closed-ended or open-ended based on answer content."""
    normalized = normalize_answer(answer)
    if normalized in ("yes", "no"):
        return "closed"
    return "open"


def validate_dataset(data_path: Path, img_dir: Path):
    """Run all validation checks on the dataset."""

    # load QA data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"total QA pairs: {len(data)}")

    # 1. image integrity
    print("\n--- image integrity check ---")
    valid_imgs, corrupted_imgs = check_image_integrity(img_dir)
    print(f"  valid images: {len(valid_imgs)}")
    print(f"  corrupted images: {len(corrupted_imgs)}")

    if corrupted_imgs:
        print(f"  WARNING: {len(corrupted_imgs)} corrupted files found!")
        for name in corrupted_imgs:
            print(f"    - {name}")

    # check that all referenced images exist
    referenced_images = set(entry["image_name"] for entry in data)
    existing_images = set(valid_imgs)
    missing = referenced_images - existing_images
    if missing:
        print(f"\n  WARNING: {len(missing)} referenced images not found on disk:")
        for name in sorted(missing)[:10]:
            print(f"    - {name}")

    # 2. label normalization check
    print("\n--- label consistency check ---")
    normalization_issues = 0
    answer_counter = Counter()

    for entry in data:
        raw_answer = entry["answer"]
        normalized = normalize_answer(raw_answer)
        answer_counter[normalized] += 1

        if raw_answer != normalized:
            normalization_issues += 1

    print(f"  answers needing normalization: {normalization_issues}")
    print(f"  unique answers (after normalization): {len(answer_counter)}")

    # show top answers
    print("\n  top 15 answers:")
    for ans, count in answer_counter.most_common(15):
        print(f"    '{ans}': {count}")

    # 3. answer type distribution
    print("\n--- answer type distribution ---")
    type_counter = Counter()
    for entry in data:
        atype = classify_answer_type(entry["answer"])
        type_counter[atype] += 1

    for atype, count in type_counter.most_common():
        pct = 100 * count / len(data)
        print(f"  {atype}: {count} ({pct:.1f}%)")

    # 4. question type distribution (if available in metadata)
    print("\n--- question type distribution ---")
    if "question_type" in data[0]:
        qtype_counter = Counter(entry.get("question_type", "unknown") for entry in data)
        for qtype, count in qtype_counter.most_common():
            pct = 100 * count / len(data)
            print(f"  {qtype}: {count} ({pct:.1f}%)")
    else:
        print("  question_type field not available in dataset metadata")

    # 5. split distribution
    print("\n--- split distribution ---")
    split_counter = Counter(entry["split"] for entry in data)
    for split, count in split_counter.most_common():
        print(f"  {split}: {count}")

    # 6. basic stats
    print("\n--- basic stats ---")
    q_lengths = [len(entry["question"].split()) for entry in data]
    a_lengths = [len(entry["answer"].split()) for entry in data]
    print(f"  avg question length: {sum(q_lengths)/len(q_lengths):.1f} words")
    print(f"  avg answer length: {sum(a_lengths)/len(a_lengths):.1f} words")
    print(f"  max question length: {max(q_lengths)} words")
    print(f"  max answer length: {max(a_lengths)} words")

    # summary
    print("\n=== validation complete ===")
    if corrupted_imgs or missing:
        print("STATUS: issues found, review warnings above")
        return False
    else:
        print("STATUS: all checks passed")
        return True


def main():
    raw_dir = Path("data/raw")
    data_path = raw_dir / "VQA_RAD_Dataset.json"
    img_dir = raw_dir / "images"

    if not data_path.exists():
        print(f"error: {data_path} not found. run download_data.py first.")
        sys.exit(1)

    if not img_dir.exists():
        print(f"error: {img_dir} not found. run download_data.py first.")
        sys.exit(1)

    ok = validate_dataset(data_path, img_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
