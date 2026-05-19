"""
Custom dataset for VQA-RAD with LLaVA-compatible preprocessing.

Handles image loading, text tokenization, and loss masking
so that loss is only computed on the answer portion.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class VQARADDataset(Dataset):
    """
    VQA-RAD dataset loader for LLaVA fine-tuning.

    Each sample returns:
      - pixel_values: preprocessed image tensor
      - input_ids: tokenized [question + answer] sequence
      - attention_mask: standard attention mask
      - labels: same as input_ids but with question tokens masked to -100
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        processor,
        tokenizer,
        split: str = "train",
        max_length: int = 512,
    ):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        self.data = [entry for entry in all_data if entry["split"] == split]
        self._normalize_answers()

    def _normalize_answers(self):
        """Normalize answer strings for consistency."""
        for entry in self.data:
            entry["answer"] = entry["answer"].strip().lower()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        # load image
        img_path = self.image_dir / entry["image_name"]
        image = Image.open(img_path).convert("RGB")

        # build prompt and target
        question = entry["question"].strip()
        answer = entry["answer"]

        prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{question} ASSISTANT:"
        full_text = f"{prompt} {answer}</s>"

        # process prompt only to get prompt_len
        prompt_inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # process full text
        full_inputs = self.processor(
            text=full_text,
            images=image,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        pixel_values = full_inputs["pixel_values"].squeeze(0)

        # loss masking: set prompt tokens to -100 so loss is only on the answer
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        # also mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_dataloaders(
    data_path: str,
    image_dir: str,
    processor,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 2,
):
    """Create train and test dataloaders."""
    train_ds = VQARADDataset(
        data_path=data_path,
        image_dir=image_dir,
        processor=processor,
        tokenizer=tokenizer,
        split="train",
        max_length=max_length,
    )

    test_ds = VQARADDataset(
        data_path=data_path,
        image_dir=image_dir,
        processor=processor,
        tokenizer=tokenizer,
        split="test",
        max_length=max_length,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"train samples: {len(train_ds)}, test samples: {len(test_ds)}")
    return train_loader, test_loader
