"""
Training pipeline for Medical VQA with QLoRA.

Supports WandB logging, gradient accumulation, and checkpoint saving.
"""

import argparse
import math
import os
import time

import torch
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup

from src.dataset import get_dataloaders
from src.model import load_model


def load_training_config(config_path: str = "configs/training_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg, epoch, global_step):
    """Run one training epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    accum_steps = cfg["training"]["gradient_accumulation_steps"]
    log_steps = cfg["logging"]["log_steps"]

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss / accum_steps
        loss.backward()

        total_loss += outputs.loss.item()
        global_step += 1

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % log_steps == 0:
            avg_loss = total_loss / (step + 1)
            current_lr = scheduler.get_last_lr()[0]
            gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

            wandb.log({
                "train/loss": avg_loss,
                "train/step_loss": outputs.loss.item(),
                "learning_rate": current_lr,
                "cuda/memory_allocated_gb": gpu_mem,
                "global_step": global_step,
            })

            print(
                f"  epoch {epoch+1} | step {step+1}/{len(train_loader)} | "
                f"loss: {avg_loss:.4f} | lr: {current_lr:.2e} | gpu: {gpu_mem:.1f}GB"
            )

    avg_epoch_loss = total_loss / len(train_loader)
    return avg_epoch_loss, global_step


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0

    for batch in test_loader:
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--qlora_config", default="configs/qlora_config.yaml")
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    train_cfg = cfg["training"]
    paths = cfg["paths"]

    # init wandb
    wandb.init(
        project=cfg["logging"]["wandb_project"],
        config=cfg,
    )

    # load model
    model, processor, tokenizer = load_model(
        model_name=cfg["model"]["name"],
        qlora_config_path=args.qlora_config,
    )

    device = next(model.parameters()).device

    # dataloaders
    data_path = os.path.join(paths["data_dir"], "VQA_RAD_Dataset.json")
    image_dir = os.path.join(paths["data_dir"], "images")

    train_loader, test_loader = get_dataloaders(
        data_path=data_path,
        image_dir=image_dir,
        processor=processor,
        tokenizer=tokenizer,
        batch_size=train_cfg["batch_size"],
        max_length=cfg["model"]["max_length"],
    )

    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # scheduler
    num_training_steps = (
        len(train_loader) // train_cfg["gradient_accumulation_steps"]
    ) * train_cfg["num_epochs"]
    num_warmup_steps = int(num_training_steps * train_cfg["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # training loop
    os.makedirs(paths["output_dir"], exist_ok=True)
    global_step = 0
    best_val_loss = float("inf")

    print(f"\nstarting training for {train_cfg['num_epochs']} epochs")
    print(f"total training steps: {num_training_steps}")

    for epoch in range(train_cfg["num_epochs"]):
        start_time = time.time()

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, cfg, epoch, global_step
        )

        val_loss = evaluate(model, test_loader, device)
        elapsed = time.time() - start_time

        wandb.log({
            "val/loss": val_loss,
            "epoch": epoch + 1,
            "epoch_time_min": elapsed / 60,
        })

        print(
            f"\nepoch {epoch+1}/{train_cfg['num_epochs']} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"time: {elapsed/60:.1f}min"
        )

        # save checkpoint if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(paths["output_dir"], "best_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  saved best model to {save_path}")

        # periodic save
        if (epoch + 1) % 1 == 0:
            save_path = os.path.join(paths["output_dir"], f"checkpoint-epoch{epoch+1}")
            model.save_pretrained(save_path)
            print(f"  saved checkpoint to {save_path}")

    wandb.finish()
    print("\ntraining complete")


if __name__ == "__main__":
    main()
