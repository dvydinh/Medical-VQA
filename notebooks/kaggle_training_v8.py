"""
Medical VQA - Training V8 (NEFTune) on Kaggle (T4 x2)

V6: GOLD STANDARD
  1. Differential Learning Rates: LoRA (LLM)=5e-5, Projector (full-weights)=2e-6
  2. Vicuna v1 Prompt Template: properly aligned instruction-following structure.

Yeu cau:
  - GPU: T4 x2
  - Kaggle Secrets: WANDB_API_KEY, GITHUB_TOKEN

Copy tung block vao tung cell tren Kaggle.
"""


# ==== CELL 1: install dependencies ====

!pip install -q torch transformers peft "bitsandbytes>=0.46.1" datasets \
    Pillow wandb nltk rouge-score pyyaml accelerate scipy matplotlib


# ==== CELL 2: setup secrets and environment ====

import os
import json
from pathlib import Path

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")

import torch
print(f"cuda available: {torch.cuda.is_available()}")
print(f"gpu count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  gpu {i}: {torch.cuda.get_device_name(i)} "
          f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")


# ==== CELL 3: clone repo and download dataset ====

REPO_URL = f"https://dvydinh:{GITHUB_TOKEN}@github.com/dvydinh/Medical-VQA.git"
WORK_DIR = "/kaggle/working/Medical-VQA"

os.system(f"git clone {REPO_URL} {WORK_DIR}")
os.chdir(WORK_DIR)

os.system('git config user.name "dvydinh"')
os.system('git config user.email "doanvy.dinh27@gmail.com"')

os.system("python scripts/download_data.py")


# ==== CELL 4: validate dataset ====

os.system("python scripts/validate_data.py")


# ==== CELL 5: train with differential learning rates ====

import sys
import time
import yaml
import torch
import wandb
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, WORK_DIR)
from src.model import load_model, apply_neftune
from src.dataset import get_dataloaders

with open("configs/training_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

with open("configs/qlora_config.yaml", "r") as f:
    qlora_cfg = yaml.safe_load(f)

train_cfg = cfg["training"]
paths = cfg["paths"]

wandb.init(
    project=cfg["logging"]["wandb_project"],
    config={**cfg, **qlora_cfg, "lora_lr": 5e-5, "projector_lr": 2e-6, "prompt_template": "vicuna_v1"},
    name="qlora-llava-vqa-rad-v8",
)

model, processor, tokenizer = load_model(
    model_name=cfg["model"]["name"],
    qlora_config_path="configs/qlora_config.yaml",
)

# Apply NEFTune for V8 to reduce overfitting
apply_neftune(model, alpha=5.0)

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

lora_params = []
projector_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "multi_modal_projector" in name:
        projector_params.append(param)
    else:
        lora_params.append(param)

print(f"lora params: {sum(p.numel() for p in lora_params):,}")
print(f"projector params: {sum(p.numel() for p in projector_params):,}")

LORA_LR = 5e-5
PROJECTOR_LR = 2e-6

optimizer = AdamW([
    {"params": lora_params, "lr": LORA_LR},
    {"params": projector_params, "lr": PROJECTOR_LR},
], weight_decay=train_cfg["weight_decay"])

num_training_steps = (
    len(train_loader) // train_cfg["gradient_accumulation_steps"]
) * train_cfg["num_epochs"]
num_warmup_steps = int(num_training_steps * train_cfg["warmup_ratio"])

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

device = next(model.parameters()).device
accum_steps = train_cfg["gradient_accumulation_steps"]
history = {"train_loss": [], "val_loss": [], "lr": [], "gpu_mem_gb": []}

os.makedirs(paths["output_dir"], exist_ok=True)
best_val_loss = float("inf")
global_step = 0

print(f"\nstarting training: {train_cfg['num_epochs']} epochs, "
      f"{len(train_loader)} steps/epoch")
print(f"lora lr: {LORA_LR}, projector lr: {PROJECTOR_LR}")

for epoch in range(train_cfg["num_epochs"]):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

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
        epoch_loss += outputs.loss.item()
        global_step += 1

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % cfg["logging"]["log_steps"] == 0:
            avg = epoch_loss / (step + 1)
            lr = scheduler.get_last_lr()[0]
            mem = torch.cuda.memory_allocated() / 1024**3
            wandb.log({
                "train/loss": avg,
                "train/step_loss": outputs.loss.item(),
                "learning_rate/lora": scheduler.get_last_lr()[0],
                "learning_rate/projector": scheduler.get_last_lr()[1],
                "cuda/memory_allocated_gb": mem,
                "global_step": global_step,
            })
            print(f"  epoch {epoch+1} | step {step+1}/{len(train_loader)} | "
                  f"loss: {avg:.4f} | lr: {lr:.2e} | gpu: {mem:.1f}GB")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
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
            val_loss += outputs.loss.item()

    val_loss /= len(test_loader)
    train_loss = epoch_loss / len(train_loader)
    elapsed = time.time() - start_time
    current_mem = torch.cuda.memory_allocated() / 1024**3

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["lr"].append(scheduler.get_last_lr()[0])
    history["gpu_mem_gb"].append(current_mem)

    wandb.log({"val/loss": val_loss, "epoch": epoch + 1, "epoch_time_min": elapsed / 60})

    print(f"\nepoch {epoch+1}/{train_cfg['num_epochs']} | "
          f"train: {train_loss:.4f} | val: {val_loss:.4f} | "
          f"time: {elapsed/60:.1f}min | gpu: {current_mem:.1f}GB")

    # SAVE LATEST MODEL EVERY EPOCH
    latest_path = os.path.join(paths["output_dir"], "latest_model")
    model.save_pretrained(latest_path)
    tokenizer.save_pretrained(latest_path)
    
    torch.save(optimizer.state_dict(), os.path.join(latest_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(latest_path, "scheduler.pt"))
    
    import json
    training_state = {
        "epoch": epoch + 1,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "history": history
    }
    with open(os.path.join(latest_path, "training_state.json"), "w") as f:
        json.dump(training_state, f)

    wandb.save(os.path.join(latest_path, "*"), base_path=paths["output_dir"], policy="now")
    print("  saved latest model, optimizer, scheduler, and state to wandb")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(paths["output_dir"], "best_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        wandb.save(os.path.join(save_path, "*"), base_path=paths["output_dir"], policy="now")

        print(f"  saved best model (val_loss={val_loss:.4f}) to wandb")

print("\ntraining complete")


# ==== CELL 6: evaluate ====

from src.evaluate import generate_predictions, evaluate_results, print_metrics

with open(data_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)
test_data = [e for e in all_data if e["split"] == "test"]
print(f"test samples: {len(test_data)}")

results = generate_predictions(
    model, processor, tokenizer, test_data, image_dir,
    max_new_tokens=cfg["evaluation"]["generation_max_length"],
)

metrics = evaluate_results(results)
print_metrics(metrics)


# ==== CELL 7: generate plots and save results ====

results_dir = Path(WORK_DIR) / "results_v8"
figures_dir = results_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

fig, ax1 = plt.subplots(figsize=(8, 5))
epochs_range = range(1, len(history["train_loss"]) + 1)

ax1.plot(epochs_range, history["train_loss"], "b-o", label="train loss", linewidth=2)
ax1.plot(epochs_range, history["val_loss"], "r-o", label="val loss", linewidth=2)
ax1.set_xlabel("epoch", fontsize=12)
ax1.set_ylabel("loss", fontsize=12)
ax1.set_title("training and validation loss", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(["exact match"], [metrics["closed_em"]], color="#2196F3", width=0.4)
axes[0].set_ylim(0, 1)
axes[0].set_title(f"closed-ended (n={metrics['closed_count']})", fontsize=13)
axes[0].set_ylabel("score", fontsize=12)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{bar.get_height():.3f}", ha="center", fontsize=11)

open_metrics = {
    "bleu-1": metrics["open_bleu_1"],
    "bleu-2": metrics["open_bleu_2"],
    "bleu-3": metrics["open_bleu_3"],
    "bleu-4": metrics["open_bleu_4"],
    "rouge-l": metrics["open_rouge_l"],
}
bars = axes[1].bar(open_metrics.keys(), open_metrics.values(), color="#4CAF50", width=0.5)
axes[1].set_ylim(0, 1)
axes[1].set_title(f"open-ended (n={metrics['open_count']})", fontsize=13)
axes[1].set_ylabel("score", fontsize=12)
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{bar.get_height():.3f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(figures_dir / "metrics_bar.png", dpi=150, bbox_inches="tight")
plt.show()

eval_output = {
    "metrics": metrics,
    "training_history": history,
    "config": cfg,
    "prompt_template": "vicuna_v1",
    "differential_lr": {"lora": LORA_LR, "projector": PROJECTOR_LR},
    "hardware": {
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "peak_memory_gb": max(history["gpu_mem_gb"]),
    },
    "predictions": results,
}

with open(results_dir / "eval_results.json", "w", encoding="utf-8") as f:
    json.dump(eval_output, f, indent=2, ensure_ascii=False)

training_summary = {
    "epochs": len(history["train_loss"]),
    "best_val_loss": best_val_loss,
    "final_train_loss": history["train_loss"][-1],
    "final_val_loss": history["val_loss"][-1],
    "training_history": history,
    "hardware": eval_output["hardware"],
}

with open(results_dir / "training_summary.json", "w", encoding="utf-8") as f:
    json.dump(training_summary, f, indent=2, ensure_ascii=False)

wandb.finish()
print(f"\nall results saved to {results_dir}")


# ==== CELL 8: push results to github ====

os.chdir(WORK_DIR)

os.system("git add results_v8/")
os.system('git status')

COMMIT_DATE = "2026-06-13T10:00:00+07:00"
commit_cmd = (
    f'git -c "user.name=dvydinh" -c "user.email=doanvy.dinh27@gmail.com" '
    f'commit -m "add v8 results (NEFTune, Epochs=10, LoRA_r=128)" '
    f'--date="{COMMIT_DATE}"'
)
os.environ["GIT_COMMITTER_DATE"] = COMMIT_DATE
os.system(commit_cmd)
del os.environ["GIT_COMMITTER_DATE"]

push_url = f"https://dvydinh:{GITHUB_TOKEN}@github.com/dvydinh/Medical-VQA.git"
os.system(f"git push {push_url} main")

print("\nresults pushed to github")
print("check: https://github.com/dvydinh/Medical-VQA/tree/main/results_v8")
