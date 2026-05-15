"""
Medical VQA - Resume Evaluation from WandB on Kaggle (T4 x2)

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
for i in range(torch.cuda.device_count()):
    print(f"  gpu {i}: {torch.cuda.get_device_name(i)}")

# ==== CELL 3: clone repo and download dataset ====

REPO_URL = f"https://dvydinh:{GITHUB_TOKEN}@github.com/dvydinh/Medical-VQA.git"
WORK_DIR = "/kaggle/working/Medical-VQA"

os.system(f"git clone {REPO_URL} {WORK_DIR}")
os.chdir(WORK_DIR)

os.system('git config user.name "dvydinh"')
os.system('git config user.email "doanvy.dinh27@gmail.com"')

os.system("python scripts/download_data.py")
os.system("python scripts/validate_data.py")

# ==== CELL 4: download best model from wandb ====

import wandb
import yaml

with open("configs/training_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

output_dir = cfg["paths"]["output_dir"]
best_model_dir = os.path.join(output_dir, "best_model")
os.makedirs(best_model_dir, exist_ok=True)

api = wandb.Api()
run = api.run("dvydinh/medicalVQA/vq6uv8at")

print("Downloading best_model from WandB...")
run.file("best_model/adapter_model.safetensors").download(root=WORK_DIR, replace=True)
run.file("best_model/adapter_config.json").download(root=WORK_DIR, replace=True)
run.file("best_model/special_tokens_map.json").download(root=WORK_DIR, replace=True)
run.file("best_model/tokenizer.json").download(root=WORK_DIR, replace=True)
run.file("best_model/tokenizer_config.json").download(root=WORK_DIR, replace=True)

print("Download complete.")

# ==== CELL 5: evaluate ====

import sys
sys.path.insert(0, WORK_DIR)

from src.model import load_model
from src.evaluate import generate_predictions, evaluate_results, print_metrics
from peft import PeftModel

data_path = os.path.join(cfg["paths"]["data_dir"], "VQA_RAD_Dataset.json")
image_dir = os.path.join(cfg["paths"]["data_dir"], "images")

with open(data_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)
test_data = [e for e in all_data if e["split"] == "test"]
print(f"test samples: {len(test_data)}")

base_model, processor, tokenizer = load_model(
    model_name=cfg["model"]["name"],
    qlora_config_path="configs/qlora_config.yaml",
)

# load the downloaded lora adapter
model = PeftModel.from_pretrained(base_model, best_model_dir)

results = generate_predictions(
    model, processor, tokenizer, test_data, image_dir,
    max_new_tokens=cfg["evaluation"]["generation_max_length"],
)

metrics = evaluate_results(results)
print_metrics(metrics)

# ==== CELL 6: generate plots and save results ====

import matplotlib.pyplot as plt

results_dir = Path(WORK_DIR) / "results_v3"
figures_dir = results_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# closed-ended
axes[0].bar(["exact match"], [metrics["closed_em"]], color="#2196F3", width=0.4)
axes[0].set_ylim(0, 1)
axes[0].set_title(f"closed-ended (n={metrics['closed_count']})", fontsize=13)
axes[0].set_ylabel("score", fontsize=12)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{bar.get_height():.3f}", ha="center", fontsize=11)

# open-ended
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
    "version": "v3_resumed",
    "metrics": metrics,
    "predictions": results,
}

with open(results_dir / "eval_results.json", "w", encoding="utf-8") as f:
    json.dump(eval_output, f, indent=2, ensure_ascii=False)
print(f"saved to {results_dir}")

# ==== CELL 7: push results to github ====

os.chdir(WORK_DIR)

os.system("git add results_v3/")

COMMIT_DATE = "2026-05-15T10:00:00+07:00"
commit_cmd = (
    f'git -c "user.name=dvydinh" -c "user.email=doanvy.dinh27@gmail.com" '
    f'commit -m "add evaluation results v3" '
    f'--date="{COMMIT_DATE}"'
)
os.environ["GIT_COMMITTER_DATE"] = COMMIT_DATE
os.system(commit_cmd)
del os.environ["GIT_COMMITTER_DATE"]

push_url = f"https://dvydinh:{GITHUB_TOKEN}@github.com/dvydinh/Medical-VQA.git"
os.system(f"git push {push_url} main")

print("results pushed to github results_v3 folder")
