# Medical VQA with Parameter-Efficient Fine-Tuning

Fine-tuning a Vision-Language Model (LLaVA-1.5-7B) for Medical Visual Question Answering on the VQA-RAD dataset using QLoRA.

## Overview

This project explores parameter-efficient fine-tuning (PEFT) of large vision-language models for clinical radiology VQA. By applying 4-bit quantization (NF4) and Low-Rank Adaptation (LoRA), we aim to reduce GPU memory requirements while maintaining competitive accuracy on medical image understanding tasks.

## Dataset

- **VQA-RAD** (Lau et al., 2018): 315 radiology images with 3,515 clinician-generated question-answer pairs
- Modalities: X-ray, CT, MRI
- Question types: closed-ended (yes/no) and open-ended

## Project Structure

```
├── data/
│   ├── raw/                  # original dataset (not tracked by git)
│   └── processed/            # cleaned and tokenized data
├── configs/                  # yaml config files
├── src/                      # core source code
│   ├── dataset.py            # data loading and preprocessing
│   ├── model.py              # model setup with quantization and LoRA
│   ├── train.py              # training pipeline
│   └── evaluate.py           # evaluation metrics
├── scripts/                  # utility scripts
├── reports/latex_report/     # IEEE format research report
├── requirements.txt
└── README.md
```

## Setup

```bash
# create virtual environment
python -m venv venv
source venv/bin/activate  # linux/mac
# venv\Scripts\activate   # windows

# install dependencies
pip install -r requirements.txt

# download dataset
python scripts/download_data.py

# validate dataset integrity
python scripts/validate_data.py
```

## Training

```bash
python -m src.train --config configs/training_config.yaml
```

## Evaluation

```bash
python -m src.evaluate --checkpoint <path_to_checkpoint>
```

## References

Lau, J. J., Gayen, S., Ben Abacha, A., & Demner-Fushman, D. (2018). A dataset of clinically generated visual questions and answers about radiology images. *Scientific Data*, 5(1), 180251.
