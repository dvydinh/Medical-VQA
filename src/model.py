"""
Model configuration with 4-bit NF4 quantization and LoRA injection.

Loads LLaVA-1.5-7B in quantized form to reduce VRAM usage,
then applies LoRA adapters to the attention projection layers.
"""

import yaml
import torch
from pathlib import Path
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_quantization_config(config_path: str = "configs/qlora_config.yaml"):
    """Load BitsAndBytes quantization config from yaml."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    quant_cfg = cfg["quantization"]
    compute_dtype = getattr(torch, quant_cfg["bnb_4bit_compute_dtype"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        llm_int8_skip_modules=["multi_modal_projector"],
    )

    return bnb_config


def load_lora_config(config_path: str = "configs/qlora_config.yaml"):
    """Load LoRA config from yaml."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    lora_cfg = cfg["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )

    return lora_config


def load_model(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    qlora_config_path: str = "configs/qlora_config.yaml",
    device_map: str = "auto",
):
    """
    Load LLaVA model with 4-bit quantization and LoRA adapters.

    Returns:
        model: PEFT-wrapped model ready for training
        processor: image processor
        tokenizer: text tokenizer
    """
    bnb_config = load_quantization_config(qlora_config_path)

    print(f"loading {model_name} with 4-bit quantization...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
    )

    # prepare for kbit training (freeze base, enable gradient checkpointing)
    model = prepare_model_for_kbit_training(model)

    # apply lora
    lora_config = load_lora_config(qlora_config_path)
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = True

    print_trainable_params(model)

    # load processor and tokenizer
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # CRITICAL FIX: enforce right padding for training so that labels[:prompt_len] = -100 correctly masks the prompt
    tokenizer.padding_side = "right"

    return model, processor, tokenizer


def print_trainable_params(model):
    """Print the number and percentage of trainable parameters."""
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    pct = 100 * trainable / total if total > 0 else 0
    print(
        f"trainable params: {trainable:,} / {total:,} ({pct:.2f}%)"
    )
