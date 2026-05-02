"""
Evaluation engine for Medical VQA.

Metrics:
  - Exact Match (EM) for closed-ended questions (yes/no)
  - BLEU-1 to BLEU-4 for open-ended questions
  - ROUGE-L for open-ended questions
"""

import argparse
import json
import os
from collections import defaultdict

import nltk
import torch
import yaml
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm

from src.model import load_model


# make sure punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def normalize_answer(text: str) -> str:
    """Lowercase, strip whitespace and trailing periods."""
    return text.strip().lower().rstrip(".")


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Exact match accuracy for closed-ended questions."""
    if not predictions:
        return 0.0

    correct = sum(
        1 for pred, ref in zip(predictions, references)
        if normalize_answer(pred) == normalize_answer(ref)
    )
    return correct / len(predictions)


def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU-1 to BLEU-4 scores."""
    smoother = SmoothingFunction().method1

    bleu_scores = {f"bleu_{i}": 0.0 for i in range(1, 5)}
    weights_map = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (0.33, 0.33, 0.33, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }

    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(normalize_answer(pred))
        ref_tokens = nltk.word_tokenize(normalize_answer(ref))

        for key, weights in weights_map.items():
            score = sentence_bleu(
                [ref_tokens], pred_tokens,
                weights=weights,
                smoothing_function=smoother,
            )
            bleu_scores[key] += score

    n = len(predictions) if predictions else 1
    return {k: v / n for k, v in bleu_scores.items()}


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Compute average ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total = 0.0

    for pred, ref in zip(predictions, references):
        scores = scorer.score(normalize_answer(ref), normalize_answer(pred))
        total += scores["rougeL"].fmeasure

    return total / len(predictions) if predictions else 0.0


def classify_question_type(answer: str) -> str:
    """Classify as closed or open based on the reference answer."""
    norm = normalize_answer(answer)
    if norm in ("yes", "no"):
        return "closed"
    return "open"


@torch.no_grad()
def generate_predictions(model, processor, tokenizer, data, image_dir, max_new_tokens=128):
    """Generate model predictions for all samples."""
    from PIL import Image

    model.eval()
    results = []

    for entry in tqdm(data, desc="generating predictions"):
        img_path = os.path.join(image_dir, entry["image_name"])
        image = Image.open(img_path).convert("RGB")

        prompt = f"<image>\nQuestion: {entry['question'].strip()}\nAnswer:"

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
        )

        # decode only the generated part
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        results.append({
            "qid": entry["qid"],
            "question": entry["question"],
            "reference": entry["answer"],
            "prediction": prediction,
            "answer_type": classify_question_type(entry["answer"]),
        })

    return results


def evaluate_results(results: list[dict]) -> dict:
    """Compute all metrics grouped by answer type."""

    # split by type
    closed_preds, closed_refs = [], []
    open_preds, open_refs = [], []

    for r in results:
        if r["answer_type"] == "closed":
            closed_preds.append(r["prediction"])
            closed_refs.append(r["reference"])
        else:
            open_preds.append(r["prediction"])
            open_refs.append(r["reference"])

    metrics = {}

    # closed-ended metrics
    metrics["closed_em"] = compute_exact_match(closed_preds, closed_refs)
    metrics["closed_count"] = len(closed_preds)

    # open-ended metrics
    bleu = compute_bleu(open_preds, open_refs)
    metrics.update({f"open_{k}": v for k, v in bleu.items()})
    metrics["open_rouge_l"] = compute_rouge_l(open_preds, open_refs)
    metrics["open_count"] = len(open_preds)

    # overall
    metrics["total_count"] = len(results)

    return metrics


def print_metrics(metrics: dict):
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 50)
    print("evaluation results")
    print("=" * 50)

    print(f"\nclosed-ended ({metrics['closed_count']} samples):")
    print(f"  exact match: {metrics['closed_em']:.4f}")

    print(f"\nopen-ended ({metrics['open_count']} samples):")
    print(f"  bleu-1: {metrics['open_bleu_1']:.4f}")
    print(f"  bleu-2: {metrics['open_bleu_2']:.4f}")
    print(f"  bleu-3: {metrics['open_bleu_3']:.4f}")
    print(f"  bleu-4: {metrics['open_bleu_4']:.4f}")
    print(f"  rouge-l: {metrics['open_rouge_l']:.4f}")

    print(f"\ntotal samples: {metrics['total_count']}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to saved model checkpoint")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--qlora_config", default="configs/qlora_config.yaml")
    parser.add_argument("--output", default="outputs/eval_results.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    paths = cfg["paths"]

    # load model from checkpoint
    model, processor, tokenizer = load_model(
        model_name=cfg["model"]["name"],
        qlora_config_path=args.qlora_config,
    )

    # load adapter weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.checkpoint)

    # load test data
    data_path = os.path.join(paths["data_dir"], "VQA_RAD_Dataset.json")
    image_dir = os.path.join(paths["data_dir"], "images")

    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    test_data = [entry for entry in all_data if entry["split"] == "test"]
    print(f"test samples: {len(test_data)}")

    # generate and evaluate
    results = generate_predictions(
        model, processor, tokenizer, test_data, image_dir,
        max_new_tokens=cfg["evaluation"]["generation_max_length"],
    )

    metrics = evaluate_results(results)
    print_metrics(metrics)

    # save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {"metrics": metrics, "predictions": results}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nresults saved to {args.output}")


if __name__ == "__main__":
    main()
