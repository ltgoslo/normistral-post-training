"""Scores texts with the model-based fluency scorer of the paper (Section 4.1).

The scorer is a NorMistral-11B sequence-classification model trained as a Bradley-Terry reward
model; it reads the text bidirectionally and outputs a single logit, which is sigmoid-normalized
into a fluency score. The paper averages the scores of 16 sampled responses per prompt. The fluency
model is not part of this repository; pass its path with --fluency_model.

Input files are the JSONL files of generate.py (scored field: --field, default "generated_text").
With --annotation_format, the inputs are in the format of outputs_for_annotation/annotation_set.jsonl
(a "responses" dict with one text per model) and every model is scored separately.
"""
import argparse
import json
import os
from statistics import mean

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROMPT_TEMPLATE = """Skriv om følgende tekst slik at den blir grammatisk korrekt:
{source_text}

Korreksjon:
{target_text}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Score the fluency of generated texts.")
    parser.add_argument("--fluency_model", type=str, required=True, help="Path of the fluency model.")
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="JSONL files to score.")
    parser.add_argument("--field", type=str, default="generated_text", help="Key of the scored text.")
    parser.add_argument("--annotation_format", action="store_true", help="Inputs have a `responses` dict with one text per model.")
    parser.add_argument("--output", type=str, default="fluency_scores.json", help="JSON file with the mean scores.")
    parser.add_argument("--sample_scores_dir", type=str, default=None, help="If given, the per-sample scores are written to this directory.")
    parser.add_argument("--max_chars", type=int, default=2048, help="Texts are truncated to this many characters.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    return parser.parse_args()


@torch.inference_mode()
def score_text(text: str, model, tokenizer, max_chars: int):
    """Returns the raw logit and the sigmoid-normalized fluency score of a text."""
    text = text.strip()[:max_chars]
    prompt = PROMPT_TEMPLATE.format(source_text=text, target_text=text)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    # an all-zero additive mask lets every token attend to every other token (bidirectional scorer)
    attention_mask = torch.zeros(1, 1, input_ids.size(1), input_ids.size(1), dtype=model.dtype, device=model.device)

    logit = model(input_ids, attention_mask=attention_mask).logits.squeeze().float()
    return logit.item(), F.sigmoid(logit).item()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.fluency_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.fluency_model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="cuda"
    )
    model.eval()
    if args.sample_scores_dir is not None:
        os.makedirs(args.sample_scores_dir, exist_ok=True)

    all_scores = {}
    for path in args.inputs:
        print(f"Processing {path}...", flush=True)
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

        sample_scores = []
        for row in tqdm(rows):
            if args.annotation_format:
                sample_scores.append({name: score_text(text, model, tokenizer, args.max_chars) for name, text in row["responses"].items()})
            else:
                sample_scores.append(score_text(row[args.field], model, tokenizer, args.max_chars))

        if args.annotation_format:
            names = sorted({name for scores in sample_scores for name in scores})
            all_scores[path] = {
                name: {
                    "mean_logit": mean(s[name][0] for s in sample_scores if name in s),
                    "mean_score": mean(s[name][1] for s in sample_scores if name in s),
                    "n": sum(name in s for s in sample_scores),
                }
                for name in names
            }
        else:
            all_scores[path] = {
                "mean_logit": mean(logit for logit, _ in sample_scores),
                "mean_score": mean(score for _, score in sample_scores),
                "n": len(sample_scores),
            }
        print(f"Fluency of {path}: {json.dumps(all_scores[path])}", flush=True)

        if args.sample_scores_dir is not None:
            with open(os.path.join(args.sample_scores_dir, os.path.basename(path)), "w") as f:
                for scores in sample_scores:
                    f.write(json.dumps(scores) + "\n")

    with open(args.output, "w") as f:
        json.dump(all_scores, f, indent=4)


if __name__ == "__main__":
    main()
